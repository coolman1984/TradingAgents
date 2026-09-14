"""Deterministic EGX portfolio planning engine.

Language models may produce dimension scores, but cannot bypass evidence,
freshness, Sharia, diversification, or action thresholds enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tradingagents.portfolio.mandate import (
    DEFAULT_EGX_BALANCED_MANDATE,
    PortfolioMandate,
)
from tradingagents.portfolio.models import (
    AdvisoryAction,
    AnalysisDimension,
    PlanAction,
    PortfolioPlan,
    PortfolioSnapshot,
    SecurityAssessment,
)

_DIMENSION_WEIGHTS = {
    AnalysisDimension.FUNDAMENTAL: 0.25,
    AnalysisDimension.VALUATION: 0.15,
    AnalysisDimension.TECHNICAL: 0.15,
    AnalysisDimension.CATALYST: 0.15,
    AnalysisDimension.LIQUIDITY: 0.15,
    AnalysisDimension.RISK: 0.15,
}
_REQUIRED_DIMENSIONS = frozenset(
    {
        AnalysisDimension.FUNDAMENTAL,
        AnalysisDimension.LIQUIDITY,
        AnalysisDimension.RISK,
    }
)


@dataclass(frozen=True)
class ScoredSecurity:
    assessment: SecurityAssessment
    score: float | None
    confidence: float
    eligible: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class EnginePolicy:
    minimum_buy_score: float = 65.0
    exit_score: float = 40.0
    minimum_confidence: float = 0.60
    minimum_replacement_edge: float = 15.0
    max_dimension_age_days: int = 120
    max_sharia_age_days: int = 220
    rebalance_tolerance: float = 0.03


DEFAULT_ENGINE_POLICY = EnginePolicy()


def _days_old(as_of: date, analysis_date: date) -> int:
    return (analysis_date - as_of).days


def score_security(
    assessment: SecurityAssessment,
    analysis_date: date,
    mandate: PortfolioMandate = DEFAULT_EGX_BALANCED_MANDATE,
    policy: EnginePolicy = DEFAULT_ENGINE_POLICY,
) -> ScoredSecurity:
    issues: list[str] = []

    if assessment.sharia_as_of > analysis_date:
        issues.append("Sharia evidence is from the future")
    elif _days_old(assessment.sharia_as_of, analysis_date) > policy.max_sharia_age_days:
        issues.append("Sharia classification is stale")

    if assessment.sharia_tier not in mandate.allowed_sharia_tiers:
        issues.append(f"Sharia tier {assessment.sharia_tier.value} is not buy-eligible")

    by_dimension = {item.dimension: item for item in assessment.dimensions}
    missing_required = _REQUIRED_DIMENSIONS - by_dimension.keys()
    if missing_required:
        labels = ", ".join(sorted(item.value for item in missing_required))
        issues.append(f"Missing required dimensions: {labels}")

    usable = {}
    for dimension, item in by_dimension.items():
        if item.as_of > analysis_date:
            issues.append(f"{dimension.value} assessment is from the future")
            continue
        if _days_old(item.as_of, analysis_date) > policy.max_dimension_age_days:
            issues.append(f"{dimension.value} assessment is stale")
            continue
        if any(ref.published_on > analysis_date for ref in item.evidence):
            issues.append(f"{dimension.value} evidence includes a future publication")
            continue
        usable[dimension] = item

    if _REQUIRED_DIMENSIONS - usable.keys():
        issues.append("One or more required dimensions have no usable point-in-time evidence")

    available_weight = sum(_DIMENSION_WEIGHTS[key] for key in usable)
    if available_weight == 0:
        return ScoredSecurity(assessment, None, 0.0, False, tuple(dict.fromkeys(issues)))

    score = sum(
        item.score * _DIMENSION_WEIGHTS[dimension]
        for dimension, item in usable.items()
    ) / available_weight
    confidence = (
        sum(
            item.confidence * _DIMENSION_WEIGHTS[dimension]
            for dimension, item in usable.items()
        )
        / available_weight
        * available_weight
    )

    if confidence < policy.minimum_confidence:
        issues.append(
            f"Confidence {confidence:.0%} is below {policy.minimum_confidence:.0%}"
        )

    eligible = (
        not issues
        and score >= policy.minimum_buy_score
        and confidence >= policy.minimum_confidence
    )
    if score < policy.minimum_buy_score:
        issues.append(f"Score {score:.1f} is below buy threshold {policy.minimum_buy_score:.1f}")

    return ScoredSecurity(
        assessment=assessment,
        score=round(score, 2),
        confidence=round(confidence, 4),
        eligible=eligible,
        issues=tuple(dict.fromkeys(issues)),
    )


def _select_diversified(
    scored: list[ScoredSecurity],
    mandate: PortfolioMandate,
) -> list[ScoredSecurity]:
    ranked = sorted(
        (item for item in scored if item.eligible and item.score is not None),
        key=lambda item: (item.score, item.confidence, item.assessment.ticker),
        reverse=True,
    )
    selected: list[ScoredSecurity] = []
    sector_counts: dict[str, int] = {}

    # Equal weighting is intentional for the small starting portfolio.  With
    # four positions at 20% each, max_sector_weight=40% permits at most two
    # names from one sector.
    target_count = mandate.min_positions
    target_weight = (1 - mandate.max_cash_weight) / target_count
    max_names_per_sector = max(1, int(mandate.max_sector_weight / target_weight))

    for item in ranked:
        sector = item.assessment.sector
        if sector_counts.get(sector, 0) >= max_names_per_sector:
            continue
        selected.append(item)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) == target_count:
            break
    return selected


def build_portfolio_plan(
    snapshot: PortfolioSnapshot,
    mandate: PortfolioMandate = DEFAULT_EGX_BALANCED_MANDATE,
    policy: EnginePolicy = DEFAULT_ENGINE_POLICY,
) -> PortfolioPlan:
    scored = [
        score_security(item, snapshot.analysis_date, mandate, policy)
        for item in snapshot.candidates
    ]
    selected = _select_diversified(scored, mandate)
    by_ticker = {item.assessment.ticker: item for item in scored}
    selected_by_ticker = {item.assessment.ticker: item for item in selected}

    total_value = snapshot.cash_egp + sum(
        item.current_value_egp for item in snapshot.positions
    )
    blocked: list[str] = []
    warnings: list[str] = []

    for item in scored:
        if item.issues:
            warnings.append(
                f"{item.assessment.ticker}: " + "; ".join(item.issues)
            )

    if len(selected) < mandate.min_positions:
        blocked.append(
            f"Only {len(selected)} eligible diversified candidates; "
            f"{mandate.min_positions} are required"
        )
        keep_cash = PlanAction(
            ticker=None,
            action=AdvisoryAction.KEEP_CASH,
            target_weight=1.0,
            value_change_egp=0,
            confidence=1.0,
            reasons=("Insufficient verified candidates; do not force investment",),
        )
        return PortfolioPlan(
            analysis_date=snapshot.analysis_date,
            investable_value_egp=total_value,
            target_cash_weight=1.0,
            actions=(keep_cash,),
            monthly_contribution_action=keep_cash,
            blocked_reasons=tuple(blocked),
            warnings=tuple(warnings),
        )

    target_cash = mandate.max_cash_weight
    target_weight = (1 - target_cash) / len(selected)
    current = {item.ticker: item for item in snapshot.positions}
    actions: list[PlanAction] = []

    for ticker, position in current.items():
        item = by_ticker.get(ticker)
        current_weight = position.current_value_egp / total_value if total_value else 0

        if item is None or item.score is None:
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=AdvisoryAction.HOLD,
                    target_weight=current_weight,
                    value_change_egp=0,
                    confidence=0,
                    reasons=("No verified current assessment; manual review required",),
                )
            )
            continue

        if ticker in selected_by_ticker:
            delta = target_weight - current_weight
            if delta > policy.rebalance_tolerance:
                action = AdvisoryAction.ADD
            elif delta < -policy.rebalance_tolerance:
                action = AdvisoryAction.REDUCE
            else:
                action = AdvisoryAction.HOLD
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=action,
                    target_weight=target_weight,
                    value_change_egp=round(delta * total_value, 2),
                    score=item.score,
                    confidence=item.confidence,
                    reasons=("Selected by verified score and diversification rules",),
                )
            )
            continue

        replacement = next(
            (
                candidate
                for candidate in selected
                if candidate.assessment.ticker not in current
                and candidate.score is not None
                and item.score is not None
                and candidate.score - item.score >= policy.minimum_replacement_edge
            ),
            None,
        )
        if replacement and snapshot.estimated_switch_cost_pct is not None:
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=AdvisoryAction.REPLACE,
                    target_weight=0,
                    value_change_egp=-position.current_value_egp,
                    score=item.score,
                    confidence=min(item.confidence, replacement.confidence),
                    replacement_ticker=replacement.assessment.ticker,
                    reasons=(
                        f"Replacement score edge is "
                        f"{replacement.score - item.score:.1f} points",
                        f"Estimated switch cost supplied: "
                        f"{snapshot.estimated_switch_cost_pct:.2%}",
                    ),
                )
            )
        elif item.score < policy.exit_score:
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=AdvisoryAction.SELL,
                    target_weight=0,
                    value_change_egp=-position.current_value_egp,
                    score=item.score,
                    confidence=item.confidence,
                    reasons=("Verified score fell below the exit threshold",),
                )
            )
        else:
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=AdvisoryAction.HOLD,
                    target_weight=current_weight,
                    value_change_egp=0,
                    score=item.score,
                    confidence=item.confidence,
                    reasons=(
                        "No cost-supported replacement edge; avoid unnecessary churn",
                    ),
                )
            )

    for item in selected:
        ticker = item.assessment.ticker
        if ticker in current:
            continue
        value = round(target_weight * total_value, 2)
        actions.append(
            PlanAction(
                ticker=ticker,
                action=AdvisoryAction.BUY,
                target_weight=target_weight,
                value_change_egp=value,
                score=item.score,
                confidence=item.confidence,
                reasons=("Eligible, high-ranked, and fits sector concentration limits",),
            )
        )

    projected_total = total_value + mandate.monthly_contribution_egp
    underweights = []
    for item in selected:
        held = current.get(item.assessment.ticker)
        held_value = held.current_value_egp if held else 0
        deficit = target_weight * projected_total - held_value
        underweights.append((deficit, item))

    best_deficit, best = max(underweights, key=lambda pair: pair[0])
    if best_deficit > 0:
        monthly_action = PlanAction(
            ticker=best.assessment.ticker,
            action=AdvisoryAction.ADD if best.assessment.ticker in current else AdvisoryAction.BUY,
            target_weight=target_weight,
            value_change_egp=min(mandate.monthly_contribution_egp, round(best_deficit, 2)),
            score=best.score,
            confidence=best.confidence,
            reasons=("Largest verified gap below target allocation",),
        )
    else:
        monthly_action = PlanAction(
            ticker=None,
            action=AdvisoryAction.KEEP_CASH,
            target_weight=target_cash,
            value_change_egp=mandate.monthly_contribution_egp,
            confidence=1.0,
            reasons=("All selected positions are at or above target",),
        )

    return PortfolioPlan(
        analysis_date=snapshot.analysis_date,
        investable_value_egp=total_value,
        target_cash_weight=target_cash,
        actions=tuple(actions),
        monthly_contribution_action=monthly_action,
        blocked_reasons=tuple(blocked),
        warnings=tuple(warnings),
    )
