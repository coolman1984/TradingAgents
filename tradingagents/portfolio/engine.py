"""Deterministic EGX portfolio planning engine.

Language models may produce dimension scores, but cannot bypass evidence,
freshness, Sharia, diversification, or action thresholds enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tradingagents.markets.egypt_sources import (
    EgyptSourceKey,
    evaluate_source_coverage,
    missing_required_sources,
    validate_evidence_source,
)
from tradingagents.portfolio.mandate import (
    DEFAULT_EGX_BALANCED_MANDATE,
    PortfolioMandate,
    ShariaTier,
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
_SHARIA_PRIORITY = {
    ShariaTier.OFFICIAL_INDEX: 2,
    ShariaTier.INDEPENDENTLY_REVIEWED: 1,
    ShariaTier.REVIEW_REQUIRED: 0,
    ShariaTier.EXCLUDED: -1,
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
    decision_ready: bool
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
    """Score one security and keep data failures separate from policy failures."""
    data_issues: list[str] = []
    policy_issues: list[str] = []

    expected_sharia_source = None
    if assessment.sharia_tier is ShariaTier.OFFICIAL_INDEX:
        expected_sharia_source = EgyptSourceKey.EGX_SHARIA_CONSTITUENTS
    elif assessment.sharia_tier is ShariaTier.INDEPENDENTLY_REVIEWED:
        expected_sharia_source = EgyptSourceKey.INDEPENDENT_SHARIA_REVIEW

    sharia_source_issues = validate_evidence_source(
        assessment.sharia_evidence,
        expected_source=expected_sharia_source,
    )
    if not assessment.sharia_evidence.content_hash:
        sharia_source_issues = (
            *sharia_source_issues,
            "Evidence content hash is required",
        )
    data_issues.extend(
        f"Sharia evidence: {issue}" for issue in sharia_source_issues
    )

    sharia_date = assessment.sharia_evidence.published_on
    if assessment.sharia_evidence.observed_at.date() > analysis_date:
        data_issues.append("Sharia evidence was not known on analysis date")
    if sharia_date > analysis_date:
        data_issues.append("Sharia evidence is from the future")
    elif _days_old(sharia_date, analysis_date) > policy.max_sharia_age_days:
        data_issues.append("Sharia classification is stale")

    if assessment.sharia_tier not in mandate.allowed_sharia_tiers:
        policy_issues.append(
            f"Sharia tier {assessment.sharia_tier.value} is not buy-eligible"
        )

    by_dimension = {item.dimension: item for item in assessment.dimensions}
    missing_required = _REQUIRED_DIMENSIONS - by_dimension.keys()
    if missing_required:
        labels = ", ".join(sorted(item.value for item in missing_required))
        data_issues.append(f"Missing required dimensions: {labels}")

    usable = {}
    for dimension, item in by_dimension.items():
        if item.as_of > analysis_date:
            data_issues.append(f"{dimension.value} assessment is from the future")
            continue
        if _days_old(item.as_of, analysis_date) > policy.max_dimension_age_days:
            data_issues.append(f"{dimension.value} assessment is stale")
            continue
        if any(ref.published_on > analysis_date for ref in item.evidence):
            data_issues.append(
                f"{dimension.value} evidence includes a future publication"
            )
            continue
        if any(ref.observed_at.date() > analysis_date for ref in item.evidence):
            data_issues.append(
                f"{dimension.value} evidence was not known on analysis date"
            )
            continue
        source_issues = tuple(
            issue
            for ref in item.evidence
            for issue in (
                *validate_evidence_source(ref),
                *(() if ref.content_hash else ("Evidence content hash is required",)),
            )
        )
        if source_issues:
            data_issues.extend(
                f"{dimension.value} evidence: {issue}" for issue in source_issues
            )
            continue
        usable[dimension] = item

    if _REQUIRED_DIMENSIONS - usable.keys():
        data_issues.append(
            "One or more required dimensions have no usable point-in-time evidence"
        )

    available_weight = sum(_DIMENSION_WEIGHTS[key] for key in usable)
    if available_weight == 0:
        issues = tuple(dict.fromkeys([*data_issues, *policy_issues]))
        return ScoredSecurity(assessment, None, 0.0, False, False, issues)

    score = sum(
        item.score * _DIMENSION_WEIGHTS[dimension]
        for dimension, item in usable.items()
    ) / available_weight

    # This is the weighted confidence multiplied by coverage.  Missing optional
    # dimensions therefore reduce confidence rather than making a sparse score
    # look as trustworthy as a complete one.
    confidence = sum(
        item.confidence * _DIMENSION_WEIGHTS[dimension]
        for dimension, item in usable.items()
    )
    if confidence < policy.minimum_confidence:
        data_issues.append(
            f"Confidence {confidence:.0%} is below {policy.minimum_confidence:.0%}"
        )

    if score < policy.minimum_buy_score:
        policy_issues.append(
            f"Score {score:.1f} is below buy threshold {policy.minimum_buy_score:.1f}"
        )

    decision_ready = not data_issues
    eligible = (
        decision_ready
        and assessment.sharia_tier in mandate.allowed_sharia_tiers
        and score >= policy.minimum_buy_score
    )
    issues = tuple(dict.fromkeys([*data_issues, *policy_issues]))
    return ScoredSecurity(
        assessment=assessment,
        score=round(score, 2),
        confidence=round(confidence, 4),
        decision_ready=decision_ready,
        eligible=eligible,
        issues=issues,
    )


def _select_diversified(
    scored: list[ScoredSecurity],
    mandate: PortfolioMandate,
) -> list[ScoredSecurity]:
    ranked = sorted(
        (item for item in scored if item.eligible and item.score is not None),
        key=lambda item: (
            _SHARIA_PRIORITY[item.assessment.sharia_tier],
            item.score,
            item.confidence,
            item.assessment.ticker,
        ),
        reverse=True,
    )
    selected: list[ScoredSecurity] = []
    sector_counts: dict[str, int] = {}

    target_count = mandate.min_positions
    equal_weight = (1 - mandate.max_cash_weight) / target_count
    max_names_per_sector = max(1, int(mandate.max_sector_weight / equal_weight))

    for item in ranked:
        sector = item.assessment.sector
        if sector_counts.get(sector, 0) >= max_names_per_sector:
            continue
        selected.append(item)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) == target_count:
            break
    return selected


def _keep_cash_plan(
    snapshot: PortfolioSnapshot,
    total_value: float,
    blocked: list[str],
    warnings: list[str],
    mandate: PortfolioMandate,
) -> PortfolioPlan:
    keep_cash = PlanAction(
        ticker=None,
        action=AdvisoryAction.KEEP_CASH,
        target_weight=1.0,
        value_change_egp=0,
        confidence=1.0,
        reasons=("Insufficient verified candidates; do not force investment",),
    )
    monthly_cash = PlanAction(
        ticker=None,
        action=AdvisoryAction.KEEP_CASH,
        target_weight=1.0,
        value_change_egp=mandate.monthly_contribution_egp,
        confidence=1.0,
        reasons=("Keep the monthly contribution liquid until evidence is complete",),
    )
    return PortfolioPlan(
        analysis_date=snapshot.analysis_date,
        investable_value_egp=total_value,
        target_cash_weight=1.0,
        actions=(keep_cash,),
        monthly_contribution_egp=mandate.monthly_contribution_egp,
        monthly_contribution_actions=(monthly_cash,),
        blocked_reasons=tuple(blocked),
        warnings=tuple(warnings),
    )


def build_portfolio_plan(
    snapshot: PortfolioSnapshot,
    mandate: PortfolioMandate = DEFAULT_EGX_BALANCED_MANDATE,
    policy: EnginePolicy = DEFAULT_ENGINE_POLICY,
) -> PortfolioPlan:
    """Build a complete advisory plan whose target weights remain internally valid."""
    scored = [
        score_security(item, snapshot.analysis_date, mandate, policy)
        for item in snapshot.candidates
    ]
    selected = _select_diversified(scored, mandate)
    by_ticker = {item.assessment.ticker: item for item in scored}
    selected_by_ticker = {item.assessment.ticker: item for item in selected}
    current = {item.ticker: item for item in snapshot.positions}

    total_value = snapshot.cash_egp + sum(
        item.current_value_egp for item in snapshot.positions
    )
    blocked: list[str] = []
    warnings: list[str] = []

    all_evidence = list(snapshot.market_evidence)
    for assessment in snapshot.candidates:
        all_evidence.append(assessment.sharia_evidence)
        for dimension in assessment.dimensions:
            all_evidence.extend(dimension.evidence)

    valid_sources, coverage_issues = evaluate_source_coverage(
        all_evidence,
        snapshot.analysis_date,
    )
    warnings.extend(coverage_issues)
    missing_sources = missing_required_sources(set(valid_sources))
    if missing_sources:
        blocked.append(
            "Missing fresh required Egyptian sources: " + ", ".join(missing_sources)
        )
        return _keep_cash_plan(snapshot, total_value, blocked, warnings, mandate)

    for item in scored:
        if item.issues:
            warnings.append(f"{item.assessment.ticker}: " + "; ".join(item.issues))

    if len(selected) < mandate.min_positions:
        blocked.append(
            f"Only {len(selected)} eligible diversified candidates; "
            f"{mandate.min_positions} are required"
        )
        return _keep_cash_plan(snapshot, total_value, blocked, warnings, mandate)

    # Decide which existing positions are locked, selected, sold, or replaced
    # before calculating targets.  This prevents buy actions from spending cash
    # already trapped in a HOLD position.
    disposition: dict[str, tuple[AdvisoryAction, ScoredSecurity | None]] = {}
    replacement_pool = [
        item for item in selected if item.assessment.ticker not in current
    ]
    reserved_replacements: set[str] = set()

    for ticker in sorted(current):
        item = by_ticker.get(ticker)
        if item is None or item.score is None or not item.decision_ready:
            disposition[ticker] = (AdvisoryAction.HOLD, None)
            warnings.append(f"{ticker}: position locked pending verified fresh data")
            continue

        if ticker in selected_by_ticker:
            disposition[ticker] = (AdvisoryAction.HOLD, None)
            continue

        if item.assessment.sharia_tier not in mandate.allowed_sharia_tiers:
            disposition[ticker] = (AdvisoryAction.SELL, None)
            continue
        if item.score < policy.exit_score:
            disposition[ticker] = (AdvisoryAction.SELL, None)
            continue

        replacement = next(
            (
                candidate
                for candidate in replacement_pool
                if candidate.assessment.ticker not in reserved_replacements
                and candidate.score is not None
                and candidate.score - item.score >= policy.minimum_replacement_edge
            ),
            None,
        )
        if replacement is not None and snapshot.estimated_switch_cost_pct is not None:
            reserved_replacements.add(replacement.assessment.ticker)
            disposition[ticker] = (AdvisoryAction.REPLACE, replacement)
        else:
            disposition[ticker] = (AdvisoryAction.HOLD, None)
            if replacement is not None:
                warnings.append(
                    f"{ticker}: replacement suppressed because switch cost is missing"
                )

    locked_value = sum(
        current[ticker].current_value_egp
        for ticker, (action, _) in disposition.items()
        if action is AdvisoryAction.HOLD and ticker not in selected_by_ticker
    )
    locked_weight = locked_value / total_value if total_value else 0
    desired_cash = mandate.max_cash_weight
    if locked_weight + desired_cash > 1:
        warnings.append("Locked positions leave no room for the normal cash reserve")
        desired_cash = max(0.0, 1 - locked_weight)

    available_for_selected = max(0.0, 1 - desired_cash - locked_weight)
    selected_target = min(
        mandate.max_position_weight,
        available_for_selected / len(selected),
    )
    actual_target_cash = max(
        desired_cash,
        1 - locked_weight - selected_target * len(selected),
    )

    actions: list[PlanAction] = []
    for ticker in sorted(current):
        position = current[ticker]
        item = by_ticker.get(ticker)
        action, replacement = disposition[ticker]
        current_weight = position.current_value_egp / total_value if total_value else 0

        if ticker in selected_by_ticker:
            chosen = selected_by_ticker[ticker]
            delta = selected_target - current_weight
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
                    target_weight=selected_target,
                    value_change_egp=round(delta * total_value, 2),
                    score=chosen.score,
                    confidence=chosen.confidence,
                    reasons=("Selected by verified score and diversification rules",),
                )
            )
            continue

        if action is AdvisoryAction.REPLACE and replacement is not None:
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=action,
                    target_weight=0,
                    value_change_egp=-position.current_value_egp,
                    score=item.score if item else None,
                    confidence=min(
                        item.confidence if item else 0,
                        replacement.confidence,
                    ),
                    replacement_ticker=replacement.assessment.ticker,
                    reasons=(
                        f"Replacement score edge is "
                        f"{replacement.score - item.score:.1f} points",
                        f"Estimated switch cost supplied: "
                        f"{snapshot.estimated_switch_cost_pct:.2%}",
                    ),
                )
            )
        elif action is AdvisoryAction.SELL:
            reason = (
                "Fresh Sharia classification is not buy-eligible"
                if item
                and item.assessment.sharia_tier not in mandate.allowed_sharia_tiers
                else "Verified score fell below the exit threshold"
            )
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=action,
                    target_weight=0,
                    value_change_egp=-position.current_value_egp,
                    score=item.score if item else None,
                    confidence=item.confidence if item else 0,
                    reasons=(reason,),
                )
            )
        else:
            actions.append(
                PlanAction(
                    ticker=ticker,
                    action=AdvisoryAction.HOLD,
                    target_weight=current_weight,
                    value_change_egp=0,
                    score=item.score if item else None,
                    confidence=item.confidence if item else 0,
                    reasons=("Position locked; no reliable evidence-backed action",),
                )
            )

    for item in selected:
        ticker = item.assessment.ticker
        if ticker in current:
            continue
        target_value = round(selected_target * total_value, 2)
        actions.append(
            PlanAction(
                ticker=ticker,
                action=AdvisoryAction.BUY,
                target_weight=selected_target,
                value_change_egp=target_value,
                score=item.score,
                confidence=item.confidence,
                reasons=("Eligible, high-ranked, and fits sector concentration limits",),
            )
        )

    # Validate the target budget produced by the plan.  Rounding is expressed in
    # currency, while this invariant remains in weights.
    target_total = (
        actual_target_cash
        + locked_weight
        + selected_target * len(selected)
    )
    if abs(target_total - 1.0) > 1e-6:
        blocked.append(f"Internal target budget mismatch: {target_total:.6f}")

    monthly_actions: list[PlanAction] = []
    contribution = mandate.monthly_contribution_egp
    allocation_base = actual_target_cash + selected_target * len(selected)
    remaining = contribution

    if allocation_base > 0:
        for item in selected:
            amount = round(contribution * selected_target / allocation_base, 2)
            amount = min(amount, remaining)
            remaining = round(remaining - amount, 2)
            if amount <= 0:
                continue
            monthly_actions.append(
                PlanAction(
                    ticker=item.assessment.ticker,
                    action=(
                        AdvisoryAction.ADD
                        if item.assessment.ticker in current
                        else AdvisoryAction.BUY
                    ),
                    target_weight=selected_target,
                    value_change_egp=amount,
                    score=item.score,
                    confidence=item.confidence,
                    reasons=("Monthly contribution follows the verified target mix",),
                )
            )

    if remaining > 0 or not monthly_actions:
        monthly_actions.append(
            PlanAction(
                ticker=None,
                action=AdvisoryAction.KEEP_CASH,
                target_weight=actual_target_cash,
                value_change_egp=remaining,
                confidence=1.0,
                reasons=("Cash share of the monthly contribution",),
            )
        )

    return PortfolioPlan(
        analysis_date=snapshot.analysis_date,
        investable_value_egp=total_value,
        target_cash_weight=actual_target_cash,
        actions=tuple(actions),
        monthly_contribution_egp=contribution,
        monthly_contribution_actions=tuple(monthly_actions),
        blocked_reasons=tuple(blocked),
        warnings=tuple(dict.fromkeys(warnings)),
    )
