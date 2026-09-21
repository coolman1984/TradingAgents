"""Advanced, deterministic opportunity intelligence for Egyptian equities.

This module does not execute trades. It converts already verified point-in-time
research into a stricter opportunity board that separates company quality,
valuation, timing, catalysts, liquidity and downside risk. The model is
deliberately deterministic so an LLM cannot bypass evidence, Sharia or risk
guardrails.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.markets.egypt import normalize_egx_equity_ticker
from tradingagents.markets.egypt_sources import validate_evidence_for_date
from tradingagents.portfolio.engine import (
    DEFAULT_ENGINE_POLICY,
    EnginePolicy,
    score_security,
)
from tradingagents.portfolio.mandate import (
    DEFAULT_EGX_BALANCED_MANDATE,
    PortfolioMandate,
)
from tradingagents.portfolio.models import (
    AnalysisDimension,
    EvidenceRef,
    PortfolioSnapshot,
    SecurityAssessment,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketRegime(str, Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"
    STRESS = "stress"


class OpportunityStatus(str, Enum):
    BLOCKED = "blocked"
    HIGH_CONVICTION_BUY = "high_conviction_buy"
    BUY = "buy"
    ADD = "add"
    WATCH = "watch"
    HOLD = "hold"
    TRIM = "trim"
    EXIT = "exit"


class MacroRegimeInput(StrictModel):
    """Observable macro/market facts used only for modest evidence-backed tilts."""

    annual_inflation_pct: float | None = None
    policy_rate_pct: float | None = None
    fx_depreciation_90d_pct: float | None = None
    market_return_90d_pct: float | None = None
    market_volatility_60d_pct: float | None = Field(default=None, ge=0)
    market_breadth_pct: float | None = Field(default=None, ge=0, le=100)
    as_of: date | None = None
    evidence: tuple[EvidenceRef, ...] = ()

    @model_validator(mode="after")
    def require_evidence_for_observations(self):
        observations = (
            self.annual_inflation_pct,
            self.policy_rate_pct,
            self.fx_depreciation_90d_pct,
            self.market_return_90d_pct,
            self.market_volatility_60d_pct,
            self.market_breadth_pct,
        )
        if any(value is not None for value in observations):
            if self.as_of is None:
                raise ValueError("macro observations require an as_of date")
            if not self.evidence:
                raise ValueError("macro observations require evidence")
        return self


class RegimeAssessment(StrictModel):
    regime: MarketRegime
    confidence: float = Field(ge=0, le=1)
    reasons: tuple[str, ...] = ()


class ValuationCase(StrictModel):
    """Price-to-value case with explicit bear/base/bull values and evidence."""

    ticker: str
    current_price_egp: float = Field(gt=0)
    fair_value_low_egp: float = Field(gt=0)
    fair_value_base_egp: float = Field(gt=0)
    fair_value_high_egp: float = Field(gt=0)
    as_of: date
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_equity_ticker(value)

    @model_validator(mode="after")
    def validate_range(self):
        if not (
            self.fair_value_low_egp
            <= self.fair_value_base_egp
            <= self.fair_value_high_egp
        ):
            raise ValueError("fair values must satisfy low <= base <= high")
        if not any(self.ticker in ref.subjects for ref in self.evidence):
            raise ValueError("valuation evidence must identify the security")
        if not any(ref.source == "egx_prices" for ref in self.evidence):
            raise ValueError("valuation case requires official EGX price evidence")
        return self

    @property
    def base_upside_pct(self) -> float:
        return self.fair_value_base_egp / self.current_price_egp - 1

    @property
    def low_case_return_pct(self) -> float:
        return self.fair_value_low_egp / self.current_price_egp - 1

    @property
    def high_case_return_pct(self) -> float:
        return self.fair_value_high_egp / self.current_price_egp - 1

    @property
    def margin_of_safety_pct(self) -> float:
        return 1 - self.current_price_egp / self.fair_value_base_egp


class OpportunityPolicy(StrictModel):
    minimum_confidence: float = Field(default=0.65, ge=0, le=1)
    high_conviction_confidence: float = Field(default=0.75, ge=0, le=1)
    buy_score: float = Field(default=68, ge=0, le=100)
    high_conviction_score: float = Field(default=78, ge=0, le=100)
    add_score: float = Field(default=72, ge=0, le=100)
    trim_score: float = Field(default=55, ge=0, le=100)
    exit_score: float = Field(default=45, ge=0, le=100)
    buy_base_upside_pct: float = 0.12
    high_conviction_base_upside_pct: float = 0.20
    max_buy_low_case_loss_pct: float = -0.22
    max_high_conviction_low_case_loss_pct: float = -0.15
    minimum_high_conviction_reward_risk: float = 1.5
    minimum_replacement_score_edge: float = 8
    minimum_replacement_return_edge_pct: float = 0.08
    replacement_safety_buffer_pct: float = 0.03

    @model_validator(mode="after")
    def validate_thresholds(self):
        if not (
            self.exit_score
            < self.trim_score
            < self.buy_score
            <= self.add_score
            < self.high_conviction_score
        ):
            raise ValueError(
                "score thresholds must rise from exit through high conviction"
            )
        if self.high_conviction_confidence < self.minimum_confidence:
            raise ValueError(
                "high-conviction confidence cannot be below normal confidence"
            )
        if self.high_conviction_base_upside_pct < self.buy_base_upside_pct:
            raise ValueError(
                "high-conviction upside cannot be below normal buy upside"
            )
        if (
            self.max_high_conviction_low_case_loss_pct
            < self.max_buy_low_case_loss_pct
        ):
            raise ValueError(
                "high-conviction downside limit must be at least as strict"
            )
        return self


DEFAULT_OPPORTUNITY_POLICY = OpportunityPolicy()


class OpportunityResult(StrictModel):
    ticker: str
    company_name: str
    sector: str
    held: bool
    status: OpportunityStatus
    composite_score: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    regime: MarketRegime
    base_upside_pct: float | None = None
    low_case_return_pct: float | None = None
    high_case_return_pct: float | None = None
    margin_of_safety_pct: float | None = None
    reward_risk_ratio: float | None = None
    reasons: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


class ReplacementDecision(StrictModel):
    current_ticker: str
    challenger_ticker: str
    replace: bool
    score_edge: float
    expected_return_edge_pct: float | None
    net_edge_after_cost_pct: float | None
    reasons: tuple[str, ...]


class OpportunityBoard(StrictModel):
    analysis_date: str
    regime: RegimeAssessment
    opportunities: tuple[OpportunityResult, ...]
    top_new_ideas: tuple[str, ...]
    holding_reviews: tuple[str, ...]
    replacements: tuple[ReplacementDecision, ...]


class OpportunityRequest(StrictModel):
    snapshot: PortfolioSnapshot
    valuations: tuple[ValuationCase, ...]
    macro: MacroRegimeInput = Field(default_factory=MacroRegimeInput)

    @model_validator(mode="after")
    def unique_valuations(self):
        tickers = [item.ticker for item in self.valuations]
        if len(tickers) != len(set(tickers)):
            raise ValueError("valuation cases contain duplicate tickers")
        candidate_tickers = {item.ticker for item in self.snapshot.candidates}
        unknown = sorted(set(tickers) - candidate_tickers)
        if unknown:
            raise ValueError(
                "valuation cases contain tickers outside the snapshot: "
                + ", ".join(unknown)
            )
        return self


_BASE_WEIGHTS = {
    AnalysisDimension.FUNDAMENTAL: 0.25,
    AnalysisDimension.VALUATION: 0.20,
    AnalysisDimension.TECHNICAL: 0.15,
    AnalysisDimension.CATALYST: 0.10,
    AnalysisDimension.LIQUIDITY: 0.10,
    AnalysisDimension.RISK: 0.20,
}

_REGIME_TILTS = {
    MarketRegime.RISK_ON: {
        AnalysisDimension.FUNDAMENTAL: -0.02,
        AnalysisDimension.VALUATION: -0.01,
        AnalysisDimension.TECHNICAL: 0.03,
        AnalysisDimension.CATALYST: 0.02,
        AnalysisDimension.LIQUIDITY: 0.00,
        AnalysisDimension.RISK: -0.02,
    },
    MarketRegime.NEUTRAL: {},
    MarketRegime.DEFENSIVE: {
        AnalysisDimension.FUNDAMENTAL: 0.03,
        AnalysisDimension.VALUATION: 0.00,
        AnalysisDimension.TECHNICAL: -0.03,
        AnalysisDimension.CATALYST: -0.02,
        AnalysisDimension.LIQUIDITY: 0.01,
        AnalysisDimension.RISK: 0.01,
    },
    MarketRegime.STRESS: {
        AnalysisDimension.FUNDAMENTAL: 0.03,
        AnalysisDimension.VALUATION: -0.04,
        AnalysisDimension.TECHNICAL: -0.04,
        AnalysisDimension.CATALYST: -0.03,
        AnalysisDimension.LIQUIDITY: 0.03,
        AnalysisDimension.RISK: 0.05,
    },
}

_STATUS_PRIORITY = {
    OpportunityStatus.HIGH_CONVICTION_BUY: 8,
    OpportunityStatus.BUY: 7,
    OpportunityStatus.ADD: 6,
    OpportunityStatus.HOLD: 5,
    OpportunityStatus.WATCH: 4,
    OpportunityStatus.TRIM: 3,
    OpportunityStatus.EXIT: 2,
    OpportunityStatus.BLOCKED: 1,
}


def classify_regime(
    data: MacroRegimeInput,
    analysis_date: date | None = None,
) -> RegimeAssessment:
    """Classify a broad market regime without pretending macro timing is precise."""

    if analysis_date is not None and data.as_of is not None:
        if data.as_of > analysis_date:
            return RegimeAssessment(
                regime=MarketRegime.NEUTRAL,
                confidence=0.0,
                reasons=("Macro observations are from the future",),
            )
        if (analysis_date - data.as_of).days > 90:
            return RegimeAssessment(
                regime=MarketRegime.NEUTRAL,
                confidence=0.0,
                reasons=("Macro observations are stale",),
            )
        evidence_issues = tuple(
            issue
            for ref in data.evidence
            for issue in validate_evidence_for_date(ref, analysis_date)
        )
        if evidence_issues:
            return RegimeAssessment(
                regime=MarketRegime.NEUTRAL,
                confidence=0.0,
                reasons=tuple(
                    dict.fromkeys(
                        f"Macro evidence: {issue}" for issue in evidence_issues
                    )
                ),
            )

    stress = 0
    risk_on = 0
    reasons: list[str] = []

    if data.market_volatility_60d_pct is not None:
        if data.market_volatility_60d_pct >= 35:
            stress += 2
            reasons.append("Market volatility is elevated")
        elif data.market_volatility_60d_pct <= 20:
            risk_on += 1
            reasons.append("Market volatility is contained")

    if data.fx_depreciation_90d_pct is not None:
        if data.fx_depreciation_90d_pct >= 15:
            stress += 2
            reasons.append("The EGP has depreciated sharply over 90 days")
        elif data.fx_depreciation_90d_pct <= 3:
            risk_on += 1
            reasons.append("FX pressure is limited over 90 days")

    if data.market_return_90d_pct is not None:
        if data.market_return_90d_pct <= -12:
            stress += 2
            reasons.append("The market has a material 90-day drawdown")
        elif data.market_return_90d_pct >= 8:
            risk_on += 2
            reasons.append("The market has positive 90-day momentum")

    if data.market_breadth_pct is not None:
        if data.market_breadth_pct < 35:
            stress += 1
            reasons.append("Market breadth is weak")
        elif data.market_breadth_pct >= 60:
            risk_on += 1
            reasons.append("Market breadth is broad")

    if (
        data.annual_inflation_pct is not None
        and data.policy_rate_pct is not None
        and data.annual_inflation_pct > data.policy_rate_pct + 3
    ):
        stress += 1
        reasons.append("Inflation materially exceeds the policy rate")

    if stress >= 4:
        regime = MarketRegime.STRESS
    elif stress >= 2:
        regime = MarketRegime.DEFENSIVE
    elif risk_on >= 3 and stress == 0:
        regime = MarketRegime.RISK_ON
    else:
        regime = MarketRegime.NEUTRAL

    separation = abs(stress - risk_on)
    confidence = min(0.90, 0.50 + 0.08 * separation)
    if not reasons:
        reasons.append("Insufficient macro evidence for a strong regime tilt")
        confidence = 0.50

    return RegimeAssessment(
        regime=regime,
        confidence=round(confidence, 2),
        reasons=tuple(reasons),
    )


def _weights_for_regime(regime: MarketRegime) -> dict[AnalysisDimension, float]:
    tilted = {
        dimension: weight + _REGIME_TILTS[regime].get(dimension, 0.0)
        for dimension, weight in _BASE_WEIGHTS.items()
    }
    total = sum(max(0.01, value) for value in tilted.values())
    return {
        dimension: max(0.01, value) / total
        for dimension, value in tilted.items()
    }


def _opportunity_score(
    assessment: SecurityAssessment,
    regime: MarketRegime,
) -> tuple[float | None, float, tuple[str, ...]]:
    by_dimension = {item.dimension: item for item in assessment.dimensions}
    weights = _weights_for_regime(regime)
    available = {
        dimension: by_dimension[dimension]
        for dimension in weights
        if dimension in by_dimension
    }
    if not available:
        return None, 0.0, ("No usable opportunity dimensions",)

    available_weight = sum(weights[dimension] for dimension in available)
    score = sum(
        item.score * weights[dimension]
        for dimension, item in available.items()
    ) / available_weight

    weighted_confidence = sum(
        item.confidence * weights[dimension]
        for dimension, item in available.items()
    )
    coverage = available_weight
    confidence = weighted_confidence * coverage

    missing = tuple(
        dimension.value for dimension in weights if dimension not in available
    )
    issues = (
        ("Missing opportunity dimensions: " + ", ".join(missing),)
        if missing
        else ()
    )
    return round(score, 2), round(confidence, 4), issues


def _valuation_metrics(
    valuation: ValuationCase,
) -> tuple[float, float, float, float]:
    downside = valuation.low_case_return_pct
    upside = valuation.base_upside_pct
    high = valuation.high_case_return_pct
    denominator = max(abs(min(downside, -0.01)), 0.01)
    reward_risk = max(upside, 0.0) / denominator
    return upside, downside, high, reward_risk


def evaluate_opportunity(
    assessment: SecurityAssessment,
    *,
    analysis_date: date,
    regime: MarketRegime = MarketRegime.NEUTRAL,
    valuation: ValuationCase | None = None,
    held: bool = False,
    mandate: PortfolioMandate = DEFAULT_EGX_BALANCED_MANDATE,
    engine_policy: EnginePolicy = DEFAULT_ENGINE_POLICY,
    policy: OpportunityPolicy = DEFAULT_OPPORTUNITY_POLICY,
) -> OpportunityResult:
    """Turn one verified security assessment into an action posture.

    A new position can never become buy-eligible without a usable valuation
    range. Existing holdings fail safe to HOLD when decision-grade evidence is
    stale or incomplete.
    """

    core = score_security(assessment, analysis_date, mandate, engine_policy)
    composite, confidence, score_issues = _opportunity_score(assessment, regime)

    blockers: list[str] = []
    risks: list[str] = list(score_issues)
    reasons: list[str] = []

    if not core.decision_ready:
        blockers.extend(
            issue
            for issue in core.issues
            if "below buy threshold" not in issue.lower()
        )

    if assessment.sharia_tier not in mandate.allowed_sharia_tiers:
        blockers.append("Fresh Sharia classification is not buy-eligible")

    valuation_usable = valuation is not None
    if valuation is not None:
        if valuation.ticker != assessment.ticker:
            blockers.append("Valuation ticker does not match the security")
            valuation_usable = False
        else:
            if valuation.as_of > analysis_date:
                risks.append("Valuation case is from the future")
                valuation_usable = False
            elif (analysis_date - valuation.as_of).days > engine_policy.max_dimension_age_days:
                risks.append("Valuation case is stale")
                valuation_usable = False
            for ref in valuation.evidence:
                issues = validate_evidence_for_date(ref, analysis_date)
                if issues:
                    risks.extend(f"Valuation evidence: {item}" for item in issues)
                    valuation_usable = False

    base_upside = low_case = high_case = margin = reward_risk = None
    if valuation_usable and valuation is not None:
        base_upside, low_case, high_case, reward_risk = _valuation_metrics(valuation)
        margin = valuation.margin_of_safety_pct

    if composite is None:
        status = OpportunityStatus.HOLD if held else OpportunityStatus.BLOCKED
        blockers.append("No decision-grade composite score")
    elif blockers:
        status = OpportunityStatus.HOLD if held else OpportunityStatus.BLOCKED
        if held:
            reasons.append("Hold until fresh evidence resolves the blocker")
    elif not valuation_usable:
        status = OpportunityStatus.HOLD if held else OpportunityStatus.WATCH
        risks.append("No usable point-in-time valuation range")
    elif held:
        assert base_upside is not None
        assert low_case is not None
        if composite < policy.exit_score or base_upside <= -0.10:
            status = OpportunityStatus.EXIT
            reasons.append("Verified economics no longer justify the position")
        elif composite < policy.trim_score or base_upside < 0.02:
            status = OpportunityStatus.TRIM
            reasons.append("Forward reward has compressed relative to risk")
        elif (
            composite >= policy.add_score
            and confidence >= policy.minimum_confidence
            and base_upside >= policy.buy_base_upside_pct
            and low_case >= policy.max_buy_low_case_loss_pct
        ):
            status = OpportunityStatus.ADD
            reasons.append("Current holding still offers attractive verified upside")
        else:
            status = OpportunityStatus.HOLD
            reasons.append("Thesis remains investable but does not clear add thresholds")
    else:
        assert base_upside is not None
        assert low_case is not None
        assert reward_risk is not None
        if (
            composite >= policy.high_conviction_score
            and confidence >= policy.high_conviction_confidence
            and base_upside >= policy.high_conviction_base_upside_pct
            and low_case >= policy.max_high_conviction_low_case_loss_pct
            and reward_risk >= policy.minimum_high_conviction_reward_risk
        ):
            status = OpportunityStatus.HIGH_CONVICTION_BUY
            reasons.append("Score, valuation, downside and confidence all clear strict gates")
        elif (
            composite >= policy.buy_score
            and confidence >= policy.minimum_confidence
            and base_upside >= policy.buy_base_upside_pct
            and low_case >= policy.max_buy_low_case_loss_pct
        ):
            status = OpportunityStatus.BUY
            reasons.append("Verified reward clears the normal buy gates")
        else:
            status = OpportunityStatus.WATCH
            reasons.append("Interesting candidate, but one or more buy gates are not cleared")

    if confidence < policy.minimum_confidence:
        risks.append(
            f"Opportunity confidence {confidence:.0%} is below "
            f"{policy.minimum_confidence:.0%}"
        )
    if low_case is not None and low_case < policy.max_buy_low_case_loss_pct:
        risks.append("Low-case valuation has excessive downside for a new buy")
    if base_upside is not None and base_upside < policy.buy_base_upside_pct:
        risks.append("Base-case upside is below the normal buy hurdle")

    return OpportunityResult(
        ticker=assessment.ticker,
        company_name=assessment.company_name,
        sector=assessment.sector,
        held=held,
        status=status,
        composite_score=composite,
        confidence=confidence,
        regime=regime,
        base_upside_pct=round(base_upside, 4) if base_upside is not None else None,
        low_case_return_pct=round(low_case, 4) if low_case is not None else None,
        high_case_return_pct=round(high_case, 4) if high_case is not None else None,
        margin_of_safety_pct=round(margin, 4) if margin is not None else None,
        reward_risk_ratio=round(reward_risk, 3) if reward_risk is not None else None,
        reasons=tuple(dict.fromkeys(reasons)),
        risks=tuple(dict.fromkeys(risks)),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def evaluate_replacement(
    current: OpportunityResult,
    challenger: OpportunityResult,
    *,
    switch_cost_pct: float | None,
    policy: OpportunityPolicy = DEFAULT_OPPORTUNITY_POLICY,
) -> ReplacementDecision:
    """Require a meaningful, cost-adjusted edge before rotating a holding."""

    reasons: list[str] = []
    score_edge = (
        (challenger.composite_score or 0) - (current.composite_score or 0)
    )

    expected_edge = None
    net_edge = None
    replace = False

    if switch_cost_pct is None:
        reasons.append("Switch cost is unknown")
    elif current.base_upside_pct is None or challenger.base_upside_pct is None:
        reasons.append("Both securities need decision-grade valuation")
    elif challenger.status not in {
        OpportunityStatus.BUY,
        OpportunityStatus.HIGH_CONVICTION_BUY,
    }:
        reasons.append("Challenger does not clear new-buy gates")
    else:
        expected_edge = challenger.base_upside_pct - current.base_upside_pct
        net_edge = (
            expected_edge
            - switch_cost_pct
            - policy.replacement_safety_buffer_pct
        )
        if score_edge < policy.minimum_replacement_score_edge:
            reasons.append("Score edge is too small to justify turnover")
        if expected_edge < policy.minimum_replacement_return_edge_pct:
            reasons.append("Expected-return edge is too small")
        if net_edge <= 0:
            reasons.append("Return edge does not survive cost and safety buffer")
        if challenger.confidence + 0.10 < current.confidence:
            reasons.append("Challenger confidence is materially weaker")

        replace = not reasons
        if replace:
            reasons.append("Challenger clears score, return, confidence and cost gates")

    return ReplacementDecision(
        current_ticker=current.ticker,
        challenger_ticker=challenger.ticker,
        replace=replace,
        score_edge=round(score_edge, 2),
        expected_return_edge_pct=(
            round(expected_edge, 4) if expected_edge is not None else None
        ),
        net_edge_after_cost_pct=(
            round(net_edge, 4) if net_edge is not None else None
        ),
        reasons=tuple(reasons),
    )


def build_opportunity_board(
    snapshot: PortfolioSnapshot,
    valuations: tuple[ValuationCase, ...],
    macro: MacroRegimeInput | None = None,
    *,
    policy: OpportunityPolicy = DEFAULT_OPPORTUNITY_POLICY,
) -> OpportunityBoard:
    """Build a market-wide opportunity board and rotation shortlist."""

    regime = classify_regime(
        macro or MacroRegimeInput(),
        snapshot.analysis_date,
    )
    current = {item.ticker for item in snapshot.positions}
    valuation_map = {item.ticker: item for item in valuations}

    opportunities = [
        evaluate_opportunity(
            candidate,
            analysis_date=snapshot.analysis_date,
            regime=regime.regime,
            valuation=valuation_map.get(candidate.ticker),
            held=candidate.ticker in current,
            policy=policy,
        )
        for candidate in snapshot.candidates
    ]
    opportunities.sort(
        key=lambda item: (
            _STATUS_PRIORITY[item.status],
            item.composite_score or -1,
            item.confidence,
            item.ticker,
        ),
        reverse=True,
    )

    top_new = tuple(
        item.ticker
        for item in opportunities
        if not item.held
        and item.status
        in {OpportunityStatus.HIGH_CONVICTION_BUY, OpportunityStatus.BUY}
    )
    holding_reviews = tuple(
        item.ticker
        for item in opportunities
        if item.held
        and item.status
        in {
            OpportunityStatus.ADD,
            OpportunityStatus.HOLD,
            OpportunityStatus.TRIM,
            OpportunityStatus.EXIT,
        }
    )

    challengers = [
        item
        for item in opportunities
        if not item.held
        and item.status
        in {OpportunityStatus.HIGH_CONVICTION_BUY, OpportunityStatus.BUY}
    ]
    held_items = [item for item in opportunities if item.held]

    replacements: list[ReplacementDecision] = []
    used_challengers: set[str] = set()
    for holding in sorted(
        held_items,
        key=lambda item: (item.composite_score or 0, item.ticker),
    ):
        best: ReplacementDecision | None = None
        for challenger in challengers:
            if challenger.ticker in used_challengers:
                continue
            decision = evaluate_replacement(
                holding,
                challenger,
                switch_cost_pct=snapshot.estimated_switch_cost_pct,
                policy=policy,
            )
            if not decision.replace:
                continue
            if best is None or decision.score_edge > best.score_edge:
                best = decision
        if best is not None:
            replacements.append(best)
            used_challengers.add(best.challenger_ticker)

    return OpportunityBoard(
        analysis_date=snapshot.analysis_date.isoformat(),
        regime=regime,
        opportunities=tuple(opportunities),
        top_new_ideas=top_new,
        holding_reviews=holding_reviews,
        replacements=tuple(replacements),
    )
