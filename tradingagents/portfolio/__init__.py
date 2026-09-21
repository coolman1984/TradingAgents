"""Portfolio-level advisory domain for the Egypt-focused product."""

from .backtest import DecisionOutcome, PerformanceSummary, summarize_outcomes
from .factors import (
    FactorInput,
    FactorScoreCard,
    apply_factor_dimensions,
    score_factor_universe,
    to_dimension_scores,
)
from .mandate import (
    DEFAULT_EGX_BALANCED_MANDATE,
    Allocation,
    PortfolioMandate,
    ShariaTier,
    validate_portfolio,
)
from .opportunity import (
    DEFAULT_OPPORTUNITY_POLICY,
    MacroRegimeInput,
    MarketRegime,
    OpportunityBoard,
    OpportunityPolicy,
    OpportunityRequest,
    OpportunityStatus,
    ValuationCase,
    build_opportunity_board,
    classify_regime,
)
from .recommendations import PortfolioAction, PortfolioRecommendation
from .thesis import (
    PillarStatus,
    ThesisAction,
    ThesisEvaluation,
    ThesisPillar,
    ThesisStatus,
    ThesisTracker,
    evaluate_thesis,
)

__all__ = [
    "DEFAULT_EGX_BALANCED_MANDATE",
    "DEFAULT_OPPORTUNITY_POLICY",
    "Allocation",
    "DecisionOutcome",
    "FactorInput",
    "FactorScoreCard",
    "MacroRegimeInput",
    "MarketRegime",
    "OpportunityBoard",
    "OpportunityPolicy",
    "OpportunityRequest",
    "OpportunityStatus",
    "PerformanceSummary",
    "PillarStatus",
    "PortfolioAction",
    "PortfolioMandate",
    "PortfolioRecommendation",
    "ShariaTier",
    "ThesisAction",
    "ThesisEvaluation",
    "ThesisPillar",
    "ThesisStatus",
    "ThesisTracker",
    "ValuationCase",
    "apply_factor_dimensions",
    "build_opportunity_board",
    "classify_regime",
    "evaluate_thesis",
    "score_factor_universe",
    "to_dimension_scores",
    "summarize_outcomes",
    "validate_portfolio",
]
