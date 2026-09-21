"""Point-in-time outcome measurement for the EGX opportunity engine."""

from __future__ import annotations

from datetime import date
from statistics import mean, median

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingagents.markets.egypt_sources import validate_evidence_for_date
from tradingagents.portfolio.models import EvidenceRef


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionOutcome(StrictModel):
    ticker: str
    signal_date: date
    exit_date: date
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    benchmark_entry: float = Field(gt=0)
    benchmark_exit: float = Field(gt=0)
    total_switch_cost_pct: float = Field(default=0.0, ge=0, le=0.20)
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.exit_date <= self.signal_date:
            raise ValueError("exit_date must be after signal_date")
        return self


class OutcomeResult(StrictModel):
    ticker: str
    gross_return_pct: float
    net_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    point_in_time_valid: bool
    evidence_issues: tuple[str, ...]


class PerformanceSummary(StrictModel):
    observations: int
    point_in_time_valid_observations: int
    hit_rate: float
    alpha_hit_rate: float
    average_net_return_pct: float
    median_net_return_pct: float
    average_alpha_pct: float
    worst_net_return_pct: float
    best_net_return_pct: float


def evaluate_outcome(outcome: DecisionOutcome) -> OutcomeResult:
    """Measure realised return after supplied switching costs and verify evidence."""

    gross = outcome.exit_price / outcome.entry_price - 1
    net = gross - outcome.total_switch_cost_pct
    benchmark = outcome.benchmark_exit / outcome.benchmark_entry - 1
    alpha = net - benchmark

    issues = tuple(
        issue
        for ref in outcome.evidence
        for issue in validate_evidence_for_date(ref, outcome.signal_date)
    )
    return OutcomeResult(
        ticker=outcome.ticker,
        gross_return_pct=round(gross, 6),
        net_return_pct=round(net, 6),
        benchmark_return_pct=round(benchmark, 6),
        alpha_pct=round(alpha, 6),
        point_in_time_valid=not issues,
        evidence_issues=tuple(dict.fromkeys(issues)),
    )


def summarize_outcomes(
    outcomes: tuple[DecisionOutcome, ...],
    *,
    require_point_in_time: bool = True,
) -> PerformanceSummary:
    """Summarise only valid observations by default to avoid flattering backtests."""

    results = [evaluate_outcome(item) for item in outcomes]
    valid = [item for item in results if item.point_in_time_valid]
    selected = valid if require_point_in_time else results

    if not selected:
        return PerformanceSummary(
            observations=len(results),
            point_in_time_valid_observations=len(valid),
            hit_rate=0,
            alpha_hit_rate=0,
            average_net_return_pct=0,
            median_net_return_pct=0,
            average_alpha_pct=0,
            worst_net_return_pct=0,
            best_net_return_pct=0,
        )

    returns = [item.net_return_pct for item in selected]
    alphas = [item.alpha_pct for item in selected]
    return PerformanceSummary(
        observations=len(results),
        point_in_time_valid_observations=len(valid),
        hit_rate=round(sum(value > 0 for value in returns) / len(returns), 4),
        alpha_hit_rate=round(sum(value > 0 for value in alphas) / len(alphas), 4),
        average_net_return_pct=round(mean(returns), 6),
        median_net_return_pct=round(median(returns), 6),
        average_alpha_pct=round(mean(alphas), 6),
        worst_net_return_pct=round(min(returns), 6),
        best_net_return_pct=round(max(returns), 6),
    )
