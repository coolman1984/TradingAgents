"""Deterministic cross-sectional factor scoring for the EGX research universe.

The language model does not assign these scores. Numeric, point-in-time inputs
are ranked against the supplied Egyptian-equity universe and converted into
the existing portfolio analysis dimensions.
"""

from __future__ import annotations

from datetime import date
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.markets.egypt import normalize_egx_equity_ticker
from tradingagents.markets.egypt_sources import validate_evidence_for_date
from tradingagents.portfolio.models import (
    AnalysisDimension,
    DimensionScore,
    EvidenceRef,
    SecurityAssessment,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactorInput(StrictModel):
    ticker: str
    sector: str = Field(min_length=1)
    as_of: date
    financial_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    price_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    catalyst_evidence: tuple[EvidenceRef, ...] = ()

    roic_pct: float | None = None
    roe_pct: float | None = None
    fcf_margin_pct: float | None = None
    revenue_growth_yoy_pct: float | None = None
    eps_growth_yoy_pct: float | None = None
    net_debt_to_ebitda: float | None = None
    cash_conversion_ratio: float | None = None

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    fcf_yield_pct: float | None = None

    momentum_12_1_pct: float | None = None
    momentum_6m_pct: float | None = None
    momentum_3m_pct: float | None = None

    median_daily_value_egp: float | None = Field(default=None, ge=0)
    trading_days_ratio_pct: float | None = Field(default=None, ge=0, le=100)

    annualized_volatility_pct: float | None = Field(default=None, ge=0)
    downside_volatility_pct: float | None = Field(default=None, ge=0)
    max_drawdown_1y_pct: float | None = None

    catalyst_score: float | None = Field(default=None, ge=0, le=100)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_equity_ticker(value)

    @field_validator("sector")
    @classmethod
    def normalize_sector(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("sector cannot be blank")
        return normalized

    @field_validator(
        "roic_pct",
        "roe_pct",
        "fcf_margin_pct",
        "revenue_growth_yoy_pct",
        "eps_growth_yoy_pct",
        "net_debt_to_ebitda",
        "cash_conversion_ratio",
        "pe_ratio",
        "pb_ratio",
        "ev_ebitda",
        "fcf_yield_pct",
        "momentum_12_1_pct",
        "momentum_6m_pct",
        "momentum_3m_pct",
        "max_drawdown_1y_pct",
    )
    @classmethod
    def finite_numbers_only(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("factor metrics must be finite")
        return value

    @model_validator(mode="after")
    def validate_evidence_identity(self):
        all_refs = (*self.financial_evidence, *self.price_evidence, *self.catalyst_evidence)
        if any(self.ticker not in ref.subjects for ref in all_refs):
            raise ValueError("all factor evidence must identify the security")
        if not any(
            ref.source in {"egx_financial_statements", "egx_disclosures"}
            for ref in self.financial_evidence
        ):
            raise ValueError("financial evidence must come from EGX financials/disclosures")
        if not all(ref.source == "egx_prices" for ref in self.price_evidence):
            raise ValueError("price evidence must use official EGX prices")
        if self.catalyst_score is not None:
            if not self.catalyst_evidence:
                raise ValueError("catalyst score requires disclosure evidence")
            if not all(ref.source == "egx_disclosures" for ref in self.catalyst_evidence):
                raise ValueError("catalyst evidence must use EGX disclosures")
        return self


class FactorScoreCard(StrictModel):
    ticker: str
    as_of: date
    dimension_scores: dict[str, float]
    dimension_confidence: dict[str, float]
    metric_percentiles: dict[str, float]
    screen_ready: bool
    issues: tuple[str, ...] = ()


_METRICS: dict[AnalysisDimension, tuple[tuple[str, bool], ...]] = {
    AnalysisDimension.FUNDAMENTAL: (
        ("roic_pct", True),
        ("roe_pct", True),
        ("fcf_margin_pct", True),
        ("revenue_growth_yoy_pct", True),
        ("eps_growth_yoy_pct", True),
        ("net_debt_to_ebitda", False),
        ("cash_conversion_ratio", True),
    ),
    AnalysisDimension.VALUATION: (
        ("pe_ratio", False),
        ("pb_ratio", False),
        ("ev_ebitda", False),
        ("fcf_yield_pct", True),
    ),
    AnalysisDimension.TECHNICAL: (
        ("momentum_12_1_pct", True),
        ("momentum_6m_pct", True),
        ("momentum_3m_pct", True),
    ),
    AnalysisDimension.LIQUIDITY: (
        ("median_daily_value_egp", True),
        ("trading_days_ratio_pct", True),
    ),
    AnalysisDimension.RISK: (
        ("annualized_volatility_pct", False),
        ("downside_volatility_pct", False),
        ("max_drawdown_1y_pct", True),
    ),
}

_MIN_METRICS = {
    AnalysisDimension.FUNDAMENTAL: 3,
    AnalysisDimension.VALUATION: 2,
    AnalysisDimension.TECHNICAL: 2,
    AnalysisDimension.LIQUIDITY: 1,
    AnalysisDimension.RISK: 2,
}

_DIMENSION_EVIDENCE = {
    AnalysisDimension.FUNDAMENTAL: "financial_evidence",
    AnalysisDimension.VALUATION: "financial_evidence",
    AnalysisDimension.TECHNICAL: "price_evidence",
    AnalysisDimension.LIQUIDITY: "price_evidence",
    AnalysisDimension.RISK: "price_evidence",
}


def _clean_metric(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if name in {"pe_ratio", "pb_ratio", "ev_ebitda"} and value <= 0:
        return None
    return value


def _percentile_map(
    values: dict[str, float],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    """Average-rank percentiles on 0..100, with ties treated identically."""
    if not values:
        return {}
    if len(values) == 1:
        ticker = next(iter(values))
        return {ticker: 50.0}

    ordered = sorted(
        values.items(),
        key=lambda item: item[1],
        reverse=not higher_is_better,
    )
    grouped: dict[float, list[int]] = {}
    for index, (_, value) in enumerate(ordered):
        grouped.setdefault(value, []).append(index)

    result: dict[str, float] = {}
    denominator = len(ordered) - 1
    for ticker, value in ordered:
        indexes = grouped[value]
        average_index = sum(indexes) / len(indexes)
        percentile = average_index / denominator * 100
        result[ticker] = round(percentile, 4)
    return result


def _sector_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _blended_percentile_map(
    records: tuple[FactorInput, ...],
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    """Blend sector-relative and market-relative ranks for valuation metrics."""

    market_values = {
        record.ticker: cleaned
        for record in records
        if (cleaned := _clean_metric(metric, getattr(record, metric))) is not None
    }
    market = _percentile_map(
        market_values,
        higher_is_better=higher_is_better,
    )
    if not market:
        return {}

    by_sector: dict[str, list[FactorInput]] = {}
    for record in records:
        by_sector.setdefault(_sector_key(record.sector), []).append(record)

    result: dict[str, float] = {}
    for members in by_sector.values():
        sector_values = {
            record.ticker: cleaned
            for record in members
            if (cleaned := _clean_metric(metric, getattr(record, metric))) is not None
        }
        if len(sector_values) < 3:
            for ticker in sector_values:
                result[ticker] = market[ticker]
            continue

        sector = _percentile_map(
            sector_values,
            higher_is_better=higher_is_better,
        )
        for ticker, sector_score in sector.items():
            result[ticker] = round(0.70 * sector_score + 0.30 * market[ticker], 4)

    return result


def _validate_record(record: FactorInput, analysis_date: date) -> tuple[str, ...]:
    issues: list[str] = []
    if record.as_of > analysis_date:
        issues.append("factor record is from the future")
    all_refs = (*record.financial_evidence, *record.price_evidence, *record.catalyst_evidence)
    for ref in all_refs:
        for issue in validate_evidence_for_date(ref, record.as_of):
            issues.append(f"{ref.source}: {issue}")
    return tuple(dict.fromkeys(issues))


def score_factor_universe(
    records: tuple[FactorInput, ...],
    analysis_date: date,
    *,
    maximum_record_age_days: int = 30,
) -> tuple[FactorScoreCard, ...]:
    """Rank a whole universe; a hand-picked tiny list should have lower confidence."""
    if not records:
        return ()
    tickers = [record.ticker for record in records]
    if len(tickers) != len(set(tickers)):
        raise ValueError("factor universe contains duplicate tickers")
    as_of_dates = {record.as_of for record in records}
    if len(as_of_dates) != 1:
        raise ValueError("factor universe records must share one as_of date")
    universe_as_of = next(iter(as_of_dates))
    if universe_as_of > analysis_date:
        raise ValueError("factor universe as_of date is from the future")
    if (analysis_date - universe_as_of).days > maximum_record_age_days:
        raise ValueError("factor universe is stale")

    metric_percentiles: dict[str, dict[str, float]] = {}
    for dimension, metrics in _METRICS.items():
        for metric, higher_is_better in metrics:
            if dimension is AnalysisDimension.VALUATION:
                metric_percentiles[metric] = _blended_percentile_map(
                    records,
                    metric,
                    higher_is_better=higher_is_better,
                )
                continue

            values = {
                record.ticker: cleaned
                for record in records
                if (cleaned := _clean_metric(metric, getattr(record, metric))) is not None
            }
            metric_percentiles[metric] = _percentile_map(
                values,
                higher_is_better=higher_is_better,
            )

    universe_confidence = min(1.0, len(records) / 20)
    cards: list[FactorScoreCard] = []

    for record in records:
        issues = list(_validate_record(record, analysis_date))
        dimensions: dict[str, float] = {}
        confidence: dict[str, float] = {}
        per_metric: dict[str, float] = {}

        for dimension, metrics in _METRICS.items():
            available = []
            for metric, _ in metrics:
                percentile = metric_percentiles[metric].get(record.ticker)
                if percentile is not None:
                    available.append(percentile)
                    per_metric[metric] = percentile

            if len(available) < _MIN_METRICS[dimension]:
                issues.append(
                    f"{dimension.value}: insufficient numeric factor coverage"
                )
                continue

            dimensions[dimension.value] = round(sum(available) / len(available), 2)
            coverage = len(available) / len(metrics)
            confidence[dimension.value] = round(coverage * universe_confidence, 4)

        if record.catalyst_score is not None:
            dimensions[AnalysisDimension.CATALYST.value] = round(record.catalyst_score, 2)
            confidence[AnalysisDimension.CATALYST.value] = round(universe_confidence, 4)

        required = {
            AnalysisDimension.FUNDAMENTAL.value,
            AnalysisDimension.VALUATION.value,
            AnalysisDimension.TECHNICAL.value,
            AnalysisDimension.LIQUIDITY.value,
            AnalysisDimension.RISK.value,
        }
        ready = not issues and required <= dimensions.keys()
        cards.append(
            FactorScoreCard(
                ticker=record.ticker,
                as_of=record.as_of,
                dimension_scores=dimensions,
                dimension_confidence=confidence,
                metric_percentiles=per_metric,
                screen_ready=ready,
                issues=tuple(dict.fromkeys(issues)),
            )
        )

    return tuple(cards)


def to_dimension_scores(
    record: FactorInput,
    card: FactorScoreCard,
) -> tuple[DimensionScore, ...]:
    """Convert deterministic factors into the existing evidence-aware score model."""
    if record.ticker != card.ticker or record.as_of != card.as_of:
        raise ValueError("factor record and score card do not match")

    results: list[DimensionScore] = []
    for dimension in AnalysisDimension:
        score = card.dimension_scores.get(dimension.value)
        if score is None:
            continue
        if dimension is AnalysisDimension.CATALYST:
            evidence = record.catalyst_evidence
        else:
            evidence = getattr(record, _DIMENSION_EVIDENCE[dimension])
        if not evidence:
            continue
        results.append(
            DimensionScore(
                dimension=dimension,
                score=score,
                confidence=card.dimension_confidence[dimension.value],
                as_of=record.as_of,
                evidence=evidence,
            )
        )
    return tuple(results)


def apply_factor_dimensions(
    assessments: tuple[SecurityAssessment, ...],
    records: tuple[FactorInput, ...],
    cards: tuple[FactorScoreCard, ...],
) -> tuple[SecurityAssessment, ...]:
    """Replace model-authored dimensions with deterministic factor dimensions."""

    record_map = {item.ticker: item for item in records}
    card_map = {item.ticker: item for item in cards}
    if len(record_map) != len(records) or len(card_map) != len(cards):
        raise ValueError("factor records and cards must have unique tickers")

    results: list[SecurityAssessment] = []
    for assessment in assessments:
        record = record_map.get(assessment.ticker)
        card = card_map.get(assessment.ticker)
        if record is None or card is None:
            results.append(assessment)
            continue
        results.append(
            assessment.model_copy(
                update={"dimensions": to_dimension_scores(record, card)}
            )
        )
    return tuple(results)
