from datetime import date, datetime, timezone

import pytest

from tradingagents.portfolio.factors import (
    FactorInput,
    apply_factor_dimensions,
    score_factor_universe,
    to_dimension_scores,
)
from tradingagents.portfolio.mandate import ShariaTier
from tradingagents.portfolio.models import (
    AnalysisDimension,
    DimensionScore,
    EvidenceRef,
    SecurityAssessment,
)


def evidence(source: str, ticker: str) -> EvidenceRef:
    host = "beta.egx.com.eg"
    return EvidenceRef(
        source=source,
        url=f"https://{host}/source",
        published_on=date(2026, 9, 14),
        observed_at=datetime(2026, 9, 14, 12, tzinfo=timezone.utc),
        authority="official",
        subjects=(ticker,),
        content_hash="d" * 64,
    )


def record(index: int, *, as_of: date = date(2026, 9, 14)) -> FactorInput:
    ticker = f"T{index:02d}"
    return FactorInput(
        ticker=ticker,
        as_of=as_of,
        financial_evidence=(evidence("egx_financial_statements", ticker),),
        price_evidence=(evidence("egx_prices", ticker),),
        roic_pct=5 + index,
        roe_pct=8 + index,
        fcf_margin_pct=3 + index,
        revenue_growth_yoy_pct=index,
        eps_growth_yoy_pct=index + 2,
        net_debt_to_ebitda=5 - index * 0.2,
        cash_conversion_ratio=0.5 + index * 0.05,
        pe_ratio=35 - index,
        pb_ratio=6 - index * 0.1,
        ev_ebitda=22 - index * 0.3,
        fcf_yield_pct=index,
        momentum_12_1_pct=index * 2,
        momentum_6m_pct=index * 1.5,
        momentum_3m_pct=index,
        median_daily_value_egp=index * 1_000_000,
        trading_days_ratio_pct=75 + index,
        annualized_volatility_pct=60 - index,
        downside_volatility_pct=50 - index,
        max_drawdown_1y_pct=-60 + index,
    )


@pytest.mark.unit
def test_factor_engine_ranks_full_universe_without_llm_scores():
    records = tuple(record(index) for index in range(1, 21))
    cards = score_factor_universe(records, date(2026, 9, 14))
    best = next(item for item in cards if item.ticker == "T20.CA")
    weak = next(item for item in cards if item.ticker == "T01.CA")

    assert best.screen_ready is True
    assert best.dimension_scores["fundamental"] > weak.dimension_scores["fundamental"]
    assert best.dimension_scores["valuation"] > weak.dimension_scores["valuation"]
    assert best.dimension_scores["technical"] > weak.dimension_scores["technical"]
    assert best.dimension_scores["liquidity"] > weak.dimension_scores["liquidity"]
    assert best.dimension_scores["risk"] > weak.dimension_scores["risk"]
    assert best.dimension_confidence["fundamental"] == pytest.approx(1.0)


@pytest.mark.unit
def test_factor_universe_rejects_mixed_dates():
    records = (
        record(1, as_of=date(2026, 9, 14)),
        record(2, as_of=date(2026, 9, 13)),
    )

    with pytest.raises(ValueError, match="share one as_of date"):
        score_factor_universe(records, date(2026, 9, 14))


@pytest.mark.unit
def test_factor_universe_rejects_stale_snapshot():
    records = tuple(record(index, as_of=date(2026, 7, 1)) for index in range(1, 4))

    with pytest.raises(ValueError, match="stale"):
        score_factor_universe(records, date(2026, 9, 14))


@pytest.mark.unit
def test_factor_scores_convert_to_existing_dimension_contract():
    records = tuple(record(index) for index in range(1, 21))
    cards = score_factor_universe(records, date(2026, 9, 14))
    chosen = records[-1]
    card = cards[-1]
    dimensions = to_dimension_scores(chosen, card)

    assert {item.dimension for item in dimensions} == {
        AnalysisDimension.FUNDAMENTAL,
        AnalysisDimension.VALUATION,
        AnalysisDimension.TECHNICAL,
        AnalysisDimension.LIQUIDITY,
        AnalysisDimension.RISK,
    }


@pytest.mark.unit
def test_factor_dimensions_can_replace_model_authored_scores():
    records = tuple(record(index) for index in range(1, 21))
    cards = score_factor_universe(records, date(2026, 9, 14))
    chosen = records[-1]
    placeholder = DimensionScore(
        dimension=AnalysisDimension.FUNDAMENTAL,
        score=50,
        confidence=0.5,
        as_of=chosen.as_of,
        evidence=chosen.financial_evidence,
    )
    assessment = SecurityAssessment(
        ticker=chosen.ticker,
        company_name="Test Company",
        sector="Industrials",
        sharia_tier=ShariaTier.OFFICIAL_INDEX,
        sharia_evidence=evidence("egx_sharia_constituents", chosen.ticker),
        dimensions=(placeholder,),
    )

    updated = apply_factor_dimensions((assessment,), records, cards)[0]

    assert len(updated.dimensions) == 5
    assert all(item.score != 50 for item in updated.dimensions)
