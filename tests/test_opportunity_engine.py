from datetime import date, datetime, timezone

import pytest

from tradingagents.portfolio.mandate import ShariaTier
from tradingagents.portfolio.models import (
    AnalysisDimension,
    DimensionScore,
    EvidenceRef,
    PortfolioPosition,
    PortfolioSnapshot,
    SecurityAssessment,
)
from tradingagents.portfolio.opportunity import (
    MacroRegimeInput,
    MarketRegime,
    OpportunityStatus,
    ValuationCase,
    build_opportunity_board,
    classify_regime,
    evaluate_opportunity,
    evaluate_replacement,
)


def evidence(source: str, ticker: str, authority: str = "official") -> EvidenceRef:
    host = "cbe.org.eg" if source == "cbe_macro" else "beta.egx.com.eg"
    return EvidenceRef(
        source=source,
        url=f"https://{host}/source",
        published_on=date(2026, 9, 1),
        observed_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        authority=authority,
        subjects=(ticker,),
        content_hash="a" * 64,
    )


def candidate(ticker: str = "EFID", score: float = 85) -> SecurityAssessment:
    refs = {
        AnalysisDimension.FUNDAMENTAL: evidence("egx_financial_statements", ticker),
        AnalysisDimension.VALUATION: evidence("egx_financial_statements", ticker),
        AnalysisDimension.TECHNICAL: evidence("egx_prices", ticker),
        AnalysisDimension.CATALYST: evidence("egx_disclosures", ticker),
        AnalysisDimension.LIQUIDITY: evidence("egx_prices", ticker),
        AnalysisDimension.RISK: evidence("cbe_macro", ticker),
    }
    return SecurityAssessment(
        ticker=ticker,
        company_name=f"{ticker} Co",
        sector="Consumer",
        sharia_tier=ShariaTier.OFFICIAL_INDEX,
        sharia_evidence=evidence("egx_sharia_constituents", ticker),
        dimensions=tuple(
            DimensionScore(
                dimension=dimension,
                score=score,
                confidence=0.90,
                as_of=date(2026, 9, 1),
                evidence=(ref,),
            )
            for dimension, ref in refs.items()
        ),
    )


def valuation(
    ticker: str = "EFID",
    *,
    current: float = 100,
    low: float = 105,
    base: float = 125,
    high: float = 145,
    as_of: date = date(2026, 9, 1),
) -> ValuationCase:
    return ValuationCase(
        ticker=ticker,
        current_price_egp=current,
        fair_value_low_egp=low,
        fair_value_base_egp=base,
        fair_value_high_egp=high,
        as_of=as_of,
        evidence=(
            evidence("egx_prices", ticker),
            evidence("egx_financial_statements", ticker),
        ),
    )


@pytest.mark.unit
def test_high_conviction_buy_requires_score_confidence_and_valuation():
    result = evaluate_opportunity(
        candidate(),
        analysis_date=date(2026, 9, 14),
        valuation=valuation(),
    )

    assert result.status is OpportunityStatus.HIGH_CONVICTION_BUY
    assert result.base_upside_pct == pytest.approx(0.25)
    assert result.low_case_return_pct == pytest.approx(0.05)
    assert result.reward_risk_ratio is not None


@pytest.mark.unit
def test_new_candidate_without_valuation_can_only_be_watch():
    result = evaluate_opportunity(
        candidate(),
        analysis_date=date(2026, 9, 14),
    )

    assert result.status is OpportunityStatus.WATCH
    assert any("valuation" in risk.lower() for risk in result.risks)


@pytest.mark.unit
def test_stale_valuation_cannot_force_exit_of_existing_holding():
    result = evaluate_opportunity(
        candidate(score=20),
        analysis_date=date(2026, 9, 14),
        valuation=valuation(as_of=date(2025, 1, 1)),
        held=True,
    )

    assert result.status is OpportunityStatus.HOLD
    assert any("stale" in risk.lower() for risk in result.risks)


@pytest.mark.unit
def test_replacement_requires_known_cost_and_material_edge():
    current = evaluate_opportunity(
        candidate("COMI", score=60),
        analysis_date=date(2026, 9, 14),
        valuation=valuation("COMI", current=100, low=85, base=105, high=120),
        held=True,
    )
    challenger = evaluate_opportunity(
        candidate("EFID", score=88),
        analysis_date=date(2026, 9, 14),
        valuation=valuation("EFID", current=100, low=105, base=135, high=150),
    )

    blocked = evaluate_replacement(current, challenger, switch_cost_pct=None)
    allowed = evaluate_replacement(current, challenger, switch_cost_pct=0.02)

    assert blocked.replace is False
    assert allowed.replace is True
    assert allowed.net_edge_after_cost_pct > 0


@pytest.mark.unit
def test_macro_classifier_uses_only_modest_regime_posture():
    stress = classify_regime(
        MacroRegimeInput(
            fx_depreciation_90d_pct=18,
            market_return_90d_pct=-15,
            market_volatility_60d_pct=40,
            market_breadth_pct=25,
            as_of=date(2026, 9, 1),
            evidence=(evidence("cbe_macro", "EFID"),),
        ),
        date(2026, 9, 14),
    )
    risk_on = classify_regime(
        MacroRegimeInput(
            fx_depreciation_90d_pct=1,
            market_return_90d_pct=12,
            market_volatility_60d_pct=17,
            market_breadth_pct=70,
            as_of=date(2026, 9, 1),
            evidence=(evidence("cbe_macro", "EFID"),),
        ),
        date(2026, 9, 14),
    )

    assert stress.regime is MarketRegime.STRESS
    assert risk_on.regime is MarketRegime.RISK_ON


@pytest.mark.unit
def test_board_separates_new_ideas_holdings_and_replacements():
    snapshot = PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=8_000,
        positions=(
            PortfolioPosition(ticker="COMI", units=10, current_value_egp=2_000),
        ),
        candidates=(candidate("COMI", 60), candidate("EFID", 88)),
        estimated_switch_cost_pct=0.02,
    )
    board = build_opportunity_board(
        snapshot,
        (
            valuation("COMI", current=100, low=85, base=105, high=120),
            valuation("EFID", current=100, low=105, base=135, high=150),
        ),
    )

    assert "EFID.CA" in board.top_new_ideas
    assert "COMI.CA" in board.holding_reviews
    assert board.replacements
    assert board.replacements[0].replace is True
