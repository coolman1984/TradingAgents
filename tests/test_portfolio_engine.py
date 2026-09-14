from datetime import date

import pytest

from tradingagents.portfolio.engine import build_portfolio_plan, score_security
from tradingagents.portfolio.mandate import ShariaTier
from tradingagents.portfolio.models import (
    AdvisoryAction,
    AnalysisDimension,
    DimensionScore,
    EvidenceRef,
    PortfolioPosition,
    PortfolioSnapshot,
    SecurityAssessment,
)


_SOURCE_URLS = {
    "egx_prices": "https://www.egx.com.eg/prices",
    "egx_disclosures": "https://www.egx.com.eg/disclosures",
    "egx_financial_statements": "https://www.egx.com.eg/financials",
    "egx_sharia_constituents": "https://www.egx.com.eg/sharia",
    "cbe_macro": "https://www.cbe.org.eg/rates",
    "capmas_inflation": "https://www.capmas.gov.eg/inflation",
    "fra_rules": "https://www.fra.gov.eg/rules",
    "independent_sharia_review": "file:///reviews/sharia-review.json",
}


def evidence(
    published_on="2026-09-01",
    authority="official",
    source="egx_disclosures",
):
    return EvidenceRef(
        source=source,
        url=_SOURCE_URLS[source],
        published_on=published_on,
        observed_at=f"{published_on}T09:00:00+03:00",
        authority=authority,
        content_hash="0" * 64,
    )


def candidate(
    ticker,
    score=80,
    sector="Sector",
    as_of="2026-09-01",
    tier=ShariaTier.OFFICIAL_INDEX,
):
    dimensions = tuple(
        DimensionScore(
            dimension=dimension,
            score=score,
            confidence=0.90,
            as_of=as_of,
            evidence=(evidence(published_on=as_of),),
        )
        for dimension in AnalysisDimension
    )
    return SecurityAssessment(
        ticker=ticker,
        company_name=ticker,
        sector=sector,
        sharia_tier=tier,
        sharia_evidence=(
            evidence(source="egx_sharia_constituents")
            if tier is ShariaTier.OFFICIAL_INDEX
            else evidence(
                source="independent_sharia_review",
                authority="secondary",
            )
        ),
        dimensions=dimensions,
    )


def market_evidence():
    return tuple(
        evidence(
            source=source,
            published_on="2026-09-14" if source == "egx_prices" else "2026-09-01",
        )
        for source in _SOURCE_URLS
        if source != "independent_sharia_review"
    )


def strong_candidates():
    return (
        candidate("EFID", 84, "Consumer"),
        candidate("SWDY", 82, "Industrials"),
        candidate("ORAS", 80, "Construction"),
        candidate("ABUK", 78, "Materials"),
    )


@pytest.mark.unit
def test_initial_plan_keeps_cash_and_equal_weights_four_names():
    snapshot = PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=10_000,
        candidates=strong_candidates(),
        market_evidence=market_evidence(),
    )
    plan = build_portfolio_plan(snapshot)
    buys = [item for item in plan.actions if item.action is AdvisoryAction.BUY]

    assert not plan.blocked_reasons
    assert len(buys) == 4
    assert all(item.target_weight == pytest.approx(0.20) for item in buys)
    assert sum(item.value_change_egp for item in buys) == pytest.approx(8_000)
    assert plan.target_cash_weight == pytest.approx(0.20)
    assert sum(
        item.value_change_egp for item in plan.monthly_contribution_actions
    ) == pytest.approx(1_000)
    assert [
        item.value_change_egp for item in plan.monthly_contribution_actions
    ] == [200, 200, 200, 200, 200]
    assert plan.monthly_contribution_actions[-1].action is AdvisoryAction.KEEP_CASH
    assert plan.advisory_only is True


@pytest.mark.unit
def test_plan_refuses_to_force_investment_with_too_few_candidates():
    snapshot = PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=10_000,
        candidates=strong_candidates()[:3],
        market_evidence=market_evidence(),
    )
    plan = build_portfolio_plan(snapshot)

    assert plan.blocked_reasons
    assert plan.actions[0].action is AdvisoryAction.KEEP_CASH
    assert plan.target_cash_weight == 1.0


@pytest.mark.unit
def test_stale_required_evidence_blocks_candidate():
    stale = candidate("EFID", as_of="2025-01-01")
    result = score_security(stale, date(2026, 9, 14))

    assert result.decision_ready is False
    assert result.eligible is False
    assert any("stale" in issue for issue in result.issues)


@pytest.mark.unit
def test_future_evidence_blocks_candidate():
    future = candidate("EFID", as_of="2026-10-01")
    result = score_security(future, date(2026, 9, 14))

    assert result.decision_ready is False
    assert result.eligible is False
    assert any("future" in issue for issue in result.issues)


@pytest.mark.unit
def test_missing_switch_cost_suppresses_replacement_and_preserves_budget():
    weak = candidate("TMGH", 60, "Real Estate")
    snapshot = PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=8_000,
        positions=(
            PortfolioPosition(ticker="TMGH", units=10, current_value_egp=2_000),
        ),
        candidates=(*strong_candidates(), weak),
        market_evidence=market_evidence(),
    )
    plan = build_portfolio_plan(snapshot)

    weak_action = next(item for item in plan.actions if item.ticker == "TMGH.CA")
    buys = [item for item in plan.actions if item.action is AdvisoryAction.BUY]
    assert weak_action.action is AdvisoryAction.HOLD
    assert sum(item.value_change_egp for item in buys) == pytest.approx(6_000)
    assert any("switch cost is missing" in warning for warning in plan.warnings)


@pytest.mark.unit
def test_costed_replacement_names_the_stronger_security():
    weak = candidate("TMGH", 60, "Real Estate")
    snapshot = PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=8_000,
        positions=(
            PortfolioPosition(ticker="TMGH", units=10, current_value_egp=2_000),
        ),
        candidates=(*strong_candidates(), weak),
        market_evidence=market_evidence(),
        estimated_switch_cost_pct=0.01,
    )
    plan = build_portfolio_plan(snapshot)

    action = next(item for item in plan.actions if item.ticker == "TMGH.CA")
    assert action.action is AdvisoryAction.REPLACE
    assert action.replacement_ticker in {item.ticker for item in strong_candidates()}
    assert action.target_weight == 0


@pytest.mark.unit
def test_stale_low_score_position_is_held_not_sold():
    stale_weak = candidate("TMGH", 20, "Real Estate", as_of="2025-01-01")
    snapshot = PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=8_000,
        positions=(
            PortfolioPosition(ticker="TMGH", units=10, current_value_egp=2_000),
        ),
        candidates=(*strong_candidates(), stale_weak),
        market_evidence=market_evidence(),
        estimated_switch_cost_pct=0.01,
    )
    plan = build_portfolio_plan(snapshot)

    action = next(item for item in plan.actions if item.ticker == "TMGH.CA")
    assert action.action is AdvisoryAction.HOLD
    assert action.value_change_egp == 0


@pytest.mark.unit
def test_official_index_tier_requires_official_evidence():
    with pytest.raises(ValueError, match="official evidence"):
        SecurityAssessment(
            ticker="EFID",
            company_name="Edita",
            sector="Consumer",
            sharia_tier=ShariaTier.OFFICIAL_INDEX,
            sharia_evidence=evidence(authority="secondary"),
            dimensions=candidate("SWDY").dimensions,
        )


@pytest.mark.unit
def test_official_sharia_tier_ranks_before_independent_review():
    candidates = (
        candidate("EFID", 70, "Consumer"),
        candidate("SWDY", 69, "Industrials"),
        candidate("ORAS", 68, "Construction"),
        candidate("ABUK", 67, "Materials"),
        candidate(
            "TMGH",
            99,
            "Real Estate",
            tier=ShariaTier.INDEPENDENTLY_REVIEWED,
        ),
    )
    plan = build_portfolio_plan(
        PortfolioSnapshot(
            analysis_date=date(2026, 9, 14),
            cash_egp=10_000,
            candidates=candidates,
            market_evidence=market_evidence(),
        )
    )

    bought = {
        action.ticker
        for action in plan.actions
        if action.action is AdvisoryAction.BUY
    }
    assert "TMGH.CA" not in bought
    assert bought == {"EFID.CA", "SWDY.CA", "ORAS.CA", "ABUK.CA"}



@pytest.mark.unit
def test_plan_blocks_when_required_market_sources_are_missing():
    plan = build_portfolio_plan(
        PortfolioSnapshot(
            analysis_date=date(2026, 9, 14),
            cash_egp=10_000,
            candidates=strong_candidates(),
        )
    )

    assert plan.blocked_reasons
    assert "Missing fresh required Egyptian sources" in plan.blocked_reasons[0]
    assert plan.actions[0].action is AdvisoryAction.KEEP_CASH
