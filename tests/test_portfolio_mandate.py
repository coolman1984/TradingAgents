import pytest

from tradingagents.portfolio.mandate import (
    DEFAULT_EGX_BALANCED_MANDATE,
    Allocation,
    ShariaTier,
    validate_portfolio,
)
from tradingagents.portfolio.recommendations import (
    PortfolioAction,
    PortfolioRecommendation,
)


def allocation(ticker, weight, sector="Industrials", tier=ShariaTier.OFFICIAL_INDEX):
    return Allocation(
        ticker=ticker,
        weight=weight,
        sector=sector,
        sharia_tier=tier,
        sharia_source="EGX33 Shariah constituent list",
        sharia_as_of="2026-09-14",
    )


@pytest.mark.unit
def test_balanced_portfolio_passes_all_guardrails():
    positions = [
        allocation("EFID", 0.20, "Consumer"),
        allocation("SWDY", 0.20, "Industrials"),
        allocation("ORAS", 0.20, "Construction"),
        allocation("ABUK", 0.20, "Materials"),
    ]
    assert validate_portfolio(positions, 0.20) == ()


@pytest.mark.unit
def test_validator_reports_sharia_concentration_and_market_failures_together():
    positions = [
        allocation("COMI", 0.30, "Financials", ShariaTier.EXCLUDED),
        allocation("AAPL.US", 0.20, "Technology"),
        allocation("SWDY", 0.20, "Industrials"),
    ]
    issues = validate_portfolio(positions, 0.10)
    joined = "\n".join(issues)
    assert "Position count" in joined
    assert "at most 25%" in joined
    assert "not buy-eligible" in joined
    assert "only Egyptian" in joined
    assert "total 100%" in joined


@pytest.mark.unit
def test_replace_requires_a_different_egyptian_security():
    with pytest.raises(ValueError, match="replacement_ticker"):
        PortfolioRecommendation(
            ticker="EFID",
            action=PortfolioAction.REPLACE,
            target_weight=0.20,
            confidence=0.80,
            rationale="A stronger risk-adjusted alternative exists.",
            evidence=("dated disclosure",),
        )


@pytest.mark.unit
def test_sell_must_target_zero_weight():
    with pytest.raises(ValueError, match="zero weight"):
        PortfolioRecommendation(
            ticker="EFID",
            action=PortfolioAction.SELL,
            target_weight=0.10,
            confidence=0.75,
            rationale="Investment thesis invalidated.",
            evidence=("dated disclosure",),
        )


@pytest.mark.unit
def test_default_mandate_matches_product_scope():
    mandate = DEFAULT_EGX_BALANCED_MANDATE
    assert mandate.starting_capital_egp == 10_000
    assert mandate.monthly_contribution_egp == 1_000
    assert mandate.max_drawdown == 0.20
    assert mandate.advisory_only is True
