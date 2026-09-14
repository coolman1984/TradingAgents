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


@pytest.mark.unit
def test_replace_rejects_same_security_in_short_and_canonical_forms():
    with pytest.raises(ValueError, match="different security"):
        PortfolioRecommendation(
            ticker="EFID",
            action=PortfolioAction.REPLACE,
            target_weight=0.20,
            confidence=0.80,
            rationale="Replacement must be genuinely different.",
            evidence=("dated disclosure",),
            replacement_ticker="EFID.CA",
        )



@pytest.mark.unit
def test_sector_concentration_is_case_insensitive():
    positions = [
        allocation("COMI", 0.25, "Banking"),
        allocation("FAIT", 0.20, " banking "),
        allocation("SWDY", 0.175, "Industrials"),
        allocation("EFID", 0.175, "Consumer"),
    ]
    issues = validate_portfolio(positions, 0.20)
    assert any("45.00%" in issue for issue in issues)


@pytest.mark.unit
def test_sharia_as_of_must_be_a_real_iso_date():
    invalid = allocation("EFID", 0.20)
    invalid = Allocation(
        ticker=invalid.ticker,
        weight=invalid.weight,
        sector=invalid.sector,
        sharia_tier=invalid.sharia_tier,
        sharia_source=invalid.sharia_source,
        sharia_as_of="unknown",
    )
    issues = validate_portfolio(
        [
            invalid,
            allocation("SWDY", 0.20),
            allocation("ORAS", 0.20),
            allocation("ABUK", 0.20),
        ],
        0.20,
    )
    assert any("ISO date" in issue for issue in issues)


@pytest.mark.unit
def test_recommendation_rejects_blank_evidence():
    with pytest.raises(ValueError, match="non-blank"):
        PortfolioRecommendation(
            ticker="EFID",
            action=PortfolioAction.HOLD,
            target_weight=0.20,
            confidence=0.80,
            rationale="Thesis remains valid.",
            evidence=("  ",),
        )


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["^CASE30", "^SHARIAH.CA"])
def test_recommendation_rejects_index_as_security(ticker):
    with pytest.raises(ValueError, match="equity ticker"):
        PortfolioRecommendation(
            ticker=ticker,
            action=PortfolioAction.BUY,
            target_weight=0.20,
            confidence=0.80,
            rationale="Indices are benchmarks, not securities.",
            evidence=("dated disclosure",),
        )
