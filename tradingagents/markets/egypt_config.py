"""Configuration overlay for the Egypt-only portfolio adviser."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio.mandate import DEFAULT_EGX_BALANCED_MANDATE


def build_egypt_config(base: dict | None = None) -> dict:
    """Build an isolated EGX advisory configuration without mutating the base."""
    config = deepcopy(DEFAULT_CONFIG if base is None else base)
    config.update(
        {
            "market_profile": "egypt",
            "portfolio_mode": True,
            "advisory_only": True,
            "output_language": "Arabic",
            "benchmark_ticker": "^SHARIAH.CA",
            "portfolio_mandate": asdict(DEFAULT_EGX_BALANCED_MANDATE),
            "required_egypt_sources": (
                "egx_prices",
                "egx_disclosures",
                "egx_financial_statements",
                "egx_sharia_constituents",
                "cbe_macro",
                "capmas_inflation",
                "fra_rules",
            ),
            # Until every required adapter is implemented and verified, callers
            # must keep this False and must not label output production-ready.
            "egypt_data_adapters_ready": False,
            "global_news_queries": [
                "Egyptian Exchange EGX listed companies disclosures",
                "Central Bank of Egypt interest rates inflation exchange rate",
                "Egypt economy fiscal policy foreign investment",
                "Egypt sectors construction fertilizers food telecom real estate",
                "Red Sea Suez Canal energy commodities impact on Egypt",
            ],
        }
    )
    benchmark_map = dict(config.get("benchmark_map", {}))
    benchmark_map[".CA"] = "^CASE30"
    config["benchmark_map"] = benchmark_map
    return config


def assert_egypt_recommendation_ready(config: dict) -> None:
    """Fail closed until authoritative Egyptian data adapters are verified."""
    if config.get("market_profile") != "egypt":
        raise ValueError("Egypt market profile is required")
    if config.get("advisory_only") is not True:
        raise ValueError("Egypt portfolio product must remain advisory-only")
    if config.get("egypt_data_adapters_ready") is not True:
        raise RuntimeError(
            "Egyptian data adapters are not verified; portfolio recommendations "
            "must not be presented as production-ready"
        )
