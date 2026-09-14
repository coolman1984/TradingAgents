from copy import deepcopy

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.markets.egypt_config import (
    assert_egypt_recommendation_ready,
    build_egypt_config,
)


@pytest.mark.unit
def test_egypt_config_is_isolated_and_advisory_only():
    base = deepcopy(DEFAULT_CONFIG)
    result = build_egypt_config(base)
    assert result is not base
    assert "market_profile" not in base
    assert result["market_profile"] == "egypt"
    assert result["portfolio_mode"] is True
    assert result["advisory_only"] is True
    assert result["output_language"] == "Arabic"
    assert result["benchmark_ticker"] == "^SHARIAH.CA"
    assert result["benchmark_map"][".CA"] == "^CASE30"


@pytest.mark.unit
def test_egypt_config_fails_closed_before_adapters_are_verified():
    config = build_egypt_config()
    with pytest.raises(RuntimeError, match="not verified"):
        assert_egypt_recommendation_ready(config)


@pytest.mark.unit
def test_egypt_config_can_be_marked_ready_after_verification():
    config = build_egypt_config()
    config["egypt_data_adapters_ready"] = True
    assert_egypt_recommendation_ready(config)
