import pytest

from tradingagents.markets.egypt import (
    EGYPT_MARKET,
    is_egx_ticker,
    normalize_egx_equity_ticker,\n    normalize_egx_ticker,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("COMI", "COMI.CA"),
        ("comi.ca", "COMI.CA"),
        (" EFID ", "EFID.CA"),
        ("^CASE30", "^CASE30"),
        ("^SHARIAH.CA", "^SHARIAH.CA"),
    ],
)
def test_normalize_egx_ticker(raw, expected):
    assert normalize_egx_ticker(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["AAPL.US", "0700.HK", "^GSPC", "", "bad symbol!"])
def test_non_egx_symbols_are_rejected(raw):
    with pytest.raises(ValueError):
        normalize_egx_ticker(raw)


@pytest.mark.unit
def test_egypt_profile_is_advisory_only():
    assert EGYPT_MARKET.advisory_only is True
    assert EGYPT_MARKET.currency == "EGP"
    assert EGYPT_MARKET.sharia_benchmark == "^SHARIAH.CA"
    assert is_egx_ticker("COMI")
    assert not is_egx_ticker("^CASE30")


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["^CASE30", "^SHARIAH.CA"])
def test_equity_normalizer_rejects_indices(ticker):
    with pytest.raises(ValueError, match="equity ticker"):
        normalize_egx_equity_ticker(ticker)
