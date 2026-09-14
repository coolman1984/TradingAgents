"""Deterministic Egyptian Exchange market rules.

This module deliberately contains no network calls and no LLM decisions.  It
is the stable boundary between portfolio logic and replaceable data adapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EGX_CODE = re.compile(r"^[A-Z0-9]{2,16}$")
_EGX_SUFFIX = ".CA"
_ALLOWED_INDICES = frozenset({"^CASE30", "^SHARIAH.CA"})


@dataclass(frozen=True)
class EgyptMarketProfile:
    market_code: str = "EGX"
    country: str = "Egypt"
    currency: str = "EGP"
    yahoo_suffix: str = _EGX_SUFFIX
    broad_benchmark: str = "^CASE30"
    sharia_benchmark: str = "^SHARIAH.CA"
    advisory_only: bool = True


EGYPT_MARKET = EgyptMarketProfile()


def normalize_egx_ticker(raw: str) -> str:
    """Return the canonical Yahoo ticker for an Egyptian listed equity.

    Plain EGX codes such as COMI become COMI.CA.  Existing .CA tickers remain
    unchanged.  Foreign exchange suffixes are rejected so an Egypt-only scan
    cannot silently analyze the wrong market.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("EGX ticker must be a non-empty string")

    ticker = raw.strip().upper()
    if ticker in _ALLOWED_INDICES:
        return ticker

    if ticker.endswith(_EGX_SUFFIX):
        base = ticker[: -len(_EGX_SUFFIX)]
        if not _EGX_CODE.fullmatch(base):
            raise ValueError(f"Invalid EGX ticker: {raw!r}")
        return ticker

    if "." in ticker or ticker.startswith("^"):
        raise ValueError(f"Non-EGX ticker is not allowed: {raw!r}")
    if not _EGX_CODE.fullmatch(ticker):
        raise ValueError(f"Invalid EGX ticker: {raw!r}")
    return f"{ticker}{_EGX_SUFFIX}"


def is_egx_ticker(raw: str) -> bool:
    """Return True only for a valid canonical EGX equity ticker."""
    try:
        ticker = normalize_egx_ticker(raw)
    except ValueError:
        return False
    return ticker.endswith(_EGX_SUFFIX) and not ticker.startswith("^")
