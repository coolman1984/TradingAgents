"""Structured advisory actions for a portfolio, never order execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tradingagents.markets.egypt import normalize_egx_ticker


class PortfolioAction(str, Enum):
    BUY = "buy"
    ADD = "add"
    HOLD = "hold"
    REDUCE = "reduce"
    SELL = "sell"
    REPLACE = "replace"


@dataclass(frozen=True)
class PortfolioRecommendation:
    ticker: str
    action: PortfolioAction
    target_weight: float
    confidence: float
    rationale: str
    evidence: tuple[str, ...]
    replacement_ticker: str | None = None

    def __post_init__(self) -> None:
        normalize_egx_ticker(self.ticker)
        if not 0 <= self.target_weight <= 1:
            raise ValueError("target_weight must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.rationale.strip():
            raise ValueError("rationale is required")
        if not self.evidence:
            raise ValueError("at least one evidence reference is required")

        if self.action is PortfolioAction.REPLACE:
            if not self.replacement_ticker:
                raise ValueError("replacement_ticker is required for replace")
            normalize_egx_ticker(self.replacement_ticker)
            if self.replacement_ticker.upper() == self.ticker.upper():
                raise ValueError("replacement must be a different security")
        elif self.replacement_ticker is not None:
            raise ValueError("replacement_ticker is only valid for replace")

        if self.action is PortfolioAction.SELL and self.target_weight != 0:
            raise ValueError("sell recommendations must target a zero weight")
