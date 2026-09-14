"""Deterministic portfolio and Sharia-policy guardrails.

Religious classification must come from dated, attributable evidence.  The
software never invents a fatwa or turns a vague model opinion into eligibility.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from math import isclose
from typing import Iterable

from tradingagents.markets.egypt import is_egx_ticker, normalize_egx_ticker


class ShariaTier(str, Enum):
    OFFICIAL_INDEX = "official_index"
    INDEPENDENTLY_REVIEWED = "independently_reviewed"
    REVIEW_REQUIRED = "review_required"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class Allocation:
    ticker: str
    weight: float
    sector: str
    sharia_tier: ShariaTier
    sharia_source: str
    sharia_as_of: str

    @property
    def canonical_ticker(self) -> str:
        return normalize_egx_ticker(self.ticker)


@dataclass(frozen=True)
class PortfolioMandate:
    starting_capital_egp: float
    monthly_contribution_egp: float
    horizon_min_months: int
    horizon_max_months: int
    max_drawdown: float
    min_positions: int
    max_positions: int
    min_cash_weight: float
    max_cash_weight: float
    max_position_weight: float
    max_sector_weight: float
    allowed_sharia_tiers: tuple[ShariaTier, ...]
    advisory_only: bool = True


DEFAULT_EGX_BALANCED_MANDATE = PortfolioMandate(
    starting_capital_egp=10_000,
    monthly_contribution_egp=1_000,
    horizon_min_months=3,
    horizon_max_months=12,
    max_drawdown=0.20,
    min_positions=4,
    max_positions=5,
    min_cash_weight=0.10,
    max_cash_weight=0.20,
    max_position_weight=0.25,
    max_sector_weight=0.40,
    allowed_sharia_tiers=(
        ShariaTier.OFFICIAL_INDEX,
        ShariaTier.INDEPENDENTLY_REVIEWED,
    ),
    advisory_only=True,
)


def validate_portfolio(
    allocations: Iterable[Allocation],
    cash_weight: float,
    mandate: PortfolioMandate = DEFAULT_EGX_BALANCED_MANDATE,
) -> tuple[str, ...]:
    """Return all mandate violations without hiding one failure behind another."""
    positions = tuple(allocations)
    issues: list[str] = []

    if not mandate.advisory_only:
        issues.append("Egypt portfolio product must remain advisory-only")

    if not mandate.min_positions <= len(positions) <= mandate.max_positions:
        issues.append(
            f"Position count must be between {mandate.min_positions} "
            f"and {mandate.max_positions}"
        )

    if not mandate.min_cash_weight <= cash_weight <= mandate.max_cash_weight:
        issues.append(
            f"Cash weight must be between {mandate.min_cash_weight:.0%} "
            f"and {mandate.max_cash_weight:.0%}"
        )

    total_weight = cash_weight + sum(item.weight for item in positions)
    if not isclose(total_weight, 1.0, abs_tol=1e-6):
        issues.append(f"Portfolio weights must total 100%, got {total_weight:.2%}")

    seen: set[str] = set()
    sector_weights: dict[str, float] = defaultdict(float)

    for item in positions:
        if not 0 < item.weight <= mandate.max_position_weight:
            issues.append(
                f"{item.ticker}: weight must be above 0% and at most "
                f"{mandate.max_position_weight:.0%}"
            )

        if not is_egx_ticker(item.ticker):
            issues.append(f"{item.ticker}: only Egyptian listed equities are allowed")
        else:
            canonical = item.canonical_ticker
            if canonical in seen:
                issues.append(f"{canonical}: duplicate position")
            seen.add(canonical)

        if item.sharia_tier not in mandate.allowed_sharia_tiers:
            issues.append(
                f"{item.ticker}: Sharia tier {item.sharia_tier.value} is not buy-eligible"
            )
        if not item.sharia_source.strip() or not item.sharia_as_of.strip():
            issues.append(
                f"{item.ticker}: dated Sharia evidence and source are required"
            )
        else:
            try:
                date.fromisoformat(item.sharia_as_of)
            except ValueError:
                issues.append(
                    f"{item.ticker}: Sharia as-of must be an ISO date (YYYY-MM-DD)"
                )

        sector = " ".join(item.sector.split()).casefold() or "unknown"
        sector_weights[sector] += item.weight

    for sector, weight in sector_weights.items():
        if weight > mandate.max_sector_weight:
            issues.append(
                f"{sector}: sector weight {weight:.2%} exceeds "
                f"{mandate.max_sector_weight:.0%}"
            )

    return tuple(issues)
