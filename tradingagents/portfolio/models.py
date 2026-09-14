"""Validated portfolio inputs and outputs.

All dates and evidence are explicit so historical analysis cannot silently use
future information.  These models contain no broker or order fields.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tradingagents.markets.egypt import normalize_egx_ticker
from tradingagents.portfolio.mandate import ShariaTier


class AnalysisDimension(str, Enum):
    FUNDAMENTAL = "fundamental"
    VALUATION = "valuation"
    TECHNICAL = "technical"
    CATALYST = "catalyst"
    LIQUIDITY = "liquidity"
    RISK = "risk"


class EvidenceRef(BaseModel):
    source: str = Field(min_length=1)
    url: str = Field(min_length=1)
    published_on: date
    observed_at: datetime
    authority: Literal["official", "company", "market_data", "secondary", "manual"]
    content_hash: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not (value.startswith("https://") or value.startswith("file://")):
            raise ValueError("evidence URL must use https:// or file://")
        return value


class DimensionScore(BaseModel):
    dimension: AnalysisDimension
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    as_of: date
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)


class SecurityAssessment(BaseModel):
    ticker: str
    company_name: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    sharia_tier: ShariaTier
    sharia_source: str = Field(min_length=1)
    sharia_as_of: date
    dimensions: tuple[DimensionScore, ...] = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_ticker(value)

    @model_validator(mode="after")
    def unique_dimensions(self):
        keys = [item.dimension for item in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError("each analysis dimension may appear only once")
        return self


class PortfolioPosition(BaseModel):
    ticker: str
    units: float = Field(ge=0)
    current_value_egp: float = Field(ge=0)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_ticker(value)


class PortfolioSnapshot(BaseModel):
    analysis_date: date
    cash_egp: float = Field(ge=0)
    positions: tuple[PortfolioPosition, ...] = ()
    candidates: tuple[SecurityAssessment, ...] = Field(min_length=1)
    estimated_switch_cost_pct: float | None = Field(default=None, ge=0, le=0.10)

    @model_validator(mode="after")
    def unique_tickers(self):
        positions = [item.ticker for item in self.positions]
        candidates = [item.ticker for item in self.candidates]
        if len(positions) != len(set(positions)):
            raise ValueError("portfolio positions contain duplicate tickers")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate assessments contain duplicate tickers")
        return self


class AdvisoryAction(str, Enum):
    BUY = "buy"
    ADD = "add"
    HOLD = "hold"
    REDUCE = "reduce"
    SELL = "sell"
    REPLACE = "replace"
    KEEP_CASH = "keep_cash"


class PlanAction(BaseModel):
    ticker: str | None
    action: AdvisoryAction
    target_weight: float = Field(ge=0, le=1)
    value_change_egp: float
    score: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    reasons: tuple[str, ...] = Field(min_length=1)
    replacement_ticker: str | None = None


class PortfolioPlan(BaseModel):
    analysis_date: date
    advisory_only: Literal[True] = True
    investable_value_egp: float = Field(ge=0)
    target_cash_weight: float = Field(ge=0, le=1)
    actions: tuple[PlanAction, ...]
    monthly_contribution_action: PlanAction
    blocked_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
