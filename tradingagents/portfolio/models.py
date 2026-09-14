"""Validated portfolio inputs and outputs.

All dates and evidence are explicit so historical analysis cannot silently use
future information.  These models contain no broker or order fields.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tradingagents.markets.egypt import normalize_egx_equity_ticker
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

    @field_validator("observed_at")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def observed_after_publication(self):
        if self.observed_at.date() < self.published_on:
            raise ValueError("evidence cannot be observed before it is published")
        return self


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
    sharia_evidence: EvidenceRef
    dimensions: tuple[DimensionScore, ...] = Field(min_length=1)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_equity_ticker(value)

    @model_validator(mode="after")
    def unique_dimensions(self):
        keys = [item.dimension for item in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError("each analysis dimension may appear only once")
        if (
            self.sharia_tier is ShariaTier.OFFICIAL_INDEX
            and self.sharia_evidence.authority != "official"
        ):
            raise ValueError("official-index classification requires official evidence")
        return self


class PortfolioPosition(BaseModel):
    ticker: str
    units: float = Field(ge=0)
    current_value_egp: float = Field(ge=0)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_equity_ticker(value)


class PortfolioSnapshot(BaseModel):
    analysis_date: date
    cash_egp: float = Field(ge=0)
    positions: tuple[PortfolioPosition, ...] = ()
    candidates: tuple[SecurityAssessment, ...] = Field(min_length=1)
    market_evidence: tuple[EvidenceRef, ...] = ()
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

    @model_validator(mode="after")
    def action_shape(self):
        if self.action is AdvisoryAction.KEEP_CASH:
            if self.ticker is not None or self.replacement_ticker is not None:
                raise ValueError("keep-cash action cannot name a security")
            return self

        if self.ticker is None:
            raise ValueError("security action requires a ticker")
        self.ticker = normalize_egx_equity_ticker(self.ticker)

        if self.action is AdvisoryAction.REPLACE:
            if self.replacement_ticker is None:
                raise ValueError("replace action requires replacement_ticker")
            self.replacement_ticker = normalize_egx_equity_ticker(self.replacement_ticker)
            if self.replacement_ticker == self.ticker:
                raise ValueError("replacement must be a different security")
        elif self.replacement_ticker is not None:
            raise ValueError("replacement_ticker is only valid for replace")

        if self.action is AdvisoryAction.SELL and self.target_weight != 0:
            raise ValueError("sell action must target zero weight")
        return self


class PortfolioPlan(BaseModel):
    analysis_date: date
    advisory_only: Literal[True] = True
    investable_value_egp: float = Field(ge=0)
    target_cash_weight: float = Field(ge=0, le=1)
    actions: tuple[PlanAction, ...] = Field(min_length=1)
    monthly_contribution_egp: float = Field(ge=0)
    monthly_contribution_actions: tuple[PlanAction, ...] = Field(min_length=1)
    blocked_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_budget(self):
        target_weight = self.target_cash_weight + sum(
            action.target_weight for action in self.actions if action.ticker is not None
        )
        if abs(target_weight - 1.0) > 1e-6:
            raise ValueError(
                f"portfolio target weights must total 100%, got {target_weight:.6f}"
            )

        monthly_values = [
            action.value_change_egp for action in self.monthly_contribution_actions
        ]
        if any(value < 0 for value in monthly_values):
            raise ValueError("monthly contribution actions cannot withdraw money")
        if abs(sum(monthly_values) - self.monthly_contribution_egp) > 0.01:
            raise ValueError("monthly contribution actions must allocate the full amount")
        return self
