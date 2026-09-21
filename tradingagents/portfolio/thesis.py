"""Falsifiable thesis tracking for EGX portfolio research."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.markets.egypt import normalize_egx_equity_ticker
from tradingagents.markets.egypt_sources import validate_evidence_for_date
from tradingagents.portfolio.models import EvidenceRef


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PillarStatus(str, Enum):
    UNTESTED = "untested"
    CONFIRMED = "confirmed"
    INTACT = "intact"
    WARNING = "warning"
    BROKEN = "broken"


class ThesisStatus(str, Enum):
    UNTESTED = "untested"
    INTACT = "intact"
    WATCH = "watch"
    IMPAIRED = "impaired"
    BROKEN = "broken"


class ThesisAction(str, Enum):
    WAIT_FOR_PROOF = "wait_for_proof"
    HOLD_THESIS = "hold_thesis"
    RE_UNDERWRITE = "re_underwrite"
    EXIT_REVIEW = "exit_review"


class ThesisPillar(StrictModel):
    name: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    higher_is_better: bool
    confirm_threshold: float
    warning_threshold: float
    break_threshold: float
    current_value: float | None = None
    as_of: date | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    next_test: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_threshold_order(self):
        if self.higher_is_better:
            if not self.break_threshold < self.warning_threshold < self.confirm_threshold:
                raise ValueError(
                    "higher-is-better thresholds must satisfy break < warning < confirm"
                )
        elif not self.confirm_threshold < self.warning_threshold < self.break_threshold:
            raise ValueError(
                "lower-is-better thresholds must satisfy confirm < warning < break"
            )
        if self.current_value is not None and (self.as_of is None or not self.evidence):
            raise ValueError("observed pillar values require dated evidence")
        return self


class ThesisTracker(StrictModel):
    ticker: str
    thesis: str = Field(min_length=1)
    pillars: tuple[ThesisPillar, ...] = Field(min_length=1)
    next_catalyst: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_egx_equity_ticker(value)


class PillarEvaluation(StrictModel):
    name: str
    metric: str
    status: PillarStatus
    current_value: float | None
    next_test: str
    reasons: tuple[str, ...]


class ThesisEvaluation(StrictModel):
    ticker: str
    status: ThesisStatus
    action: ThesisAction
    pillars: tuple[PillarEvaluation, ...]
    broken_pillars: tuple[str, ...]
    warning_pillars: tuple[str, ...]
    next_proof_points: tuple[str, ...]


def _evaluate_value(pillar: ThesisPillar) -> PillarStatus:
    if pillar.current_value is None:
        return PillarStatus.UNTESTED

    value = pillar.current_value
    if pillar.higher_is_better:
        if value <= pillar.break_threshold:
            return PillarStatus.BROKEN
        if value <= pillar.warning_threshold:
            return PillarStatus.WARNING
        if value >= pillar.confirm_threshold:
            return PillarStatus.CONFIRMED
        return PillarStatus.INTACT

    if value >= pillar.break_threshold:
        return PillarStatus.BROKEN
    if value >= pillar.warning_threshold:
        return PillarStatus.WARNING
    if value <= pillar.confirm_threshold:
        return PillarStatus.CONFIRMED
    return PillarStatus.INTACT


def evaluate_thesis(
    tracker: ThesisTracker,
    analysis_date: date,
    *,
    maximum_age_days: int = 220,
) -> ThesisEvaluation:
    """Evaluate a thesis without allowing stale evidence to trigger an exit."""

    evaluations: list[PillarEvaluation] = []
    broken: list[str] = []
    warnings: list[str] = []

    for pillar in tracker.pillars:
        reasons: list[str] = []
        status = _evaluate_value(pillar)

        if pillar.current_value is not None:
            assert pillar.as_of is not None
            stale = (
                pillar.as_of > analysis_date
                or (analysis_date - pillar.as_of).days > maximum_age_days
            )
            evidence_issues = tuple(
                issue
                for ref in pillar.evidence
                for issue in validate_evidence_for_date(ref, analysis_date)
            )
            if stale or evidence_issues:
                status = PillarStatus.UNTESTED
                if stale:
                    reasons.append("Pillar observation is stale or from the future")
                reasons.extend(f"Evidence: {item}" for item in evidence_issues)

        if status is PillarStatus.BROKEN:
            broken.append(pillar.name)
            reasons.append("Observed value crossed the explicit break threshold")
        elif status is PillarStatus.WARNING:
            warnings.append(pillar.name)
            reasons.append("Observed value crossed the warning threshold")
        elif status is PillarStatus.CONFIRMED:
            reasons.append("Observed value clears the confirmation threshold")
        elif status is PillarStatus.INTACT:
            reasons.append("Observed value remains between warning and confirmation")
        else:
            reasons.append("Fresh evidence is not sufficient to test this pillar")

        evaluations.append(
            PillarEvaluation(
                name=pillar.name,
                metric=pillar.metric,
                status=status,
                current_value=pillar.current_value,
                next_test=pillar.next_test,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    untested = [x for x in evaluations if x.status is PillarStatus.UNTESTED]
    if broken:
        status = ThesisStatus.BROKEN
        action = ThesisAction.EXIT_REVIEW
    elif len(warnings) >= 2:
        status = ThesisStatus.IMPAIRED
        action = ThesisAction.RE_UNDERWRITE
    elif warnings:
        status = ThesisStatus.WATCH
        action = ThesisAction.RE_UNDERWRITE
    elif untested:
        status = ThesisStatus.UNTESTED
        action = ThesisAction.WAIT_FOR_PROOF
    else:
        status = ThesisStatus.INTACT
        action = ThesisAction.HOLD_THESIS

    return ThesisEvaluation(
        ticker=tracker.ticker,
        status=status,
        action=action,
        pillars=tuple(evaluations),
        broken_pillars=tuple(broken),
        warning_pillars=tuple(warnings),
        next_proof_points=tuple(item.next_test for item in tracker.pillars),
    )
