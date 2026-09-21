"""Connected, advisory-only synthesis for human portfolio review.

This module never sends orders. It combines independent research layers and
flags which holdings or candidates deserve human attention.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingagents.portfolio.opportunity import (
    OpportunityBoard,
    OpportunityRequest,
    OpportunityStatus,
    build_opportunity_board,
)
from tradingagents.portfolio.thesis import (
    ThesisEvaluation,
    ThesisStatus,
    ThesisTracker,
    evaluate_thesis,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewPosture(str, Enum):
    INITIATE_REVIEW = "initiate_review"
    INCREASE_REVIEW = "increase_review"
    MAINTAIN = "maintain"
    REDUCE_REVIEW = "reduce_review"
    CLOSE_REVIEW = "close_review"
    ROTATE_REVIEW = "rotate_review"
    RE_UNDERWRITE = "re_underwrite"
    WATCH = "watch"
    WAIT_FOR_EVIDENCE = "wait_for_evidence"


class StrategyRequest(StrictModel):
    opportunity: OpportunityRequest
    theses: tuple[ThesisTracker, ...] = ()

    @model_validator(mode="after")
    def validate_thesis_universe(self):
        thesis_tickers = [item.ticker for item in self.theses]
        if len(thesis_tickers) != len(set(thesis_tickers)):
            raise ValueError("strategy request contains duplicate thesis tickers")
        candidate_tickers = {
            item.ticker for item in self.opportunity.snapshot.candidates
        }
        unknown = sorted(set(thesis_tickers) - candidate_tickers)
        if unknown:
            raise ValueError(
                "thesis tickers are outside the opportunity universe: "
                + ", ".join(unknown)
            )
        return self


class StrategyDirective(StrictModel):
    ticker: str
    posture: ReviewPosture
    alternative_ticker: str | None = None
    opportunity_status: OpportunityStatus
    thesis_status: ThesisStatus | None = None
    confidence: float = Field(ge=0, le=1)
    reasons: tuple[str, ...] = Field(min_length=1)


class StrategyPlan(StrictModel):
    analysis_date: str
    advisory_only: Literal[True] = True
    opportunity_board: OpportunityBoard
    thesis_evaluations: tuple[ThesisEvaluation, ...]
    directives: tuple[StrategyDirective, ...]
    new_money_priority: tuple[str, ...]
    review_queue: tuple[str, ...]


def _evaluate_theses(
    request: StrategyRequest,
) -> dict[str, ThesisEvaluation]:
    analysis_date = request.opportunity.snapshot.analysis_date
    return {
        tracker.ticker: evaluate_thesis(tracker, analysis_date)
        for tracker in request.theses
    }


def build_strategy_plan(request: StrategyRequest) -> StrategyPlan:
    """Combine opportunity, thesis and rotation evidence for human review."""
    snapshot = request.opportunity.snapshot
    board = build_opportunity_board(
        snapshot,
        request.opportunity.valuations,
        request.opportunity.macro,
    )
    theses = _evaluate_theses(request)
    rotations = {item.current_ticker: item for item in board.replacements}

    directives: list[StrategyDirective] = []
    review_queue: list[str] = []

    for item in board.opportunities:
        thesis = theses.get(item.ticker)
        thesis_status = thesis.status if thesis is not None else None
        rotation = rotations.get(item.ticker)
        reasons: list[str] = []

        if item.blockers:
            posture = ReviewPosture.WAIT_FOR_EVIDENCE
            reasons.extend(item.blockers)
        elif not item.held:
            if item.status in {
                OpportunityStatus.HIGH_CONVICTION_BUY,
                OpportunityStatus.BUY,
            }:
                posture = ReviewPosture.INITIATE_REVIEW
                reasons.extend(item.reasons or ("Candidate clears entry gates",))
            else:
                posture = ReviewPosture.WATCH
                reasons.extend(item.reasons or item.risks or ("Watch candidate",))
        elif thesis_status is ThesisStatus.BROKEN:
            posture = ReviewPosture.CLOSE_REVIEW
            reasons.append("A fresh, explicit thesis pillar is broken")
            reasons.extend(thesis.broken_pillars)
        elif rotation is not None:
            posture = ReviewPosture.ROTATE_REVIEW
            reasons.extend(rotation.reasons)
        elif item.status is OpportunityStatus.EXIT:
            posture = ReviewPosture.CLOSE_REVIEW
            reasons.extend(item.reasons)
        elif item.status is OpportunityStatus.TRIM:
            posture = ReviewPosture.REDUCE_REVIEW
            reasons.extend(item.reasons)
        elif thesis_status in {ThesisStatus.IMPAIRED, ThesisStatus.WATCH}:
            posture = ReviewPosture.RE_UNDERWRITE
            reasons.append("Thesis evidence weakened and needs fresh underwriting")
            reasons.extend(thesis.warning_pillars)
        elif item.status is OpportunityStatus.ADD:
            if thesis_status is ThesisStatus.UNTESTED:
                posture = ReviewPosture.WAIT_FOR_EVIDENCE
                reasons.append("Wait for fresh thesis proof before increasing exposure")
            else:
                posture = ReviewPosture.INCREASE_REVIEW
                reasons.extend(item.reasons)
        else:
            posture = ReviewPosture.MAINTAIN
            reasons.extend(item.reasons or ("No stronger posture is evidence-backed",))

        if thesis is None and item.held:
            reasons.append("No explicit thesis tracker is attached to this holding")

        if posture in {
            ReviewPosture.ROTATE_REVIEW,
            ReviewPosture.RE_UNDERWRITE,
            ReviewPosture.REDUCE_REVIEW,
            ReviewPosture.CLOSE_REVIEW,
            ReviewPosture.WAIT_FOR_EVIDENCE,
        }:
            review_queue.append(item.ticker)

        directives.append(
            StrategyDirective(
                ticker=item.ticker,
                posture=posture,
                alternative_ticker=(
                    rotation.challenger_ticker if rotation is not None else None
                ),
                opportunity_status=item.status,
                thesis_status=thesis_status,
                confidence=item.confidence,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    priority = tuple(
        directive.ticker
        for directive in directives
        if directive.posture in {
            ReviewPosture.INCREASE_REVIEW,
            ReviewPosture.INITIATE_REVIEW,
        }
    )

    return StrategyPlan(
        analysis_date=snapshot.analysis_date.isoformat(),
        opportunity_board=board,
        thesis_evaluations=tuple(theses.values()),
        directives=tuple(directives),
        new_money_priority=priority,
        review_queue=tuple(dict.fromkeys(review_queue)),
    )
