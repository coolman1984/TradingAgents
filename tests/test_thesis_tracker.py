from datetime import date, datetime, timezone

import pytest

from tradingagents.portfolio.models import EvidenceRef
from tradingagents.portfolio.thesis import (
    PillarStatus,
    ThesisAction,
    ThesisPillar,
    ThesisStatus,
    ThesisTracker,
    evaluate_thesis,
)


def evidence(published_on: date = date(2026, 9, 1)) -> EvidenceRef:
    return EvidenceRef(
        source="egx_financial_statements",
        url="https://beta.egx.com.eg/source",
        published_on=published_on,
        observed_at=datetime(
            published_on.year,
            published_on.month,
            published_on.day,
            12,
            tzinfo=timezone.utc,
        ),
        authority="official",
        subjects=("EFID",),
        content_hash="b" * 64,
    )


def pillar(name: str, value: float, *, as_of: date = date(2026, 9, 1)):
    return ThesisPillar(
        name=name,
        metric="revenue_growth_pct",
        higher_is_better=True,
        confirm_threshold=20,
        warning_threshold=10,
        break_threshold=0,
        current_value=value,
        as_of=as_of,
        evidence=(evidence(as_of),),
        next_test=f"Recheck {name} at next results",
    )


@pytest.mark.unit
def test_broken_pillar_escalates_to_exit_review():
    tracker = ThesisTracker(
        ticker="EFID",
        thesis="Growth remains durable",
        pillars=(pillar("growth", -2),),
    )
    result = evaluate_thesis(tracker, date(2026, 9, 14))

    assert result.status is ThesisStatus.BROKEN
    assert result.action is ThesisAction.EXIT_REVIEW
    assert result.pillars[0].status is PillarStatus.BROKEN


@pytest.mark.unit
def test_two_warning_pillars_impair_thesis():
    tracker = ThesisTracker(
        ticker="EFID",
        thesis="Growth remains durable",
        pillars=(pillar("growth", 5), pillar("margin", 7)),
    )
    result = evaluate_thesis(tracker, date(2026, 9, 14))

    assert result.status is ThesisStatus.IMPAIRED
    assert result.action is ThesisAction.RE_UNDERWRITE


@pytest.mark.unit
def test_stale_broken_value_is_not_allowed_to_trigger_exit():
    tracker = ThesisTracker(
        ticker="EFID",
        thesis="Growth remains durable",
        pillars=(pillar("growth", -20, as_of=date(2025, 1, 1)),),
    )
    result = evaluate_thesis(tracker, date(2026, 9, 14))

    assert result.status is ThesisStatus.UNTESTED
    assert result.action is ThesisAction.WAIT_FOR_PROOF
    assert result.pillars[0].status is PillarStatus.UNTESTED
