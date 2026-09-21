from datetime import date, datetime, timezone

import pytest

from tradingagents.portfolio.backtest import (
    DecisionOutcome,
    evaluate_outcome,
    summarize_outcomes,
)
from tradingagents.portfolio.models import EvidenceRef


def evidence(published_on: date) -> EvidenceRef:
    return EvidenceRef(
        source="egx_prices",
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
        content_hash="c" * 64,
    )


@pytest.mark.unit
def test_outcome_is_net_of_cost_and_benchmark_relative():
    item = DecisionOutcome(
        ticker="EFID",
        signal_date=date(2026, 1, 1),
        exit_date=date(2026, 4, 1),
        entry_price=100,
        exit_price=120,
        benchmark_entry=1000,
        benchmark_exit=1100,
        total_switch_cost_pct=0.02,
        evidence=(evidence(date(2026, 1, 1)),),
    )
    result = evaluate_outcome(item)

    assert result.point_in_time_valid is True
    assert result.gross_return_pct == pytest.approx(0.20)
    assert result.net_return_pct == pytest.approx(0.18)
    assert result.alpha_pct == pytest.approx(0.08)


@pytest.mark.unit
def test_future_evidence_is_excluded_from_default_summary():
    valid = DecisionOutcome(
        ticker="EFID",
        signal_date=date(2026, 1, 1),
        exit_date=date(2026, 4, 1),
        entry_price=100,
        exit_price=110,
        benchmark_entry=1000,
        benchmark_exit=1050,
        evidence=(evidence(date(2026, 1, 1)),),
    )
    leaked = DecisionOutcome(
        ticker="EFID",
        signal_date=date(2026, 1, 1),
        exit_date=date(2026, 4, 1),
        entry_price=100,
        exit_price=200,
        benchmark_entry=1000,
        benchmark_exit=1050,
        evidence=(evidence(date(2026, 2, 1)),),
    )

    summary = summarize_outcomes((valid, leaked))

    assert summary.observations == 2
    assert summary.point_in_time_valid_observations == 1
    assert summary.average_net_return_pct == pytest.approx(0.10)
