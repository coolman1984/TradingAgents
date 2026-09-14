from datetime import date, datetime

import pytest
from pydantic import ValidationError

from tradingagents.portfolio.models import (
    AdvisoryAction,
    EvidenceRef,
    PlanAction,
    PortfolioPlan,
)


def test_evidence_time_requires_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        EvidenceRef(
            source="egx_disclosures",
            url="https://www.egx.com.eg/disclosures",
            published_on=date(2026, 9, 1),
            observed_at=datetime(2026, 9, 1, 12),
            authority="official",
        )


def test_plan_action_rejects_benchmark_as_security():
    with pytest.raises(ValidationError, match="equity ticker"):
        PlanAction(
            ticker="^CASE30",
            action=AdvisoryAction.BUY,
            target_weight=0.20,
            value_change_egp=2_000,
            confidence=0.80,
            reasons=("Benchmark is not an equity",),
        )


def test_plan_rejects_incomplete_monthly_budget():
    keep_cash = PlanAction(
        ticker=None,
        action=AdvisoryAction.KEEP_CASH,
        target_weight=1,
        value_change_egp=500,
        confidence=1,
        reasons=("Test",),
    )
    with pytest.raises(ValidationError, match="allocate the full amount"):
        PortfolioPlan(
            analysis_date=date(2026, 9, 14),
            investable_value_egp=10_000,
            target_cash_weight=1,
            actions=(keep_cash,),
            monthly_contribution_egp=1_000,
            monthly_contribution_actions=(keep_cash,),
        )


def test_plan_rejects_target_weight_budget_mismatch():
    buy = PlanAction(
        ticker="EFID",
        action=AdvisoryAction.BUY,
        target_weight=0.50,
        value_change_egp=5_000,
        confidence=0.80,
        reasons=("Test",),
    )
    monthly = PlanAction(
        ticker=None,
        action=AdvisoryAction.KEEP_CASH,
        target_weight=0.20,
        value_change_egp=1_000,
        confidence=1,
        reasons=("Test",),
    )
    with pytest.raises(ValidationError, match="target weights"):
        PortfolioPlan(
            analysis_date=date(2026, 9, 14),
            investable_value_egp=10_000,
            target_cash_weight=0.20,
            actions=(buy,),
            monthly_contribution_egp=1_000,
            monthly_contribution_actions=(monthly,),
        )
