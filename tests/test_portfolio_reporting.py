import json
from datetime import date, datetime, timezone

from typer.testing import CliRunner

from cli.egx_portfolio import app
from tradingagents.markets.egypt_sources import EgyptSourceKey
from tradingagents.portfolio.evidence_store import EvidenceDocument, EvidenceStore
from tradingagents.portfolio.mandate import ShariaTier
from tradingagents.portfolio.models import (
    AdvisoryAction,
    AnalysisDimension,
    DimensionScore,
    EvidenceRef,
    PlanAction,
    PortfolioPlan,
    PortfolioSnapshot,
    SecurityAssessment,
)
from tradingagents.portfolio.reporting import (
    render_arabic_markdown,
    save_plan_report,
)

runner = CliRunner()


def blocked_snapshot(evidence: EvidenceRef) -> PortfolioSnapshot:
    candidate = SecurityAssessment(
        ticker="COMI",
        company_name="Commercial International Bank",
        sector="Banks",
        sharia_tier=ShariaTier.OFFICIAL_INDEX,
        sharia_evidence=evidence,
        dimensions=(
            DimensionScore(
                dimension=AnalysisDimension.FUNDAMENTAL,
                score=80,
                confidence=0.9,
                as_of=date(2026, 9, 1),
                evidence=(evidence,),
            ),
        ),
    )
    return PortfolioSnapshot(
        analysis_date=date(2026, 9, 14),
        cash_egp=10_000,
        candidates=(candidate,),
    )


def keep_cash_plan() -> PortfolioPlan:
    action = PlanAction(
        ticker=None,
        action=AdvisoryAction.KEEP_CASH,
        target_weight=1,
        value_change_egp=1_000,
        confidence=1,
        reasons=("No safe verified allocation is currently available",),
    )
    return PortfolioPlan(
        analysis_date=date(2026, 9, 14),
        investable_value_egp=10_000,
        target_cash_weight=1,
        actions=(action,),
        monthly_contribution_egp=1_000,
        monthly_contribution_actions=(action,),
        blocked_reasons=("Insufficient evidence",),
    )


def test_arabic_report_is_advisory_only():
    report = render_arabic_markdown(keep_cash_plan())
    assert "لا تنفذ أي أمر شراء أو بيع" in report
    assert "الاحتفاظ بالسيولة" in report
    assert "10,000.00 جنيه" in report


def test_report_refuses_overwrite_without_force(tmp_path):
    plan = keep_cash_plan()
    json_path, markdown_path = save_plan_report(plan, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["advisory_only"] is True
    assert markdown_path.exists()

    try:
        save_plan_report(plan, tmp_path)
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("existing reports must not be overwritten")


def test_cli_writes_auditable_blocked_plan_and_returns_code_two(tmp_path):
    database = tmp_path / "evidence.sqlite3"
    stored = EvidenceStore(database).add(
        EvidenceDocument(
            source_key=EgyptSourceKey.EGX_SHARIA_CONSTITUENTS,
            source_url="https://beta.egx.com.eg/ar/media-center",
            published_on=date(2026, 9, 1),
            observed_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
            authority="official",
            subjects=("COMI",),
            payload={"ticker": "COMI.CA"},
        )
    )
    input_path = tmp_path / "snapshot.json"
    input_path.write_text(
        blocked_snapshot(stored.reference).model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "plan",
            str(input_path),
            "--output",
            str(output),
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 2
    assert (output / "portfolio_plan.json").exists()
    assert (output / "portfolio_plan.md").exists()
    payload = json.loads((output / "portfolio_plan.json").read_text(encoding="utf-8"))
    assert payload["blocked_reasons"]


def test_cli_rejects_invalid_json(tmp_path):
    input_path = tmp_path / "snapshot.json"
    input_path.write_text("{broken", encoding="utf-8")

    result = runner.invoke(app, ["plan", str(input_path)])

    assert result.exit_code == 1
    assert "Input or report error" in result.output



def test_cli_rejects_fabricated_snapshot_hash(tmp_path):
    evidence = EvidenceRef(
        source="egx_sharia_constituents",
        url="https://beta.egx.com.eg/ar/media-center",
        published_on=date(2026, 9, 1),
        observed_at=datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        authority="official",
        subjects=("COMI",),
        content_hash="0" * 64,
    )
    input_path = tmp_path / "snapshot.json"
    input_path.write_text(
        blocked_snapshot(evidence).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "plan",
            str(input_path),
            "--database",
            str(tmp_path / "empty.sqlite3"),
        ],
    )

    assert result.exit_code == 1
    assert "not present in the verified store" in result.output
