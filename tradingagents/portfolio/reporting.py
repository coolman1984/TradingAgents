"""Human-readable and machine-readable output for EGX advisory plans."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tradingagents.portfolio.models import AdvisoryAction, PlanAction, PortfolioPlan

_ACTION_ARABIC = {
    AdvisoryAction.BUY: "شراء",
    AdvisoryAction.ADD: "زيادة",
    AdvisoryAction.HOLD: "احتفاظ",
    AdvisoryAction.REDUCE: "تخفيض",
    AdvisoryAction.SELL: "بيع",
    AdvisoryAction.REPLACE: "استبدال",
    AdvisoryAction.KEEP_CASH: "الاحتفاظ بالسيولة",
}


def _safe_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _action_row(action: PlanAction) -> str:
    ticker = action.ticker or "—"
    if action.replacement_ticker:
        ticker = f"{ticker} ← {action.replacement_ticker}"
    score = "—" if action.score is None else f"{action.score:.1f}"
    reason = "؛ ".join(action.reasons)
    return (
        f"| {_safe_cell(ticker)} | {_ACTION_ARABIC[action.action]} | "
        f"{action.target_weight:.1%} | {action.value_change_egp:,.2f} | "
        f"{score} | {action.confidence:.0%} | {_safe_cell(reason)} |"
    )


def render_arabic_markdown(plan: PortfolioPlan) -> str:
    """Render a deterministic Arabic advisory report."""
    lines = [
        "# خطة المحفظة المصرية",
        "",
        "> هذه توصية بحثية فقط، ولا تنفذ أي أمر شراء أو بيع.",
        "",
        f"- تاريخ التحليل: {plan.analysis_date.isoformat()}",
        f"- قيمة المحفظة: {plan.investable_value_egp:,.2f} جنيه",
        f"- السيولة المستهدفة: {plan.target_cash_weight:.1%}",
        "",
        "## الإجراءات",
        "",
        "| السهم | القرار | الوزن المستهدف | التغير بالجنيه | الدرجة | الثقة | السبب |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(_action_row(action) for action in plan.actions)

    monthly = plan.monthly_contribution_action
    lines.extend(
        [
            "",
            "## مساهمة الشهر",
            "",
            _action_row(monthly),
        ]
    )

    if plan.blocked_reasons:
        lines.extend(["", "## أسباب حجب التوصية", ""])
        lines.extend(f"- {_safe_cell(reason)}" for reason in plan.blocked_reasons)
    if plan.warnings:
        lines.extend(["", "## تنبيهات البيانات", ""])
        lines.extend(f"- {_safe_cell(warning)}" for warning in plan.warnings)
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_plan_report(
    plan: PortfolioPlan,
    output_directory: str | Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Atomically write JSON and Markdown, refusing accidental overwrites."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "portfolio_plan.json"
    markdown_path = output / "portfolio_plan.md"

    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing report files: {names}")

    _atomic_write(json_path, plan.model_dump_json(indent=2))
    _atomic_write(markdown_path, render_arabic_markdown(plan))
    return json_path, markdown_path
