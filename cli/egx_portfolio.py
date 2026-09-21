"""Non-interactive command line interface for the EGX portfolio adviser."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from pydantic import ValidationError

from tradingagents.dataflows.egypt_ingestion import import_evidence_bundle
from tradingagents.markets.egypt_sources import EgyptSourceKey
from tradingagents.portfolio.engine import build_portfolio_plan
from tradingagents.portfolio.evidence_store import EvidenceStore
from tradingagents.portfolio.models import PortfolioSnapshot
from tradingagents.portfolio.opportunity import (
    OpportunityRequest,
    build_opportunity_board,
)
from tradingagents.portfolio.reporting import save_plan_report
from tradingagents.portfolio.thesis import ThesisTracker, evaluate_thesis

app = typer.Typer(
    name="egx-portfolio",
    help="Build evidence-gated Egyptian portfolio advice without executing trades.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Evidence-first portfolio planning for Egyptian listed equities."""


def _verify_snapshot_evidence(
    snapshot: PortfolioSnapshot,
    store: EvidenceStore,
) -> None:
    references = list(snapshot.market_evidence)
    for assessment in snapshot.candidates:
        references.append(assessment.sharia_evidence)
        for dimension in assessment.dimensions:
            references.extend(dimension.evidence)

    for reference in references:
        store.require(reference)


def _verify_opportunity_evidence(
    request: OpportunityRequest,
    store: EvidenceStore,
) -> None:
    _verify_snapshot_evidence(request.snapshot, store)
    for valuation in request.valuations:
        for reference in valuation.evidence:
            store.require(reference)


def _write_json_model(model, output_file: Path, *, force: bool) -> None:
    if output_file.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(model.model_dump_json(indent=2), encoding="utf-8")


@app.command("snapshot-schema")
def snapshot_schema() -> None:
    """Print the strict PortfolioSnapshot JSON schema."""
    typer.echo(
        json.dumps(
            PortfolioSnapshot.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("import-evidence")
def import_evidence(
    bundle_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Strict version-1 evidence bundle in UTF-8 JSON.",
    ),
    database: Path = typer.Option(
        Path("egx-evidence.sqlite3"),
        "--database",
        "-d",
        dir_okay=False,
        resolve_path=True,
        help="SQLite evidence database.",
    ),
    expected_source: EgyptSourceKey | None = typer.Option(
        None,
        "--expected-source",
        help="Reject the bundle unless it contains this source.",
    ),
) -> None:
    """Validate and idempotently store an offline evidence bundle."""
    try:
        stored = import_evidence_bundle(
            bundle_file,
            EvidenceStore(database),
            expected_source=expected_source,
        )
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Evidence import error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Stored evidence record: {stored.record_id}")
    typer.echo(f"SHA-256: {stored.reference.content_hash}")


@app.command("opportunity-schema")
def opportunity_schema() -> None:
    """Print the strict opportunity-board request JSON schema."""
    typer.echo(
        json.dumps(
            OpportunityRequest.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("opportunities")
def opportunities(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="UTF-8 JSON file containing an OpportunityRequest.",
    ),
    output_file: Path = typer.Option(
        Path("egx-output/opportunity_board.json"),
        "--output",
        "-o",
        dir_okay=False,
        resolve_path=True,
        help="JSON opportunity board output.",
    ),
    database: Path = typer.Option(
        Path("egx-evidence.sqlite3"),
        "--database",
        "-d",
        dir_okay=False,
        resolve_path=True,
        help="Verified SQLite evidence database.",
    ),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Rank new ideas, review holdings and test replacement economics."""
    try:
        request = OpportunityRequest.model_validate_json(
            input_file.read_text(encoding="utf-8")
        )
        store = EvidenceStore(database)
        _verify_opportunity_evidence(request, store)
        board = build_opportunity_board(
            request.snapshot,
            request.valuations,
            request.macro,
        )
        _write_json_model(board, output_file, force=force)
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Opportunity error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Opportunity board: {output_file}")
    typer.echo(f"Regime: {board.regime.regime.value}")
    typer.echo(f"Top new ideas: {', '.join(board.top_new_ideas) or 'none'}")


@app.command("thesis-check")
def thesis_check(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="UTF-8 JSON file containing a ThesisTracker.",
    ),
    analysis_date: str = typer.Option(..., "--analysis-date"),
    output_file: Path = typer.Option(
        Path("egx-output/thesis_evaluation.json"),
        "--output",
        "-o",
        dir_okay=False,
        resolve_path=True,
    ),
    database: Path = typer.Option(
        Path("egx-evidence.sqlite3"),
        "--database",
        "-d",
        dir_okay=False,
        resolve_path=True,
    ),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Evaluate falsifiable thesis pillars against fresh verified evidence."""
    try:
        tracker = ThesisTracker.model_validate_json(
            input_file.read_text(encoding="utf-8")
        )
        store = EvidenceStore(database)
        for pillar in tracker.pillars:
            for reference in pillar.evidence:
                store.require(reference)
        try:
            parsed_analysis_date = date.fromisoformat(analysis_date)
        except ValueError as exc:
            raise ValueError("analysis-date must use YYYY-MM-DD") from exc
        evaluation = evaluate_thesis(tracker, parsed_analysis_date)
        _write_json_model(evaluation, output_file, force=force)
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Thesis error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Thesis evaluation: {output_file}")
    typer.echo(f"Status: {evaluation.status.value}")


@app.command()
def plan(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="UTF-8 JSON file containing a PortfolioSnapshot.",
    ),
    output_directory: Path = typer.Option(
        Path("egx-output"),
        "--output",
        "-o",
        file_okay=False,
        resolve_path=True,
        help="Directory for the JSON and Arabic Markdown reports.",
    ),
    database: Path = typer.Option(
        Path("egx-evidence.sqlite3"),
        "--database",
        "-d",
        dir_okay=False,
        resolve_path=True,
        help="Verified SQLite evidence database used by every snapshot reference.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace an existing report in the output directory.",
    ),
) -> None:
    """Validate a snapshot and create a deterministic advisory plan."""
    try:
        raw = json.loads(input_file.read_text(encoding="utf-8"))
        snapshot = PortfolioSnapshot.model_validate(raw)
        _verify_snapshot_evidence(snapshot, EvidenceStore(database))
        portfolio_plan = build_portfolio_plan(snapshot)
        json_path, markdown_path = save_plan_report(
            portfolio_plan,
            output_directory,
            force=force,
        )
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        typer.echo(f"Input or report error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"JSON: {json_path}")
    typer.echo(f"Arabic report: {markdown_path}")
    if portfolio_plan.blocked_reasons:
        typer.echo(
            "Recommendation blocked: verified evidence is incomplete or unsafe.",
            err=True,
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
