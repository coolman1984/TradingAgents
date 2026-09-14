"""Non-interactive command line interface for the EGX portfolio adviser."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from tradingagents.portfolio.engine import build_portfolio_plan
from tradingagents.portfolio.models import PortfolioSnapshot
from tradingagents.portfolio.reporting import save_plan_report

app = typer.Typer(
    name="egx-portfolio",
    help="Build evidence-gated Egyptian portfolio advice without executing trades.",
    no_args_is_help=True,
)


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
