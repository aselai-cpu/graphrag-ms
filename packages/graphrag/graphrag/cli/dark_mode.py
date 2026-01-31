# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI commands for dark mode analysis and management."""

import sys
from pathlib import Path

import typer

from graphrag.index.graph.dark_mode import MetricsAnalyzer

# Create typer app for dark mode commands
app = typer.Typer(
    help="Dark mode analysis and management commands",
    no_args_is_help=True,
)


@app.command("analyze")
def analyze_command(
    metrics_file: Path = typer.Argument(
        ...,
        help="Path to dark mode metrics log (JSON lines format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    min_operations: int = typer.Option(
        1000,
        "--min-operations",
        help="Minimum operations required for cutover",
    ),
    max_error_rate: float = typer.Option(
        0.01,
        "--max-error-rate",
        help="Maximum shadow error rate for cutover",
    ),
    min_pass_rate: float = typer.Option(
        0.95,
        "--min-pass-rate",
        help="Minimum comparison pass rate for cutover",
    ),
    max_latency_ratio: float = typer.Option(
        2.0,
        "--max-latency-ratio",
        help="Maximum shadow/primary latency ratio for cutover",
    ),
    export_csv: Path | None = typer.Option(
        None,
        "--export-csv",
        help="Export metrics to CSV file",
    ),
):
    """Analyze dark mode metrics and check cutover readiness.

    Examples:

        graphrag dark-mode analyze output/dark_mode_metrics.jsonl

        graphrag dark-mode analyze metrics.jsonl --min-operations 500

        graphrag dark-mode analyze metrics.jsonl --export-csv analysis.csv
    """
    try:
        analyzer = MetricsAnalyzer(metrics_file)
        analyzer.load_metrics()

        analysis = analyzer.analyze(
            min_operations=min_operations,
            max_error_rate=max_error_rate,
            min_pass_rate=min_pass_rate,
            max_latency_ratio=max_latency_ratio,
        )

        analyzer.print_summary(analysis)

        if export_csv:
            analyzer.export_to_csv(export_csv)
            typer.echo(f"\n✅ Exported metrics to {export_csv}")

        # Exit with appropriate code
        if analysis.ready_for_cutover:
            typer.echo("\n✅ Ready for cutover to shadow backend!")
            raise typer.Exit(0)
        else:
            typer.echo("\n❌ Not ready for cutover - see blocking reasons above")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command("summary")
def summary_command(
    metrics_file: Path = typer.Argument(
        ...,
        help="Path to dark mode metrics log (JSON lines format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """Show quick summary of dark mode execution.

    Examples:

        graphrag dark-mode summary output/dark_mode_metrics.jsonl
    """
    try:
        analyzer = MetricsAnalyzer(metrics_file)
        analyzer.load_metrics()
        analysis = analyzer.analyze()

        typer.echo(f"📊 Total operations: {analysis.total_operations}")
        typer.echo(f"✅ Pass rate: {analysis.comparison_pass_rate:.2%}")
        typer.echo(f"⚠️  Error rate: {analysis.shadow_error_rate:.2%}")
        typer.echo(f"⚡ Latency ratio: {analysis.avg_latency_ratio:.2f}x")
        typer.echo()

        if analysis.ready_for_cutover:
            typer.echo("✅ Ready for cutover!")
        else:
            typer.echo("❌ Not ready for cutover")
            typer.echo("Blocking reasons:")
            for reason in analysis.blocking_reasons:
                typer.echo(f"  - {reason}")

    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command("check-cutover")
def check_cutover_command(
    metrics_file: Path = typer.Argument(
        ...,
        help="Path to dark mode metrics log (JSON lines format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    min_operations: int = typer.Option(
        1000,
        "--min-operations",
        help="Minimum operations required",
    ),
    max_error_rate: float = typer.Option(
        0.01,
        "--max-error-rate",
        help="Maximum error rate",
    ),
    min_pass_rate: float = typer.Option(
        0.95,
        "--min-pass-rate",
        help="Minimum pass rate",
    ),
    max_latency_ratio: float = typer.Option(
        2.0,
        "--max-latency-ratio",
        help="Maximum latency ratio",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed analysis",
    ),
):
    """Check if ready for cutover to shadow backend.

    Returns exit code 0 if ready, 1 if not ready.

    Examples:

        graphrag dark-mode check-cutover metrics.jsonl

        graphrag dark-mode check-cutover metrics.jsonl --verbose

        graphrag dark-mode check-cutover metrics.jsonl --min-operations 500
    """
    try:
        analyzer = MetricsAnalyzer(metrics_file)
        analyzer.load_metrics()

        analysis = analyzer.analyze(
            min_operations=min_operations,
            max_error_rate=max_error_rate,
            min_pass_rate=min_pass_rate,
            max_latency_ratio=max_latency_ratio,
        )

        if verbose:
            analyzer.print_summary(analysis)
        else:
            if analysis.ready_for_cutover:
                typer.echo("✅ READY FOR CUTOVER")
                typer.echo(f"  Operations: {analysis.total_operations}")
                typer.echo(f"  Pass rate: {analysis.comparison_pass_rate:.2%}")
                typer.echo(f"  Error rate: {analysis.shadow_error_rate:.2%}")
                typer.echo(f"  Latency: {analysis.avg_latency_ratio:.2f}x")
            else:
                typer.echo("❌ NOT READY FOR CUTOVER")
                for reason in analysis.blocking_reasons:
                    typer.echo(f"  - {reason}")

        raise typer.Exit(0 if analysis.ready_for_cutover else 1)

    except FileNotFoundError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {e}", err=True)
        raise typer.Exit(1)
