"""CLI commands for aggregator training and inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
import typer
from typing import Annotated

log = structlog.get_logger(__name__)

aggregator_app = typer.Typer(help="Aggregator training and inspection")


@aggregator_app.command("train")
def aggregator_train(
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD (exclusive val start)")] = None,
    val_end: Annotated[Optional[str], typer.Option(help="Validation end date YYYY-MM-DD")] = None,
    symbol: Annotated[str, typer.Option(help="Futures symbol")] = "BTCUSDT",
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Model output dir (default: data/models/aggregator)")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Train the LightGBM meta-learner aggregator on historical feature data.

    Loads pre-computed aggregator feature rows (produced by building the full
    feature pipeline + agent opinions) and trains with purged k-fold CV.
    Saves the model artefact to ``output_dir``.

    Examples::

        intraday aggregator train --start 2026-01-01 --end 2026-05-01 \\
            --val_end 2026-06-01
    """
    from datetime import date

    import polars as pl
    from rich import print as rprint

    from intraday.aggregator.meta_learner import MetaLearner

    out_dir = output_dir or (data_dir / "models" / "aggregator")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve date range
    start_date = date.fromisoformat(start) if start else date(2026, 1, 1)
    end_date = date.fromisoformat(end) if end else date.today()

    rprint(f"[yellow]Aggregator train[/yellow]  {start_date} → {end_date}  (symbol={symbol})")
    rprint(f"Output: {out_dir}")

    # Load pre-built aggregator rows
    agg_dir = data_dir / "aggregator_rows" / symbol
    if not agg_dir.exists():
        rprint(
            "[red]No aggregator rows found.[/red] "
            "Run the backtest with --save-agg-rows first to produce training data."
        )
        raise typer.Exit(code=1)

    row_files = sorted(agg_dir.glob("*.parquet"))
    if not row_files:
        rprint(f"[red]No parquet files in {agg_dir}[/red]")
        raise typer.Exit(code=1)

    df = pl.read_parquet(row_files)
    rprint(f"Loaded {len(df):,} aggregator rows from {len(row_files)} files.")

    # Filter by date range using ts_ms
    start_ms = int(start_date.strftime("%s")) * 1000 if start else 0
    end_ms = int(end_date.strftime("%s")) * 1000 if end else int(1e18)
    df = df.filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms))
    rprint(f"After date filter: {len(df):,} rows.")

    if len(df) < 200:
        rprint("[red]Insufficient data for training (need ≥200 rows).[/red]")
        raise typer.Exit(code=1)

    if "label" not in df.columns:
        rprint("[red]Column 'label' not found. Aggregator rows must include a binary label.[/red]")
        raise typer.Exit(code=1)

    from intraday.aggregator.features import AGGREGATOR_FEATURE_COLS

    feature_cols = [c for c in AGGREGATOR_FEATURE_COLS if c in df.columns]
    X = df.select(feature_cols)
    y = df["label"]
    ts = df["ts_ms"]

    ml = MetaLearner(model_dir=out_dir)
    metrics = ml.fit(X, y, ts=ts, n_folds=5, embargo_pct=0.01)

    rprint(f"\n[green]Training complete![/green]")
    rprint(f"  AUC    : {metrics['auc']:.4f}")
    rprint(f"  Brier  : {metrics['brier']:.4f}")
    rprint(f"  ECE    : {metrics['ece']:.4f}")

    model_path = out_dir / "meta_learner.pkl"
    ml.save(model_path)
    rprint(f"\nModel saved → {model_path}")

    # Print feature importance top-10
    fi = ml.feature_importance().head(10)
    rprint("\n[bold]Top-10 feature importances:[/bold]")
    for row in fi.iter_rows(named=True):
        rprint(f"  {row['feature']:<30} {row['importance']:.1f}")


@aggregator_app.command("inspect")
def aggregator_inspect(
    version: Annotated[str, typer.Option(help="Model version tag or 'latest'")] = "latest",
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
) -> None:
    """Inspect a saved aggregator meta-learner: show feature importance and OOF metrics.

    Examples::

        intraday aggregator inspect --version latest
    """
    from rich import print as rprint
    from rich.table import Table
    from rich.console import Console

    from intraday.aggregator.meta_learner import MetaLearner

    console = Console()

    models_dir = data_dir / "models" / "aggregator"
    if version == "latest":
        candidates = sorted(models_dir.glob("meta_learner*.pkl"))
        if not candidates:
            rprint(f"[red]No model found in {models_dir}[/red]")
            raise typer.Exit(code=1)
        model_path = candidates[-1]
    else:
        model_path = models_dir / f"meta_learner_{version}.pkl"
        if not model_path.exists():
            model_path = models_dir / "meta_learner.pkl"

    if not model_path.exists():
        rprint(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(code=1)

    rprint(f"Loading model from [cyan]{model_path}[/cyan]")
    ml = MetaLearner.load(model_path)

    rprint(f"\n[bold]Model info[/bold]")
    rprint(f"  Fold models : {len(ml._models)}")
    rprint(f"  Features    : {len(ml._feature_cols)}")
    rprint(f"  Threshold   : {ml._threshold:.4f}")

    fi = ml.feature_importance()
    table = Table(title="Feature Importances (gain, top 20)")
    table.add_column("Rank", justify="right")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")

    for rank, row in enumerate(fi.head(20).iter_rows(named=True), start=1):
        table.add_row(str(rank), row["feature"], f"{row['importance']:.1f}")

    console.print(table)


__all__ = ["aggregator_app"]
