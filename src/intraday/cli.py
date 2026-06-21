"""CLI entry point for intraday trading system."""

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from intraday.data import (
    CaptureConfig,
    Checkpoint,
    get_checkpoint_path,
    capture_live,
)
from intraday.agents.cli import agents_app
from intraday.aggregator.cli import aggregator_app
from intraday.data.binance_bulk import BulkKind, download_bulk
from intraday.forecast.cli import forecast_app
from intraday.models.cli import ml_app
from intraday.rl.cli import rl_app
from intraday.sim.cli import backtest_app
from intraday.utils.logging import setup_logging

app = typer.Typer(name="intraday", help="BTC/USD multi-agent intraday trading system")
data_app = typer.Typer(help="Data acquisition and management")
features_app = typer.Typer(help="Feature engineering")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(backtest_app, name="backtest")
app.add_typer(forecast_app, name="forecast")
app.add_typer(agents_app, name="agent")
app.add_typer(aggregator_app, name="train")
app.add_typer(ml_app, name="ml")
app.add_typer(rl_app, name="rl")

console = Console()

ALL_KINDS: list[BulkKind] = ["aggTrades", "klines_1m", "klines_5m", "bookDepth", "metrics"]


@app.callback()
def main(
    ctx: typer.Context,
    log_level: Annotated[str, typer.Option(help="Log level")] = "info",
    quiet: Annotated[bool, typer.Option(help="Suppress console logs")] = False,
) -> None:
    setup_logging(log_level=log_level, console=not quiet)


# ---------------------------------------------------------------------------
# data download-bulk
# ---------------------------------------------------------------------------

@data_app.command("download-bulk")
def data_download_bulk(
    symbol: Annotated[str, typer.Option(help="Futures symbol")] = "BTCUSDT",
    kinds: Annotated[
        str,
        typer.Option(help=f"Comma-separated kinds: {','.join(ALL_KINDS)} or 'all'"),
    ] = "all",
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD (default: yesterday)")] = None,
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    concurrency: Annotated[int, typer.Option(help="Parallel downloads")] = 8,
) -> None:
    """Download bulk historical data from data.binance.vision.

    Downloads daily Parquet files (aggTrades, klines, bookDepth, metrics).
    Skips files that already exist. Safe to re-run.

    Examples:
        # Download last 1 month of all kinds
        intraday data download-bulk --start 2026-05-20

        # Download only trades and klines for a specific range
        intraday data download-bulk --kinds aggTrades,klines_1m,klines_5m \\
            --start 2026-01-01 --end 2026-06-19
    """
    selected_kinds: list[BulkKind] = ALL_KINDS if kinds == "all" else kinds.split(",")  # type: ignore
    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    rprint(f"[yellow]Downloading {selected_kinds} from data.binance.vision[/yellow]")
    rprint(f"Range: {start_date or 'auto'} → {end_date or 'yesterday'}")
    rprint(f"Concurrency: {concurrency} parallel downloads")

    counts = asyncio.run(
        download_bulk(
            symbol=symbol,
            kinds=selected_kinds,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            max_concurrent=concurrency,
        )
    )
    rprint("\n[green]✓[/green] Download complete!")
    for k, n in counts.items():
        rprint(f"  {k}: {n} new days")


# ---------------------------------------------------------------------------
# data live-capture
# ---------------------------------------------------------------------------

@data_app.command("live-capture")
def data_live_capture(
    symbol: Annotated[str, typer.Option(help="Futures symbol")] = "BTCUSDT",
    streams: Annotated[
        str,
        typer.Option(help="Comma-separated: aggTrade,depth,mark_price,liquidations"),
    ] = "aggTrade,depth,mark_price",
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    flush_interval: Annotated[int, typer.Option(help="Flush interval in seconds")] = 60,
) -> None:
    """Start live WebSocket capture from Binance futures streams.

    Uses BTCUSDT perpetual futures (fstream.binance.com).
    Runs indefinitely until interrupted (Ctrl+C).

    Examples:
        intraday data live-capture --streams aggTrade,depth,mark_price
    """
    config = CaptureConfig(
        symbol=symbol,
        streams=streams.split(","),
        data_dir=data_dir,
        flush_interval_s=flush_interval,
    )

    rprint(f"[yellow]Starting live capture for {symbol}...[/yellow]")
    rprint(f"Streams: {', '.join(config.streams)}")
    rprint(f"Data dir: {data_dir}")
    rprint("[dim]Press Ctrl+C to stop[/dim]\n")

    asyncio.run(capture_live(config))


# ---------------------------------------------------------------------------
# data summary / checkpoint
# ---------------------------------------------------------------------------

@data_app.command("summary")
def data_summary(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
) -> None:
    """Show summary of downloaded data."""
    checkpoint_path = get_checkpoint_path(data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)

    # Also scan raw parquet files since bulk downloads bypass the checkpoint
    raw_dir = data_dir / "raw" / "binance"
    table = Table(title="Raw Data Summary")
    table.add_column("Kind", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Date range")
    table.add_column("Size")

    if raw_dir.exists():
        for kind_dir in sorted(raw_dir.iterdir()):
            if not kind_dir.is_dir():
                continue
            for symbol_dir in kind_dir.iterdir():
                files = sorted(symbol_dir.glob("*.parquet"))
                if not files:
                    continue
                dates = [f.stem for f in files]
                total_kb = sum(f.stat().st_size for f in files) / 1024
                size_str = f"{total_kb/1024:.1f} MB" if total_kb > 1024 else f"{total_kb:.0f} KB"
                table.add_row(
                    f"{kind_dir.name}/{symbol_dir.name}",
                    str(len(files)),
                    f"{dates[0]} → {dates[-1]}",
                    size_str,
                )

    console.print(table)


# ---------------------------------------------------------------------------
# features compute
# ---------------------------------------------------------------------------

@features_app.command("compute")
def features_compute(
    symbol: Annotated[str, typer.Option(help="Symbol")] = "BTCUSDT",
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD")] = None,
    data_dir: Annotated[Path, typer.Option(help="Data root")] = Path("data"),
    force: Annotated[bool, typer.Option(help="Recompute existing days")] = False,
    vpin_bucket: Annotated[float, typer.Option(help="VPIN bucket size in BTC")] = 100.0,
    vpin_window: Annotated[int,   typer.Option(help="VPIN rolling window (buckets)")] = 50,
    workers: Annotated[int, typer.Option(help="Parallel workers (0 = sequential)")] = 0,
    warmup_days: Annotated[int, typer.Option(help="Warmup days per chunk to initialize rolling state")] = 14,
) -> None:
    """Compute feature rows (price, volume, depth, VPIN, Hawkes) from raw Parquet.

    By default uses all available CPU cores for parallel processing. Each worker
    handles an independent date-range chunk with a warmup window to initialize
    rolling state (VPIN, Hawkes, RSI).

    Output: data/features/{symbol}/YYYY-MM-DD.parquet (288 rows per day)

    Examples:
        intraday features compute                          # auto-continue, all cores
        intraday features compute --start 2020-09-10      # full history, all cores
        intraday features compute --workers 0             # sequential (exact state)
    """
    import os
    from intraday.features.pipeline import TransformationPipeline

    if start:
        start_date = date.fromisoformat(start)
    else:
        # Auto-detect: start from day after the last computed feature file
        features_dir = data_dir / "features" / symbol
        existing = sorted(features_dir.glob("*.parquet")) if features_dir.exists() else []
        if existing:
            from datetime import timedelta
            start_date = date.fromisoformat(existing[-1].stem) + timedelta(days=1)
        else:
            start_date = date(2020, 9, 10)
    end_date = date.fromisoformat(end) if end else date.today()

    n_workers = workers if workers > 0 else os.cpu_count() or 1
    mode = f"{n_workers} parallel workers (warmup={warmup_days}d)" if n_workers > 1 else "sequential"

    rprint(f"[yellow]Computing features for {symbol}[/yellow]")
    rprint(f"Range     : {start_date} → {end_date}")
    rprint(f"Mode      : {mode}")
    rprint(f"VPIN      : bucket={vpin_bucket} BTC, window={vpin_window} buckets")
    rprint(f"Hawkes    : α=1.0, β=10.0/s, μ=6.0")
    rprint(f"Output    : {data_dir}/features/{symbol}/")

    pipeline = TransformationPipeline(
        data_dir=data_dir, symbol=symbol,
        vpin_bucket_btc=vpin_bucket, vpin_window=vpin_window,
    )

    if n_workers > 1:
        total = pipeline.run_parallel(start_date, end_date, force=force,
                                      workers=n_workers, warmup_days=warmup_days)
    else:
        total = pipeline.run(start_date, end_date, force=force)

    rprint(f"\n[green]✓[/green] {total:,} feature rows written")


@features_app.command("summary")
def features_summary(
    data_dir: Annotated[Path, typer.Option(help="Data root")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="Symbol")] = "BTCUSDT",
) -> None:
    """Show feature store summary."""
    features_dir = data_dir / "features" / symbol
    if not features_dir.exists():
        rprint("[yellow]No features yet. Run: intraday features compute[/yellow]")
        return

    files = sorted(features_dir.glob("*.parquet"))
    if not files:
        rprint("[yellow]No feature files found.[/yellow]")
        return

    total_rows = 0
    for f in files:
        import polars as pl
        df = pl.scan_parquet(f).select("bar_time_ms").collect()
        total_rows += len(df)

    rprint(f"Feature files : {len(files)}")
    rprint(f"Date range    : {files[0].stem} → {files[-1].stem}")
    rprint(f"Total rows    : {total_rows:,}")
    rprint(f"Directory     : {features_dir}")


if __name__ == "__main__":
    app()
