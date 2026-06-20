"""CLI entry point for intraday trading system."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from intraday.data import (
    CaptureConfig,
    Checkpoint,
    DownloadConfig,
    capture_live,
    download_historical,
    get_checkpoint_path,
)
from intraday.utils.logging import setup_logging

app = typer.Typer(name="intraday", help="BTC/USD multi-agent intraday trading system")
data_app = typer.Typer(help="Data acquisition and management")
app.add_typer(data_app, name="data")

console = Console()


# Global options
@app.callback()
def main(
    ctx: typer.Context,
    log_level: Annotated[str, typer.Option(help="Log level")] = "info",
    quiet: Annotated[bool, typer.Option(help="Suppress console logs")] = False,
) -> None:
    """Global configuration."""
    setup_logging(log_level=log_level, console=not quiet)


# Data commands


@data_app.command("download")
def data_download(
    symbol: Annotated[str, typer.Option(help="Trading symbol")] = "BTCUSDT",
    venue: Annotated[str, typer.Option(help="Exchange venue")] = "binance",
    kind: Annotated[
        str,
        typer.Option(
            help="Data type: klines_1m, klines_5m, klines_15m, klines_1h, funding, open_interest"
        ),
    ] = "klines_5m",
    start: Annotated[Optional[str], typer.Option(help="Start date (YYYY-MM-DD)")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date (YYYY-MM-DD)")] = None,
    offset: Annotated[int, typer.Option(help="Offset from checkpoint in milliseconds")] = 0,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    force: Annotated[bool, typer.Option(help="Re-download even if exists")] = False,
) -> None:
    """Download historical data from Binance API.

    Examples:
        # Download 12 months of 5m klines
        intraday data download --kind klines_5m --start 2024-01-01 --end 2024-12-31

        # Resume from checkpoint
        intraday data download --kind klines_5m

        # Start from checkpoint + 1 week
        intraday data download --kind klines_5m --offset 604800000
    """
    config = DownloadConfig(
        symbol=symbol,
        venue=venue,
        kind=kind,
        start=datetime.fromisoformat(start) if start else None,
        end=datetime.fromisoformat(end) if end else None,
        offset_from_checkpoint=offset,
        data_dir=data_dir,
        force=force,
    )

    asyncio.run(download_historical(config))
    rprint("[green]✓[/green] Download complete!")


@data_app.command("live-capture")
def data_live_capture(
    symbol: Annotated[str, typer.Option(help="Trading symbol")] = "BTCUSDT",
    venue: Annotated[str, typer.Option(help="Exchange venue")] = "binance",
    streams: Annotated[
        str,
        typer.Option(help="Comma-separated streams: trade,depth,mark_price,liquidations"),
    ] = "trade,depth,mark_price",
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    flush_interval: Annotated[int, typer.Option(help="Flush interval in seconds")] = 60,
) -> None:
    """Start live WebSocket data capture.

    Runs indefinitely until interrupted (Ctrl+C).

    Examples:
        # Capture trades and depth
        intraday data live-capture --streams trade,depth

        # All streams
        intraday data live-capture --streams trade,depth,mark_price,liquidations
    """
    config = CaptureConfig(
        symbol=symbol,
        venue=venue,
        streams=streams.split(","),
        data_dir=data_dir,
        flush_interval_s=flush_interval,
    )

    rprint(f"[yellow]Starting live capture for {symbol}...[/yellow]")
    rprint(f"Streams: {', '.join(config.streams)}")
    rprint(f"Data dir: {data_dir}")
    rprint("[dim]Press Ctrl+C to stop[/dim]\n")

    asyncio.run(capture_live(config))


@data_app.command("summary")
def data_summary(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    symbol: Annotated[Optional[str], typer.Option(help="Filter by symbol")] = None,
) -> None:
    """Show summary of downloaded data."""
    checkpoint_path = get_checkpoint_path(data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)

    if not checkpoint.entries:
        rprint("[yellow]No data found. Run 'intraday data download' first.[/yellow]")
        return

    # Create table
    table = Table(title="Data Summary")
    table.add_column("Venue/Kind/Symbol", style="cyan")
    table.add_column("Start", style="green")
    table.add_column("End", style="green")
    table.add_column("Records", justify="right", style="magenta")
    table.add_column("Files", justify="right", style="blue")

    for key, entry in sorted(checkpoint.entries.items()):
        if symbol and entry.symbol != symbol:
            continue

        table.add_row(
            key,
            entry.start_time.strftime("%Y-%m-%d %H:%M"),
            entry.end_time.strftime("%Y-%m-%d %H:%M"),
            f"{entry.num_records:,}",
            str(entry.num_files),
        )

    console.print(table)


@data_app.command("checkpoint")
def data_checkpoint(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
) -> None:
    """Show checkpoint details."""
    checkpoint_path = get_checkpoint_path(data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)

    rprint(checkpoint.summary())


if __name__ == "__main__":
    app()
