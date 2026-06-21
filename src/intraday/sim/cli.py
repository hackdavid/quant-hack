"""CLI for backtesting and simulation.

All data loaded from data_dir/raw/binance/{kind}/BTCUSDT/YYYY-MM-DD.parquet.
Events from all kinds are merged by ts_ms order before simulation.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import structlog
import typer
from rich.console import Console
from rich.table import Table

from intraday.sim.account import Account
from intraday.sim.book import LocalOrderBook
from intraday.sim.events import BarEvent, DepthEvent, Event, FundingEvent, MarkEvent, TradeEvent
from intraday.sim.latency import LatencyModel
from intraday.sim.loop import SimulatorLoop
from intraday.sim.matching import MatchingEngine
from intraday.sim.strategies.registry import get_strategy

# Eagerly import strategies to populate registry
import intraday.sim.strategies.v0_buy_hold  # noqa: F401
import intraday.sim.strategies.v1_random  # noqa: F401

log = structlog.get_logger(__name__)
console = Console()

backtest_app = typer.Typer(help="Backtesting and simulation")


def _date_range(start: date, end: date) -> list[date]:
    from datetime import timedelta
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _load_bar_events(data_dir: Path, symbol: str, day: date) -> list[BarEvent]:
    path = data_dir / "raw" / "binance" / "klines_5m" / symbol / f"{day.isoformat()}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    events: list[BarEvent] = []
    for row in df.iter_rows(named=True):
        # Handle various column naming conventions from bulk download
        ts = row.get("open_time_ms") or row.get("open_time") or row.get("ts_ms") or 0
        events.append(BarEvent(
            ts_ms=int(ts),
            symbol=symbol,
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0)),
        ))
    return events


def _load_trade_events(data_dir: Path, symbol: str, day: date) -> list[TradeEvent]:
    path = data_dir / "raw" / "binance" / "aggTrades" / symbol / f"{day.isoformat()}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    events: list[TradeEvent] = []
    for row in df.iter_rows(named=True):
        ts = row.get("time_ms") or row.get("T") or row.get("ts_ms") or 0
        price = float(row.get("price") or row.get("p") or 0)
        qty = float(row.get("quantity") or row.get("q") or 0)
        is_bm = bool(row.get("is_buyer_maker") or row.get("m") or False)
        if price <= 0 or qty <= 0:
            continue
        events.append(TradeEvent(
            ts_ms=int(ts),
            price=price,
            qty_base=qty,
            is_buyer_maker=is_bm,
        ))
    return events


def _load_depth_events(
    data_dir: Path, symbol: str, day: date,
    bar_close_series: dict[int, float] | None = None,
) -> list[DepthEvent]:
    """Load depth snapshots from bookDepth parquet.

    Our bookDepth is stored as %-band aggregates (bid_02pct, bid_1pct …), not raw L2.
    We reconstruct approximate synthetic levels using the most recent bar close as mid.
    Levels are placed at band mid-points and assigned incremental depth.
    """
    path = data_dir / "raw" / "binance" / "bookDepth" / symbol / f"{day.isoformat()}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    events: list[DepthEvent] = []

    # Sort by timestamp to allow interpolation of mid price
    ts_col = "snapshot_time_ms" if "snapshot_time_ms" in df.columns else "ts_ms"
    if ts_col not in df.columns:
        return []

    sorted_ts = sorted(bar_close_series.keys()) if bar_close_series else []

    def _last_close(ts_ms: int) -> float:
        if not sorted_ts:
            return 0.0
        lo, hi = 0, len(sorted_ts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sorted_ts[mid] <= ts_ms:
                lo = mid
            else:
                hi = mid - 1
        return bar_close_series[sorted_ts[lo]] if bar_close_series else 0.0

    # %-band columns present in the file
    BID_BANDS = [("bid_02pct", 0.002), ("bid_1pct", 0.01), ("bid_2pct", 0.02)]
    ASK_BANDS = [("ask_02pct", 0.002), ("ask_1pct", 0.01), ("ask_2pct", 0.02)]

    for row in df.iter_rows(named=True):
        ts = int(row.get(ts_col) or 0)
        if ts == 0:
            continue
        mid = _last_close(ts)
        if mid <= 0:
            continue

        # Build bid levels (incremental depth per band)
        bids: list[tuple[float, float]] = []
        prev_depth = 0.0
        for col, offset_pct in BID_BANDS:
            total_depth = float(row.get(col) or 0.0)
            incremental = max(total_depth - prev_depth, 0.0)
            if incremental > 0:
                price = round(mid * (1.0 - offset_pct), 1)
                bids.append((price, incremental))
            prev_depth = total_depth

        # Build ask levels
        asks: list[tuple[float, float]] = []
        prev_depth = 0.0
        for col, offset_pct in ASK_BANDS:
            total_depth = float(row.get(col) or 0.0)
            incremental = max(total_depth - prev_depth, 0.0)
            if incremental > 0:
                price = round(mid * (1.0 + offset_pct), 1)
                asks.append((price, incremental))
            prev_depth = total_depth

        if bids or asks:
            events.append(DepthEvent(
                ts_ms=ts,
                bids=bids,
                asks=asks,
                is_snapshot=True,
            ))
    return events


def _load_funding_events(data_dir: Path, symbol: str, day: date) -> list[FundingEvent]:
    path = data_dir / "raw" / "binance" / "metrics" / symbol / f"{day.isoformat()}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    events: list[FundingEvent] = []
    for row in df.iter_rows(named=True):
        ts = row.get("time_ms") or row.get("timestamp") or row.get("ts_ms") or 0
        rate = float(row.get("funding_rate") or row.get("lastFundingRate") or 0)
        mark = float(row.get("mark_price") or row.get("markPrice") or 0)
        if mark <= 0:
            continue
        events.append(FundingEvent(
            ts_ms=int(ts),
            funding_rate=rate,
            mark_price=mark,
        ))
    return events


def _build_event_stream(
    data_dir: Path,
    symbol: str,
    start: date,
    end: date,
) -> list[Event]:
    all_events: list[Event] = []
    # Build bar close series first — used to reconstruct synthetic depth levels
    bar_close_series: dict[int, float] = {}
    for day in _date_range(start, end):
        bars = _load_bar_events(data_dir, symbol, day)
        for b in bars:
            bar_close_series[b.ts_ms] = b.close
        all_events.extend(bars)

    for day in _date_range(start, end):
        all_events.extend(_load_trade_events(data_dir, symbol, day))
        all_events.extend(_load_depth_events(data_dir, symbol, day, bar_close_series))
        all_events.extend(_load_funding_events(data_dir, symbol, day))

    all_events.sort(key=lambda e: e.ts_ms)
    log.info("sim.events_loaded", total=len(all_events), symbol=symbol)
    return all_events


@backtest_app.command("run")
def backtest_run(
    strategy: Annotated[str, typer.Option("--strategy", help="Strategy name")] = "v0_buy_hold",
    symbol: Annotated[str, typer.Option("--symbol", help="Trading symbol")] = "BTCUSDT",
    start: Annotated[str, typer.Option("--start", help="Start date YYYY-MM-DD")] = "",
    end: Annotated[str, typer.Option("--end", help="End date YYYY-MM-DD")] = "",
    capital: Annotated[float, typer.Option("--capital", help="Starting USDT capital")] = 10_000.0,
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Data root directory")] = Path("data"),
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    report: Annotated[bool, typer.Option("--report", help="Write HTML report")] = False,
) -> None:
    """Run backtest with a named strategy. Writes run dir to runs/."""
    from intraday.utils.logging import setup_logging
    setup_logging()

    if not start:
        typer.echo("--start is required (YYYY-MM-DD)", err=True)
        raise typer.Exit(1)
    if not end:
        end = date.today().isoformat()

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    run_id = f"{strategy}_{start}_{end}_{uuid.uuid4().hex[:6]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("sim.run_started", run_id=run_id, strategy=strategy, start=start, end=end, capital=capital)

    events = _build_event_stream(data_dir, symbol, start_date, end_date)
    if not events:
        typer.echo(f"No events found in {data_dir} for {symbol} {start}→{end}", err=True)
        raise typer.Exit(1)

    strategy_cls = get_strategy(strategy)
    strategy_inst = strategy_cls()

    sim = SimulatorLoop(
        events=events,
        book=LocalOrderBook(),
        matching=MatchingEngine(),
        account=Account(cash_quote=capital),
        strategy=strategy_inst,
        latency=LatencyModel(seed=seed),
        run_id=run_id,
        seed=seed,
    )

    t0 = time.monotonic()
    result = sim.run()
    elapsed = time.monotonic() - t0

    # Write artifacts
    from intraday.sim.reports import write_metrics_json
    write_metrics_json(result, run_dir)
    (run_dir / "run_result.json").write_text(result.model_dump_json(indent=2))

    log.info("sim.run_complete", run_id=run_id, elapsed_s=elapsed, net_pnl=result.net_pnl_quote)

    # Print summary table
    table = Table(title=f"Backtest: {run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Strategy", strategy)
    table.add_row("Symbol", symbol)
    table.add_row("Period", f"{start} → {end}")
    table.add_row("Events", f"{result.n_events:,}")
    table.add_row("Orders", str(result.n_orders))
    table.add_row("Fills", str(result.n_fills))
    table.add_row("Fill Rate", f"{result.fill_rate:.1%}")
    table.add_row("Gross PnL", f"{result.gross_pnl_quote:+.2f} USDT")
    table.add_row("Net PnL", f"{result.net_pnl_quote:+.2f} USDT")
    table.add_row("Fees", f"{result.fees_paid_quote:.2f} USDT")
    table.add_row("Funding", f"{result.funding_paid_quote:.2f} USDT")
    table.add_row("Max DD", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("Sharpe", f"{result.sharpe:.2f}")
    table.add_row("Sortino", f"{result.sortino:.2f}")
    table.add_row("Calmar", f"{result.calmar:.2f}")
    table.add_row("Avg Slippage", f"{result.avg_slippage_bps:.2f} bps")
    table.add_row("Run Dir", str(run_dir))
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    console.print(table)

    if report:
        from intraday.sim.reports import write_report_html
        write_report_html(result, [], [], run_dir)
        console.print(f"[green]HTML report:[/green] {run_dir / 'report.html'}")


@backtest_app.command("replay")
def backtrack_replay(
    run_id: Annotated[str, typer.Argument(help="Run ID to replay")],
    speed: Annotated[str, typer.Option("--speed", help="Playback speed multiplier e.g. 10x")] = "10x",
    render: Annotated[str, typer.Option("--render", help="Output format: console")] = "console",
) -> None:
    """Replay a run's decisions from runs/{run_id}/decisions.jsonl."""
    from intraday.utils.logging import setup_logging
    setup_logging()

    run_dir = Path("runs") / run_id
    decisions_path = run_dir / "decisions.jsonl"

    if not decisions_path.exists():
        typer.echo(f"decisions.jsonl not found at {decisions_path}", err=True)
        raise typer.Exit(1)

    multiplier_str = speed.rstrip("x")
    try:
        multiplier = float(multiplier_str)
    except ValueError:
        multiplier = 10.0

    lines = decisions_path.read_text().splitlines()
    log.info("replay.started", run_id=run_id, speed=speed, n_lines=len(lines))

    prev_ts: int | None = None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts_ms = record.get("ts_ms", 0)
        if prev_ts is not None and multiplier > 0:
            delta_s = (ts_ms - prev_ts) / 1000.0 / multiplier
            if 0 < delta_s < 5.0:
                time.sleep(delta_s)
        prev_ts = ts_ms

        if render == "console":
            from rich.panel import Panel
            ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
            content = f"ts: {ts_dt}\n" + "\n".join(f"  {k}: {v}" for k, v in record.items() if k != "ts_ms")
            console.print(Panel(content, title=f"[cyan]{run_id}[/cyan]", expand=False))


@backtest_app.command("compare")
def backtest_compare(
    runs: Annotated[str, typer.Argument(help="Comma-separated run IDs")],
) -> None:
    """Compare metrics.json files across multiple run IDs."""
    from intraday.utils.logging import setup_logging
    setup_logging()

    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    if len(run_ids) < 2:
        typer.echo("Provide at least two comma-separated run IDs", err=True)
        raise typer.Exit(1)

    metrics_list: list[tuple[str, dict]] = []
    for rid in run_ids:
        path = Path("runs") / rid / "metrics.json"
        if not path.exists():
            typer.echo(f"metrics.json not found for {rid}", err=True)
            continue
        data = json.loads(path.read_text())
        metrics_list.append((rid, data))

    if not metrics_list:
        raise typer.Exit(1)

    all_keys = list(metrics_list[0][1].keys())

    table = Table(title="Run Comparison")
    table.add_column("Metric", style="cyan")
    for rid, _ in metrics_list:
        table.add_column(rid[:20], justify="right")

    for key in all_keys:
        row = [key]
        for _, m in metrics_list:
            val = m.get(key, "N/A")
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        table.add_row(*row)

    console.print(table)


__all__ = ["backtest_app"]
