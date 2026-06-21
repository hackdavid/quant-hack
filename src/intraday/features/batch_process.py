"""Batch feature computation from historical raw Parquet files.

Processes one day at a time:
  raw/binance/aggTrades/BTCUSDT/2026-06-19.parquet
  raw/binance/klines_1m/BTCUSDT/2026-06-19.parquet
  raw/binance/klines_5m/BTCUSDT/2026-06-19.parquet
  raw/binance/bookDepth/BTCUSDT/2026-06-19.parquet
  raw/binance/metrics/BTCUSDT/2026-06-19.parquet
      →
  features/BTCUSDT/2026-06-19.parquet

Uses the same FeatureCalculator as live trading — identical computation path.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import polars as pl

from intraday.features.calculator import (
    AggTrade,
    DepthBands,
    FeatureCalculator,
    KlineBar,
    MetricsUpdate,
)
from intraday.features.schema import FEATURE_ROW_SCHEMA, FeatureRow
from intraday.utils.logging import get_logger

logger = get_logger(__name__)


def _load_agg_trades(path: Path) -> list[AggTrade]:
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    return [
        AggTrade(
            time_ms=row["time_ms"],
            price=row["price"],
            quantity=row["quantity"],
            is_buyer_maker=row["is_buyer_maker"],
        )
        for row in df.iter_rows(named=True)
    ]


def _load_klines(path: Path) -> list[KlineBar]:
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    return [
        KlineBar(
            open_time_ms=row["open_time_ms"],
            close_time_ms=row["close_time_ms"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            trade_count=row["trade_count"],
            taker_buy_volume=row["taker_buy_volume"],
        )
        for row in df.iter_rows(named=True)
    ]


def _load_depth(path: Path) -> list[DepthBands]:
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    bands = []
    for row in df.iter_rows(named=True):
        bands.append(DepthBands(
            snapshot_time_ms=row["snapshot_time_ms"],
            bid_02pct=row.get("bid_02pct") or 0.0,
            bid_1pct=row.get("bid_1pct") or 0.0,
            bid_2pct=row.get("bid_2pct") or 0.0,
            bid_3pct=row.get("bid_3pct") or 0.0,
            bid_4pct=row.get("bid_4pct") or 0.0,
            bid_5pct=row.get("bid_5pct") or 0.0,
            ask_02pct=row.get("ask_02pct") or 0.0,
            ask_1pct=row.get("ask_1pct") or 0.0,
            ask_2pct=row.get("ask_2pct") or 0.0,
            ask_3pct=row.get("ask_3pct") or 0.0,
            ask_4pct=row.get("ask_4pct") or 0.0,
            ask_5pct=row.get("ask_5pct") or 0.0,
        ))
    return bands


def _load_metrics(path: Path) -> list[MetricsUpdate]:
    if not path.exists():
        return []
    df = pl.read_parquet(path)
    return [
        MetricsUpdate(
            create_time_ms=row["create_time_ms"],
            oi_btc=row["oi_btc"],
            oi_usd=row["oi_usd"],
            ls_count_ratio=row["ls_count_ratio"],
            taker_ls_vol_ratio=row["taker_ls_vol_ratio"],
            top_ls_count=row.get("top_ls_count") or 0.0,
            top_ls_value=row.get("top_ls_value") or 0.0,
        )
        for row in df.iter_rows(named=True)
    ]


def process_day(
    symbol: str,
    day: date,
    raw_dir: Path,
    features_dir: Path,
    calc: Optional[FeatureCalculator] = None,
    force: bool = False,
) -> tuple[FeatureCalculator, list[FeatureRow]]:
    """Process one day of raw data into feature rows.

    Pass `calc` from the previous day to carry over rolling state across days.
    Returns (updated_calc, rows_for_this_day).
    Rows include only those with bar_time_ms within this calendar day.
    """
    out_path = features_dir / symbol / f"{day.isoformat()}.parquet"
    if out_path.exists() and not force:
        logger.info("features.skip", day=day.isoformat(), reason="already exists")
        return calc or FeatureCalculator(symbol), []

    raw_base = raw_dir / "binance"

    # Load all events for this day
    trades = _load_agg_trades(raw_base / "aggTrades" / symbol / f"{day.isoformat()}.parquet")
    klines_1m = _load_klines(raw_base / "klines_1m" / symbol / f"{day.isoformat()}.parquet")
    klines_5m = _load_klines(raw_base / "klines_5m" / symbol / f"{day.isoformat()}.parquet")
    depths = _load_depth(raw_base / "bookDepth" / symbol / f"{day.isoformat()}.parquet")
    metrics = _load_metrics(raw_base / "metrics" / symbol / f"{day.isoformat()}.parquet")

    if not klines_5m:
        logger.warning("features.no_klines5m", day=day.isoformat())
        return calc or FeatureCalculator(symbol), []

    # Merge all events into a single time-sorted stream
    # Tag each event with its timestamp for merging
    events: list[tuple[int, str, object]] = []
    for t in trades:
        events.append((t.time_ms, "trade", t))
    for k in klines_1m:
        events.append((k.close_time_ms, "kline_1m", k))
    for k in klines_5m:
        events.append((k.close_time_ms, "kline_5m", k))
    for d in depths:
        events.append((d.snapshot_time_ms, "depth", d))
    for m in metrics:
        events.append((m.create_time_ms, "metrics", m))

    events.sort(key=lambda e: e[0])

    if calc is None:
        calc = FeatureCalculator(symbol)

    rows: list[FeatureRow] = []
    for _, kind, event in events:
        if kind == "trade":
            calc.on_trade(event)
        elif kind == "kline_1m":
            calc.on_kline_1m(event)
        elif kind == "depth":
            calc.on_depth(event)
        elif kind == "metrics":
            calc.on_metrics(event)
        elif kind == "kline_5m":
            row = calc.on_kline_5m(event)
            if row is not None:
                rows.append(row)

    # Flush any remaining rows that have enough future data
    rows.extend(calc.flush())

    if not rows:
        logger.warning("features.no_rows", day=day.isoformat())
        return calc, []

    # Write to Parquet
    records = [r.model_dump() for r in rows]
    df = pl.DataFrame(records, schema=FEATURE_ROW_SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")

    logger.info(
        "features.saved",
        day=day.isoformat(),
        rows=len(rows),
        size_kb=round(out_path.stat().st_size / 1024, 1),
    )
    return calc, rows


def process_range(
    symbol: str,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    features_dir: Path,
    force: bool = False,
) -> int:
    """Process all days from start_date to end_date. Returns total feature rows written."""
    calc = FeatureCalculator(symbol)
    total = 0
    day = start_date
    while day <= end_date:
        calc, rows = process_day(symbol, day, raw_dir, features_dir, calc, force)
        total += len(rows)
        day += timedelta(days=1)
    logger.info("features.range_complete", start=start_date.isoformat(), end=end_date.isoformat(), total_rows=total)
    return total
