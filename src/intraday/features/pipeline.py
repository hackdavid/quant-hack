"""TransformationPipeline — raw Parquet → feature Parquet.

Single entry point that:
  1. Discovers raw data files for each day
  2. Merges all event streams (trades, depth, metrics, klines) into one
     time-sorted stream — identical to how live WS events arrive
  3. Feeds the merged stream through FeatureCalculator
  4. Carries rolling state (VPIN buckets, Hawkes intensities, price windows)
     across day boundaries — no artificial resets at midnight
  5. Writes one feature Parquet per calendar day, sorted by bar_time_ms

Parallel mode:
  Splits the date range into N equal chunks, each with a configurable warmup
  window so rolling state (VPIN, Hawkes) is properly initialized.
  Each chunk runs in its own OS process — no shared memory, no locking.

Output:
  data/features/BTCUSDT/2026-05-20.parquet  ← 288 rows (one per 5m bar)
  data/features/BTCUSDT/2026-05-21.parquet
  ...
"""

import math
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import polars as pl

from intraday.features.calculator import (
    AggTrade,
    DepthBands,
    Event,
    FeatureCalculator,
    KlineBar,
    MetricsUpdate,
)
from intraday.features.schema import FEATURE_ROW_SCHEMA, FeatureRow
from intraday.utils.logging import get_logger

logger = get_logger(__name__)

_MS_PER_DAY = 86_400_000


# ---------------------------------------------------------------------------
# Raw Parquet → event stream loaders
# ---------------------------------------------------------------------------

def _load_events(raw_base: Path, symbol: str, day: date) -> list[tuple[int, str, Event]]:
    """Load all raw events for one day. Returns list of (time_ms, kind, event)."""
    events: list[tuple[int, str, Event]] = []
    d = day.isoformat()

    # aggTrades
    p = raw_base / "aggTrades" / symbol / f"{d}.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        for r in df.iter_rows(named=True):
            events.append((
                r["time_ms"], "trade",
                AggTrade(r["time_ms"], r["price"], r["quantity"], r["is_buyer_maker"])
            ))

    # klines_1m
    p = raw_base / "klines_1m" / symbol / f"{d}.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        for r in df.iter_rows(named=True):
            events.append((
                r["close_time_ms"], "kline_1m",
                KlineBar(
                    open_time_ms=r["open_time_ms"],
                    close_time_ms=r["close_time_ms"],
                    open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                    volume=r["volume"], trade_count=r["trade_count"],
                    taker_buy_volume=r["taker_buy_volume"], interval="1m",
                )
            ))

    # klines_5m — bar close triggers feature emission
    p = raw_base / "klines_5m" / symbol / f"{d}.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        for r in df.iter_rows(named=True):
            events.append((
                r["close_time_ms"], "kline_5m",
                KlineBar(
                    open_time_ms=r["open_time_ms"],
                    close_time_ms=r["close_time_ms"],
                    open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                    volume=r["volume"], trade_count=r["trade_count"],
                    taker_buy_volume=r["taker_buy_volume"], interval="5m",
                )
            ))

    # bookDepth (wide Parquet → DepthBands)
    p = raw_base / "bookDepth" / symbol / f"{d}.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        for r in df.iter_rows(named=True):
            events.append((
                r["snapshot_time_ms"], "depth",
                DepthBands(
                    snapshot_time_ms=r["snapshot_time_ms"],
                    bid_02pct=r.get("bid_02pct") or 0.0,
                    bid_1pct=r.get("bid_1pct") or 0.0,
                    bid_2pct=r.get("bid_2pct") or 0.0,
                    bid_3pct=r.get("bid_3pct") or 0.0,
                    bid_4pct=r.get("bid_4pct") or 0.0,
                    bid_5pct=r.get("bid_5pct") or 0.0,
                    ask_02pct=r.get("ask_02pct") or 0.0,
                    ask_1pct=r.get("ask_1pct") or 0.0,
                    ask_2pct=r.get("ask_2pct") or 0.0,
                    ask_3pct=r.get("ask_3pct") or 0.0,
                    ask_4pct=r.get("ask_4pct") or 0.0,
                    ask_5pct=r.get("ask_5pct") or 0.0,
                )
            ))

    # metrics (5-min OI, L/S ratios)
    p = raw_base / "metrics" / symbol / f"{d}.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        for r in df.iter_rows(named=True):
            events.append((
                r["create_time_ms"], "metrics",
                MetricsUpdate(
                    create_time_ms=r["create_time_ms"],
                    oi_btc=r["oi_btc"],
                    oi_usd=r["oi_usd"],
                    ls_count_ratio=r["ls_count_ratio"],
                    taker_ls_vol_ratio=r["taker_ls_vol_ratio"],
                    top_ls_count=r.get("top_ls_count") or 0.0,
                    top_ls_value=r.get("top_ls_value") or 0.0,
                )
            ))

    # Sort by timestamp. For equal timestamps dispatch order: trade < depth < metrics < kline
    _ORDER = {"trade": 0, "depth": 1, "metrics": 2, "kline_1m": 3, "kline_5m": 4}
    events.sort(key=lambda e: (e[0], _ORDER.get(e[1], 9)))
    return events


def _process_day(
    day: date,
    calc: FeatureCalculator,
    raw_base: Path,
    features_dir: Path,
    symbol: str,
    force: bool,
) -> int:
    """Process one day: feed events into calc, write parquet. Returns rows written."""
    out_path = features_dir / f"{day.isoformat()}.parquet"
    if out_path.exists() and not force:
        return 0

    events = _load_events(raw_base, symbol, day)
    if not events:
        return 0

    rows: list[FeatureRow] = []
    for _, _kind, event in events:
        result = calc.dispatch(event)
        if result is not None:
            rows.append(result)

    rows.extend(calc.flush())
    if not rows:
        return 0

    day_start_ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    day_end_ms = day_start_ms + _MS_PER_DAY
    rows_today = [r for r in rows if day_start_ms <= r.bar_time_ms < day_end_ms]
    if not rows_today:
        return 0

    df = pl.DataFrame([r.model_dump() for r in rows_today], schema=FEATURE_ROW_SCHEMA).sort("bar_time_ms")
    df.write_parquet(out_path, compression="zstd")
    return len(rows_today)


# ---------------------------------------------------------------------------
# Module-level worker — must be at top level to be picklable by multiprocessing
# ---------------------------------------------------------------------------

def _chunk_worker(args: tuple) -> tuple[int, int]:
    """Process one date-range chunk. Returns (chunk_id, total_rows_written).

    Steps:
      1. Warmup phase: replay `warmup_days` days before the chunk start to
         initialize rolling state (VPIN buckets, Hawkes, price windows).
         No files are written during warmup.
      2. Production phase: process chunk_start..chunk_end and write parquet.
    """
    (
        chunk_id,
        data_dir_str, symbol,
        warmup_start_iso, chunk_start_iso, chunk_end_iso,
        calc_kwargs, force,
    ) = args

    data_dir = Path(data_dir_str)
    raw_base = data_dir / "raw" / "binance"
    features_dir = data_dir / "features" / symbol
    features_dir.mkdir(parents=True, exist_ok=True)

    warmup_start = date.fromisoformat(warmup_start_iso)
    chunk_start  = date.fromisoformat(chunk_start_iso)
    chunk_end    = date.fromisoformat(chunk_end_iso)

    calc = FeatureCalculator(**calc_kwargs)

    # ── Warmup: feed events, discard output ──────────────────────────────
    day = warmup_start
    while day < chunk_start:
        events = _load_events(raw_base, symbol, day)
        for _, _kind, event in events:
            calc.dispatch(event)
        day += timedelta(days=1)

    # ── Production: write feature parquet files ───────────────────────────
    total = 0
    day = chunk_start
    while day <= chunk_end:
        written = _process_day(day, calc, raw_base, features_dir, symbol, force)
        total += written
        day += timedelta(days=1)

    return chunk_id, total


# ---------------------------------------------------------------------------
# TransformationPipeline
# ---------------------------------------------------------------------------

class TransformationPipeline:
    """End-to-end pipeline: raw Parquet → feature Parquet."""

    def __init__(
        self,
        data_dir: Path,
        symbol: str = "BTCUSDT",
        vpin_bucket_btc: float = 100.0,
        vpin_window: int = 50,
        hawkes_alpha: float = 1.0,
        hawkes_beta: float = 10.0,
        hawkes_mu: float = 6.0,
    ) -> None:
        self.data_dir     = Path(data_dir)
        self.raw_base     = self.data_dir / "raw" / "binance"
        self.features_dir = self.data_dir / "features" / symbol
        self.symbol       = symbol
        self._calc_kwargs = dict(
            symbol=symbol,
            vpin_bucket_btc=vpin_bucket_btc,
            vpin_window=vpin_window,
            hawkes_alpha=hawkes_alpha,
            hawkes_beta=hawkes_beta,
            hawkes_mu=hawkes_mu,
        )

    # ── Sequential run ────────────────────────────────────────────────────

    def run(
        self,
        start_date: date,
        end_date: date,
        force: bool = False,
    ) -> int:
        """Process all days sequentially. Rolling state is exact — no warmup gap."""
        self.features_dir.mkdir(parents=True, exist_ok=True)
        calc = FeatureCalculator(**self._calc_kwargs)
        total = 0
        day = start_date
        while day <= end_date:
            written = _process_day(day, calc, self.raw_base, self.features_dir, self.symbol, force)
            total += written
            day += timedelta(days=1)

        logger.info("pipeline.complete",
                    start=start_date.isoformat(), end=end_date.isoformat(), total_rows=total)
        return total

    # ── Parallel run ──────────────────────────────────────────────────────

    def run_parallel(
        self,
        start_date: date,
        end_date: date,
        force: bool = False,
        workers: int = 8,
        warmup_days: int = 14,
    ) -> int:
        """Split date range into `workers` chunks and process in parallel.

        Each chunk warms up `warmup_days` before its start so rolling state
        (VPIN, Hawkes, RSI) is properly initialized. Rolling state within each
        chunk is fully correct; the only approximation is the very first few bars
        of each chunk boundary (resolved by warmup).

        Note: GPU (A100) is not used here — Hawkes/VPIN are sequential state
        machines that don't benefit from GPU tensor ops. All 16 CPU cores are
        used instead for ~16x wall-clock speedup.
        """
        self.features_dir.mkdir(parents=True, exist_ok=True)

        total_days = (end_date - start_date).days + 1
        actual_workers = min(workers, total_days)
        chunk_size = math.ceil(total_days / actual_workers)

        job_args = []
        for i in range(actual_workers):
            chunk_start = start_date + timedelta(days=i * chunk_size)
            if chunk_start > end_date:
                break
            chunk_end = min(start_date + timedelta(days=(i + 1) * chunk_size - 1), end_date)
            # Clamp warmup so it never goes before the very first available date
            warmup_start = max(start_date, chunk_start - timedelta(days=warmup_days))

            job_args.append((
                i,
                str(self.data_dir), self.symbol,
                warmup_start.isoformat(), chunk_start.isoformat(), chunk_end.isoformat(),
                self._calc_kwargs, force,
            ))

        logger.info(
            "pipeline.parallel_start",
            workers=len(job_args),
            warmup_days=warmup_days,
            total_days=total_days,
            chunk_size=chunk_size,
        )

        with Pool(processes=len(job_args)) as pool:
            results = pool.map(_chunk_worker, job_args)

        total = sum(rows for _, rows in results)
        logger.info("pipeline.parallel_complete",
                    workers=len(job_args), total_rows=total,
                    start=start_date.isoformat(), end=end_date.isoformat())
        return total
