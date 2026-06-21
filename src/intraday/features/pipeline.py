"""TransformationPipeline — raw Parquet → feature Parquet.

Single entry point that:
  1. Discovers raw data files for each day
  2. Merges all event streams (trades, depth, metrics, klines) into one
     time-sorted stream — identical to how live WS events arrive
  3. Feeds the merged stream through FeatureCalculator
  4. Carries rolling state (VPIN buckets, Hawkes intensities, price windows)
     across day boundaries — no artificial resets at midnight
  5. Writes one feature Parquet per calendar day, sorted by bar_time_ms

Output:
  data/features/BTCUSDT/2026-05-20.parquet  ← 288 rows (one per 5m bar)
  data/features/BTCUSDT/2026-05-21.parquet
  ...

Iterate tick-by-tick (bar-by-bar) across days via LazyFeatureStore.
"""

from datetime import date, timedelta
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
from intraday.features.schema import FeatureRow
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


# ---------------------------------------------------------------------------
# TransformationPipeline
# ---------------------------------------------------------------------------

class TransformationPipeline:
    """End-to-end pipeline: raw Parquet → feature Parquet.

    Example:
        pipeline = TransformationPipeline(Path("data"))
        total = pipeline.run(date(2026, 5, 20), date(2026, 6, 19))
        print(f"{total} feature rows written")
    """

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
        self.data_dir    = Path(data_dir)
        self.raw_base    = self.data_dir / "raw" / "binance"
        self.features_dir = self.data_dir / "features" / symbol
        self.symbol      = symbol
        self._calc_kwargs = dict(
            symbol=symbol,
            vpin_bucket_btc=vpin_bucket_btc,
            vpin_window=vpin_window,
            hawkes_alpha=hawkes_alpha,
            hawkes_beta=hawkes_beta,
            hawkes_mu=hawkes_mu,
        )

    def run(
        self,
        start_date: date,
        end_date: date,
        force: bool = False,
    ) -> int:
        """Process all days in chronological order. Returns total rows written.

        Rolling state (VPIN, Hawkes, price windows) carries across day boundaries.
        """
        self.features_dir.mkdir(parents=True, exist_ok=True)
        calc = FeatureCalculator(**self._calc_kwargs)
        total = 0
        day = start_date

        while day <= end_date:
            rows_written = self._run_day(day, calc, force)
            total += rows_written
            day += timedelta(days=1)

        logger.info(
            "pipeline.complete",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            total_rows=total,
        )
        return total

    def _run_day(self, day: date, calc: FeatureCalculator, force: bool) -> int:
        out_path = self.features_dir / f"{day.isoformat()}.parquet"
        if out_path.exists() and not force:
            logger.debug("pipeline.skip", day=day.isoformat())
            return 0

        events = _load_events(self.raw_base, self.symbol, day)
        if not events:
            logger.warning("pipeline.no_data", day=day.isoformat())
            return 0

        rows: list[FeatureRow] = []
        for _, kind, event in events:
            result = calc.dispatch(event)
            if result is not None:
                rows.append(result)

        # Flush remaining rows that have enough future data
        rows.extend(calc.flush())

        if not rows:
            logger.warning("pipeline.no_rows", day=day.isoformat())
            return 0

        # Keep only rows belonging to this calendar day
        day_start_ms = int(day.strftime("%s")) * 1000  # midnight UTC
        # Safer: compute from date parts
        from datetime import datetime, timezone
        day_start_ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
        day_end_ms   = day_start_ms + _MS_PER_DAY

        rows_today = [r for r in rows if day_start_ms <= r.bar_time_ms < day_end_ms]

        if not rows_today:
            logger.debug("pipeline.all_rows_spillover", day=day.isoformat(), total=len(rows))
            return 0

        df = pl.DataFrame([r.model_dump() for r in rows_today]).sort("bar_time_ms")
        df.write_parquet(out_path, compression="zstd")

        logger.info(
            "pipeline.day_done",
            day=day.isoformat(),
            rows=len(rows_today),
            size_kb=round(out_path.stat().st_size / 1024, 1),
        )
        return len(rows_today)
