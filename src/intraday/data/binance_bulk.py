"""Bulk historical data downloader from data.binance.vision.

Downloads daily zip files in parallel, parses CSV, writes daily Parquet.
All datasets use Binance BTCUSDT perpetual futures (USDM).

Datasets and their Parquet schemas:
  aggTrades  → time_ms, price, quantity, is_buyer_maker
  klines_1m  → open_time_ms, open, high, low, close, volume, close_time_ms,
               quote_volume, trade_count, taker_buy_volume, taker_buy_quote_volume
  klines_5m  → same as klines_1m
  bookDepth  → snapshot_time_ms, bid_02pct, bid_1pct, ..., ask_02pct, ask_1pct, ...
  metrics    → create_time_ms, oi_btc, oi_usd, top_ls_count, top_ls_value,
               ls_count_ratio, taker_ls_vol_ratio
"""

import asyncio
import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx
import polars as pl

from intraday.utils.logging import get_logger

logger = get_logger(__name__)

BINANCE_DATA_BASE = "https://data.binance.vision"

# URL template for each dataset kind
# {symbol} = "BTCUSDT", {date} = "2026-06-19"
_URL_TEMPLATES: dict[str, str] = {
    "aggTrades": "data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip",
    "klines_1m": "data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{date}.zip",
    "klines_5m": "data/futures/um/daily/klines/{symbol}/5m/{symbol}-5m-{date}.zip",
    "bookDepth": "data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip",
    "metrics":   "data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date}.zip",
}

BulkKind = Literal["aggTrades", "klines_1m", "klines_5m", "bookDepth", "metrics"]

# Depth band percentage values in the CSV → column names
_DEPTH_BAND_COLS: dict[float, str] = {
    -0.2: "bid_02pct", -1.0: "bid_1pct", -2.0: "bid_2pct",
    -3.0: "bid_3pct", -4.0: "bid_4pct", -5.0: "bid_5pct",
     0.2: "ask_02pct",  1.0: "ask_1pct",  2.0: "ask_2pct",
     3.0: "ask_3pct",   4.0: "ask_4pct",   5.0: "ask_5pct",
}


# ---------------------------------------------------------------------------
# CSV → Polars transforms (one per dataset kind)
# ---------------------------------------------------------------------------

def _parse_aggTrades(raw: bytes) -> pl.DataFrame:
    df = pl.read_csv(
        raw,
        schema_overrides={
            "agg_trade_id": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "transact_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
        },
    )
    return (
        df.rename({"transact_time": "time_ms"})
        .select(["time_ms", "price", "quantity", "is_buyer_maker"])
        .sort("time_ms")
    )


def _parse_klines(raw: bytes) -> pl.DataFrame:
    df = pl.read_csv(
        raw,
        schema_overrides={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Int64,
            "quote_volume": pl.Float64,
            "count": pl.Int32,
            "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64,
            "ignore": pl.Int32,
        },
    )
    return (
        df.rename({"open_time": "open_time_ms", "close_time": "close_time_ms", "count": "trade_count"})
        .drop("ignore")
        .sort("open_time_ms")
    )


def _parse_bookDepth(raw: bytes) -> pl.DataFrame:
    df = pl.read_csv(
        raw,
        schema_overrides={
            "timestamp": pl.String,
            "percentage": pl.Float64,
            "depth": pl.Float64,
            "notional": pl.Float64,
        },
    )
    # Parse timestamp string "YYYY-MM-DD HH:MM:SS" → epoch ms
    df = df.with_columns(
        pl.col("timestamp")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
        .dt.epoch(time_unit="ms")
        .alias("snapshot_time_ms")
    ).drop("timestamp")

    # Pivot long → wide: one row per snapshot
    timestamps = df.select("snapshot_time_ms").unique().sort("snapshot_time_ms")
    frames = [timestamps]
    for pct, col_name in _DEPTH_BAND_COLS.items():
        band = (
            df.filter(pl.col("percentage") == pct)
            .select(["snapshot_time_ms", "depth"])
            .rename({"depth": col_name})
        )
        frames.append(band)

    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.join(frame, on="snapshot_time_ms", how="left")

    return wide.sort("snapshot_time_ms")


def _parse_metrics(raw: bytes) -> pl.DataFrame:
    df = pl.read_csv(
        raw,
        schema_overrides={
            "create_time": pl.String,
            "symbol": pl.String,
            "sum_open_interest": pl.Float64,
            "sum_open_interest_value": pl.Float64,
            "count_toptrader_long_short_ratio": pl.Float64,
            "sum_toptrader_long_short_ratio": pl.Float64,
            "count_long_short_ratio": pl.Float64,
            "sum_taker_long_short_vol_ratio": pl.Float64,
        },
    )
    df = df.with_columns(
        pl.col("create_time")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
        .dt.epoch(time_unit="ms")
        .alias("create_time_ms")
    ).drop(["create_time", "symbol"])

    return (
        df.rename({
            "sum_open_interest":              "oi_btc",
            "sum_open_interest_value":        "oi_usd",
            "count_toptrader_long_short_ratio": "top_ls_count",
            "sum_toptrader_long_short_ratio":  "top_ls_value",
            "count_long_short_ratio":         "ls_count_ratio",
            "sum_taker_long_short_vol_ratio": "taker_ls_vol_ratio",
        })
        .sort("create_time_ms")
    )


_PARSERS = {
    "aggTrades": _parse_aggTrades,
    "klines_1m": _parse_klines,
    "klines_5m": _parse_klines,
    "bookDepth": _parse_bookDepth,
    "metrics": _parse_metrics,
}


# ---------------------------------------------------------------------------
# Single-day download
# ---------------------------------------------------------------------------

def _output_path(symbol: str, kind: BulkKind, day: date, data_dir: Path) -> Path:
    return data_dir / "raw" / "binance" / kind / symbol / f"{day.isoformat()}.parquet"


async def _download_day(
    client: httpx.AsyncClient,
    symbol: str,
    kind: BulkKind,
    day: date,
    data_dir: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download one day of one kind. Returns True if downloaded, False if skipped."""
    out_path = _output_path(symbol, kind, day, data_dir)
    if out_path.exists():
        return False  # already done

    url_path = _URL_TEMPLATES[kind].format(symbol=symbol, date=day.isoformat())
    url = f"{BINANCE_DATA_BASE}/{url_path}"

    async with semaphore:
        try:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code == 404:
                logger.debug("bulk.not_found", kind=kind, date=day.isoformat())
                return False
            resp.raise_for_status()

            # Unzip in memory
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            raw_csv = zf.read(csv_name)

            # Parse and save
            df = _PARSERS[kind](raw_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_path, compression="zstd")

            logger.info(
                "bulk.saved",
                kind=kind,
                date=day.isoformat(),
                rows=len(df),
                size_kb=round(out_path.stat().st_size / 1024, 1),
            )
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("bulk.rate_limited", wait_s=10)
                await asyncio.sleep(10)
                return await _download_day(client, symbol, kind, day, data_dir, semaphore)
            logger.error("bulk.http_error", kind=kind, date=day.isoformat(), status=e.response.status_code)
            return False
        except Exception as e:
            logger.error("bulk.error", kind=kind, date=day.isoformat(), error=str(e))
            return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def download_bulk(
    symbol: str = "BTCUSDT",
    kinds: list[BulkKind] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    data_dir: Path = Path("data"),
    max_concurrent: int = 8,
) -> dict[str, int]:
    """Download bulk data from data.binance.vision in parallel.

    Downloads all (kind, date) pairs concurrently up to max_concurrent.
    Skips files that already exist on disk.

    Returns dict of {kind: days_downloaded}.
    """
    if kinds is None:
        kinds = ["aggTrades", "klines_1m", "klines_5m", "bookDepth", "metrics"]
    if end_date is None:
        end_date = date.today() - timedelta(days=1)  # yesterday (today not yet complete)
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    logger.info(
        "bulk.started",
        symbol=symbol,
        kinds=kinds,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        total_tasks=len(kinds) * len(days),
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    counts: dict[str, int] = {k: 0 for k in kinds}

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            _download_day(client, symbol, kind, day, data_dir, semaphore)
            for kind in kinds
            for day in days
        ]
        results = await asyncio.gather(*tasks)

    # Tally results
    idx = 0
    for kind in kinds:
        for _ in days:
            if results[idx]:
                counts[kind] += 1
            idx += 1

    logger.info("bulk.complete", counts=counts)
    return counts


def depth_bands_from_top20(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> dict[str, float]:
    """Convert top-20 WS depth levels to the same %-band format as historical bookDepth.

    Used during live paper trading so the feature calculator receives identical
    format regardless of data source.

    Args:
        bids: list of (price, quantity) sorted best-first (descending price)
        asks: list of (price, quantity) sorted best-first (ascending price)

    Returns dict with keys matching _DEPTH_BAND_COLS values (e.g. "bid_02pct").
    """
    if not bids or not asks:
        return {}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2

    result: dict[str, float] = {}
    for pct, col_name in _DEPTH_BAND_COLS.items():
        if pct < 0:
            # Bid side: cumulative depth at prices >= mid*(1 + pct/100)
            threshold = mid * (1 + pct / 100)  # pct is negative → below mid
            depth = sum(qty for price, qty in bids if price >= threshold)
        else:
            # Ask side: cumulative depth at prices <= mid*(1 + pct/100)
            threshold = mid * (1 + pct / 100)
            depth = sum(qty for price, qty in asks if price <= threshold)
        result[col_name] = depth

    return result
