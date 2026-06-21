"""Historical data download — supports OKX and Binance venues.

OKX is the default venue (Binance is geo-restricted on many servers).
Data is saved as Parquet partitioned by year/month with checkpoint tracking.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
import polars as pl
from pydantic import BaseModel

from intraday.data.checkpoint import Checkpoint, get_checkpoint_path
from intraday.data.schemas import FundingRate, Kline, OpenInterest
from intraday.utils.logging import get_logger

logger = get_logger(__name__)

AnyRecord = Kline | FundingRate | OpenInterest


class DownloadConfig(BaseModel):
    """Configuration for data download."""

    symbol: str = "BTCUSDT"
    venue: Literal["binance", "okx"] = "okx"
    kind: Literal["klines_1m", "klines_5m", "klines_15m", "klines_1h", "funding", "open_interest"]

    start: datetime | None = None
    end: datetime | None = None
    offset_from_checkpoint: int = 0

    data_dir: Path = Path("data")
    rate_limit_ms: int = 150
    force: bool = False


# --- API bases ---
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
OKX_BASE = "https://www.okx.com"

_OKX_INTERVAL = {
    "klines_1m": "1m",
    "klines_5m": "5m",
    "klines_15m": "15m",
    "klines_1h": "1H",
}
_OKX_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1H": 3_600_000,
}
_OKX_SYMBOL = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
}


def _okx_symbol(symbol: str) -> str:
    return _OKX_SYMBOL.get(symbol, symbol)


def _record_ts_ms(record: AnyRecord) -> int:
    if isinstance(record, Kline):
        return record.open_time_ms
    if isinstance(record, FundingRate):
        return record.funding_time_ms
    return record.time_ms


# ---------------------------------------------------------------------------
# OKX download helpers (backwards pagination: after= means ts < cursor)
# ---------------------------------------------------------------------------

async def _okx_klines_page(
    client: httpx.AsyncClient, inst_id: str, bar: str, after_ms: int, limit: int = 100
) -> list[Kline]:
    resp = await client.get(
        f"{OKX_BASE}/api/v5/market/history-candles",
        params={"instId": inst_id, "bar": bar, "after": after_ms, "limit": limit},
    )
    resp.raise_for_status()
    body = resp.json()
    if body["code"] != "0":
        raise RuntimeError(f"OKX klines error {body['code']}: {body.get('msg')}")

    interval_ms = _OKX_INTERVAL_MS[bar]
    results = []
    for row in body["data"]:
        ts = int(row[0])
        results.append(Kline(
            symbol=inst_id,
            interval=bar.lower(),
            open_time_ms=ts,
            close_time_ms=ts + interval_ms - 1,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
            num_trades=0,
            taker_buy_base_volume=0.0,
            taker_buy_quote_volume=0.0,
        ))
    return results  # newest-first


async def _okx_funding_page(
    client: httpx.AsyncClient, inst_id: str, after_ms: int, limit: int = 100
) -> list[FundingRate]:
    resp = await client.get(
        f"{OKX_BASE}/api/v5/public/funding-rate-history",
        params={"instId": inst_id, "after": after_ms, "limit": limit},
    )
    resp.raise_for_status()
    body = resp.json()
    if body["code"] != "0":
        raise RuntimeError(f"OKX funding error {body['code']}: {body.get('msg')}")

    results = []
    for row in body["data"]:
        results.append(FundingRate(
            symbol=inst_id,
            funding_time_ms=int(row["fundingTime"]),
            funding_rate=float(row["realizedRate"]),
        ))
    return results  # newest-first


async def _okx_oi_all(
    client: httpx.AsyncClient, inst_id: str, start_ms: int, end_ms: int, limit: int = 100
) -> list[OpenInterest]:
    """Fetch OI history using 1D period (covers months in a single request).

    OKX's rubik OI endpoint does not support backwards-cursor pagination for
    sub-daily periods. Daily granularity returns up to 100 days in one shot.
    """
    resp = await client.get(
        f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-history",
        params={"instId": inst_id, "period": "1D", "limit": limit},
    )
    resp.raise_for_status()
    body = resp.json()
    if body["code"] != "0":
        raise RuntimeError(f"OKX OI error {body['code']}: {body.get('msg')}")

    results = []
    for row in body["data"]:
        ts = int(row[0])
        if start_ms <= ts <= end_ms:
            results.append(OpenInterest(
                symbol=inst_id,
                time_ms=ts,
                open_interest=float(row[2]),      # BTC
                open_interest_usd=float(row[3]),  # USDT
            ))
    results.sort(key=lambda r: r.time_ms)
    return results


async def _collect_okx(config: DownloadConfig, start_ms: int, end_ms: int) -> list[AnyRecord]:
    """Collect all OKX records in [start_ms, end_ms].

    OI uses a single daily-granularity request (the endpoint does not support
    backwards cursor pagination for sub-daily periods).
    Klines and funding use backwards pagination with the 'after' cursor.
    """
    inst_id = _okx_symbol(config.symbol)

    # OI: single request, daily granularity
    if config.kind == "open_interest":
        async with httpx.AsyncClient(timeout=30.0) as client:
            records = await _okx_oi_all(client, inst_id, start_ms, end_ms)
        logger.info("download.okx_oi_done", records=len(records), kind="open_interest")
        return records

    # Klines / funding: backwards cursor pagination
    all_records: list[AnyRecord] = []
    cursor_ms = end_ms + 1
    prev_cursor = -1  # stuck-cursor guard

    async with httpx.AsyncClient(timeout=30.0) as client:
        while cursor_ms > start_ms:
            if cursor_ms == prev_cursor:
                logger.warning("download.cursor_stuck", cursor_ms=cursor_ms, kind=config.kind)
                break
            prev_cursor = cursor_ms

            try:
                if config.kind.startswith("klines_"):
                    bar = _OKX_INTERVAL[config.kind]
                    page = await _okx_klines_page(client, inst_id, bar, cursor_ms)
                else:
                    page = await _okx_funding_page(client, inst_id, cursor_ms)

                if not page:
                    break

                in_range = [r for r in page if _record_ts_ms(r) >= start_ms]
                all_records.extend(in_range)

                oldest_ts = min(_record_ts_ms(r) for r in page)
                logger.info(
                    "download.okx_page",
                    kind=config.kind,
                    page_records=len(in_range),
                    total_records=len(all_records),
                    oldest_dt=datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc).isoformat(),
                )

                if oldest_ts <= start_ms:
                    break

                cursor_ms = oldest_ts
                await asyncio.sleep(config.rate_limit_ms / 1000)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("download.rate_limited", wait_s=10)
                    await asyncio.sleep(10)
                    continue
                logger.error("download.http_error", status=e.response.status_code, kind=config.kind)
                raise

    # Sort chronologically and deduplicate
    all_records.sort(key=_record_ts_ms)
    seen: set[int] = set()
    unique: list[AnyRecord] = []
    for r in all_records:
        ts = _record_ts_ms(r)
        if ts not in seen:
            seen.add(ts)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Binance download helpers (forwards pagination)
# ---------------------------------------------------------------------------

async def _binance_klines_page(
    client: httpx.AsyncClient, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000
) -> list[Kline]:
    resp = await client.get(
        f"{BINANCE_SPOT_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit},
    )
    resp.raise_for_status()
    results = []
    for row in resp.json():
        results.append(Kline(
            symbol=symbol,
            interval=interval,
            open_time_ms=row[0],
            close_time_ms=row[6],
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
            num_trades=row[8],
            taker_buy_base_volume=float(row[9]),
            taker_buy_quote_volume=float(row[10]),
        ))
    return results


async def _binance_funding_page(
    client: httpx.AsyncClient, symbol: str, start_ms: int, end_ms: int, limit: int = 1000
) -> list[FundingRate]:
    resp = await client.get(
        f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit},
    )
    resp.raise_for_status()
    results = []
    for row in resp.json():
        results.append(FundingRate(
            symbol=row["symbol"],
            funding_time_ms=row["fundingTime"],
            funding_rate=float(row["fundingRate"]),
        ))
    return results


async def _binance_oi_page(
    client: httpx.AsyncClient, symbol: str, start_ms: int, end_ms: int, limit: int = 500
) -> list[OpenInterest]:
    resp = await client.get(
        f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": "5m", "startTime": start_ms, "endTime": end_ms, "limit": limit},
    )
    resp.raise_for_status()
    results = []
    for row in resp.json():
        results.append(OpenInterest(
            symbol=row["symbol"],
            time_ms=row["timestamp"],
            open_interest=float(row["sumOpenInterest"]),
            open_interest_usd=float(row["sumOpenInterestValue"]),
        ))
    return results


async def _collect_binance(config: DownloadConfig, start_ms: int, end_ms: int) -> list[AnyRecord]:
    """Collect all Binance records in [start_ms, end_ms] using forward pagination."""
    batch_size = 1000
    all_records: list[AnyRecord] = []
    current_ms = start_ms

    async with httpx.AsyncClient(timeout=30.0) as client:
        while current_ms < end_ms:
            batch_end_ms = min(current_ms + (batch_size * 60 * 1000), end_ms)
            try:
                if config.kind.startswith("klines_"):
                    interval = config.kind.split("_")[1]
                    page = await _binance_klines_page(client, config.symbol, interval, current_ms, batch_end_ms)
                elif config.kind == "funding":
                    page = await _binance_funding_page(client, config.symbol, current_ms, batch_end_ms)
                else:
                    page = await _binance_oi_page(client, config.symbol, current_ms, batch_end_ms)

                if not page:
                    break

                all_records.extend(page)
                last_ts = _record_ts_ms(page[-1])
                logger.info(
                    "download.binance_page",
                    kind=config.kind,
                    page_records=len(page),
                    total_records=len(all_records),
                    progress_pct=(current_ms - start_ms) / (end_ms - start_ms) * 100,
                )
                current_ms = last_ts + 1
                await asyncio.sleep(config.rate_limit_ms / 1000)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("download.rate_limited", wait_s=60)
                    await asyncio.sleep(60)
                    continue
                logger.error("download.http_error", status=e.response.status_code, kind=config.kind)
                raise

    return all_records


# ---------------------------------------------------------------------------
# Shared save + checkpoint logic
# ---------------------------------------------------------------------------

def save_to_parquet(
    data: list[AnyRecord],
    output_path: Path,
) -> None:
    """Save records to Parquet (zstd compressed)."""
    if not data:
        logger.warning("save_to_parquet.empty", path=str(output_path))
        return

    records = [item.model_dump() for item in data]
    df = pl.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    logger.info(
        "save_to_parquet.success",
        path=str(output_path),
        num_records=len(data),
        file_size_mb=round(output_path.stat().st_size / 1024 / 1024, 3),
    )


def _save_and_checkpoint(records: list[AnyRecord], config: DownloadConfig) -> None:
    """Group records by year-month, write one Parquet per month, update checkpoint."""
    by_month: defaultdict[str, list[AnyRecord]] = defaultdict(list)
    for r in records:
        ts = _record_ts_ms(r)
        month_key = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y/%Y-%m")
        by_month[month_key].append(r)

    file_paths: list[str] = []
    for partition, month_records in sorted(by_month.items()):
        output_path = (
            config.data_dir / "raw" / config.venue / config.kind / config.symbol / partition
        ).with_suffix(".parquet")
        save_to_parquet(month_records, output_path)
        file_paths.append(str(output_path))

    # Single checkpoint update covering the full collected range
    checkpoint_path = get_checkpoint_path(config.data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)
    first_ts = _record_ts_ms(records[0])
    last_ts = _record_ts_ms(records[-1])
    # Update once per file written
    for fp in file_paths:
        checkpoint.update(
            config.symbol, config.venue, config.kind,
            first_ts, last_ts, len(records), fp,
        )
    checkpoint.save(checkpoint_path)
    logger.info(
        "download.checkpoint_saved",
        kind=config.kind,
        total_records=len(records),
        files=len(file_paths),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def download_historical(config: DownloadConfig) -> None:
    """Download historical data with checkpoint tracking."""
    logger.info(
        "download.started",
        symbol=config.symbol,
        venue=config.venue,
        kind=config.kind,
        start=config.start.isoformat() if config.start else "checkpoint",
        end=config.end.isoformat() if config.end else "now",
    )

    # Determine time range
    checkpoint_path = get_checkpoint_path(config.data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)

    if config.start:
        start_ms = int(config.start.timestamp() * 1000)
    else:
        start_ms = checkpoint.get_next_start_time_ms(
            config.symbol, config.venue, config.kind, requested_start_ms=None
        ) + config.offset_from_checkpoint

    end_ms = (
        int(config.end.timestamp() * 1000)
        if config.end
        else int(datetime.now(timezone.utc).timestamp() * 1000)
    )

    logger.info(
        "download.range",
        start_dt=datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
        end_dt=datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
    )

    if not config.force and checkpoint.has_data(
        config.symbol, config.venue, config.kind, start_ms, end_ms
    ):
        logger.info("download.already_exists", message="Use --force to re-download")
        return

    # Fetch all records
    if config.venue == "okx":
        records = await _collect_okx(config, start_ms, end_ms)
    else:
        records = await _collect_binance(config, start_ms, end_ms)

    if not records:
        logger.warning("download.empty", start_ms=start_ms, end_ms=end_ms)
        return

    _save_and_checkpoint(records, config)

    logger.info(
        "download.complete",
        kind=config.kind,
        total_records=len(records),
    )
