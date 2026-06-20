"""Historical data download from Binance REST API.

Supports:
- Klines (1m, 5m, 15m, 1h)
- Funding rates
- Open interest
- Checkpoint-based pagination

Data is saved as Parquet partitioned by year/month.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx
import polars as pl
from pydantic import BaseModel

from intraday.data.checkpoint import Checkpoint, get_checkpoint_path
from intraday.data.schemas import FundingRate, Kline, OpenInterest
from intraday.utils.logging import get_logger

logger = get_logger(__name__)


class DownloadConfig(BaseModel):
    """Configuration for data download."""

    symbol: str = "BTCUSDT"
    venue: Literal["binance"] = "binance"
    kind: Literal["klines_1m", "klines_5m", "klines_15m", "klines_1h", "funding", "open_interest"]

    start: datetime | None = None  # If None, resume from checkpoint
    end: datetime | None = None  # If None, download until now
    offset_from_checkpoint: int = 0  # Start from checkpoint + N milliseconds

    data_dir: Path = Path("data")
    batch_size: int = 1000  # Records per API request
    rate_limit_ms: int = 200  # Min ms between requests (5 req/s)

    force: bool = False  # Re-download even if checkpoint exists


# Binance API endpoints
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"


async def download_klines(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list[Kline]:
    """Download klines from Binance spot API.

    API: GET /api/v3/klines
    Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
    resp = await client.get(url, params=params)
    resp.raise_for_status()

    data = resp.json()
    klines = []

    for row in data:
        klines.append(
            Kline(
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
            )
        )

    return klines


async def download_funding_rate(
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list[FundingRate]:
    """Download funding rate history from Binance futures API.

    API: GET /fapi/v1/fundingRate
    Docs: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
    """
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
    resp = await client.get(url, params=params)
    resp.raise_for_status()

    data = resp.json()
    rates = []

    for row in data:
        rates.append(
            FundingRate(
                symbol=row["symbol"],
                funding_time_ms=row["fundingTime"],
                funding_rate=float(row["fundingRate"]),
            )
        )

    return rates


async def download_open_interest(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 500,
) -> list[OpenInterest]:
    """Download open interest history from Binance futures API.

    API: GET /fapi/v1/openInterestHist
    Docs: https://binance-docs.github.io/apidocs/futures/en/#open-interest-statistics-market_data
    """
    params = {
        "symbol": symbol,
        "period": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    url = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
    resp = await client.get(url, params=params)
    resp.raise_for_status()

    data = resp.json()
    oi_records = []

    for row in data:
        oi_records.append(
            OpenInterest(
                symbol=row["symbol"],
                time_ms=row["timestamp"],
                open_interest=float(row["sumOpenInterest"]),
                open_interest_usd=float(row["sumOpenInterestValue"]),
            )
        )

    return oi_records


def save_to_parquet(
    data: list[Kline] | list[FundingRate] | list[OpenInterest],
    output_path: Path,
) -> None:
    """Save data to Parquet using Polars.

    Validates schema via Pydantic, converts to Polars DataFrame, writes Parquet.
    """
    if not data:
        logger.warning("save_to_parquet.empty", path=str(output_path))
        return

    # Convert Pydantic models to dicts
    records = [item.model_dump() for item in data]

    # Create Polars DataFrame
    df = pl.DataFrame(records)

    # Write to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")

    logger.info(
        "save_to_parquet.success",
        path=str(output_path),
        num_records=len(data),
        file_size_mb=output_path.stat().st_size / 1024 / 1024,
    )


async def download_historical(config: DownloadConfig) -> None:
    """Download historical data with checkpoint tracking.

    Supports:
    - Resume from checkpoint (start=None)
    - Offset from checkpoint (offset_from_checkpoint=N)
    - Custom range (start/end provided)
    - Force re-download (force=True)
    """
    logger.info(
        "download.started",
        symbol=config.symbol,
        venue=config.venue,
        kind=config.kind,
        start=config.start.isoformat() if config.start else "checkpoint",
        end=config.end.isoformat() if config.end else "now",
    )

    # Load checkpoint
    checkpoint_path = get_checkpoint_path(config.data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)

    # Determine start time
    if config.start:
        start_ms = int(config.start.timestamp() * 1000)
    else:
        # Resume from checkpoint (with optional offset)
        start_ms = checkpoint.get_next_start_time_ms(
            config.symbol,
            config.venue,
            config.kind,
            requested_start_ms=None,  # Will use checkpoint end
        )
        start_ms += config.offset_from_checkpoint

    # Determine end time
    if config.end:
        end_ms = int(config.end.timestamp() * 1000)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    logger.info(
        "download.range",
        start_ms=start_ms,
        end_ms=end_ms,
        start_dt=datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
        end_dt=datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
    )

    # Check if already downloaded (unless force=True)
    if not config.force and checkpoint.has_data(
        config.symbol, config.venue, config.kind, start_ms, end_ms
    ):
        logger.info("download.already_exists", message="Data already downloaded, use --force to re-download")
        return

    # Download in batches
    async with httpx.AsyncClient(timeout=30.0) as client:
        current_ms = start_ms
        total_records = 0

        while current_ms < end_ms:
            # Calculate batch window
            batch_end_ms = min(current_ms + (config.batch_size * 60 * 1000), end_ms)

            # Download based on kind
            try:
                if config.kind.startswith("klines_"):
                    interval = config.kind.split("_")[1]  # "1m", "5m", etc.
                    data = await download_klines(
                        client,
                        config.symbol,
                        interval,
                        current_ms,
                        batch_end_ms,
                        config.batch_size,
                    )
                elif config.kind == "funding":
                    data = await download_funding_rate(
                        client,
                        config.symbol,
                        current_ms,
                        batch_end_ms,
                        config.batch_size,
                    )
                elif config.kind == "open_interest":
                    data = await download_open_interest(
                        client,
                        config.symbol,
                        "5m",  # Default interval
                        current_ms,
                        batch_end_ms,
                        config.batch_size,
                    )
                else:
                    raise ValueError(f"Unsupported kind: {config.kind}")

                if not data:
                    logger.warning("download.no_data", current_ms=current_ms, batch_end_ms=batch_end_ms)
                    break

                # Save to Parquet (partitioned by year-month)
                first_time = datetime.fromtimestamp(data[0].open_time_ms / 1000 if isinstance(data[0], Kline) else data[0].time_ms / 1000, tz=timezone.utc)
                partition = first_time.strftime("%Y/%Y-%m")
                output_path = (
                    config.data_dir
                    / "raw"
                    / config.venue
                    / config.kind
                    / config.symbol
                    / partition
                ).with_suffix(".parquet")

                save_to_parquet(data, output_path)

                # Update checkpoint
                last_record = data[-1]
                if isinstance(last_record, Kline):
                    batch_actual_end_ms = last_record.close_time_ms
                elif isinstance(last_record, FundingRate):
                    batch_actual_end_ms = last_record.funding_time_ms
                else:
                    batch_actual_end_ms = last_record.time_ms

                checkpoint.update(
                    config.symbol,
                    config.venue,
                    config.kind,
                    data[0].open_time_ms if isinstance(data[0], Kline) else data[0].time_ms if hasattr(data[0], 'time_ms') else data[0].funding_time_ms,
                    batch_actual_end_ms,
                    len(data),
                    str(output_path),
                )
                checkpoint.save(checkpoint_path)

                total_records += len(data)
                logger.info(
                    "download.batch_complete",
                    batch_records=len(data),
                    total_records=total_records,
                    progress_pct=(current_ms - start_ms) / (end_ms - start_ms) * 100,
                )

                # Move to next batch
                current_ms = batch_actual_end_ms + 1

                # Rate limiting
                await asyncio.sleep(config.rate_limit_ms / 1000)

            except httpx.HTTPStatusError as e:
                logger.error(
                    "download.http_error",
                    status_code=e.response.status_code,
                    message=str(e),
                    current_ms=current_ms,
                )
                if e.response.status_code == 429:
                    logger.warning("download.rate_limited", wait_s=60)
                    await asyncio.sleep(60)
                    continue
                raise

            except Exception as e:
                logger.error("download.error", error=str(e), current_ms=current_ms)
                raise

    logger.info(
        "download.complete",
        total_records=total_records,
        duration_s=(end_ms - start_ms) / 1000,
    )
