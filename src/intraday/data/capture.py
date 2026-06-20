"""Live WebSocket data capture from Binance.

Captures:
- Trade stream (@trade, @aggTrade)
- Depth stream (@depth@100ms, @depth20@100ms)
- Mark price stream (@markPrice) - includes funding rate
- Liquidation stream (@forceOrder)

Features:
- Auto-reconnect with exponential backoff
- Buffer in memory, flush to Parquet every 60s
- Gap detection and logging
- Checkpoint tracking
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import websockets
from pydantic import BaseModel

from intraday.data.checkpoint import Checkpoint, get_checkpoint_path
from intraday.data.schemas import Depth, DepthLevel, Liquidation, MarkPrice, Trade
from intraday.utils.logging import get_logger

logger = get_logger(__name__)


class CaptureConfig(BaseModel):
    """Configuration for live data capture."""

    symbol: str = "BTCUSDT"
    venue: Literal["binance"] = "binance"
    streams: list[Literal["trade", "depth", "mark_price", "liquidations"]] = [
        "trade",
        "depth",
        "mark_price",
    ]

    data_dir: Path = Path("data")
    flush_interval_s: int = 60  # Flush to disk every N seconds
    max_buffer_size: int = 10000  # Max events in buffer before forced flush
    reconnect_delay_s: int = 5  # Initial reconnect delay
    max_reconnect_delay_s: int = 60  # Max reconnect delay


# Binance WebSocket base URLs
BINANCE_WS_SPOT = "wss://stream.binance.com:9443/ws"
BINANCE_WS_FUTURES = "wss://fstream.binance.com/ws"


class StreamBuffer:
    """In-memory buffer for stream data before flush to disk."""

    def __init__(self) -> None:
        self.trades: list[Trade] = []
        self.depths: list[Depth] = []
        self.mark_prices: list[MarkPrice] = []
        self.liquidations: list[Liquidation] = []
        self.last_flush = datetime.now(timezone.utc)
        self.last_event_time_ms: dict[str, int] = {}  # Track gaps

    def add_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self._update_last_event_time("trade", trade.time_ms)

    def add_depth(self, depth: Depth) -> None:
        self.depths.append(depth)
        self._update_last_event_time("depth", depth.time_ms)

    def add_mark_price(self, mark_price: MarkPrice) -> None:
        self.mark_prices.append(mark_price)
        self._update_last_event_time("mark_price", mark_price.time_ms)

    def add_liquidation(self, liquidation: Liquidation) -> None:
        self.liquidations.append(liquidation)
        self._update_last_event_time("liquidations", liquidation.time_ms)

    def _update_last_event_time(self, stream: str, time_ms: int) -> None:
        """Track last event time for gap detection."""
        if stream in self.last_event_time_ms:
            gap_ms = time_ms - self.last_event_time_ms[stream]
            if gap_ms > 5000:  # > 5s gap
                logger.warning(
                    "stream.gap_detected",
                    stream=stream,
                    gap_ms=gap_ms,
                    gap_s=gap_ms / 1000,
                )
        self.last_event_time_ms[stream] = time_ms

    def size(self) -> int:
        """Total events in buffer."""
        return len(self.trades) + len(self.depths) + len(self.mark_prices) + len(self.liquidations)

    def clear(self) -> None:
        """Clear all buffers."""
        self.trades.clear()
        self.depths.clear()
        self.mark_prices.clear()
        self.liquidations.clear()
        self.last_flush = datetime.now(timezone.utc)


async def parse_trade_message(msg: dict, symbol: str) -> Trade | None:
    """Parse trade stream message.

    Format: https://binance-docs.github.io/apidocs/spot/en/#trade-streams
    """
    try:
        return Trade(
            symbol=symbol,
            trade_id=msg["t"],
            price=float(msg["p"]),
            quantity=float(msg["q"]),
            time_ms=msg["T"],
            is_buyer_maker=msg["m"],
        )
    except (KeyError, ValueError) as e:
        logger.error("parse.trade_error", error=str(e), msg=msg)
        return None


async def parse_depth_message(msg: dict, symbol: str) -> Depth | None:
    """Parse depth stream message.

    Format: https://binance-docs.github.io/apidocs/spot/en/#partial-book-depth-streams
    """
    try:
        bids = [DepthLevel(price=float(b[0]), quantity=float(b[1])) for b in msg["bids"]]
        asks = [DepthLevel(price=float(a[0]), quantity=float(a[1])) for a in msg["asks"]]

        return Depth(
            symbol=symbol,
            time_ms=msg.get("E", int(datetime.now(timezone.utc).timestamp() * 1000)),
            last_update_id=msg["lastUpdateId"],
            bids=bids,
            asks=asks,
        )
    except (KeyError, ValueError) as e:
        logger.error("parse.depth_error", error=str(e), msg=msg)
        return None


async def parse_mark_price_message(msg: dict, symbol: str) -> MarkPrice | None:
    """Parse mark price stream message.

    Format: https://binance-docs.github.io/apidocs/futures/en/#mark-price-stream
    """
    try:
        return MarkPrice(
            symbol=symbol,
            time_ms=msg["E"],
            mark_price=float(msg["p"]),
            index_price=float(msg["i"]),
            estimated_settle_price=float(msg.get("P", 0)) or None,
            last_funding_rate=float(msg["r"]),
            next_funding_time_ms=msg["T"],
        )
    except (KeyError, ValueError) as e:
        logger.error("parse.mark_price_error", error=str(e), msg=msg)
        return None


async def parse_liquidation_message(msg: dict) -> Liquidation | None:
    """Parse liquidation stream message.

    Format: https://binance-docs.github.io/apidocs/futures/en/#liquidation-order-streams
    """
    try:
        order = msg["o"]
        return Liquidation(
            symbol=order["s"],
            time_ms=msg["E"],
            side=order["S"],
            order_type=order["o"],
            price=float(order["p"]),
            quantity=float(order["q"]),
            average_price=float(order["ap"]),
        )
    except (KeyError, ValueError) as e:
        logger.error("parse.liquidation_error", error=str(e), msg=msg)
        return None


async def flush_buffer_to_disk(
    buffer: StreamBuffer,
    symbol: str,
    venue: str,
    data_dir: Path,
    checkpoint: Checkpoint,
    checkpoint_path: Path,
) -> None:
    """Flush buffer to Parquet files partitioned by date."""
    if buffer.size() == 0:
        return

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    logger.info(
        "flush.started",
        buffer_size=buffer.size(),
        trades=len(buffer.trades),
        depths=len(buffer.depths),
        mark_prices=len(buffer.mark_prices),
        liquidations=len(buffer.liquidations),
    )

    # Import here to avoid circular dependency
    from intraday.data.download import save_to_parquet

    # Flush trades
    if buffer.trades:
        path = data_dir / "raw" / venue / "trades" / symbol / f"{date_str}.parquet"
        save_to_parquet(buffer.trades, path)
        checkpoint.update(
            symbol,
            venue,
            "trades",
            buffer.trades[0].time_ms,
            buffer.trades[-1].time_ms,
            len(buffer.trades),
            str(path),
        )

    # Flush depths
    if buffer.depths:
        path = data_dir / "raw" / venue / "depth" / symbol / f"{date_str}.parquet"
        save_to_parquet(buffer.depths, path)
        checkpoint.update(
            symbol,
            venue,
            "depth",
            buffer.depths[0].time_ms,
            buffer.depths[-1].time_ms,
            len(buffer.depths),
            str(path),
        )

    # Flush mark prices
    if buffer.mark_prices:
        path = data_dir / "raw" / venue / "mark_price" / symbol / f"{date_str}.parquet"
        save_to_parquet(buffer.mark_prices, path)
        checkpoint.update(
            symbol,
            venue,
            "mark_price",
            buffer.mark_prices[0].time_ms,
            buffer.mark_prices[-1].time_ms,
            len(buffer.mark_prices),
            str(path),
        )

    # Flush liquidations
    if buffer.liquidations:
        path = data_dir / "raw" / venue / "liquidations" / symbol / f"{date_str}.parquet"
        save_to_parquet(buffer.liquidations, path)
        checkpoint.update(
            symbol,
            venue,
            "liquidations",
            buffer.liquidations[0].time_ms,
            buffer.liquidations[-1].time_ms,
            len(buffer.liquidations),
            str(path),
        )

    # Save checkpoint
    checkpoint.save(checkpoint_path)

    logger.info("flush.complete", duration_s=(datetime.now(timezone.utc) - now).total_seconds())

    # Clear buffer
    buffer.clear()


async def subscribe_stream(
    stream_name: str,
    symbol: str,
    buffer: StreamBuffer,
    config: CaptureConfig,
) -> None:
    """Subscribe to a WebSocket stream with auto-reconnect."""
    symbol_lower = symbol.lower()
    reconnect_delay = config.reconnect_delay_s

    # Determine WebSocket URL based on stream type
    if stream_name in ["trade", "depth"]:
        base_url = BINANCE_WS_SPOT
    else:
        base_url = BINANCE_WS_FUTURES

    # Construct stream URL
    if stream_name == "trade":
        url = f"{base_url}/{symbol_lower}@trade"
    elif stream_name == "depth":
        url = f"{base_url}/{symbol_lower}@depth20@100ms"
    elif stream_name == "mark_price":
        url = f"{base_url}/{symbol_lower}@markPrice@1s"
    elif stream_name == "liquidations":
        url = f"{base_url}/{symbol_lower}@forceOrder"
    else:
        raise ValueError(f"Unknown stream: {stream_name}")

    while True:
        try:
            logger.info("stream.connecting", stream=stream_name, url=url)

            async with websockets.connect(url) as ws:
                logger.info("stream.connected", stream=stream_name)
                reconnect_delay = config.reconnect_delay_s  # Reset delay on successful connect

                async for message in ws:
                    data = json.loads(message)

                    # Parse based on stream type
                    if stream_name == "trade":
                        trade = await parse_trade_message(data, symbol)
                        if trade:
                            buffer.add_trade(trade)
                    elif stream_name == "depth":
                        depth = await parse_depth_message(data, symbol)
                        if depth:
                            buffer.add_depth(depth)
                    elif stream_name == "mark_price":
                        mark_price = await parse_mark_price_message(data, symbol)
                        if mark_price:
                            buffer.add_mark_price(mark_price)
                    elif stream_name == "liquidations":
                        if data.get("e") == "forceOrder":
                            liquidation = await parse_liquidation_message(data)
                            if liquidation:
                                buffer.add_liquidation(liquidation)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(
                "stream.disconnected",
                stream=stream_name,
                reason=str(e),
                reconnect_in_s=reconnect_delay,
            )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, config.max_reconnect_delay_s)

        except Exception as e:
            logger.error(
                "stream.error",
                stream=stream_name,
                error=str(e),
                reconnect_in_s=reconnect_delay,
            )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, config.max_reconnect_delay_s)


async def periodic_flush(
    buffer: StreamBuffer,
    config: CaptureConfig,
    checkpoint: Checkpoint,
    checkpoint_path: Path,
) -> None:
    """Periodically flush buffer to disk."""
    while True:
        await asyncio.sleep(config.flush_interval_s)

        # Check if buffer needs flushing
        elapsed = (datetime.now(timezone.utc) - buffer.last_flush).total_seconds()
        if buffer.size() > 0 and (elapsed >= config.flush_interval_s or buffer.size() >= config.max_buffer_size):
            await flush_buffer_to_disk(
                buffer,
                config.symbol,
                config.venue,
                config.data_dir,
                checkpoint,
                checkpoint_path,
            )


async def capture_live(config: CaptureConfig) -> None:
    """Start live data capture from WebSocket streams.

    Runs indefinitely until interrupted (Ctrl+C).
    """
    logger.info(
        "capture.started",
        symbol=config.symbol,
        venue=config.venue,
        streams=config.streams,
    )

    # Load checkpoint
    checkpoint_path = get_checkpoint_path(config.data_dir)
    checkpoint = Checkpoint.load(checkpoint_path)

    # Create buffer
    buffer = StreamBuffer()

    # Create tasks for each stream + periodic flush
    tasks = []

    for stream in config.streams:
        task = asyncio.create_task(subscribe_stream(stream, config.symbol, buffer, config))
        tasks.append(task)

    # Add periodic flush task
    flush_task = asyncio.create_task(periodic_flush(buffer, config, checkpoint, checkpoint_path))
    tasks.append(flush_task)

    # Run all tasks
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("capture.interrupted", message="Flushing buffer before exit...")
        await flush_buffer_to_disk(
            buffer,
            config.symbol,
            config.venue,
            config.data_dir,
            checkpoint,
            checkpoint_path,
        )
        logger.info("capture.stopped")
