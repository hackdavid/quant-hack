"""Pydantic schemas for all market data types.

These schemas define the exact structure of data we can get from Binance
in paper trading mode (WebSocket streams + REST API).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Kline(BaseModel):
    """OHLCV kline/candlestick data.

    Available from:
    - WS: btcusdt@kline_1m, btcusdt@kline_5m
    - REST: /api/v3/klines
    """

    symbol: str
    interval: Literal["1m", "5m", "15m", "1h", "4h", "1d"]
    open_time_ms: int = Field(description="Kline open time in milliseconds")
    close_time_ms: int = Field(description="Kline close time in milliseconds")

    open: float
    high: float
    low: float
    close: float
    volume: float  # Base asset volume
    quote_volume: float  # Quote asset volume

    num_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    @field_validator("open_time_ms", "close_time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def open_time(self) -> datetime:
        return datetime.fromtimestamp(self.open_time_ms / 1000)

    @property
    def close_time(self) -> datetime:
        return datetime.fromtimestamp(self.close_time_ms / 1000)


class Trade(BaseModel):
    """Individual trade (aggTrade format).

    Available from:
    - WS: btcusdt@trade, btcusdt@aggTrade
    - REST: /api/v3/aggTrades
    """

    symbol: str
    trade_id: int
    price: float
    quantity: float
    time_ms: int = Field(description="Trade execution time in milliseconds")

    is_buyer_maker: bool = Field(
        description="True if buyer is maker (sell order filled), False if seller is maker (buy order filled)"
    )

    @field_validator("time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def side(self) -> Literal["buy", "sell"]:
        """Aggressive side (taker)."""
        return "sell" if self.is_buyer_maker else "buy"

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.time_ms / 1000)


class DepthLevel(BaseModel):
    """Single level in order book."""

    price: float
    quantity: float


class Depth(BaseModel):
    """Order book depth snapshot.

    Available from:
    - WS: btcusdt@depth@100ms (partial book depth, top 20 levels)
    - WS: btcusdt@depth (full book depth updates)
    - REST: /api/v3/depth
    """

    symbol: str
    time_ms: int = Field(description="Snapshot time in milliseconds")
    last_update_id: int

    bids: list[DepthLevel] = Field(description="Bid levels, sorted highest to lowest")
    asks: list[DepthLevel] = Field(description="Ask levels, sorted lowest to highest")

    @field_validator("time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.time_ms / 1000)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread_bps(self) -> float | None:
        """Spread in basis points."""
        if self.best_bid and self.best_ask and self.mid_price:
            return ((self.best_ask - self.best_bid) / self.mid_price) * 10000
        return None


class FundingRate(BaseModel):
    """Futures funding rate.

    Available from:
    - WS: btcusdt@markPrice (includes funding rate)
    - REST: /fapi/v1/fundingRate
    - REST: /fapi/v1/premiumIndex
    """

    symbol: str
    funding_time_ms: int = Field(description="Funding settlement time in milliseconds")
    funding_rate: float = Field(description="Funding rate (positive = longs pay shorts)")
    mark_price: float | None = Field(default=None, description="Mark price at funding time")

    @field_validator("funding_time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def funding_time(self) -> datetime:
        return datetime.fromtimestamp(self.funding_time_ms / 1000)

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (funding happens every 8h = 3x per day = 1095x per year)."""
        return self.funding_rate * 1095


class OpenInterest(BaseModel):
    """Futures open interest.

    Available from:
    - WS: btcusdt@openInterest (unreliable, use REST)
    - REST: /fapi/v1/openInterest
    """

    symbol: str
    time_ms: int = Field(description="Snapshot time in milliseconds")
    open_interest: float = Field(description="Total open interest in contracts (BTC)")
    open_interest_usd: float | None = Field(
        default=None, description="Total open interest in USD notional"
    )

    @field_validator("time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.time_ms / 1000)


class MarkPrice(BaseModel):
    """Mark price and funding info (real-time).

    Available from:
    - WS: btcusdt@markPrice (1s updates)
    - REST: /fapi/v1/premiumIndex
    """

    symbol: str
    time_ms: int
    mark_price: float
    index_price: float = Field(description="Index price (spot composite)")
    estimated_settle_price: float | None = None
    last_funding_rate: float
    next_funding_time_ms: int

    @field_validator("time_ms", "next_funding_time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.time_ms / 1000)

    @property
    def next_funding_time(self) -> datetime:
        return datetime.fromtimestamp(self.next_funding_time_ms / 1000)

    @property
    def premium_bps(self) -> float:
        """Premium of mark over index in basis points."""
        return ((self.mark_price - self.index_price) / self.index_price) * 10000


class Liquidation(BaseModel):
    """Liquidation event.

    Available from:
    - WS: btcusdt@forceOrder
    """

    symbol: str
    time_ms: int
    side: Literal["BUY", "SELL"] = Field(
        description="Order side (BUY = long liquidation, SELL = short liquidation)"
    )
    order_type: str
    price: float = Field(description="Liquidation price")
    quantity: float = Field(description="Liquidated quantity")
    average_price: float = Field(description="Average fill price")

    @field_validator("time_ms")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.time_ms / 1000)

    @property
    def notional_usd(self) -> float:
        return self.quantity * self.average_price


# Export all schemas
__all__ = [
    "Kline",
    "Trade",
    "DepthLevel",
    "Depth",
    "FundingRate",
    "OpenInterest",
    "MarkPrice",
    "Liquidation",
]
