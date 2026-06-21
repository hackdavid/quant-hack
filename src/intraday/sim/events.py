"""Event types for the backtest simulator.

All events are frozen dataclasses keyed by ts_ms (UTC epoch milliseconds).
The Union type alias Event is used throughout the simulator for type narrowing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


class IntradayError(Exception):
    """Base exception for all intraday errors."""


class SimulatorError(IntradayError):
    """Raised when the simulator encounters an unrecoverable state."""


@dataclass(frozen=True)
class BarEvent:
    ts_ms: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    kind: str = field(default="bar", init=False)


@dataclass(frozen=True)
class TradeEvent:
    ts_ms: int
    price: float
    qty_base: float
    is_buyer_maker: bool
    kind: str = field(default="trade", init=False)


@dataclass(frozen=True)
class DepthEvent:
    ts_ms: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    is_snapshot: bool
    kind: str = field(default="depth", init=False)


@dataclass(frozen=True)
class FundingEvent:
    ts_ms: int
    funding_rate: float
    mark_price: float
    kind: str = field(default="funding", init=False)


@dataclass(frozen=True)
class MarkEvent:
    ts_ms: int
    mark_price: float
    kind: str = field(default="mark", init=False)


Event = Union[BarEvent, TradeEvent, DepthEvent, FundingEvent, MarkEvent]

__all__ = [
    "IntradayError",
    "SimulatorError",
    "BarEvent",
    "TradeEvent",
    "DepthEvent",
    "FundingEvent",
    "MarkEvent",
    "Event",
]
