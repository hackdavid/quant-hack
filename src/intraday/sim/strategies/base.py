"""Abstract base class for all backtest strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intraday.sim.events import Event
    from intraday.sim.loop import Fill, OrderRequest, StrategyContext


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def on_event(self, event: "Event", ctx: "StrategyContext") -> list["OrderRequest"]:
        """Called for every market event. Return list of orders to submit."""
        ...

    def on_fill(self, fill: "Fill", ctx: "StrategyContext") -> None:
        """Called when one of our orders is filled."""

    def on_cancel(self, order_id: str, ctx: "StrategyContext") -> None:
        """Called when one of our orders is cancelled."""


__all__ = ["Strategy"]
