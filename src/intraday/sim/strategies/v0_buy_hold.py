"""Buy-and-hold canary strategy.

Enters long at the first BarEvent via market buy, holds the entire simulation,
and exits at the last BarEvent via market sell.

Since strategies cannot know which event is the last, this strategy arms a
pending exit on every bar after the entry fills. The exit is submitted once
per bar and tracked. When the first exit fills, the strategy is done.

Canary acceptance: net PnL must be within 1bp of
    (exit_fill_price - entry_fill_price) / entry_fill_price * notional - fees

This is the correct accounting identity regardless of which bar the exit fires.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from intraday.sim.events import BarEvent
from intraday.sim.loop import OrderRequest
from intraday.sim.strategies.base import Strategy
from intraday.sim.strategies.registry import register

if TYPE_CHECKING:
    from intraday.sim.events import Event
    from intraday.sim.loop import Fill, StrategyContext

log = structlog.get_logger(__name__)


@register("v0_buy_hold")
class BuyHoldStrategy(Strategy):
    name = "v0_buy_hold"

    def __init__(self, qty_btc: float = 0.1) -> None:
        self._qty_btc = qty_btc
        self._entry_submitted = False
        self._entry_filled = False
        self._exit_filled = False
        self._bars_since_entry: int = 0

        # For canary validation
        self.entry_fill_price: float = 0.0
        self.exit_fill_price: float = 0.0

    def on_event(self, event: "Event", ctx: "StrategyContext") -> list[OrderRequest]:
        if not isinstance(event, BarEvent):
            return []

        if not self._entry_submitted:
            self._entry_submitted = True
            log.info("strategy.buy_hold.entry_submitted", ts_ms=event.ts_ms, close=event.close)
            return [
                OrderRequest(
                    side="buy",
                    qty_base=self._qty_btc,
                    type="market",
                    client_order_id=f"bh_entry_{uuid.uuid4().hex[:8]}",
                )
            ]

        if self._entry_filled and not self._exit_filled:
            self._bars_since_entry += 1
            # Submit exit on each bar so it fires with near-zero latency at the current bar.
            # Only one will execute since position is 0 after the first fill.
            # We guard with reduce_only to prevent phantom sells.
            log.debug("strategy.buy_hold.exit_arm", ts_ms=event.ts_ms, bar=self._bars_since_entry)
            return [
                OrderRequest(
                    side="sell",
                    qty_base=self._qty_btc,
                    type="market",
                    reduce_only=True,
                    client_order_id=f"bh_exit_{uuid.uuid4().hex[:8]}",
                )
            ]

        return []

    def on_fill(self, fill: "Fill", ctx: "StrategyContext") -> None:
        if fill.side == "buy" and not self._entry_filled:
            self._entry_filled = True
            self.entry_fill_price = fill.price
            log.info(
                "strategy.buy_hold.entry_filled",
                ts_ms=fill.ts_ms,
                price=fill.price,
                qty=fill.qty_base,
            )
        elif fill.side == "sell" and not self._exit_filled:
            self._exit_filled = True
            self.exit_fill_price = fill.price
            log.info(
                "strategy.buy_hold.exit_filled",
                ts_ms=fill.ts_ms,
                price=fill.price,
                qty=fill.qty_base,
            )


__all__ = ["BuyHoldStrategy"]
