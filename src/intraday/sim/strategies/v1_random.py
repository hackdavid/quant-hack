"""Random position-flipping strategy.

At every BarEvent, with probability 0.05, randomly flips the current position:
long → flat → short → flat → long, using IOC market orders.
Seeded for reproducibility via numpy.random.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
import structlog

from intraday.sim.events import BarEvent
from intraday.sim.loop import OrderRequest
from intraday.sim.strategies.base import Strategy
from intraday.sim.strategies.registry import register

if TYPE_CHECKING:
    from intraday.sim.events import Event
    from intraday.sim.loop import Fill, StrategyContext

log = structlog.get_logger(__name__)

FLIP_PROB: float = 0.05
QTY_BTC: float = 0.1


@register("v1_random")
class RandomStrategy(Strategy):
    name = "v1_random"

    def __init__(self, qty_btc: float = QTY_BTC, seed: int = 42) -> None:
        self._qty_btc = qty_btc
        self._rng = np.random.default_rng(seed)
        self._position: float = 0.0  # signed, tracked locally

    def on_event(self, event: "Event", ctx: "StrategyContext") -> list[OrderRequest]:
        if not isinstance(event, BarEvent):
            return []

        if self._rng.random() >= FLIP_PROB:
            return []

        orders: list[OrderRequest] = []

        if self._position > 0:
            # Close long
            orders.append(OrderRequest(
                side="sell",
                qty_base=self._qty_btc,
                type="ioc",
                reduce_only=True,
                time_in_force="IOC",
                client_order_id=f"rnd_close_{uuid.uuid4().hex[:8]}",
            ))
            self._position = 0.0
        elif self._position < 0:
            # Close short
            orders.append(OrderRequest(
                side="buy",
                qty_base=self._qty_btc,
                type="ioc",
                reduce_only=True,
                time_in_force="IOC",
                client_order_id=f"rnd_close_{uuid.uuid4().hex[:8]}",
            ))
            self._position = 0.0
        else:
            # Flat: go long or short with equal probability
            if self._rng.random() > 0.5:
                orders.append(OrderRequest(
                    side="buy",
                    qty_base=self._qty_btc,
                    type="ioc",
                    time_in_force="IOC",
                    client_order_id=f"rnd_long_{uuid.uuid4().hex[:8]}",
                ))
                self._position = self._qty_btc
            else:
                orders.append(OrderRequest(
                    side="sell",
                    qty_base=self._qty_btc,
                    type="ioc",
                    time_in_force="IOC",
                    client_order_id=f"rnd_short_{uuid.uuid4().hex[:8]}",
                ))
                self._position = -self._qty_btc

        log.debug(
            "strategy.random.flip",
            ts_ms=event.ts_ms,
            new_position=self._position,
            n_orders=len(orders),
        )
        return orders

    def on_fill(self, fill: "Fill", ctx: "StrategyContext") -> None:
        pass


__all__ = ["RandomStrategy"]
