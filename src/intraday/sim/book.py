"""Local order book maintained from depth events.

Tracks only the top-of-book data needed for matching and strategy signals.
Level qty=0 means the level is removed.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class LocalOrderBook:
    def __init__(self) -> None:
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

    def apply_snapshot(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        self._bids = {price: qty for price, qty in bids if qty > 0}
        self._asks = {price: qty for price, qty in asks if qty > 0}

    def apply_diff(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        for price, qty in bids:
            if qty == 0.0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = qty
        for price, qty in asks:
            if qty == 0.0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

    def mid_price(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        return (max(self._bids) + min(self._asks)) / 2.0

    def spread_bps(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        best_b = max(self._bids)
        best_a = min(self._asks)
        mid = (best_b + best_a) / 2.0
        if mid == 0.0:
            return 0.0
        return ((best_a - best_b) / mid) * 10_000.0

    def best_bid(self) -> tuple[float, float]:
        if not self._bids:
            return (0.0, 0.0)
        price = max(self._bids)
        return (price, self._bids[price])

    def best_ask(self) -> tuple[float, float]:
        if not self._asks:
            return (0.0, 0.0)
        price = min(self._asks)
        return (price, self._asks[price])

    def depth_within_pct(self, side: str, pct: float) -> float:
        """Total qty within pct% of mid price on the given side."""
        mid = self.mid_price()
        if mid == 0.0:
            return 0.0
        threshold = pct / 100.0
        if side == "bid":
            cutoff = mid * (1.0 - threshold)
            return sum(qty for price, qty in self._bids.items() if price >= cutoff)
        else:
            cutoff = mid * (1.0 + threshold)
            return sum(qty for price, qty in self._asks.items() if price <= cutoff)

    @property
    def bid_levels(self) -> dict[float, float]:
        return self._bids

    @property
    def ask_levels(self) -> dict[float, float]:
        return self._asks


__all__ = ["LocalOrderBook"]
