"""Queue-aware limit order book matching engine.

Queue position is tracked per-order. On trade events: orders at the traded
price lose queue ahead of them proportionally to trade size. On depth diffs
where a level shrinks without a trade: queue is aged proportionally to the
reduction (non-trade queue consumption models hidden activity).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from intraday.sim.events import DepthEvent, TradeEvent
from intraday.sim.market_impact import adjusted_fill_price

if TYPE_CHECKING:
    from intraday.sim.book import LocalOrderBook
    from intraday.sim.loop import Fill, OrderRequest

log = structlog.get_logger(__name__)


@dataclass
class OrderState:
    order_id: str
    side: str
    qty_base: float
    limit_price: float
    type: str
    queue_position: float
    filled_qty: float
    ts_submitted_ms: int
    client_order_id: str = ""


class MatchingEngine:
    def __init__(self) -> None:
        self._orders: dict[str, OrderState] = {}
        # Track previous depth levels to detect shrinkage without a trade
        self._prev_bid_levels: dict[float, float] = {}
        self._prev_ask_levels: dict[float, float] = {}
        # Track traded prices in current "tick" to distinguish trade-driven from depth-driven changes
        self._last_trade_price: float | None = None
        self._last_trade_qty: float = 0.0

    def place_order(self, req: "OrderRequest", book: "LocalOrderBook", ts_ms: int) -> str:
        from intraday.sim.loop import OrderRequest

        order_id = str(uuid.uuid4())
        if req.limit_price is None:
            raise ValueError("Limit order requires limit_price")

        # Queue position = total quantity ahead at this price level
        if req.side == "buy":
            queue_pos = book.bid_levels.get(req.limit_price, 0.0)
        else:
            queue_pos = book.ask_levels.get(req.limit_price, 0.0)

        state = OrderState(
            order_id=order_id,
            side=req.side,
            qty_base=req.qty_base,
            limit_price=req.limit_price,
            type=req.type,
            queue_position=queue_pos,
            filled_qty=0.0,
            ts_submitted_ms=ts_ms,
            client_order_id=req.client_order_id,
        )
        self._orders[order_id] = state
        log.debug("matching.order_placed", order_id=order_id, side=req.side, price=req.limit_price, qty=req.qty_base, queue=queue_pos)
        return order_id

    def on_trade_event(self, ev: TradeEvent, book: "LocalOrderBook") -> list["Fill"]:
        from intraday.sim.loop import Fill

        self._last_trade_price = ev.price
        self._last_trade_qty = ev.qty_base

        fills: list[Fill] = []
        for order_id, order in list(self._orders.items()):
            if order.filled_qty >= order.qty_base:
                continue
            # Only fill resting orders on the opposite side of the aggressor
            # is_buyer_maker=True means the buy was resting (maker), so sell was aggressor
            # We fill resting bids when a sell trade hits, resting asks when a buy trade hits
            if ev.is_buyer_maker and order.side == "buy" and order.limit_price >= ev.price:
                order.queue_position = max(0.0, order.queue_position - ev.qty_base)
                if order.queue_position <= 0.0:
                    fill_qty = min(order.qty_base - order.filled_qty, ev.qty_base)
                    if fill_qty > 0:
                        fill = Fill(
                            ts_ms=ev.ts_ms,
                            order_id=order_id,
                            side=order.side,
                            qty_base=fill_qty,
                            price=order.limit_price,
                            is_maker=True,
                            fee_quote=0.0,
                        )
                        order.filled_qty += fill_qty
                        fills.append(fill)
                        log.debug("matching.limit_filled", order_id=order_id, qty=fill_qty, price=order.limit_price)
                        if order.filled_qty >= order.qty_base:
                            del self._orders[order_id]

            elif not ev.is_buyer_maker and order.side == "sell" and order.limit_price <= ev.price:
                order.queue_position = max(0.0, order.queue_position - ev.qty_base)
                if order.queue_position <= 0.0:
                    fill_qty = min(order.qty_base - order.filled_qty, ev.qty_base)
                    if fill_qty > 0:
                        fill = Fill(
                            ts_ms=ev.ts_ms,
                            order_id=order_id,
                            side=order.side,
                            qty_base=fill_qty,
                            price=order.limit_price,
                            is_maker=True,
                            fee_quote=0.0,
                        )
                        order.filled_qty += fill_qty
                        fills.append(fill)
                        log.debug("matching.limit_filled", order_id=order_id, qty=fill_qty, price=order.limit_price)
                        if order.filled_qty >= order.qty_base:
                            del self._orders[order_id]

        return fills

    def on_depth_event(self, ev: DepthEvent, book: "LocalOrderBook") -> list["Fill"]:
        """Proportional queue ageing on non-trade level shrinkage."""
        fills: list[Fill] = []

        if ev.is_snapshot:
            self._prev_bid_levels = {p: q for p, q in ev.bids}
            self._prev_ask_levels = {p: q for p, q in ev.asks}
            return fills

        traded_price = self._last_trade_price

        # Check bid side shrinkage
        for price, new_qty in ev.bids:
            old_qty = self._prev_bid_levels.get(price, 0.0)
            if new_qty == 0.0:
                new_qty = 0.0
            if old_qty > 0 and new_qty < old_qty and price != traded_price:
                reduction = old_qty - new_qty
                frac = reduction / old_qty
                for order_id, order in list(self._orders.items()):
                    if order.side == "buy" and order.limit_price == price:
                        order.queue_position = max(0.0, order.queue_position * (1.0 - frac))

        # Check ask side shrinkage
        for price, new_qty in ev.asks:
            old_qty = self._prev_ask_levels.get(price, 0.0)
            if new_qty == 0.0:
                new_qty = 0.0
            if old_qty > 0 and new_qty < old_qty and price != traded_price:
                reduction = old_qty - new_qty
                frac = reduction / old_qty
                for order_id, order in list(self._orders.items()):
                    if order.side == "sell" and order.limit_price == price:
                        order.queue_position = max(0.0, order.queue_position * (1.0 - frac))

        # Update prev levels from the diff
        for price, qty in ev.bids:
            if qty == 0.0:
                self._prev_bid_levels.pop(price, None)
            else:
                self._prev_bid_levels[price] = qty
        for price, qty in ev.asks:
            if qty == 0.0:
                self._prev_ask_levels.pop(price, None)
            else:
                self._prev_ask_levels[price] = qty

        # Reset last trade tracker after processing
        self._last_trade_price = None
        self._last_trade_qty = 0.0

        return fills

    def on_market_order(self, req: "OrderRequest", book: "LocalOrderBook", ts_ms: int) -> list["Fill"]:
        """Walk the book and fill at impact-adjusted prices."""
        from intraday.sim.loop import Fill

        fills: list[Fill] = []
        remaining = req.qty_base

        if req.side == "buy":
            levels = sorted(book.ask_levels.items())
        else:
            levels = sorted(book.bid_levels.items(), reverse=True)

        if not levels:
            log.warning("matching.no_liquidity", side=req.side, qty=req.qty_base)
            return fills

        order_id = str(uuid.uuid4())

        for level_price, level_qty in levels:
            if remaining <= 0:
                break
            fill_qty = min(remaining, level_qty)
            impact_price = adjusted_fill_price(level_price, req.qty_base, req.side)
            fill = Fill(
                ts_ms=ts_ms,
                order_id=order_id,
                side=req.side,
                qty_base=fill_qty,
                price=impact_price,
                is_maker=False,
                fee_quote=0.0,
            )
            fills.append(fill)
            remaining -= fill_qty

        return fills

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            del self._orders[order_id]
            log.debug("matching.order_cancelled", order_id=order_id)
            return True
        return False

    def open_orders(self) -> list[OrderState]:
        return list(self._orders.values())


__all__ = ["OrderState", "MatchingEngine"]
