"""4-dim continuous action → 6 discrete execution intents."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from intraday.sim.book import LocalOrderBook
    from intraday.sim.loop import OrderRequest


@dataclass
class ExecutionAction:
    order_type: str       # "post_only" | "limit_ioc" | "market" | "cancel_all"
    tick_offset: float    # [-5..+5] ticks from microprice
    child_size_pct: float # fraction of remaining target (0..1)
    urgency: float        # 0..1 — if >0.7, cancel resting and re-send


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def decode_action(a: np.ndarray) -> ExecutionAction:
    """a shape (4,) in [-1, 1]:
    a[0]: order_type (sigmoid applied, then bucketed)
           <0.33 → post_only, 0.33..0.66 → limit_ioc, >0.66 → market
    a[1]: tick_offset = tanh(a[1]) * 5  (range -5..+5)
    a[2]: child_size_pct = sigmoid(a[2])
    a[3]: urgency = sigmoid(a[3])
    """
    a = np.asarray(a, dtype=np.float64)
    assert a.shape == (4,), f"Action must be shape (4,), got {a.shape}"

    order_type_val = _sigmoid(float(a[0]))
    if order_type_val < 0.33:
        order_type = "post_only"
    elif order_type_val < 0.66:
        order_type = "limit_ioc"
    else:
        order_type = "market"

    tick_offset = float(np.tanh(a[1])) * 5.0

    child_size_pct = _sigmoid(float(a[2]))
    child_size_pct = max(0.0, min(1.0, child_size_pct))

    urgency = _sigmoid(float(a[3]))
    urgency = max(0.0, min(1.0, urgency))

    return ExecutionAction(
        order_type=order_type,
        tick_offset=tick_offset,
        child_size_pct=child_size_pct,
        urgency=urgency,
    )


def action_to_order_requests(
    action: ExecutionAction,
    *,
    remaining_qty_base: float,
    side: str,
    book: "LocalOrderBook",
    tick_size: float = 0.1,
) -> list:  # list[OrderRequest]
    """Convert decoded ExecutionAction into a list of OrderRequests.

    - If remaining_qty_base <= 0, returns [].
    - cancel_all maps to empty list (caller should handle cancellation).
    - post_only: limit order placed away from mid to earn maker rebate.
    - limit_ioc: limit IOC order near mid.
    - market: market order for immediate fill.
    - urgency > 0.7 triggers market order regardless of order_type.
    """
    from intraday.sim.loop import OrderRequest

    if remaining_qty_base <= 0.0:
        return []

    child_qty = remaining_qty_base * max(action.child_size_pct, 0.01)
    if child_qty <= 0.0:
        return []

    mid = book.mid_price()
    if mid <= 0.0:
        return []

    cid = f"rl_{action.order_type}_{uuid.uuid4().hex[:8]}"

    effective_order_type = action.order_type
    if action.urgency > 0.7:
        effective_order_type = "market"

    if effective_order_type == "cancel_all":
        return []

    if effective_order_type == "market":
        return [
            OrderRequest(
                side=side,
                qty_base=child_qty,
                type="market",
                limit_price=None,
                time_in_force="GTC",
                client_order_id=cid,
            )
        ]

    tick_offset_signed = action.tick_offset
    if side == "buy":
        # Negative offset moves limit below mid (safer for maker)
        limit_price = mid - tick_offset_signed * tick_size
    else:
        # Positive offset moves limit above mid (safer for maker)
        limit_price = mid + tick_offset_signed * tick_size

    limit_price = round(limit_price / tick_size) * tick_size
    limit_price = round(limit_price, 1)
    limit_price = max(tick_size, limit_price)

    if effective_order_type == "post_only":
        return [
            OrderRequest(
                side=side,
                qty_base=child_qty,
                type="post_only",
                limit_price=limit_price,
                time_in_force="GTC",
                client_order_id=cid,
            )
        ]
    else:
        # limit_ioc
        return [
            OrderRequest(
                side=side,
                qty_base=child_qty,
                type="ioc",
                limit_price=limit_price,
                time_in_force="IOC",
                client_order_id=cid,
            )
        ]


__all__ = ["ExecutionAction", "decode_action", "action_to_order_requests"]
