"""Almgren-Chriss optimal execution baseline."""

from __future__ import annotations

import math

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class AlmgrenChrissBaseline:
    """Splits target into N child orders along a cosine trajectory.

    Each child: post-only at microprice for first half of its time slot,
    then IOC for the remainder.  Used as the benchmark against which the
    RL policy is compared.
    """

    def __init__(self, n_slices: int = 10, tick_size: float = 0.1) -> None:
        self._n_slices = n_slices
        self._tick_size = tick_size
        self._schedule: list[dict] = []
        self._target_qty_base: float = 0.0
        self._side: str = "buy"
        self._window_seconds: float = 300.0
        log.debug(
            "almgren_chriss_baseline.init",
            n_slices=n_slices,
            tick_size=tick_size,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        *,
        target_qty_base: float,
        side: str,
        window_seconds: float,
    ) -> list[dict]:
        """Return a list of scheduled child orders: [{t_seconds, qty, order_type}].

        Cosine schedule:
          qty_i = target * (cos(pi*i/N) - cos(pi*(i+1)/N)) / 2
        which integrates to target over the whole window and front-loads
        execution (higher quantities earlier in the window).
        """
        n = self._n_slices
        slices: list[dict] = []
        total = 0.0

        for i in range(n):
            weight = (math.cos(math.pi * i / n) - math.cos(math.pi * (i + 1) / n)) / 2.0
            qty = abs(target_qty_base) * weight
            t_seconds = (i + 0.5) * window_seconds / n  # midpoint of slice
            half_slot = window_seconds / n / 2.0

            slices.append(
                {
                    "t_seconds": t_seconds,
                    "qty": qty,
                    "order_type": "post_only",
                    "ioc_fallback_at": t_seconds + half_slot,
                }
            )
            total += qty

        # Normalise to ensure we hit the exact target
        if total > 0.0 and abs(total - abs(target_qty_base)) > 1e-8:
            scale = abs(target_qty_base) / total
            for s in slices:
                s["qty"] *= scale

        self._schedule = slices
        self._target_qty_base = target_qty_base
        self._side = side
        self._window_seconds = window_seconds

        log.debug(
            "almgren_chriss_baseline.plan",
            side=side,
            target_qty=target_qty_base,
            window_seconds=window_seconds,
            n_slices=n,
        )
        return slices

    def step(self, state_vec: np.ndarray, elapsed_s: float) -> dict:
        """Return action dict for the current elapsed time.

        Finds the next un-executed slice whose scheduled time has been reached.
        Falls back to IOC if we are past the ioc_fallback_at time.

        Returns a dict compatible with the ExecutionAction fields:
          {order_type, tick_offset, child_size_pct, urgency}
        """
        if not self._schedule:
            return {
                "order_type": "market",
                "tick_offset": 0.0,
                "child_size_pct": 1.0,
                "urgency": 1.0,
            }

        remaining_qty = float(self._target_qty_base)
        for s in self._schedule:
            if elapsed_s >= s["t_seconds"]:
                remaining_qty -= s["qty"]

        if abs(remaining_qty) < 1e-10:
            return {
                "order_type": "post_only",
                "tick_offset": -1.0,
                "child_size_pct": 0.0,
                "urgency": 0.0,
            }

        # Determine next due slice
        next_slice: dict | None = None
        for s in self._schedule:
            if elapsed_s < s["t_seconds"]:
                next_slice = s
                break

        if next_slice is None:
            # All slices past due — submit market order for remainder
            return {
                "order_type": "market",
                "tick_offset": 0.0,
                "child_size_pct": 1.0,
                "urgency": 1.0,
            }

        if elapsed_s >= next_slice.get("ioc_fallback_at", next_slice["t_seconds"] + 15.0):
            order_type = "limit_ioc"
            tick_offset = 0.0
            urgency = 0.5
        else:
            order_type = "post_only"
            tick_offset = -2.0  # 2 ticks passive for buys
            urgency = 0.1

        total_remaining_abs = abs(remaining_qty)
        target_abs = max(abs(self._target_qty_base), 1e-10)
        child_size_pct = min(next_slice["qty"] / target_abs, 1.0)

        return {
            "order_type": order_type,
            "tick_offset": tick_offset,
            "child_size_pct": child_size_pct,
            "urgency": urgency,
        }


__all__ = ["AlmgrenChrissBaseline"]
