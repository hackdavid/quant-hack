"""Full strategy V6: identical to V5 but execution replaced by RLExecutionPolicy.

Decision logic (forecast → agents → aggregator → Kelly sizing) is inherited
from V5FullNoRL.  Only the order-submission step differs: instead of naive
post-only at microprice ± 3 ticks, the RL policy selects order type, offset,
child size, and urgency from the current execution state vector.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import structlog

from intraday.rl.action import decode_action, action_to_order_requests
from intraday.rl.predict import RLExecutionPolicy
from intraday.rl.state import build_state_vector
from intraday.sim.loop import OrderRequest
from intraday.sim.strategies.base import Strategy
from intraday.sim.strategies.registry import register
from intraday.sim.strategies.v5_full_no_rl import V5FullNoRL

if TYPE_CHECKING:
    from intraday.aggregator.decision import Decision
    from intraday.sim.events import Event
    from intraday.sim.loop import Fill, StrategyContext

log = structlog.get_logger(__name__)

_TICK_SIZE: float = 0.1
_WINDOW_SECONDS: float = 300.0  # 5-minute execution window
_MIN_NOTIONAL_USD: float = 1.0


@register("v6_full_with_rl")
class V6FullWithRL(Strategy):
    """Full pipeline V6: same decision logic as V5, RL-based execution.

    The V5 decision machinery (forecast + agents + aggregator + Kelly sizing)
    is reused without modification.  The _post_only_orders method is overridden
    via composition: instead of calling V5's method we call the RL policy.
    """

    name = "v6_full_with_rl"

    def __init__(
        self,
        *,
        rl_policy: RLExecutionPolicy,
        # V5 kwargs forwarded to inner instance
        forecast_model: Any = None,
        orderflow_agent: Any = None,
        regime_agent: Any = None,
        risk_agent: Any = None,
        stay_out: Any = None,
        decision_engine: Any,
        sizing_engine: Any,
        horizon_minutes: int = 15,
    ) -> None:
        self._rl_policy = rl_policy

        # Inner V5 instance provides all decision / sizing logic
        self._v5 = V5FullNoRL(
            forecast_model=forecast_model,
            orderflow_agent=orderflow_agent,
            regime_agent=regime_agent,
            risk_agent=risk_agent,
            stay_out=stay_out,
            decision_engine=decision_engine,
            sizing_engine=sizing_engine,
            horizon_minutes=horizon_minutes,
        )

        # Execution tracking
        self._current_decision: Decision | None = None
        self._window_start_ms: int = 0
        self._window_end_ms: int = 0
        self._target_usd: float = 0.0
        self._filled_usd: float = 0.0
        self._filled_qty_base: float = 0.0
        self._target_qty_base: float = 0.0
        self._recent_fills: list[Fill] = []
        self._cancel_count: int = 0
        self._equity_usd: float = 100_000.0

        # Forward fill/cancel callbacks to inner V5 for position tracking
        log.info("v6_full_with_rl.init")

    # ── Main event handler ────────────────────────────────────────────────────

    def on_event(self, event: "Event", ctx: "StrategyContext") -> list[OrderRequest]:
        """Same decision logic as V5.  Only execution differs: RL policy."""
        from intraday.sim.events import BarEvent

        if not isinstance(event, BarEvent):
            return []

        ts_ms = event.ts_ms

        # ── Stale order handling (delegate to v5 internals) ───────────────────
        stale_ioc = self._v5._cancel_stale_orders(ts_ms)
        if stale_ioc:
            self._cancel_count += len(stale_ioc)

        # ── Feature / forecast / agent / decision (all from v5) ──────────────
        features = self._v5._build_features(event, ctx)
        forecast = self._v5._run_forecast(features, ts_ms)
        if forecast is None:
            return stale_ioc

        opinions = self._v5._run_agents(features)

        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        spread_bps = ctx.book.spread_bps() if ctx.book.mid_price() > 0 else 0.0
        realized_vol = float(features.get("realized_vol_30m", 0.0))
        funding_rate = float(features.get("funding_rate", 0.0))

        from intraday.aggregator.features import build_aggregator_row
        agg_row = build_aggregator_row(
            forecast,
            opinions,
            spread_bps=spread_bps,
            realized_vol_30m=realized_vol,
            funding_rate=funding_rate,
            hour_utc=dt.hour,
            minute_of_hour=dt.minute,
            day_of_week=dt.weekday(),
        )

        decision = self._v5._decision_engine.decide(agg_row, forecast)

        log.debug(
            "v6.decision",
            side=decision.side,
            confidence=round(decision.confidence, 4),
            ts_ms=ts_ms,
        )

        new_orders: list[OrderRequest] = []

        if decision.side == "flat" and self._v5._position_base != 0.0:
            new_orders.extend(self._v5._exit_orders(ctx))
            self._reset_execution_state()
            return stale_ioc + new_orders

        if decision.side == "flat":
            return stale_ioc

        # ── Sizing (same as v5) ───────────────────────────────────────────────
        rk_opinion = opinions.get("risk")
        risk_multiplier = 1.0
        if rk_opinion is not None:
            risk_multiplier = float(rk_opinion.payload.get("risk_multiplier", 1.0))

        vol_bps = realized_vol * 100.0
        edge_bps = abs(forecast.expected_move_sigma) * vol_bps
        self._equity_usd = ctx.account.equity(
            ctx.mark_price if ctx.mark_price > 0 else event.close
        )

        size_usd = self._v5._sizing_engine.size_usd(
            decision,
            expected_edge_bps=edge_bps,
            vol_30m_bps=vol_bps,
            risk_multiplier=risk_multiplier,
            account_equity_usd=self._equity_usd,
        )

        if abs(size_usd) < _MIN_NOTIONAL_USD or ctx.mark_price <= 0:
            return stale_ioc + new_orders

        # ── Execution via RL policy ───────────────────────────────────────────
        if self._current_decision is None or decision.side != (
            "long" if self._current_decision.side == "long" else "short"
        ):
            # New execution window
            self._current_decision = decision
            self._window_start_ms = ts_ms
            self._window_end_ms = ts_ms + int(_WINDOW_SECONDS * 1000)
            self._target_usd = size_usd
            self._target_qty_base = abs(size_usd) / ctx.mark_price
            self._filled_usd = 0.0
            self._filled_qty_base = 0.0
            self._cancel_count = 0
            log.debug(
                "v6.new_execution_window",
                side=decision.side,
                target_usd=round(size_usd, 2),
                target_qty=round(self._target_qty_base, 6),
                ts_ms=ts_ms,
            )

        remaining_qty = max(self._target_qty_base - self._filled_qty_base, 0.0)
        if remaining_qty <= 0.0:
            return stale_ioc

        # Determine vol regime from spread
        if spread_bps < 0.5:
            vol_regime_id = 0
        elif spread_bps > 2.0:
            vol_regime_id = 2
        else:
            vol_regime_id = 1

        book_features = {
            "spread_bps": spread_bps,
            "ofi": float(features.get("ofi", 0.0)),
            "queue_imbalance": float(features.get("queue_imbalance", 0.0)),
            "vpin": float(features.get("vpin", 0.5)),
            "microprice_drift_5m_z": float(features.get("microprice_drift_5m_z", 0.0)),
            "recent_cancel_rate": min(self._cancel_count / 10.0, 1.0),
        }

        state_vec = build_state_vector(
            ts_ms=ts_ms,
            window_start_ms=self._window_start_ms,
            window_end_ms=self._window_end_ms,
            target_usd=self._target_usd,
            filled_usd=self._filled_usd,
            book_features=book_features,
            vol_regime_id=vol_regime_id,
            forecast_confidence=decision.confidence,
            recent_fills=self._recent_fills[-5:],
            equity_usd=self._equity_usd,
        )

        action_vec = self._rl_policy.act(state_vec)
        exec_action = decode_action(action_vec)

        side_str = "buy" if decision.side == "long" else "sell"
        rl_orders = action_to_order_requests(
            exec_action,
            remaining_qty_base=remaining_qty,
            side=side_str,
            book=ctx.book,
            tick_size=_TICK_SIZE,
        )

        # Track submitted orders in v5's pending dict so cancel-and-replace works
        for req in rl_orders:
            self._v5._pending[req.client_order_id] = (ts_ms, req.side, req.qty_base)

        log.debug(
            "v6.rl_execution",
            order_type=exec_action.order_type,
            tick_offset=round(exec_action.tick_offset, 2),
            child_size_pct=round(exec_action.child_size_pct, 3),
            urgency=round(exec_action.urgency, 3),
            n_orders=len(rl_orders),
            ts_ms=ts_ms,
        )

        new_orders.extend(rl_orders)
        return stale_ioc + new_orders

    # ── Fill / cancel tracking ─────────────────────────────────────────────────

    def on_fill(self, fill: "Fill", ctx: "StrategyContext") -> None:
        """Track fill for RL state; forward to V5 for position tracking."""
        self._v5.on_fill(fill, ctx)
        self._recent_fills.append(fill)
        if len(self._recent_fills) > 20:
            self._recent_fills = self._recent_fills[-20:]

        self._filled_qty_base += fill.qty_base
        self._filled_usd += fill.qty_base * fill.price

        log.debug(
            "v6.fill",
            side=fill.side,
            qty=round(fill.qty_base, 8),
            price=fill.price,
            filled_pct=round(
                self._filled_qty_base / max(self._target_qty_base, 1e-10) * 100, 1
            ),
        )

    def on_cancel(self, order_id: str, ctx: "StrategyContext") -> None:
        """Forward cancel to V5 and increment cancel counter."""
        self._v5.on_cancel(order_id, ctx)
        self._cancel_count += 1

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_execution_state(self) -> None:
        """Reset per-window execution tracking."""
        self._current_decision = None
        self._target_usd = 0.0
        self._filled_usd = 0.0
        self._filled_qty_base = 0.0
        self._target_qty_base = 0.0
        self._cancel_count = 0
        self._recent_fills = []


__all__ = ["V6FullWithRL"]
