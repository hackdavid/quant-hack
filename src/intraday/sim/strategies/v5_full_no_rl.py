"""Full strategy: ForecastAgent + 4 specialist agents + Aggregator + Kelly sizing.

No RL yet.

Execution: simple post-only at microprice, 3-tick offset, cancel-and-replace
after 30 s, fall back to IOC.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from intraday.aggregator.features import build_aggregator_row
from intraday.sim.events import BarEvent
from intraday.sim.loop import OrderRequest
from intraday.sim.strategies.base import Strategy
from intraday.sim.strategies.registry import register

if TYPE_CHECKING:
    from intraday.aggregator.decision import Decision, DecisionEngine
    from intraday.aggregator.sizing import SizingEngine
    from intraday.agents.base import AgentOpinion
    from intraday.sim.events import Event
    from intraday.sim.loop import Fill, StrategyContext

log = structlog.get_logger(__name__)

# Tick size for BTC/USDT perpetual futures (0.1 USD)
_TICK_SIZE: float = 0.1
# Number of ticks away from microprice for post-only orders
_POST_ONLY_TICKS: int = 3
# Seconds after which an unfilled post-only order is cancelled and replaced with IOC
_CANCEL_REPLACE_SECS: int = 30
# Minimum notional for submitting an order (avoid dust)
_MIN_NOTIONAL_USD: float = 1.0


@register("v5_full_no_rl")
class V5FullNoRL(Strategy):
    """Full pipeline: forecast → agents → aggregator → Kelly sizing → post-only."""

    name = "v5_full_no_rl"

    def __init__(
        self,
        *,
        forecast_model: Any,          # ForecastModel instance (or None)
        orderflow_agent: Any,         # OrderflowAgent (or None)
        regime_agent: Any,            # RegimeAgent (or None)
        risk_agent: Any,              # RiskAgent (or None)
        stay_out: Any,                # StayOutDetector (or None)
        decision_engine: "DecisionEngine",
        sizing_engine: "SizingEngine",
        horizon_minutes: int = 15,
    ) -> None:
        self._forecast_model = forecast_model
        self._orderflow_agent = orderflow_agent
        self._regime_agent = regime_agent
        self._risk_agent = risk_agent
        self._stay_out = stay_out
        self._decision_engine = decision_engine
        self._sizing_engine = sizing_engine
        self._horizon_minutes = horizon_minutes

        # Order tracking: client_order_id → (submit_ts_ms, side, qty_base)
        self._pending: dict[str, tuple[int, str, float]] = {}
        # Signed position tracked locally (mirrors account)
        self._position_base: float = 0.0

        log.info(
            "v5_full_no_rl.init",
            has_forecast=forecast_model is not None,
            has_orderflow=orderflow_agent is not None,
            has_regime=regime_agent is not None,
            has_risk=risk_agent is not None,
            has_stay_out=stay_out is not None,
            horizon_minutes=horizon_minutes,
        )

    # ── Main event handler ────────────────────────────────────────────────────

    def on_event(self, event: "Event", ctx: "StrategyContext") -> list[OrderRequest]:
        """Only acts on BarEvent (bar_5m)."""
        if not isinstance(event, BarEvent):
            return []

        ts_ms = event.ts_ms

        # ── Cancel-and-replace: check for stale post-only orders ─────────────
        stale_ioc = self._cancel_stale_orders(ts_ms)

        # ── Build feature dict from ctx ───────────────────────────────────────
        features = self._build_features(event, ctx)

        # ── Forecast ──────────────────────────────────────────────────────────
        forecast = self._run_forecast(features, ts_ms)
        if forecast is None:
            return stale_ioc

        # ── Agent opinions ────────────────────────────────────────────────────
        opinions = self._run_agents(features)

        # ── Log agent opinions ────────────────────────────────────────────────
        for agent_name, opinion in opinions.items():
            log.debug(
                "agent.opinion",
                agent=agent_name,
                confidence=round(opinion.confidence, 4),
                payload=opinion.payload,
                ts_ms=ts_ms,
            )

        # ── Aggregator feature row ────────────────────────────────────────────
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        spread_bps = ctx.book.spread_bps() if ctx.book.mid_price() > 0 else 0.0
        realized_vol = float(features.get("realized_vol_30m", 0.0))
        funding_rate = float(features.get("funding_rate", 0.0))

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

        # ── Decision ──────────────────────────────────────────────────────────
        decision = self._decision_engine.decide(agg_row, forecast)

        log.debug(
            "aggregator.decision",
            side=decision.side,
            confidence=round(decision.confidence, 4),
            reason=decision.reason,
            ts_ms=ts_ms,
        )

        # ── Position management ───────────────────────────────────────────────
        new_orders: list[OrderRequest] = []

        if decision.side == "flat" and self._position_base != 0.0:
            # Exit existing position
            new_orders.extend(self._exit_orders(ctx))
        elif decision.side != "flat":
            # Compute size
            rk_opinion = opinions.get("risk")
            risk_multiplier = 1.0
            if rk_opinion is not None:
                risk_multiplier = float(rk_opinion.payload.get("risk_multiplier", 1.0))

            # Convert expected move in sigma to bps using vol
            vol_bps = realized_vol * 100.0  # rough conversion: 1% vol = 100 bps
            edge_bps = abs(forecast.expected_move_sigma) * vol_bps

            account_equity = ctx.account.equity(ctx.mark_price if ctx.mark_price > 0 else event.close)

            size_usd = self._sizing_engine.size_usd(
                decision,
                expected_edge_bps=edge_bps,
                vol_30m_bps=vol_bps,
                risk_multiplier=risk_multiplier,
                account_equity_usd=account_equity,
            )

            log.debug(
                "sizing.computed",
                size_usd=round(size_usd, 2),
                edge_bps=round(edge_bps, 4),
                vol_bps=round(vol_bps, 4),
                risk_multiplier=risk_multiplier,
                ts_ms=ts_ms,
            )

            if abs(size_usd) >= _MIN_NOTIONAL_USD and ctx.mark_price > 0:
                post_orders = self._post_only_orders(decision, size_usd, ctx)
                new_orders.extend(post_orders)

        return stale_ioc + new_orders

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_features(self, event: BarEvent, ctx: "StrategyContext") -> dict[str, Any]:
        """Combine bar event fields + feature snapshot into a feature dict."""
        features: dict[str, Any] = {
            "bar_time_ms": event.ts_ms,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
            "log_ret_5m": 0.0,  # will be overridden by feature_snapshot if available
        }
        if ctx.feature_snapshot is not None:
            features.update(ctx.feature_snapshot)
        return features

    def _run_forecast(self, features: dict[str, Any], ts_ms: int) -> Any | None:
        """Run forecast model. Returns None (with warning) if model unavailable."""
        if self._forecast_model is None:
            log.warning("v5.forecast_model_missing", ts_ms=ts_ms)
            return None
        try:
            return self._forecast_model.predict(features)
        except Exception as exc:
            log.warning("v5.forecast_error", error=str(exc), ts_ms=ts_ms)
            return None

    def _run_agents(self, features: dict[str, Any]) -> dict[str, "AgentOpinion"]:
        """Run all specialist agents. Missing agents are silently skipped."""
        opinions: dict[str, "AgentOpinion"] = {}
        agent_map = {
            "orderflow": self._orderflow_agent,
            "regime": self._regime_agent,
            "risk": self._risk_agent,
            "stay_out": self._stay_out,
        }
        for name, agent in agent_map.items():
            if agent is None:
                log.debug("v5.agent_missing", agent=name)
                continue
            try:
                opinions[name] = agent.predict(features)
            except Exception as exc:
                log.warning("v5.agent_error", agent=name, error=str(exc))
        return opinions

    def _cancel_stale_orders(self, ts_ms: int) -> list[OrderRequest]:
        """Return IOC fallback orders for any post-only orders older than 30 s."""
        ioc_orders: list[OrderRequest] = []
        stale_ids: list[str] = []
        stale_cutoff_ms = ts_ms - _CANCEL_REPLACE_SECS * 1000

        for cid, (submit_ts, side, qty) in self._pending.items():
            if submit_ts <= stale_cutoff_ms:
                stale_ids.append(cid)
                ioc_orders.append(
                    OrderRequest(
                        side=side,
                        qty_base=qty,
                        type="ioc",
                        time_in_force="IOC",
                        client_order_id=f"ioc_fallback_{uuid.uuid4().hex[:8]}",
                    )
                )
                log.debug(
                    "v5.cancel_replace",
                    original_cid=cid,
                    age_ms=ts_ms - submit_ts,
                    side=side,
                    qty=qty,
                )

        for cid in stale_ids:
            del self._pending[cid]

        return ioc_orders

    def _post_only_orders(
        self,
        decision: "Decision",
        size_usd: float,
        ctx: "StrategyContext",
    ) -> list[OrderRequest]:
        """Build post-only limit orders at microprice ± 3 ticks."""
        mid = ctx.book.mid_price()
        if mid <= 0:
            return []

        is_long = size_usd > 0
        side: str = "buy" if is_long else "sell"
        qty_base = abs(size_usd) / mid

        if is_long:
            # Buy slightly below mid to post as maker
            limit_price = round(mid - _POST_ONLY_TICKS * _TICK_SIZE, 1)
        else:
            # Sell slightly above mid to post as maker
            limit_price = round(mid + _POST_ONLY_TICKS * _TICK_SIZE, 1)

        cid = f"v5_po_{side}_{uuid.uuid4().hex[:8]}"
        self._pending[cid] = (ctx.ts_ms, side, qty_base)

        log.debug(
            "v5.post_only_submitted",
            side=side,
            qty=round(qty_base, 8),
            limit_price=limit_price,
            mid=mid,
            ts_ms=ctx.ts_ms,
        )

        return [
            OrderRequest(
                side=side,
                qty_base=qty_base,
                type="post_only",
                limit_price=limit_price,
                time_in_force="GTC",
                client_order_id=cid,
            )
        ]

    def _exit_orders(self, ctx: "StrategyContext") -> list[OrderRequest]:
        """Submit IOC market order to flatten position."""
        if self._position_base == 0.0:
            return []

        side = "sell" if self._position_base > 0 else "buy"
        qty = abs(self._position_base)
        cid = f"v5_exit_{uuid.uuid4().hex[:8]}"

        log.debug(
            "v5.exit_submitted",
            side=side,
            qty=round(qty, 8),
            ts_ms=ctx.ts_ms,
        )

        return [
            OrderRequest(
                side=side,
                qty_base=qty,
                type="ioc",
                time_in_force="IOC",
                reduce_only=True,
                client_order_id=cid,
            )
        ]

    # ── Fill tracking ─────────────────────────────────────────────────────────

    def on_fill(self, fill: "Fill", ctx: "StrategyContext") -> None:
        """Update local position tracker and remove pending order."""
        sign = 1.0 if fill.side == "buy" else -1.0
        self._position_base += sign * fill.qty_base

        # Remove from pending if it was a tracked post-only order
        self._pending.pop(fill.order_id, None)

        log.debug(
            "v5.fill_received",
            side=fill.side,
            qty=round(fill.qty_base, 8),
            price=fill.price,
            position_base=round(self._position_base, 8),
            ts_ms=fill.ts_ms,
        )


__all__ = ["V5FullNoRL"]
