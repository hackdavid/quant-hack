"""Rule-based risk agent.

Hard caps only — no learning. Can override every other agent's output.
All logic is deterministic and stateless between predict() calls
(except for daily reset state which is managed explicitly).
"""

import time
from typing import Any

import structlog
from pydantic import BaseModel

from intraday.agents.base import Agent, AgentOpinion

log = structlog.get_logger(__name__)


class RiskConfig(BaseModel):
    """Configuration knobs for the risk agent."""

    max_daily_drawdown_pct: float = 2.0
    max_position_usd: float = 200.0
    max_trades_per_day: int = 50
    high_vol_threshold_sigma: float = 2.0  # realized_vol z-score


class RiskAgent(Agent):
    """Rule-based risk agent that enforces hard trading constraints."""

    name = "risk"

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._trades_today: int = 0
        self._peak_equity_today: float = 0.0
        log.debug(
            "risk_agent_init",
            max_dd=self.config.max_daily_drawdown_pct,
            max_pos=self.config.max_position_usd,
            max_trades=self.config.max_trades_per_day,
        )

    def reset_daily(self, equity: float) -> None:
        """Call once at the start of each trading day."""
        self._trades_today = 0
        self._peak_equity_today = equity
        log.info("risk_daily_reset", equity=equity)

    def predict(self, features: dict[str, Any]) -> AgentOpinion:
        """Evaluate current risk state and return hard constraints.

        Expected feature keys:
            drawdown_today_pct    — current intraday drawdown as a positive pct
            vol_regime            — "low" / "normal" / "high"
            n_trades_today        — number of completed trades today
            position_exposure_usd — absolute USD value of open position
            equity_usd            — current account equity
        """
        t0 = time.perf_counter()
        ts_ms = int(features.get("bar_time_ms") or (time.time() * 1000))

        dd_pct = float(features.get("drawdown_today_pct") or 0.0)
        vol_regime: str = str(features.get("vol_regime") or "normal")
        n_trades: int = int(features.get("n_trades_today") or 0)
        pos_usd: float = float(features.get("position_exposure_usd") or 0.0)
        equity: float = float(features.get("equity_usd") or 0.0)

        # Derive max_position_size_btc from equity and max position USD cap
        btc_price = float(features.get("close") or 1.0)
        if btc_price < 1.0:
            btc_price = 1.0
        max_pos_btc = self.config.max_position_usd / btc_price

        stop_trading = False
        allow_trade = True
        risk_multiplier = 1.0
        reason = ""

        # ── Priority 1: daily drawdown hard stop ──────────────────────────
        if dd_pct > self.config.max_daily_drawdown_pct:
            stop_trading = True
            allow_trade = False
            risk_multiplier = 0.0
            reason = "daily DD limit"

        # ── Priority 2: high volatility → half size ───────────────────────
        elif vol_regime == "high":
            risk_multiplier = 0.5
            reason = "high vol — half size"

        # ── Priority 3: trade count limit ─────────────────────────────────
        elif n_trades > self.config.max_trades_per_day:
            allow_trade = False
            reason = "max trades reached"

        # ── Priority 4: position size cap ─────────────────────────────────
        elif pos_usd > self.config.max_position_usd:
            allow_trade = False
            reason = "position limit"

        # ── Normal ────────────────────────────────────────────────────────
        else:
            risk_multiplier = 1.0
            allow_trade = True

        inference_ms = (time.perf_counter() - t0) * 1000.0

        payload: dict[str, Any] = {
            "max_position_size_btc": round(max_pos_btc * risk_multiplier, 8),
            "risk_multiplier": risk_multiplier,
            "allow_trade": allow_trade,
            "stop_trading": stop_trading,
            "reason": reason,
        }

        log.debug(
            "risk_predict",
            stop_trading=stop_trading,
            allow_trade=allow_trade,
            risk_multiplier=risk_multiplier,
            reason=reason,
            dd_pct=dd_pct,
            vol_regime=vol_regime,
            inference_ms=round(inference_ms, 3),
        )

        return AgentOpinion(
            agent=self.name,
            ts_ms=ts_ms,
            payload=payload,
            confidence=1.0,
            inference_ms=inference_ms,
        )
