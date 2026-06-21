"""Decision model and engine: translate aggregated features into a trade decision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
import structlog
from pydantic import BaseModel

from intraday.forecast.output import ForecastOutput

if TYPE_CHECKING:
    from intraday.aggregator.meta_learner import MetaLearner

log = structlog.get_logger(__name__)


class Decision(BaseModel):
    """A trade decision produced by the DecisionEngine."""

    ts_ms: int
    side: Literal["long", "short", "flat"]
    confidence: float     # meta-learner P(profitable)
    horizon_minutes: int = 15
    reason: str = ""      # for logging / diagnostics


class DecisionEngine:
    """Convert aggregator features + forecast → Decision.

    Logic (Phase 6 section 6):
    1. If rk_stop_trading or so_mode == "stay_out"  → flat
    2. If not fc_meta_act                            → flat
    3. p_correct = meta_learner.predict_proba(row)
    4. If p_correct < threshold                      → flat
    5. side = "long" if fc_p_up > fc_p_down else "short"
    """

    def __init__(
        self,
        meta_learner: "MetaLearner",
        *,
        threshold: float = 0.55,
    ) -> None:
        self._meta = meta_learner
        self._threshold = threshold
        log.debug("decision_engine_init", threshold=threshold)

    def decide(
        self,
        aggregator_row: dict,
        forecast: ForecastOutput,
    ) -> Decision:
        """Produce a Decision from the aggregator feature row and the forecast."""
        ts_ms = forecast.ts_ms

        # ── Gate 1: hard risk / stay-out blocks ──────────────────────────────
        if aggregator_row.get("rk_stop_trading", 0):
            log.debug("decision.flat", reason="rk_stop_trading", ts_ms=ts_ms)
            return Decision(
                ts_ms=ts_ms, side="flat", confidence=0.0,
                horizon_minutes=forecast.horizon_minutes,
                reason="rk_stop_trading",
            )

        if aggregator_row.get("so_mode", "normal") == "stay_out":
            log.debug("decision.flat", reason="so_mode=stay_out", ts_ms=ts_ms)
            return Decision(
                ts_ms=ts_ms, side="flat", confidence=0.0,
                horizon_minutes=forecast.horizon_minutes,
                reason="so_mode=stay_out",
            )

        # ── Gate 2: meta-label gate ───────────────────────────────────────────
        if not aggregator_row.get("fc_meta_act", 1):
            log.debug("decision.flat", reason="fc_meta_act=False", ts_ms=ts_ms)
            return Decision(
                ts_ms=ts_ms, side="flat", confidence=0.0,
                horizon_minutes=forecast.horizon_minutes,
                reason="fc_meta_act=False",
            )

        # ── Gate 3/4: meta-learner probability ───────────────────────────────
        row_df = pl.DataFrame([aggregator_row])
        p_arr = self._meta.predict_proba(row_df)
        p_correct = float(p_arr[0]) if len(p_arr) > 0 else 0.0

        if p_correct < self._threshold:
            log.debug(
                "decision.flat",
                reason="p_correct_below_threshold",
                p_correct=round(p_correct, 4),
                threshold=self._threshold,
                ts_ms=ts_ms,
            )
            return Decision(
                ts_ms=ts_ms, side="flat", confidence=p_correct,
                horizon_minutes=forecast.horizon_minutes,
                reason=f"p_correct={p_correct:.4f}<{self._threshold}",
            )

        # ── Gate 5: directional decision ─────────────────────────────────────
        side: Literal["long", "short"] = (
            "long" if forecast.p_up_05sigma > forecast.p_down_05sigma else "short"
        )
        log.debug(
            "decision.trade",
            side=side,
            p_correct=round(p_correct, 4),
            p_up=round(forecast.p_up_05sigma, 4),
            p_down=round(forecast.p_down_05sigma, 4),
            ts_ms=ts_ms,
        )
        return Decision(
            ts_ms=ts_ms,
            side=side,
            confidence=p_correct,
            horizon_minutes=forecast.horizon_minutes,
            reason=f"p_correct={p_correct:.4f}",
        )


__all__ = ["Decision", "DecisionEngine"]
