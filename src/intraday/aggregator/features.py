"""Build the aggregator feature row from all agent opinions + forecast + raw context."""

from __future__ import annotations

import polars as pl

from intraday.agents.base import AgentOpinion
from intraday.forecast.output import ForecastOutput

# ---------------------------------------------------------------------------
# Column schema (exact order matters — matches meta-learner training input)
# ---------------------------------------------------------------------------

AGGREGATOR_FEATURE_COLS: list[str] = [
    # Forecast
    "fc_p_up",
    "fc_p_down",
    "fc_expected_move_sigma",
    "fc_confidence",
    "fc_meta_act",
    "fc_meta_p_correct",
    # Orderflow agent
    "of_flow_bias",
    "of_flow_strength",
    "of_step_away",
    "of_vpin",
    # Regime agent
    "rg_regime",
    "rg_max_prob",
    "rg_is_transition",
    "rg_vol_regime",
    # Risk agent
    "rk_risk_multiplier",
    "rk_allow_trade",
    "rk_stop_trading",
    # Stay-out detector
    "so_mode",
    "so_score",
    # Raw market context
    "spread_bps",
    "realized_vol_30m",
    "funding_rate",
    "hour_utc",
    "minute_of_hour",
    "day_of_week",
]


def build_aggregator_row(
    forecast: ForecastOutput,
    opinions: dict[str, AgentOpinion],
    *,
    spread_bps: float,
    realized_vol_30m: float,
    funding_rate: float,
    hour_utc: int,
    minute_of_hour: int,
    day_of_week: int,
) -> dict:
    """Return a flat dict matching the aggregator feature schema.

    ``opinions`` must contain at minimum the keys:
        ``"orderflow"``, ``"regime"``, ``"risk"``, ``"stay_out"``

    Missing opinion keys produce safe neutral values so callers can pass a
    partial dict without crashing.
    """
    # ── Forecast features ────────────────────────────────────────────────────
    fc_meta_act_int = 1 if forecast.meta_act else 0

    row: dict = {
        "fc_p_up": forecast.p_up_05sigma,
        "fc_p_down": forecast.p_down_05sigma,
        "fc_expected_move_sigma": forecast.expected_move_sigma,
        "fc_confidence": forecast.confidence,
        "fc_meta_act": fc_meta_act_int,
        "fc_meta_p_correct": forecast.meta_p_correct,
    }

    # ── Orderflow agent ──────────────────────────────────────────────────────
    of_opinion = opinions.get("orderflow")
    if of_opinion is not None:
        of_payload = of_opinion.payload
        row["of_flow_bias"] = float(of_payload.get("flow_bias", 0.0))
        row["of_flow_strength"] = float(of_payload.get("flow_strength", 0.0))
        step_away_raw = of_payload.get("step_away", False)
        row["of_step_away"] = 1 if step_away_raw else 0
        row["of_vpin"] = float(of_payload.get("vpin", 0.0))
    else:
        row["of_flow_bias"] = 0.0
        row["of_flow_strength"] = 0.0
        row["of_step_away"] = 0
        row["of_vpin"] = 0.0

    # ── Regime agent ─────────────────────────────────────────────────────────
    rg_opinion = opinions.get("regime")
    if rg_opinion is not None:
        rg_payload = rg_opinion.payload
        row["rg_regime"] = str(rg_payload.get("regime", "unknown"))
        # max probability across all regime classes
        regime_probs = rg_payload.get("regime_probs", {})
        row["rg_max_prob"] = float(max(regime_probs.values())) if regime_probs else float(rg_opinion.confidence)
        is_trans_raw = rg_payload.get("is_transition", False)
        row["rg_is_transition"] = 1 if is_trans_raw else 0
        row["rg_vol_regime"] = str(rg_payload.get("vol_regime", "normal"))
    else:
        row["rg_regime"] = "unknown"
        row["rg_max_prob"] = 0.0
        row["rg_is_transition"] = 0
        row["rg_vol_regime"] = "normal"

    # ── Risk agent ───────────────────────────────────────────────────────────
    rk_opinion = opinions.get("risk")
    if rk_opinion is not None:
        rk_payload = rk_opinion.payload
        row["rk_risk_multiplier"] = float(rk_payload.get("risk_multiplier", 1.0))
        allow_raw = rk_payload.get("allow_trade", True)
        row["rk_allow_trade"] = 1 if allow_raw else 0
        stop_raw = rk_payload.get("stop_trading", False)
        row["rk_stop_trading"] = 1 if stop_raw else 0
    else:
        row["rk_risk_multiplier"] = 1.0
        row["rk_allow_trade"] = 1
        row["rk_stop_trading"] = 0

    # ── Stay-out detector ────────────────────────────────────────────────────
    so_opinion = opinions.get("stay_out")
    if so_opinion is not None:
        so_payload = so_opinion.payload
        row["so_mode"] = str(so_payload.get("mode", "normal"))
        row["so_score"] = float(so_payload.get("score", 0.0))
    else:
        row["so_mode"] = "normal"
        row["so_score"] = 0.0

    # ── Raw market context ───────────────────────────────────────────────────
    row["spread_bps"] = float(spread_bps)
    row["realized_vol_30m"] = float(realized_vol_30m)
    row["funding_rate"] = float(funding_rate)
    row["hour_utc"] = int(hour_utc)
    row["minute_of_hour"] = int(minute_of_hour)
    row["day_of_week"] = int(day_of_week)

    return row


__all__ = ["AGGREGATOR_FEATURE_COLS", "build_aggregator_row"]
