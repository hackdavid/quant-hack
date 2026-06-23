#!/usr/bin/env python3
"""Run the full V6 pipeline on raw bar data and output JSON with English explanations.

Usage:
    # Single day file
    uv run python run_pipeline_json.py \
        --transformer-run models/transformer/20260623T132957Z \
        --features-file data/features/BTCUSDT/2026-01-01.parquet \
        --output output.json

    # Live stream (last N bars from stdin)
    cat bars.jsonl | uv run python run_pipeline_json.py \
        --transformer-run models/transformer/20260623T132957Z \
        --stdin \
        --output -

Output JSON schema:
    {
      "ts_ms": 1704067200000,
      "timestamp_utc": "2024-01-01 00:00:00",
      "agents": {
        "forecast": { ... "explanation": "..." },
        "orderflow": { ... "explanation": "..." },
        "regime": { ... "explanation": "..." },
        "risk": { ... "explanation": "..." },
        "stay_out": { ... "explanation": "..." }
      },
      "aggregator": { ... "explanation": "..." },
      "decision": { ... "explanation": "..." },
      "rl_execution": { ... "explanation": "..." },
      "feature_summary": { ... }
    }
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import typer
from rich import print as rprint

from intraday.agents.forecast import ForecastAgent
from intraday.agents.orderflow import OrderflowAgent
from intraday.agents.regime import RegimeAgent
from intraday.agents.risk import RiskAgent
from intraday.agents.stay_out import StayOutDetector
from intraday.aggregator.decision import DecisionEngine
from intraday.aggregator.features import build_aggregator_row, AGGREGATOR_FEATURE_COLS
from intraday.aggregator.meta_learner import MetaLearner
from intraday.forecast.output import ForecastOutput

log = structlog.get_logger(__name__)

app = typer.Typer()


def _load_rl_policy(model_path: Path):
    """Load CQL policy from d3rlpy checkpoint."""
    try:
        import d3rlpy
        import torch
    except ImportError:
        return None
    if not model_path.exists():
        return None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _orig_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load  # type: ignore
    try:
        cql = d3rlpy.load_learnable(str(model_path), device=device)
    except Exception:
        return None
    finally:
        torch.load = _orig_torch_load  # type: ignore
    return cql


def _explain_forecast(fc: ForecastOutput) -> dict:
    """Return forecast dict with human-readable explanation."""
    bias = "up" if fc.p_up_05sigma > 0.5 else "down"
    strength = "strong" if fc.confidence > 0.3 else "moderate" if fc.confidence > 0.15 else "weak"
    return {
        "p_up": round(fc.p_up_05sigma, 4),
        "p_down": round(fc.p_down_05sigma, 4),
        "expected_move_sigma": round(fc.expected_move_sigma, 4),
        "confidence": round(fc.confidence, 4),
        "meta_act": fc.meta_act,
        "meta_p_correct": round(fc.meta_p_correct, 4),
        "explanation": (
            f"Transformer predicts {bias} bias ({round(fc.p_up_05sigma*100,1)}% up). "
            f"Signal strength is {strength} (confidence={round(fc.confidence,2)}). "
            f"Expected move = {round(fc.expected_move_sigma,2)} sigma over 15min."
        ),
    }


def _explain_orderflow(op: Any) -> dict:
    """Return orderflow dict with human-readable explanation."""
    p = op.payload
    bias = p.get("flow_bias", 0.0)
    strength = p.get("flow_strength", 0.0)
    away = p.get("step_away", False)
    vpin = p.get("vpin", 0.0)
    side = "buy" if bias > 0.05 else "sell" if bias < -0.05 else "neutral"
    return {
        "flow_bias": round(bias, 4),
        "flow_strength": round(strength, 4),
        "step_away": away,
        "vpin": round(vpin, 4),
        "explanation": (
            f"Orderflow shows {side} pressure (bias={round(bias,3)}). "
            f"VPIN={round(vpin,3)} indicates {'high' if vpin > 0.6 else 'low'} toxicity. "
            f"{'Step away — avoid tight spreads' if away else 'Normal execution acceptable'}."
        ),
    }


def _explain_regime(op: Any) -> dict:
    """Return regime dict with human-readable explanation."""
    p = op.payload
    regime = p.get("regime", "unknown")
    max_prob = max(p.get("regime_probs", {}).values()) if p.get("regime_probs") else op.confidence
    is_trans = p.get("is_transition", False)
    vol = p.get("vol_regime", "normal")
    return {
        "regime": regime,
        "max_prob": round(max_prob, 4),
        "is_transition": is_trans,
        "vol_regime": vol,
        "explanation": (
            f"Market regime: {regime} (confidence={round(max_prob,2)}). "
            f"Volatility regime is {vol}. "
            f"{'Transition state — avoid new positions' if is_trans else 'Stable regime'}."
        ),
    }


def _explain_risk(op: Any) -> dict:
    """Return risk dict with human-readable explanation."""
    p = op.payload
    mult = p.get("risk_multiplier", 1.0)
    allow = p.get("allow_trade", True)
    stop = p.get("stop_trading", False)
    sizing = "normal" if mult == 1.0 else "reduced" if mult < 1.0 else "increased"
    return {
        "risk_multiplier": round(mult, 4),
        "allow_trade": allow,
        "stop_trading": stop,
        "explanation": (
            f"Risk: {sizing} sizing (mult={round(mult,2)}). "
            f"{'Trading HALTED' if stop else 'Trading allowed' if allow else 'Trading paused'}."
        ),
    }


def _explain_stay_out(op: Any) -> dict:
    """Return stay-out dict with human-readable explanation."""
    p = op.payload
    mode = p.get("mode", "normal")
    score = p.get("score", 0.0)
    return {
        "mode": mode,
        "score": round(score, 4),
        "explanation": (
            f"Stay-out detector: {mode} (score={round(score,2)}). "
            f"{'Do not trade — elevated risk conditions' if mode == 'stay_out' else 'Market conditions normal'}."
        ),
    }


def _explain_aggregator(row: dict) -> dict:
    """Return aggregator dict with human-readable explanation."""
    return {
        "fc_p_up": round(row.get("fc_p_up", 0.0), 4),
        "fc_p_down": round(row.get("fc_p_down", 0.0), 4),
        "fc_confidence": round(row.get("fc_confidence", 0.0), 4),
        "of_flow_bias": round(row.get("of_flow_bias", 0.0), 4),
        "rg_regime": row.get("rg_regime", "unknown"),
        "rg_vol_regime": row.get("rg_vol_regime", "normal"),
        "rk_allow_trade": bool(row.get("rk_allow_trade", 1)),
        "rk_stop_trading": bool(row.get("rk_stop_trading", 0)),
        "so_mode": row.get("so_mode", "normal"),
        "spread_bps": round(row.get("spread_bps", 0.0), 4),
        "realized_vol_30m": round(row.get("realized_vol_30m", 0.0), 4),
        "funding_rate": round(row.get("funding_rate", 0.0), 4),
        "hour_utc": row.get("hour_utc", 0),
        "day_of_week": row.get("day_of_week", 0),
        "explanation": (
            f"Aggregated view: forecast confidence={round(row.get('fc_confidence',0),2)}, "
            f"orderflow bias={round(row.get('of_flow_bias',0),2)}, "
            f"regime={row.get('rg_regime','unknown')}, "
            f"risk={'HALTED' if row.get('rk_stop_trading') else 'OK' if row.get('rk_allow_trade') else 'PAUSED'}, "
            f"stay_out={row.get('so_mode','normal')}."
        ),
    }


def _explain_decision(dec: Any) -> dict:
    """Return decision dict with human-readable explanation."""
    return {
        "side": dec.side,
        "confidence": round(dec.confidence, 4),
        "horizon_minutes": dec.horizon_minutes,
        "reason": dec.reason,
        "explanation": (
            f"Decision: {dec.side.upper()} with confidence={round(dec.confidence,2)}. "
            f"Horizon: {dec.horizon_minutes}min. Reason: {dec.reason}."
        ),
    }


def _explain_rl_execution(action: np.ndarray | None) -> dict:
    """Return RL execution dict with human-readable explanation."""
    if action is None:
        return {
            "size": None,
            "aggressiveness": None,
            "hold_time": None,
            "stop_pct": None,
            "explanation": "RL execution policy not loaded.",
        }
    size, agg, hold, stop = float(action[0]), float(action[1]), float(action[2]), float(action[3])
    size_desc = "full" if size > 0.8 else "half" if size > 0.4 else "small" if size > 0.1 else "flat"
    agg_desc = "aggressive" if agg > 0.5 else "passive"
    return {
        "size": round(size, 4),
        "aggressiveness": round(agg, 4),
        "hold_time": round(hold, 4),
        "stop_pct": round(stop, 4),
        "explanation": (
            f"RL execution: {size_desc} size ({round(size,2)}), {agg_desc} fill ({round(agg,2)}), "
            f"hold={round(hold,1)}min, stop={round(stop*100,1)}%"
        ),
    }


def _explain_features(row: dict) -> dict:
    """Return a human-readable summary of the raw features."""
    return {
        "close": round(row.get("close", 0.0), 2),
        "realized_vol_30m": round(row.get("realized_vol_30m", 0.0), 4),
        "rsi_14": round(row.get("rsi_14", 50.0), 2),
        "taker_buy_ratio_5m": round(row.get("taker_buy_ratio_5m", 0.5), 4),
        "spread_bps": round(row.get("spread_bps", 0.0), 4),
        "funding_rate": round(row.get("funding_rate", 0.0), 4),
        "explanation": (
            f"Bar: close={row.get('close',0)}, vol={round(row.get('realized_vol_30m',0)*100,2)}%, "
            f"RSI={round(row.get('rsi_14',50),1)}, taker_buy={round(row.get('taker_buy_ratio_5m',0.5)*100,1)}%, "
            f"spread={round(row.get('spread_bps',0),2)}bps, funding={round(row.get('funding_rate',0)*100,3)}%"
        ),
    }


def _process_bar(
    row: dict,
    feat_window: pl.DataFrame,
    forecast_agent: ForecastAgent,
    orderflow_agent: OrderflowAgent,
    regime_agent: RegimeAgent,
    risk_agent: RiskAgent,
    stay_out: StayOutDetector,
    decision_engine: DecisionEngine,
    rl_policy: Any | None,
) -> dict:
    """Process a single bar and return the full JSON pipeline output."""
    ts_ms = int(row["bar_time_ms"])
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

    try:
        forecast_opinion = forecast_agent.predict(feat_window)
    except Exception as exc:
        log.warning("forecast_error", ts_ms=ts_ms, error=str(exc))
        return {"error": f"forecast_error: {exc}"}

    opinions = {
        "orderflow": orderflow_agent.predict(feat_window),
        "regime": regime_agent.predict(feat_window) if regime_agent else None,
        "risk": risk_agent.predict(feat_window),
        "stay_out": stay_out.predict(feat_window),
    }
    opinions = {k: v for k, v in opinions.items() if v is not None}

    fc_payload = forecast_opinion.payload
    prob_up = fc_payload.get("forecast_prob_up", 0.5)
    p_up = max(0.0, min(1.0, prob_up))
    p_down = 1.0 - p_up
    forecast = ForecastOutput(
        ts_ms=ts_ms,
        horizon_minutes=15,
        p_bins=[p_down, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_up],
        p_up_05sigma=p_up,
        p_down_05sigma=p_down,
        expected_move_sigma=(p_up - 0.5) * 2.0,
        confidence=abs(p_up - 0.5) * 2.0,
        meta_act=p_up > 0.52 or p_up < 0.48,
        meta_p_correct=abs(p_up - 0.5) * 2.0,
        model_version="forecast_agent",
        inference_ms=forecast_opinion.inference_ms,
    )

    spread_bps = float(row.get("spread_bps", 0.0))
    realized_vol = float(row.get("realized_vol_30m", 0.0))
    funding_rate = float(row.get("funding_rate", 0.0))

    agg_row = build_aggregator_row(
        forecast=forecast,
        opinions=opinions,
        spread_bps=spread_bps,
        realized_vol_30m=realized_vol,
        funding_rate=funding_rate,
        hour_utc=dt.hour,
        minute_of_hour=dt.minute,
        day_of_week=dt.weekday(),
    )

    try:
        decision = decision_engine.decide(agg_row, forecast)
    except Exception as exc:
        log.warning("decide_error", ts_ms=ts_ms, error=str(exc))
        decision = None

    # RL execution
    rl_action = None
    if rl_policy is not None:
        string_cols = {"rg_regime", "rg_vol_regime", "so_mode", "reason", "side"}
        numeric_cols = [c for c in AGGREGATOR_FEATURE_COLS if c in agg_row and c not in string_cols]
        state = np.array([float(agg_row.get(c, 0.0)) for c in numeric_cols], dtype=np.float32)
        from intraday.rl.state import STATE_DIM
        if len(state) < STATE_DIM:
            state = np.concatenate([state, np.zeros(STATE_DIM - len(state), dtype=np.float32)])
        elif len(state) > STATE_DIM:
            state = state[:STATE_DIM]
        rl_action = rl_policy.predict(state.reshape(1, -1))[0]

    return {
        "ts_ms": ts_ms,
        "timestamp_utc": dt.isoformat(),
        "agents": {
            "forecast": _explain_forecast(forecast),
            "orderflow": _explain_orderflow(opinions["orderflow"]),
            "regime": _explain_regime(opinions["regime"]) if "regime" in opinions else None,
            "risk": _explain_risk(opinions["risk"]),
            "stay_out": _explain_stay_out(opinions["stay_out"]),
        },
        "aggregator": _explain_aggregator(agg_row),
        "decision": _explain_decision(decision) if decision else None,
        "rl_execution": _explain_rl_execution(rl_action),
        "feature_summary": _explain_features(row),
    }


@app.command()
def main(
    transformer_run: Path = typer.Option(..., help="Path to transformer run dir (contains best.pt)"),
    features_file: Path = typer.Option(None, help="Path to a single .parquet features file"),
    data_dir: Path = typer.Option(Path("data"), help="Data root directory"),
    symbol: str = typer.Option("BTCUSDT", help="Symbol"),
    date: str = typer.Option(None, help="Date to process (YYYY-MM-DD), overrides features_file"),
    bar_index: int = typer.Option(-1, help="Which bar to process (0=first, -1=last). If -2, process all bars."),
    rl_policy_path: Path = typer.Option(Path("data/models/rl/cql_v1/cql_policy/cql.d3"), help="RL policy checkpoint"),
    use_rl: bool = typer.Option(True, help="Load RL execution policy"),
    output: Path = typer.Option(Path("pipeline_output.json"), help="Output JSON file (- for stdout)"),
) -> None:
    from intraday.utils.logging import setup_logging
    setup_logging(log_level="info", console=True)

    # Determine features file
    if features_file is None:
        if date is None:
            rprint("[red]Error: either --features-file or --date must be specified[/red]")
            raise typer.Exit(1)
        features_file = data_dir / "features" / symbol / f"{date}.parquet"
    if not features_file.exists():
        rprint(f"[red]Features file not found: {features_file}[/red]")
        raise typer.Exit(1)

    # Load agents
    rprint("[yellow]Loading pipeline components...[/yellow]")
    forecast_agent = ForecastAgent(run_dir=transformer_run)
    orderflow_agent = OrderflowAgent()
    risk_agent = RiskAgent()
    stay_out = StayOutDetector()

    regime_agent = RegimeAgent.load(data_dir / "models" / "regime.pkl")
    meta_learner = MetaLearner.load(data_dir / "models" / "aggregator" / "meta_learner.pkl")
    decision_engine = DecisionEngine(meta_learner=meta_learner, threshold=meta_learner._threshold)
    rprint("[green]✓ All agents loaded[/green]")

    rl_policy = None
    if use_rl and rl_policy_path.exists():
        rl_policy = _load_rl_policy(rl_policy_path)
        rprint("[green]✓ RL policy loaded[/green]")

    # Load features
    df = pl.read_parquet(features_file).sort("bar_time_ms")
    n = len(df)
    rprint(f"[yellow]Loaded {n} bars from {features_file}[/yellow]")

    if bar_index == -2:
        indices = list(range(n))
    elif bar_index == -1:
        indices = [n - 1]
    elif 0 <= bar_index < n:
        indices = [bar_index]
    else:
        rprint(f"[red]Invalid bar_index {bar_index} (file has {n} bars)[/red]")
        raise typer.Exit(1)

    results: list[dict] = []
    for i in indices:
        row = df.row(i, named=True)
        start_idx = max(0, i - 127)
        feat_window = df.slice(start_idx, i - start_idx + 1)
        if len(feat_window) < 128:
            continue
        result = _process_bar(
            row, feat_window,
            forecast_agent, orderflow_agent, regime_agent,
            risk_agent, stay_out, decision_engine, rl_policy,
        )
        results.append(result)
        rprint(f"[green]Processed bar {i}/{n} (ts={result['timestamp_utc']})[/green]")

    output_data = results[0] if len(results) == 1 else results
    json_str = json.dumps(output_data, indent=2, default=str)

    if str(output) == "-":
        print(json_str)
    else:
        output.write_text(json_str)
        rprint(f"[green]Saved output → {output}[/green]")


if __name__ == "__main__":
    app()
