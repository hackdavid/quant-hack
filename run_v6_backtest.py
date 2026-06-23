#!/usr/bin/env python3
"""Full V6 pipeline backtest: load all models → run on historical data → backtest.

Usage:
    python run_v6_backtest.py \
        --transformer-run models/transformer/20260623T132957Z \
        --start 2026-01-01 --end 2026-05-31 \
        --output-dir data
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import typer
from rich import print as rprint
from rich.console import Console

from intraday.agents.forecast import ForecastAgent
from intraday.agents.orderflow import OrderflowAgent
from intraday.agents.regime import RegimeAgent
from intraday.agents.risk import RiskAgent
from intraday.agents.stay_out import StayOutDetector
from intraday.aggregator.decision import DecisionEngine
from intraday.aggregator.features import build_aggregator_row, AGGREGATOR_FEATURE_COLS
from intraday.aggregator.meta_learner import MetaLearner
from intraday.backtest.engine import BacktestEngine
from intraday.forecast.output import ForecastOutput
from intraday.risk.agent import RiskAgent as OldRiskAgent

log = structlog.get_logger(__name__)
console = Console()

app = typer.Typer()


def _load_rl_policy(model_path: Path):
    """Load CQL policy from d3rlpy checkpoint."""
    try:
        import d3rlpy
        import torch
    except ImportError:
        raise ImportError("d3rlpy not installed. Run: uv add d3rlpy")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Monkey-patch torch.load to work around PyTorch 2.6 weights_only default
    _orig_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load  # type: ignore
    try:
        cql = d3rlpy.load_learnable(str(model_path), device=device)
    finally:
        torch.load = _orig_torch_load  # type: ignore
    return cql


def _process_day(
    day_df: pl.DataFrame,
    forecast_agent: ForecastAgent,
    orderflow_agent: OrderflowAgent,
    regime_agent: RegimeAgent,
    risk_agent: RiskAgent,
    stay_out: StayOutDetector,
    decision_engine: DecisionEngine,
    rl_policy: Any | None,
) -> list[dict]:
    """Process one day and return list of decision records with backtest fields."""
    records: list[dict] = []
    feat_df = day_df.sort("bar_time_ms")
    n_bars = len(feat_df)

    for i in range(n_bars):
        row = feat_df.row(i, named=True)
        ts_ms = int(row["bar_time_ms"])

        start_idx = max(0, i - 127)
        feat_window = feat_df.slice(start_idx, i - start_idx + 1)
        if len(feat_window) < 128:
            continue

        try:
            forecast_opinion = forecast_agent.predict(feat_window)
        except Exception as exc:
            log.warning("forecast_error", ts_ms=ts_ms, error=str(exc))
            continue

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
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

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
        _fwd_dir = row.get("fwd_direction_5m")
        agg_row["fwd_direction_5m"] = float(_fwd_dir) if _fwd_dir is not None else 0.0
        _fwd_ret = row.get("fwd_ret_5m")
        agg_row["fwd_return_5m"] = float(_fwd_ret) if _fwd_ret is not None else 0.0

        # Decision engine
        try:
            decision = decision_engine.decide(agg_row, forecast)
        except Exception as exc:
            log.warning("decide_error", ts_ms=ts_ms, error=str(exc))
            continue

        # RL execution override
        if rl_policy is not None:
            # Build numeric state vector from aggregator row (skip string cols)
            string_cols = {"rg_regime", "rg_vol_regime", "so_mode", "reason", "side"}
            numeric_cols = [c for c in AGGREGATOR_FEATURE_COLS if c in agg_row and c not in string_cols]
            state = np.array([float(agg_row.get(c, 0.0)) for c in numeric_cols], dtype=np.float32)
            # Pad or truncate to RL state dim
            from intraday.rl.state import STATE_DIM
            if len(state) < STATE_DIM:
                state = np.concatenate([state, np.zeros(STATE_DIM - len(state), dtype=np.float32)])
            elif len(state) > STATE_DIM:
                state = state[:STATE_DIM]
            action = rl_policy.predict(state.reshape(1, -1))[0]
            # action is [size, aggressiveness, hold_time, stop_pct]
            rl_size = float(action[0])
            rl_aggressive = float(action[1])
            # Override side if RL strongly disagrees
            if rl_size < 0.1:
                decision.side = "flat"
                decision.confidence *= 0.5
            else:
                decision.confidence = min(1.0, decision.confidence * (1 + rl_aggressive))

        records.append({
            "ts_ms": ts_ms,
            "side": decision.side,
            "confidence": decision.confidence,
            "horizon_minutes": decision.horizon_minutes,
            "reason": decision.reason,
            "prob_up": p_up,
            "fwd_return_5m": agg_row["fwd_return_5m"],
            "close": float(row.get("close", 0.0)),
        })

    return records


@app.command()
def main(
    transformer_run: Path = typer.Option(..., help="Path to transformer run dir"),
    data_dir: Path = typer.Option(Path("data"), help="Data root directory"),
    symbol: str = typer.Option("BTCUSDT", help="Symbol"),
    start: str = typer.Option("2026-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2026-05-31", help="End date YYYY-MM-DD"),
    rl_policy_path: Path = typer.Option(Path("data/models/rl/cql_v1/cql_policy/cql.d3"), help="RL policy checkpoint"),
    use_rl: bool = typer.Option(True, help="Use RL execution policy"),
    backtest: bool = typer.Option(True, help="Run backtest on results"),
    save_results: bool = typer.Option(True, help="Save results.parquet"),
) -> None:
    from intraday.utils.logging import setup_logging
    setup_logging(log_level="info", console=True)

    features_dir = data_dir / "features" / symbol
    if not features_dir.exists():
        rprint(f"[red]Features not found: {features_dir}[/red]")
        raise typer.Exit(1)

    # ── Load all agents ──────────────────────────────────────────────────────
    rprint("[yellow]Loading V6 pipeline components...[/yellow]")
    forecast_agent = ForecastAgent(run_dir=transformer_run)
    orderflow_agent = OrderflowAgent()
    risk_agent = RiskAgent()
    stay_out = StayOutDetector()

    regime_agent = RegimeAgent.load(data_dir / "models" / "regime.pkl")
    rprint("[green]✓ Regime agent loaded[/green]")

    meta_learner = MetaLearner.load(data_dir / "models" / "aggregator" / "meta_learner.pkl")
    rprint(f"[green]✓ Meta-learner loaded (threshold={meta_learner._threshold:.3f})[/green]")

    decision_engine = DecisionEngine(meta_learner=meta_learner, threshold=meta_learner._threshold)
    rprint("[green]✓ Decision engine ready[/green]")

    # ── Load RL policy ───────────────────────────────────────────────────────
    rl_policy = None
    if use_rl and rl_policy_path.exists():
        rl_policy = _load_rl_policy(rl_policy_path)
        rprint(f"[green]✓ RL CQL policy loaded from {rl_policy_path}[/green]")
    elif use_rl:
        rprint(f"[red]✗ RL policy not found at {rl_policy_path}, skipping[/red]")

    # ── Run inference on each bar ──────────────────────────────────────────
    rprint(f"[yellow]Running V6 pipeline on {start} → {end}...[/yellow]")
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    day_files = [
        f for f in sorted(features_dir.glob("*.parquet"))
        if start_date <= date.fromisoformat(f.stem) <= end_date
    ]

    rprint(f"Processing {len(day_files)} day files...")
    all_records: list[dict] = []
    total_bars = 0

    for f in day_files:
        day_df = pl.read_parquet(f).sort("bar_time_ms")
        records = _process_day(
            day_df,
            forecast_agent,
            orderflow_agent,
            regime_agent,
            risk_agent,
            stay_out,
            decision_engine,
            rl_policy,
        )
        all_records.extend(records)
        total_bars += len(day_df)

    rprint(f"[green]Processed {total_bars:,} bars, generated {len(all_records):,} decisions[/green]")

    if not all_records:
        rprint("[red]No records generated.[/red]")
        raise typer.Exit(1)

    # ── Backtest ─────────────────────────────────────────────────────────────
    if backtest:
        rprint("[yellow]Running backtest...[/yellow]")
        probs = np.array([r["prob_up"] for r in all_records], dtype=np.float64)
        fwd_returns = np.array([r["fwd_return_5m"] for r in all_records], dtype=np.float64)
        timestamps = np.array([r["ts_ms"] for r in all_records], dtype=np.int64)

        engine = BacktestEngine(threshold=meta_learner._threshold)
        result = engine.run(probs, fwd_returns, timestamps)
        m = engine.metrics(result)

        rprint(f"\n[bold]V6 Backtest Results ({start} → {end})[/bold]")
        rprint(f"  Total return     : {m['total_return_pct']:>8.2f}%")
        rprint(f"  Ann. return      : {m['ann_return_pct']:>8.2f}%")
        rprint(f"  Sharpe (ann.)    : {m['sharpe']:>8.3f}")
        rprint(f"  Calmar           : {m['calmar']:>8.3f}")
        rprint(f"  Max drawdown     : {m['max_drawdown_pct']:>8.2f}%")
        rprint(f"  Trades           : {m['n_trades']:>8d}")
        rprint(f"  Win rate         : {m['win_rate_pct']:>8.1f}%")
        rprint(f"  Profit factor    : {m['profit_factor']:>8.3f}")
        rprint(f"  Avg win          : {m['avg_win_pct']:>8.3f}%")
        rprint(f"  Avg loss         : {m['avg_loss_pct']:>8.3f}%")
        rprint(f"  Time in market   : {m['pct_time_in_market']:>8.1f}%")
        rprint(f"  Bars processed   : {m['n_bars']:>8d}")

        # Side breakdown
        n_long = sum(1 for r in all_records if r["side"] == "long")
        n_short = sum(1 for r in all_records if r["side"] == "short")
        n_flat = len(all_records) - n_long - n_short
        rprint(f"\n  Direction breakdown:")
        rprint(f"    long={n_long:,}  short={n_short:,}  flat={n_flat:,}")

    # ── Save results ─────────────────────────────────────────────────────────
    if save_results:
        results_dir = data_dir / "v6_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(all_records).write_parquet(results_dir / "v6_decisions.parquet")
        rprint(f"[green]Saved results → {results_dir / 'v6_decisions.parquet'}[/green]")

    # ── Paper trading readiness check ───────────────────────────────────────
    rprint("\n[bold]Paper Trading Readiness Check:[/bold]")
    checks = {
        "Transformer model": transformer_run / "best.pt",
        "Meta-learner": data_dir / "models" / "aggregator" / "meta_learner.pkl",
        "Regime agent": data_dir / "models" / "regime.pkl",
        "RL policy": rl_policy_path,
    }
    all_ready = True
    for name, path in checks.items():
        ok = path.exists()
        emoji = "[green]✓[/green]" if ok else "[red]✗[/red]"
        rprint(f"  {emoji} {name}: {path}")
        if not ok:
            all_ready = False

    if all_ready:
        rprint("\n[green]All systems ready for paper trading![/green]")
        rprint("  Run: python scripts/run_paper_trade.py")
    else:
        rprint("\n[red]Some components missing. Please re-run training.[/red]")


if __name__ == "__main__":
    app()
