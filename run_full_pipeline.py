#!/usr/bin/env python3
"""End-to-end pipeline: load models → run backtest → save aggregator rows + decisions.

Usage:
    python run_full_pipeline.py \\
        --transformer-run models/transformer/20260623T132957Z \\
        --start 2026-01-01 --end 2026-05-31 \\
        --output-dir data

Outputs:
    data/aggregator_rows/BTCUSDT/YYYY-MM-DD.parquet
    data/decisions/decisions.parquet
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

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
from intraday.aggregator.features import build_aggregator_row
from intraday.aggregator.meta_learner import MetaLearner
from intraday.forecast.output import ForecastOutput

log = structlog.get_logger(__name__)
console = Console()

app = typer.Typer()


def _build_meta_labels(agg_rows: list[dict]) -> list[int]:
    """Build binary labels: 1 if the 5-min forward direction was up, else 0."""
    labels: list[int] = []
    for row in agg_rows:
        fwd_dir = row.get("fwd_direction_5m", 0)
        labels.append(1 if fwd_dir > 0 else 0)
    return labels


def _process_day(
    day_df: pl.DataFrame,
    forecast_agent: ForecastAgent,
    orderflow_agent: OrderflowAgent,
    regime_agent: RegimeAgent,
    risk_agent: RiskAgent,
    stay_out: StayOutDetector,
    decision_engine: DecisionEngine,
) -> tuple[list[dict], list[dict]]:
    """Process one day of bars and return (agg_rows, decisions)."""
    agg_rows: list[dict] = []
    decisions: list[dict] = []

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
            meta_act=p_up > 0.55 or p_up < 0.45,
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
        # Add forward direction for meta-labeling
        fwd_dir = row.get("fwd_direction_5m")
        agg_row["fwd_direction_5m"] = float(fwd_dir) if fwd_dir is not None else 0.0
        agg_rows.append(agg_row)

        try:
            decision = decision_engine.decide(agg_row, forecast)
        except Exception as exc:
            log.warning("decide_error", ts_ms=ts_ms, error=str(exc))
            continue

        decisions.append({
            "ts_ms": decision.ts_ms,
            "side": decision.side,
            "confidence": decision.confidence,
            "horizon_minutes": decision.horizon_minutes,
            "reason": decision.reason,
        })

    return agg_rows, decisions


@app.command()
def main(
    transformer_run: Path = typer.Option(..., help="Path to transformer run dir (contains best.pt)"),
    data_dir: Path = typer.Option(Path("data"), help="Data root directory"),
    symbol: str = typer.Option("BTCUSDT", help="Symbol"),
    start: str = typer.Option("2026-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2026-05-31", help="End date YYYY-MM-DD"),
    fit_regime: bool = typer.Option(True, help="Fit RegimeAgent on features"),
    train_meta: bool = typer.Option(True, help="Train meta-learner on saved rows"),
    save_decisions: bool = typer.Option(True, help="Save decisions.parquet for RL"),
    workers: int = typer.Option(4, help="Parallel workers for day processing"),
) -> None:
    from intraday.utils.logging import setup_logging
    setup_logging(log_level="info", console=True)

    features_dir = data_dir / "features" / symbol
    if not features_dir.exists():
        rprint(f"[red]Features not found: {features_dir}[/red]")
        raise typer.Exit(1)

    # ── Load / fit agents ───────────────────────────────────────────────────
    rprint("[yellow]Loading agents...[/yellow]")

    forecast_agent = ForecastAgent(run_dir=transformer_run)
    orderflow_agent = OrderflowAgent()
    risk_agent = RiskAgent()
    stay_out = StayOutDetector()

    regime_agent: RegimeAgent | None = None
    regime_path = data_dir / "models" / "regime.pkl"
    if regime_path.exists() and not fit_regime:
        regime_agent = RegimeAgent.load(regime_path)
        rprint(f"[green]Loaded regime agent from {regime_path}[/green]")
    else:
        rprint("[yellow]Fitting RegimeAgent...[/yellow]")
        all_files = sorted(features_dir.glob("*.parquet"))
        train_files = [f for f in all_files if start <= f.stem <= end]
        if not train_files:
            rprint("[red]No feature files in range.[/red]")
            raise typer.Exit(1)
        train_df = pl.concat([pl.read_parquet(f) for f in train_files]).sort("bar_time_ms")
        regime_agent = RegimeAgent().fit(train_df)
        regime_agent.save(regime_path)
        rprint(f"[green]Regime agent fitted and saved to {regime_path}[/green]")

    # ── Decision engine (meta-learner will be loaded after training) ────────
    meta_learner = MetaLearner()
    decision_engine = DecisionEngine(meta_learner=meta_learner, threshold=0.55)

    # ── Run inference on each bar ───────────────────────────────────────────
    rprint(f"[yellow]Running full pipeline on {start} → {end}...[/yellow]")

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    day_files = [
        f for f in sorted(features_dir.glob("*.parquet"))
        if start_date <= date.fromisoformat(f.stem) <= end_date
    ]

    rprint(f"Processing {len(day_files)} day files...")

    all_agg_rows: list[dict] = []
    all_decisions: list[dict] = []
    total_bars = 0

    for f in day_files:
        day_df = pl.read_parquet(f).sort("bar_time_ms")
        agg_rows, decisions = _process_day(
            day_df,
            forecast_agent,
            orderflow_agent,
            regime_agent,
            risk_agent,
            stay_out,
            decision_engine,
        )
        all_agg_rows.extend(agg_rows)
        all_decisions.extend(decisions)
        total_bars += len(day_df)

    rprint(f"[green]Processed {total_bars:,} bars, generated {len(all_agg_rows):,} aggregator rows[/green]")

    # ── Save aggregator rows ────────────────────────────────────────────────
    if all_agg_rows:
        agg_dir = data_dir / "aggregator_rows" / symbol
        agg_dir.mkdir(parents=True, exist_ok=True)
        by_date: dict[str, list[dict]] = {}
        for row in all_agg_rows:
            d = datetime.fromtimestamp(row["ts_ms"] / 1000, tz=timezone.utc).date().isoformat()
            by_date.setdefault(d, []).append(row)
        for d, rows in by_date.items():
            pl.DataFrame(rows).write_parquet(agg_dir / f"{d}.parquet")
        rprint(f"[green]Saved aggregator rows → {agg_dir}[/green]")

    # ── Train meta-learner ──────────────────────────────────────────────────
    if train_meta and all_agg_rows:
        rprint("[yellow]Training meta-learner...[/yellow]")
        labels = _build_meta_labels(all_agg_rows)
        df = pl.DataFrame(all_agg_rows)
        df = df.with_columns(pl.Series("label", labels))

        from intraday.aggregator.features import AGGREGATOR_FEATURE_COLS
        feature_cols = [c for c in AGGREGATOR_FEATURE_COLS if c in df.columns]
        X = df.select(feature_cols)
        y = df["label"]
        ts = df["ts_ms"]

        meta_learner = MetaLearner(model_dir=data_dir / "models" / "aggregator")
        metrics = meta_learner.fit(X, y, ts=ts, n_folds=5, embargo_pct=0.01)
        meta_learner.save(data_dir / "models" / "aggregator" / "meta_learner.pkl")
        rprint(f"[green]Meta-learner trained: AUC={metrics['auc']:.4f} Brier={metrics['brier']:.4f} ECE={metrics['ece']:.4f}[/green]")

        fi = meta_learner.feature_importance().head(10)
        rprint("[bold]Top-10 feature importances:[/bold]")
        for row in fi.iter_rows(named=True):
            rprint(f"  {row['feature']:<30} {row['importance']:.1f}")

    # ── Reload meta-learner and regenerate decisions ───────────────────────
    if train_meta and all_agg_rows:
        meta_learner = MetaLearner.load(data_dir / "models" / "aggregator" / "meta_learner.pkl")
        # Use the meta-learner's own calibrated threshold (usually ~0.3–0.5)
        decision_engine = DecisionEngine(meta_learner=meta_learner, threshold=meta_learner._threshold)

        rprint("[yellow]Regenerating decisions with trained meta-learner...[/yellow]")
        all_decisions = []
        for row in all_agg_rows:
            forecast = ForecastOutput(
                ts_ms=row["ts_ms"],
                horizon_minutes=15,
                p_bins=[row["fc_p_down"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, row["fc_p_up"]],
                p_up_05sigma=row["fc_p_up"],
                p_down_05sigma=row["fc_p_down"],
                expected_move_sigma=row["fc_expected_move_sigma"],
                confidence=row["fc_confidence"],
                meta_act=bool(row["fc_meta_act"]),
                meta_p_correct=row["fc_meta_p_correct"],
                model_version="forecast_agent",
                inference_ms=0.0,
            )
            try:
                decision = decision_engine.decide(row, forecast)
                all_decisions.append({
                    "ts_ms": decision.ts_ms,
                    "side": decision.side,
                    "confidence": decision.confidence,
                    "horizon_minutes": decision.horizon_minutes,
                    "reason": decision.reason,
                })
            except Exception as exc:
                log.warning("decide_error", ts_ms=row["ts_ms"], error=str(exc))
        rprint(f"[green]Regenerated {len(all_decisions)} decisions[/green]")
        n_long = sum(1 for d in all_decisions if d["side"] == "long")
        n_short = sum(1 for d in all_decisions if d["side"] == "short")
        rprint(f"[green]  long={n_long}, short={n_short}, flat={len(all_decisions)-n_long-n_short}[/green]")

    # ── Save decisions ──────────────────────────────────────────────────────
    if all_decisions and save_decisions:
        dec_dir = data_dir / "decisions"
        dec_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(all_decisions).write_parquet(dec_dir / "decisions.parquet")
        rprint(f"[green]Saved {len(all_decisions):,} decisions → {dec_dir / 'decisions.parquet'}[/green]")


if __name__ == "__main__":
    app()
