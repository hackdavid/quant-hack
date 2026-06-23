#!/usr/bin/env python3
"""Backtest a trained model on the validation split.

Examples:
    # Transformer only
    python scripts/run_backtest.py --transformer-dir models/transformer/20260623T132957Z

    # LightGBM only
    python scripts/run_backtest.py --lgb-dir models/lgb

    # Ensemble (both)
    python scripts/run_backtest.py \\
        --transformer-dir models/transformer/20260623T132957Z \\
        --lgb-dir models/lgb

    # GBM ensemble output
    python scripts/run_backtest.py \\
        --transformer-dir models/transformer/20260623T132957Z \\
        --lgb-ensemble-dir models/gbm_ensemble

    # Sweep thresholds to find optimal
    python scripts/run_backtest.py --transformer-dir models/transformer/20260623T132957Z --sweep
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from intraday.backtest.engine import BacktestEngine
from intraday.signal.combiner import SignalCombiner
from intraday.features.schema import ALL_FEATURES


def load_val_data(features_dir: str, val_frac: float, ret_threshold: float):
    """Load the same val split used during training."""
    files = sorted(glob.glob(f"{features_dir}/*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("bar_time_ms")
    n  = len(df)
    df = df.slice(int(n * (1 - val_frac)), n)
    df = df.filter(
        (pl.col("fwd_ret_15m") > ret_threshold) | (pl.col("fwd_ret_15m") < -ret_threshold)
    )
    return df


def generate_predictions(
    combiner: SignalCombiner,
    df: pl.DataFrame,
    seq_len: int,
    batch_every: int = 50,
) -> np.ndarray:
    """Run inference on every row of the val DataFrame."""
    n     = len(df)
    probs = np.full(n, 0.5)

    for i in range(seq_len, n):
        window = df.slice(i - seq_len, seq_len + 1)   # seq_len history + current
        try:
            prob, _ = combiner.predict_from_df(window)
            probs[i] = prob
        except Exception:
            probs[i] = 0.5
        if (i - seq_len) % batch_every == 0:
            pct = (i - seq_len) / max(n - seq_len, 1) * 100
            print(f"  inference {i-seq_len}/{n-seq_len}  ({pct:.0f}%)", end="\r", flush=True)

    print()
    return probs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transformer-dir",  default=None)
    p.add_argument("--lgb-dir",          default=None)
    p.add_argument("--lgb-ensemble-dir", default=None)
    p.add_argument("--features-dir",     default="data/features/BTCUSDT")
    p.add_argument("--val-frac",         type=float, default=0.15)
    p.add_argument("--ret-threshold",    type=float, default=0.0005)
    p.add_argument("--threshold",        type=float, default=0.55,
                   help="Min prob to enter a trade (1-t for shorts)")
    p.add_argument("--sweep",            action="store_true",
                   help="Sweep thresholds 0.50–0.65 to find best Sharpe")
    p.add_argument("--output-dir",       default="models/backtest_results")
    args = p.parse_args()

    if not args.transformer_dir and not args.lgb_dir and not args.lgb_ensemble_dir:
        p.error("Provide at least one of --transformer-dir / --lgb-dir / --lgb-ensemble-dir")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading models...")
    combiner = SignalCombiner(
        transformer_dir  = args.transformer_dir,
        lgb_dir          = args.lgb_dir,
        lgb_ensemble_dir = args.lgb_ensemble_dir,
    )

    # ── Load val data ──────────────────────────────────────────────────────────
    print(f"\nLoading val data ({args.val_frac*100:.0f}% holdout)...")
    df = load_val_data(args.features_dir, args.val_frac, args.ret_threshold)
    print(f"  {len(df):,} labeled bars")

    fwd_returns = df["fwd_ret_15m"].to_numpy().astype(np.float64)
    timestamps  = df["bar_time_ms"].to_numpy()

    # ── Determine seq_len (from transformer or default) ────────────────────────
    seq_len = 128
    if combiner._transformer:
        seq_len = combiner._transformer.seq_len

    # ── Generate predictions ───────────────────────────────────────────────────
    print(f"\nGenerating predictions (seq_len={seq_len})...")
    probs = generate_predictions(combiner, df, seq_len)

    # Trim to rows with valid predictions
    valid = probs != 0.5
    valid[:seq_len] = False
    probs_v   = probs[valid]
    returns_v = fwd_returns[valid]
    ts_v      = timestamps[valid]

    print(f"  Valid predictions: {valid.sum():,} / {len(probs):,}")
    print(f"  Prob range: [{probs_v.min():.3f}, {probs_v.max():.3f}]  mean={probs_v.mean():.3f}")

    # ── Backtest ───────────────────────────────────────────────────────────────
    engine = BacktestEngine()

    if args.sweep:
        print("\nThreshold sweep:")
        print(f"  {'Threshold':>10}  {'AUC≈':>6}  {'Sharpe':>7}  {'Return%':>8}  {'DD%':>7}  {'Trades':>7}  {'WinRate%':>9}")
        best_sharpe = -999.0
        best_t      = args.threshold

        for t in np.arange(0.50, 0.66, 0.01):
            result = engine.run(probs_v, returns_v, timestamps=ts_v)
            engine.threshold = t
            result = engine.run(probs_v, returns_v, timestamps=ts_v)
            m = engine.metrics(result)
            marker = " ←" if m["sharpe"] > best_sharpe else ""
            print(f"  {t:>10.2f}  {'':>6}  {m['sharpe']:>7.3f}  "
                  f"{m['total_return_pct']:>8.2f}  {m['max_drawdown_pct']:>7.2f}  "
                  f"{m['n_trades']:>7}  {m['win_rate_pct']:>9.1f}{marker}")
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_t = t

        engine.threshold = best_t
        print(f"\nUsing best threshold: {best_t:.2f}  (Sharpe {best_sharpe:.3f})")
    else:
        engine.threshold = args.threshold

    result = engine.run(probs_v, returns_v, timestamps=ts_v)
    m = BacktestEngine.print_report(result, title=f"Backtest  (threshold={engine.threshold:.2f})")

    # ── Save results ───────────────────────────────────────────────────────────
    tag = Path(args.transformer_dir).name if args.transformer_dir else "lgb"
    result_path = out / f"result_{tag}.json"
    result_path.write_text(json.dumps({**m, "threshold": engine.threshold,
                                        "n_valid_bars": int(valid.sum())}, indent=2))
    print(f"Results saved → {result_path}")

    # Equity curve as CSV
    eq_path = out / f"equity_{tag}.csv"
    import csv
    with open(eq_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar_time_ms", "equity", "signal", "prob"])
        for i, (ts, eq, sig, pr) in enumerate(zip(ts_v, result.equity, result.signals, result.probs)):
            w.writerow([int(ts), round(float(eq), 4), int(sig), round(float(pr), 4)])
    print(f"Equity curve → {eq_path}")


if __name__ == "__main__":
    main()
