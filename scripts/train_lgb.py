#!/usr/bin/env python3
"""LightGBM baseline with lagged microstructure features.

Usage:
    python scripts/train_lgb.py
    python scripts/train_lgb.py --features-dir data/features/BTCUSDT --output-dir models/lgb
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ── Feature groups ────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_60m",
    "realized_vol_30m", "rsi_14",
    "vol_5m", "taker_buy_ratio_5m", "trade_count_5m", "avg_trade_size_5m",
    "depth_imbalance_1pct",
    "vpin_50", "vpin_bucket_imbalance",
    "hawkes_buy_intensity", "hawkes_sell_intensity", "hawkes_net",
    "oi_btc", "oi_change_1h", "ls_count_ratio", "taker_ls_vol_ratio",
]

# Features worth lagging — those with highest LightGBM gain in initial run
LAG_FEATURES = [
    "ls_count_ratio", "oi_change_1h", "depth_imbalance_1pct",
    "vpin_50", "taker_buy_ratio_5m", "realized_vol_30m",
    "hawkes_net", "log_ret_5m", "log_ret_15m",
]

LAG_PERIODS = [1, 3, 6, 12, 24]      # × 5 min = 5m, 15m, 30m, 1h, 2h ago

ROLL_FEATURES = [
    "taker_buy_ratio_5m", "vpin_50", "ls_count_ratio",
    "hawkes_net", "depth_imbalance_1pct", "oi_change_1h",
]
ROLL_WINDOWS = [6, 12, 24, 48]        # × 5 min = 30m, 1h, 2h, 4h


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []

    for feat in LAG_FEATURES:
        if feat not in df.columns:
            continue
        for lag in LAG_PERIODS:
            exprs.append(pl.col(feat).shift(lag).alias(f"{feat}_lag{lag}"))

    for feat in ROLL_FEATURES:
        if feat not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            exprs.append(pl.col(feat).rolling_mean(w).alias(f"{feat}_rmean{w}"))
            exprs.append(pl.col(feat).rolling_std(w).alias(f"{feat}_rstd{w}"))

    # Cyclical time features
    ms = pl.col("bar_time_ms")
    hour = (ms // 3_600_000) % 24
    dow  = (ms // 86_400_000) % 7
    exprs += [
        hour.cast(pl.Float32).alias("hour_utc"),
        (hour.cast(pl.Float64) * (2 * 3.14159265 / 24)).sin().cast(pl.Float32).alias("sin_hour"),
        (hour.cast(pl.Float64) * (2 * 3.14159265 / 24)).cos().cast(pl.Float32).alias("cos_hour"),
        dow.cast(pl.Float32).alias("day_of_week"),
        (dow.cast(pl.Float64) * (2 * 3.14159265 / 7)).sin().cast(pl.Float32).alias("sin_dow"),
        (dow.cast(pl.Float64) * (2 * 3.14159265 / 7)).cos().cast(pl.Float32).alias("cos_dow"),
    ]

    return df.with_columns(exprs)


def make_labels(df: pl.DataFrame, ret_col: str = "fwd_ret_15m",
                threshold: float = 0.0005) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col(ret_col) > threshold).then(1)
          .when(pl.col(ret_col) < -threshold).then(0)
          .otherwise(None)
          .cast(pl.Int8)
          .alias("label")
    )
    return df.drop_nulls(subset=["label", ret_col])


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    lag_cols  = [f"{f}_lag{l}"   for f in LAG_FEATURES  for l in LAG_PERIODS  if f"{f}_lag{l}" in df.columns]
    roll_cols = [f"{f}_rmean{w}" for f in ROLL_FEATURES  for w in ROLL_WINDOWS if f"{f}_rmean{w}" in df.columns]
    roll_cols += [f"{f}_rstd{w}" for f in ROLL_FEATURES  for w in ROLL_WINDOWS if f"{f}_rstd{w}" in df.columns]
    time_cols = ["hour_utc", "sin_hour", "cos_hour", "day_of_week", "sin_dow", "cos_dow"]
    base = [c for c in BASE_FEATURES if c in df.columns]
    return base + lag_cols + roll_cols + [c for c in time_cols if c in df.columns]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir",  default="data/features/BTCUSDT")
    parser.add_argument("--output-dir",    default="models/lgb")
    parser.add_argument("--ret-threshold", type=float, default=0.0005)
    parser.add_argument("--val-frac",      type=float, default=0.15)
    parser.add_argument("--n-estimators",  type=int,   default=2000)
    parser.add_argument("--early-stop",    type=int,   default=75)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading features...")
    files = sorted(glob.glob(f"{args.features_dir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {args.features_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("bar_time_ms")
    print(f"  {len(df):,} bars  ({len(files)} files)")

    # ── Feature engineering ───────────────────────────────────────────────
    print("Building lag / rolling features...")
    df = build_features(df)
    df = make_labels(df, threshold=args.ret_threshold)
    feat_cols = get_feature_cols(df)
    print(f"  {len(df):,} labeled rows  balance={df['label'].mean():.3f}")
    print(f"  {len(feat_cols)} total features")

    # ── Time split ────────────────────────────────────────────────────────
    n = len(df)
    split = int(n * (1 - args.val_frac))
    train_df, val_df = df[:split], df[split:]
    print(f"  train={len(train_df):,}  val={len(val_df):,}")

    def to_xy(frame: pl.DataFrame):
        X = frame.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)
        y = frame["label"].to_numpy().astype(int)
        return X, y

    X_tr, y_tr = to_xy(train_df)
    X_val, y_val = to_xy(val_df)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nTraining LightGBM (up to {args.n_estimators} rounds, early stop {args.early_stop})...")

    dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feat_cols)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain)

    params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "learning_rate":     0.03,
        "num_leaves":        127,
        "max_depth":         -1,
        "min_child_samples": 100,
        "feature_fraction":  0.7,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "lambda_l2":         1.0,
        "min_split_gain":    0.001,
        "verbosity":         -1,
        "n_jobs":            -1,
        "seed":              42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stop, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    val_prob = model.predict(X_val)
    val_auc  = roc_auc_score(y_val, val_prob)
    val_ll   = log_loss(y_val, val_prob)

    # Decile breakdown
    thresholds = np.percentile(val_prob, [10, 30, 50, 70, 90])
    print(f"\n── Results ───────────────────────────────")
    print(f"  Val AUC:      {val_auc:.4f}")
    print(f"  Val log-loss: {val_ll:.4f}")
    print(f"  Best iter:    {model.best_iteration}")

    # Feature importance (top 30)
    importance = sorted(
        zip(feat_cols, model.feature_importance("gain").tolist()),
        key=lambda x: -x[1],
    )
    print("\nTop 30 features by gain:")
    for name, gain in importance[:30]:
        print(f"  {name:<45s}  {gain:>12.1f}")

    # ── Save ──────────────────────────────────────────────────────────────
    model_path = out / "lgb_model.txt"
    model.save_model(str(model_path))

    meta = {
        "val_auc":        val_auc,
        "val_log_loss":   val_ll,
        "best_iteration": model.best_iteration,
        "n_features":     len(feat_cols),
        "feat_cols":      feat_cols,
        "ret_threshold":  args.ret_threshold,
        "val_frac":       args.val_frac,
        "top_features":   [{"name": n, "gain": g} for n, g in importance[:30]],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved → {model_path}")
    print(f"Saved → {out / 'meta.json'}")


if __name__ == "__main__":
    main()
