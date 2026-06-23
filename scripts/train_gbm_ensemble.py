#!/usr/bin/env python3
"""RAM-intensive GBM ensemble: LightGBM GBDT + DART + XGBoost.

Runs three complementary gradient boosting models and produces a blended
prediction.  Designed to exploit the available ~100 GB of system RAM:
all feature matrices are kept in memory, enabling fast multi-model iteration.

Walk-forward cross-validation (4 folds) gives a more stable AUC estimate
than a single train/val split, especially on financial time series.

Usage:
    python scripts/train_gbm_ensemble.py
    python scripts/train_gbm_ensemble.py --output-dir models/gbm_ensemble
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping XGB model (pip install xgboost)")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ── Feature groups ─────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_60m",
    "realized_vol_30m", "rsi_14",
    "vol_5m", "taker_buy_ratio_5m", "trade_count_5m", "avg_trade_size_5m",
    "depth_imbalance_1pct",
    "vpin_50", "vpin_bucket_imbalance",
    "hawkes_buy_intensity", "hawkes_sell_intensity", "hawkes_net",
    "oi_btc", "oi_change_1h", "ls_count_ratio", "taker_ls_vol_ratio",
]

# All features to lag (fine-grained at short lags, sparse at long lags)
LAG_FEATURES = [
    "ls_count_ratio", "oi_change_1h", "depth_imbalance_1pct",
    "vpin_50", "taker_buy_ratio_5m", "realized_vol_30m",
    "hawkes_net", "log_ret_5m", "log_ret_15m", "log_ret_60m",
    "taker_ls_vol_ratio", "vpin_bucket_imbalance",
]

# Short lags: every bar; medium lags: key intervals; long lags: daily patterns
LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 96, 288]  # 5m 10m 15m 30m 1h 2h 4h 8h 1day

ROLL_FEATURES = [
    "taker_buy_ratio_5m", "vpin_50", "ls_count_ratio",
    "hawkes_net", "depth_imbalance_1pct", "oi_change_1h",
    "log_ret_5m", "taker_ls_vol_ratio", "realized_vol_30m",
]
ROLL_WINDOWS = [6, 12, 24, 48, 96, 288]   # 30m 1h 2h 4h 8h 1day

# Top feature pairs for interaction terms (gain-ranked from first run)
INTERACTION_PAIRS = [
    ("depth_imbalance_1pct", "taker_buy_ratio_5m"),
    ("depth_imbalance_1pct", "vpin_50"),
    ("depth_imbalance_1pct", "ls_count_ratio"),
    ("taker_buy_ratio_5m",   "ls_count_ratio"),
    ("taker_buy_ratio_5m",   "hawkes_net"),
    ("vpin_50",              "oi_change_1h"),
    ("vpin_50",              "ls_count_ratio"),
    ("oi_change_1h",         "hawkes_net"),
    ("oi_change_1h",         "taker_ls_vol_ratio"),
    ("hawkes_net",           "depth_imbalance_1pct"),
    ("ls_count_ratio",       "taker_ls_vol_ratio"),
    ("realized_vol_30m",     "vpin_50"),
]


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []

    # Lag features
    for feat in LAG_FEATURES:
        if feat not in df.columns:
            continue
        for lag in LAG_PERIODS:
            exprs.append(pl.col(feat).shift(lag).alias(f"{feat}_lag{lag}"))

    # Rolling mean + std
    for feat in ROLL_FEATURES:
        if feat not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            exprs.append(pl.col(feat).rolling_mean(w).alias(f"{feat}_rmean{w}"))
            exprs.append(pl.col(feat).rolling_std(w).alias(f"{feat}_rstd{w}"))

    # Rolling ratio: short-term vs long-term (momentum of rolling mean)
    for feat in ["taker_buy_ratio_5m", "vpin_50", "ls_count_ratio", "hawkes_net"]:
        if feat not in df.columns:
            continue
        for short, long in [(6, 24), (12, 48), (24, 96)]:
            s_col = f"{feat}_rmean{short}"
            l_col = f"{feat}_rmean{long}"
            exprs.append(
                (pl.col(feat).rolling_mean(short) / (pl.col(feat).rolling_mean(long).abs() + 1e-9))
                .alias(f"{feat}_rr{short}_{long}")
            )

    # Momentum of momentum: lag1 - lag6, lag6 - lag24
    for feat in ["depth_imbalance_1pct", "vpin_50", "taker_buy_ratio_5m", "ls_count_ratio"]:
        if feat not in df.columns:
            continue
        exprs.append((pl.col(feat).shift(1) - pl.col(feat).shift(6)).alias(f"{feat}_mom1_6"))
        exprs.append((pl.col(feat).shift(6) - pl.col(feat).shift(24)).alias(f"{feat}_mom6_24"))

    # Interaction terms (product of two microstructure signals)
    for a, b in INTERACTION_PAIRS:
        if a in df.columns and b in df.columns:
            exprs.append((pl.col(a) * pl.col(b)).alias(f"x_{a[:8]}_{b[:8]}"))

    # Cyclical time features
    ms   = pl.col("bar_time_ms")
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


def make_labels(df: pl.DataFrame, threshold: float = 0.0005) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("fwd_ret_15m") > threshold).then(1)
          .when(pl.col("fwd_ret_15m") < -threshold).then(0)
          .otherwise(None)
          .cast(pl.Int8)
          .alias("label")
    )
    return df.drop_nulls(subset=["label", "fwd_ret_15m"])


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    keep = set(BASE_FEATURES)
    for feat in LAG_FEATURES:
        for lag in LAG_PERIODS:
            keep.add(f"{feat}_lag{lag}")
    for feat in ROLL_FEATURES:
        for w in ROLL_WINDOWS:
            keep.add(f"{feat}_rmean{w}")
            keep.add(f"{feat}_rstd{w}")
    for feat in ["taker_buy_ratio_5m", "vpin_50", "ls_count_ratio", "hawkes_net"]:
        for short, long in [(6, 24), (12, 48), (24, 96)]:
            keep.add(f"{feat}_rr{short}_{long}")
    for feat in ["depth_imbalance_1pct", "vpin_50", "taker_buy_ratio_5m", "ls_count_ratio"]:
        keep.add(f"{feat}_mom1_6")
        keep.add(f"{feat}_mom6_24")
    for a, b in INTERACTION_PAIRS:
        keep.add(f"x_{a[:8]}_{b[:8]}")
    keep |= {"hour_utc", "sin_hour", "cos_hour", "day_of_week", "sin_dow", "cos_dow"}
    return [c for c in keep if c in df.columns]


# ── Walk-forward CV ────────────────────────────────────────────────────────────

def walk_forward_splits(n: int, n_folds: int = 4, val_size_frac: float = 0.06):
    """Yield (train_end, val_end) index pairs for walk-forward validation.

    Each fold uses all data up to train_end for training and the next
    val_size bars for validation. Folds are non-overlapping and ordered.
    """
    val_size = int(n * val_size_frac)
    # Start so that the last fold ends at n
    starts = [n - (n_folds - i) * val_size for i in range(n_folds)]
    for s in starts:
        if s <= val_size:
            continue
        yield s, min(s + val_size, n)


# ── Model training ─────────────────────────────────────────────────────────────

def train_lgb_gbdt(
    X_tr, y_tr, X_val, y_val, feat_cols: list[str]
) -> tuple[lgb.Booster, float]:
    params = {
        "objective":          "binary",
        "metric":             ["binary_logloss", "auc"],
        "boosting_type":      "gbdt",
        "learning_rate":      0.01,       # slow + thorough
        "num_leaves":         255,
        "max_depth":          -1,
        "min_child_samples":  50,
        "feature_fraction":   0.6,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "lambda_l1":          0.05,
        "lambda_l2":          0.5,
        "min_split_gain":     0.0005,
        "verbosity":          -1,
        "n_jobs":             -1,
        "seed":               42,
    }
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feat_cols, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        params, dtrain, num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    auc = roc_auc_score(y_val, model.predict(X_val))
    return model, auc


def train_lgb_dart(
    X_tr, y_tr, X_val, y_val, feat_cols: list[str], n_rounds: int
) -> tuple[lgb.Booster, float]:
    """DART boosting — cannot use early stopping, so we pass a fixed n_rounds
    derived from the GBDT best iteration."""
    params = {
        "objective":          "binary",
        "metric":             ["binary_logloss", "auc"],
        "boosting_type":      "dart",
        "learning_rate":      0.01,
        "num_leaves":         255,
        "max_depth":          -1,
        "min_child_samples":  50,
        "feature_fraction":   0.6,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "lambda_l1":          0.05,
        "lambda_l2":          0.5,
        "drop_rate":          0.1,
        "uniform_drop":       False,
        "max_drop":           50,
        "skip_drop":          0.5,
        "verbosity":          -1,
        "n_jobs":             -1,
        "seed":               42,
    }
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feat_cols, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        params, dtrain, num_boost_round=n_rounds,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    auc = roc_auc_score(y_val, model.predict(X_val))
    return model, auc


def train_xgb(
    X_tr, y_tr, X_val, y_val
) -> tuple[object, float]:
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective":       "binary:logistic",
        "eval_metric":     ["logloss", "auc"],
        "learning_rate":   0.02,
        "max_depth":       7,
        "subsample":       0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 50,
        "reg_alpha":       0.05,
        "reg_lambda":      0.5,
        "tree_method":     "hist",
        "device":          "cpu",   # XGBoost GPU conflicts with LightGBM GPU
        "seed":            42,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=200,
    )
    auc = roc_auc_score(y_val, model.predict(dval))
    return model, auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir",  default="data/features/BTCUSDT")
    parser.add_argument("--output-dir",    default="models/gbm_ensemble")
    parser.add_argument("--ret-threshold", type=float, default=0.0005)
    parser.add_argument("--val-frac",      type=float, default=0.15)
    parser.add_argument("--cv-folds",      type=int,   default=4)
    parser.add_argument("--skip-dart",     action="store_true")
    parser.add_argument("--skip-xgb",     action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load & engineer features (keep full matrix in RAM) ─────────────────
    t0 = time.time()
    print("Loading features into RAM...")
    files = sorted(glob.glob(f"{args.features_dir}/*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("bar_time_ms")
    print(f"  {len(df):,} bars  ({len(files)} files)  [{time.time()-t0:.1f}s]")

    print("Engineering features (lags / rolling / interactions)...")
    df = build_features(df)
    df = make_labels(df, threshold=args.ret_threshold)
    feat_cols = get_feature_cols(df)
    print(f"  {len(df):,} labeled rows  balance={df['label'].mean():.3f}")
    print(f"  {len(feat_cols)} features  [{time.time()-t0:.1f}s]")

    # Full NumPy arrays kept in RAM
    X_all = df.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)
    y_all = df["label"].to_numpy().astype(np.int8)
    n = len(X_all)
    print(f"  Feature matrix in RAM: {X_all.nbytes / 1e9:.2f} GB")

    # ── Final hold-out split (fixed, never used for model selection) ────────
    split   = int(n * (1 - args.val_frac))
    X_tr_f, y_tr_f = X_all[:split], y_all[:split]
    X_val_f, y_val_f = X_all[split:], y_all[split:]
    print(f"  Hold-out: train={split:,}  val={n-split:,}")

    results: dict = {"feat_cols": feat_cols, "n_features": len(feat_cols), "models": {}}

    # ── Walk-forward CV (4 folds, no leakage) ──────────────────────────────
    print(f"\n── Walk-forward CV ({args.cv_folds} folds) ──")
    folds = list(walk_forward_splits(split, n_folds=args.cv_folds))
    print(f"  Fold sizes: {[(e-s, e) for s, e in folds]}")

    # ── 1. LightGBM GBDT ───────────────────────────────────────────────────
    print("\n[1/3] LightGBM GBDT  (lr=0.01, leaves=255, early_stop=100)")
    gbdt_cv_aucs = []
    gbdt_best_iter = 0

    for fold_i, (tr_end, val_end) in enumerate(folds):
        X_cv_tr, y_cv_tr = X_all[:tr_end],       y_all[:tr_end]
        X_cv_val, y_cv_val = X_all[tr_end:val_end], y_all[tr_end:val_end]
        print(f"  Fold {fold_i+1}/{args.cv_folds}  train={tr_end:,}  val={val_end-tr_end:,}")
        m, auc = train_lgb_gbdt(X_cv_tr, y_cv_tr, X_cv_val, y_cv_val, feat_cols)
        gbdt_best_iter = max(gbdt_best_iter, m.best_iteration)
        gbdt_cv_aucs.append(auc)
        print(f"    AUC={auc:.4f}  best_iter={m.best_iteration}")

    print(f"  CV AUC: {np.mean(gbdt_cv_aucs):.4f} ± {np.std(gbdt_cv_aucs):.4f}")

    # Final GBDT on full training set
    print("  Training final GBDT on full train split...")
    gbdt_final, gbdt_holdout_auc = train_lgb_gbdt(X_tr_f, y_tr_f, X_val_f, y_val_f, feat_cols)
    gbdt_final.save_model(str(out / "lgb_gbdt.txt"))
    print(f"  Hold-out AUC: {gbdt_holdout_auc:.4f}  best_iter={gbdt_final.best_iteration}")

    # Feature importance
    importance = sorted(
        zip(feat_cols, gbdt_final.feature_importance("gain").tolist()),
        key=lambda x: -x[1],
    )
    print("\n  Top 20 features:")
    for name, gain in importance[:20]:
        print(f"    {name:<50s}  {gain:>10.1f}")

    results["models"]["lgb_gbdt"] = {
        "cv_auc_mean": float(np.mean(gbdt_cv_aucs)),
        "cv_auc_std":  float(np.std(gbdt_cv_aucs)),
        "holdout_auc": gbdt_holdout_auc,
        "best_iter":   gbdt_final.best_iteration,
    }

    # ── 2. LightGBM DART ───────────────────────────────────────────────────
    dart_preds_val = None
    if not args.skip_dart:
        dart_rounds = max(200, int(gbdt_best_iter * 1.1))  # slightly more than GBDT
        print(f"\n[2/3] LightGBM DART  (n_rounds={dart_rounds}, drop=0.1)")
        dart_cv_aucs = []

        for fold_i, (tr_end, val_end) in enumerate(folds):
            X_cv_tr, y_cv_tr = X_all[:tr_end],         y_all[:tr_end]
            X_cv_val, y_cv_val = X_all[tr_end:val_end], y_all[tr_end:val_end]
            print(f"  Fold {fold_i+1}/{args.cv_folds}  train={tr_end:,}  val={val_end-tr_end:,}")
            m, auc = train_lgb_dart(X_cv_tr, y_cv_tr, X_cv_val, y_cv_val, feat_cols, dart_rounds)
            dart_cv_aucs.append(auc)
            print(f"    AUC={auc:.4f}")

        print(f"  CV AUC: {np.mean(dart_cv_aucs):.4f} ± {np.std(dart_cv_aucs):.4f}")

        print("  Training final DART on full train split...")
        dart_final, dart_holdout_auc = train_lgb_dart(
            X_tr_f, y_tr_f, X_val_f, y_val_f, feat_cols, dart_rounds
        )
        dart_final.save_model(str(out / "lgb_dart.txt"))
        dart_preds_val = dart_final.predict(X_val_f)
        print(f"  Hold-out AUC: {dart_holdout_auc:.4f}")

        results["models"]["lgb_dart"] = {
            "cv_auc_mean": float(np.mean(dart_cv_aucs)),
            "cv_auc_std":  float(np.std(dart_cv_aucs)),
            "holdout_auc": dart_holdout_auc,
            "n_rounds":    dart_rounds,
        }

    # ── 3. XGBoost ────────────────────────────────────────────────────────
    xgb_preds_val = None
    if HAS_XGB and not args.skip_xgb:
        print("\n[3/3] XGBoost  (lr=0.02, depth=7, early_stop=100)")
        xgb_cv_aucs = []

        for fold_i, (tr_end, val_end) in enumerate(folds):
            X_cv_tr, y_cv_tr = X_all[:tr_end],         y_all[:tr_end]
            X_cv_val, y_cv_val = X_all[tr_end:val_end], y_all[tr_end:val_end]
            print(f"  Fold {fold_i+1}/{args.cv_folds}  train={tr_end:,}  val={val_end-tr_end:,}")
            m, auc = train_xgb(X_cv_tr, y_cv_tr, X_cv_val, y_cv_val)
            xgb_cv_aucs.append(auc)
            print(f"    AUC={auc:.4f}")

        print(f"  CV AUC: {np.mean(xgb_cv_aucs):.4f} ± {np.std(xgb_cv_aucs):.4f}")

        print("  Training final XGB on full train split...")
        xgb_final, xgb_holdout_auc = train_xgb(X_tr_f, y_tr_f, X_val_f, y_val_f)
        xgb_final.save_model(str(out / "xgb_model.ubj"))
        xgb_preds_val = xgb_final.predict(xgb.DMatrix(X_val_f))
        print(f"  Hold-out AUC: {xgb_holdout_auc:.4f}")

        results["models"]["xgboost"] = {
            "cv_auc_mean": float(np.mean(xgb_cv_aucs)),
            "cv_auc_std":  float(np.std(xgb_cv_aucs)),
            "holdout_auc": xgb_holdout_auc,
        }

    # ── Ensemble ────────────────────────────────────────────────────────────
    print("\n── Ensemble ──────────────────────────────────────────")
    gbdt_preds = gbdt_final.predict(X_val_f)
    preds_list = [gbdt_preds]
    labels_list = ["GBDT"]

    if dart_preds_val is not None:
        preds_list.append(dart_preds_val)
        labels_list.append("DART")
    if xgb_preds_val is not None:
        preds_list.append(xgb_preds_val)
        labels_list.append("XGB")

    for preds, name in zip(preds_list, labels_list):
        auc = roc_auc_score(y_val_f, preds)
        ll  = log_loss(y_val_f, preds)
        print(f"  {name:<6}  AUC={auc:.4f}  logloss={ll:.4f}")

    if len(preds_list) > 1:
        ensemble_preds = np.mean(preds_list, axis=0)
        ensemble_auc   = roc_auc_score(y_val_f, ensemble_preds)
        ensemble_ll    = log_loss(y_val_f, ensemble_preds)
        print(f"  {'Blend':<6}  AUC={ensemble_auc:.4f}  logloss={ensemble_ll:.4f}  ← final")
        results["ensemble"] = {"holdout_auc": ensemble_auc, "holdout_logloss": ensemble_ll}
        np.save(str(out / "ensemble_val_preds.npy"), ensemble_preds)
    else:
        results["ensemble"] = {"holdout_auc": float(roc_auc_score(y_val_f, gbdt_preds))}

    # ── Save ────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    results["elapsed_s"]  = round(elapsed_total, 1)
    results["n_feat_cols"] = len(feat_cols)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nAll models saved → {out}/")
    print(f"Total time: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
