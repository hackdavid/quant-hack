#!/usr/bin/env python3
"""Push trained model checkpoints to HuggingFace dataset.

Repository: https://huggingface.co/datasets/ibrahimdaud/binance-btcusdt

Checkpoint layout on HuggingFace:
    checkpoints/
    ├── transformer_v<N>/          CryptoTransformer (best.pt by val AUC)
    │   ├── best.pt                74 MB  — load for inference
    │   ├── config.json            hyperparameters + feature list
    │   ├── history.json           per-epoch train/val loss + AUC curve
    │   └── MODEL_CARD.md          when to use, how to load, performance
    ├── lgb_baseline/              LightGBM GBDT baseline (fast, 2 min train)
    │   ├── lgb_model.txt          LightGBM model (text format)
    │   ├── meta.json              feature list, val AUC, best_iteration
    │   └── MODEL_CARD.md
    └── gbm_ensemble/              3-model GBM blend (GBDT + DART + XGBoost)
        ├── lgb_gbdt.txt
        ├── lgb_dart.txt           (if trained)
        ├── xgb_model.ubj          (if trained)
        ├── results.json           per-model + ensemble AUC
        └── MODEL_CARD.md

Usage:
    # Use env variable (recommended — never commit token to git)
    export HUGGINGFACE_TOKEN="hf_..."
    python scripts/push_checkpoints_hf.py

    # Or pass directly
    python scripts/push_checkpoints_hf.py --token hf_...

    # Push specific checkpoint only
    python scripts/push_checkpoints_hf.py --only transformer
    python scripts/push_checkpoints_hf.py --only lgb
    python scripts/push_checkpoints_hf.py --only gbm_ensemble
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID  = "ibrahimdaud/binance-btcusdt"
HF_REPO_TYPE = "dataset"

REPO_ROOT = Path(__file__).parents[1]


# ── Model card templates ───────────────────────────────────────────────────────

def _transformer_card(cfg: dict, history: list[dict], run_id: str) -> str:
    best = max(history, key=lambda x: x["val_auc"])
    return f"""# CryptoTransformer — BTC/USDT 5-min Direction Forecast

## What it does
Predicts the direction of BTC/USDT price over the next **15 minutes** from a
sequence of **{cfg['seq_len']} × 5-min bars** (= {cfg['seq_len']*5//60}h context window).

Output: probability P(up), where up = `fwd_ret_15m > 0.05%`.
- **P > 0.55** → go long
- **P < 0.45** → go short
- **0.45 ≤ P ≤ 0.55** → no trade

## When to use
- Real-time inference every 5-min bar close
- As one signal in a multi-model ensemble (pair with LGB for +AUC)
- **Do NOT use** for horizons longer than 30 min (trained on 15-min labels)

## Performance (validation set, last 15% of data)
| Metric | Value |
|--------|-------|
| Best val AUC | **{best['val_auc']:.4f}** (epoch {best['epoch']}) |
| Best val loss | {best['val_loss']:.4f} |
| Epoch time | ~{best['elapsed_s']:.0f}s on A100 |
| Early stopped | Yes (patience=10) |

## Architecture
```
Input: (batch, {cfg['seq_len']}, {cfg['n_features']} features + {cfg['n_time_feat']} time)
  → Linear projection → d_model={cfg['d_model']}
  → LocalConvBlock (causal depthwise conv, dilation 1+2)
  → CLS token prepended
  → Sinusoidal positional encoding
  → {cfg['n_layers']} × Pre-LN TransformerEncoderLayer
      (d={cfg['d_model']}, heads={cfg['n_heads']}, ffn={cfg['dim_ff']}, dropout={cfg['dropout']})
  → CLS output → Linear({cfg['d_model']}, 128) → GELU → Linear(128, 2)
Total params: {cfg.get('n_params', 0):,}
```

## Training config
```json
{json.dumps({k: v for k, v in cfg.items() if k not in ('feat_cols',)}, indent=2)}
```

## Input features (in order)
```python
feat_cols = {cfg.get('feat_cols', [])}
```

## How to load
```python
import torch, json
from intraday.forecast.transformer_model import CryptoTransformer

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["config"]

model = CryptoTransformer(
    n_features=cfg["n_features"], n_time_feat=cfg["n_time_feat"],
    d_model=cfg["d_model"], n_heads=cfg["n_heads"],
    n_layers=cfg["n_layers"], dim_ff=cfg["dim_ff"],
    seq_len=cfg["seq_len"], dropout=0.0,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

norm_mean = ckpt["norm_mean"]   # (n_features,) float32
norm_std  = ckpt["norm_std"]    # (n_features,) float32
```

## Run ID
`{run_id}`
"""


def _lgb_card(meta: dict) -> str:
    feat_count = len(meta.get("feat_cols", []))
    return f"""# LightGBM Baseline — BTC/USDT 15-min Direction

## What it does
Classifies whether BTC/USDT will rise >0.05% or fall >0.05% over the next 15 min
using **{feat_count} features** (base + lags + rolling stats).

Output: probability P(up) ∈ [0, 1].

## When to use
- Fast baseline or fallback when transformer is unavailable
- Feature importance analysis (gain-based ranking)
- Ensemble component alongside CryptoTransformer
- **Instant** inference (< 1ms per bar)
- **No GPU required**

## Performance
| Metric | Value |
|--------|-------|
| Val AUC | **{meta.get('val_auc', 0):.4f}** |
| Val log-loss | {meta.get('val_log_loss', 0):.4f} |
| Best iteration | {meta.get('best_iteration', 0)} |
| Val fraction | last {meta.get('val_frac', 0.15)*100:.0f}% of data |
| Features | {feat_count} (base + lags t-1/3/6/12/24 + rolling 30m/1h/2h/4h) |

## Feature engineering
- **Base**: 20 raw 5-min features (see feature schema)
- **Lags**: t-1, t-3, t-6, t-12, t-24 bars for 9 key microstructure signals
- **Rolling**: mean + std over 6, 12, 24, 48 bar windows for 6 signals
- **Time**: hour_utc, sin/cos(hour), day_of_week, sin/cos(dow)

## Top features (by GBDT gain)
`depth_imbalance_1pct`, `log_ret_15m`, `rsi_14`, `log_ret_60m`,
`taker_buy_ratio_5m_rmean6`, `oi_btc`, `vpin_50_rstd48`

## How to load
```python
import lightgbm as lgb
import json

model    = lgb.Booster(model_file="lgb_model.txt")
meta     = json.load(open("meta.json"))
feat_cols = meta["feat_cols"]

# Build feature row (single-row numpy array in feat_cols order)
prob_up = model.predict(X_row)[0]   # X_row shape (1, n_features)
```
"""


def _gbm_ensemble_card(results: dict) -> str:
    models = results.get("models", {})
    ens    = results.get("ensemble", {})
    return f"""# GBM Ensemble — BTC/USDT 15-min Direction

## What it does
Three gradient boosting models trained on **{results.get('n_feat_cols', 200)}+ features**,
blended by simple average. Walk-forward 4-fold CV prevents data leakage.

Output: probability P(up) ∈ [0, 1].

## When to use
- Best standalone GBM prediction (outperforms LGB baseline)
- Ensemble component alongside CryptoTransformer
- When you want interpretable feature importance + strong baseline

## Performance
| Model | CV AUC | Hold-out AUC |
|-------|--------|-------------|
| LGB GBDT | {models.get('lgb_gbdt',{}).get('cv_auc_mean',0):.4f} ± {models.get('lgb_gbdt',{}).get('cv_auc_std',0):.4f} | {models.get('lgb_gbdt',{}).get('holdout_auc',0):.4f} |
| LGB DART | {models.get('lgb_dart',{}).get('cv_auc_mean',0):.4f} ± {models.get('lgb_dart',{}).get('cv_auc_std',0):.4f} | {models.get('lgb_dart',{}).get('holdout_auc',0):.4f} |
| XGBoost | {models.get('xgboost',{}).get('cv_auc_mean',0):.4f} ± {models.get('xgboost',{}).get('cv_auc_std',0):.4f} | {models.get('xgboost',{}).get('holdout_auc',0):.4f} |
| **Ensemble** | — | **{ens.get('holdout_auc',0):.4f}** |

## Models
- **lgb_gbdt.txt** — Primary model. Use when only one model needed.
- **lgb_dart.txt** — DART (dropout trees). Better regularisation, use for ensemble.
- **xgb_model.ubj** — XGBoost. Diverse gradient algorithm, ensemble diversity.

## How to load
```python
import lightgbm as lgb, xgboost as xgb, json, numpy as np

gbdt = lgb.Booster(model_file="lgb_gbdt.txt")
dart = lgb.Booster(model_file="lgb_dart.txt")
xgb_ = xgb.Booster(); xgb_.load_model("xgb_model.ubj")

results   = json.load(open("results.json"))
feat_cols = results["feat_cols"]

# Inference
p_gbdt = gbdt.predict(X)[0]
p_dart = dart.predict(X)[0]
p_xgb  = xgb_.predict(xgb.DMatrix(X))[0]
prob_up = np.mean([p_gbdt, p_dart, p_xgb])
```
"""


# ── Upload helpers ─────────────────────────────────────────────────────────────

def upload_file(api, local: Path, remote: str, repo_id: str, commit_msg: str) -> None:
    if not local.exists():
        print(f"  SKIP (not found): {local}")
        return
    size_mb = local.stat().st_size / 1e6
    print(f"  Uploading {local.name} ({size_mb:.1f} MB) → {remote}")
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        commit_message=commit_msg,
    )


def upload_text(api, content: str, remote: str, repo_id: str, commit_msg: str) -> None:
    import io
    print(f"  Writing {remote}")
    api.upload_file(
        path_or_fileobj=io.BytesIO(content.encode()),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        commit_message=commit_msg,
    )


# ── Push functions ─────────────────────────────────────────────────────────────

def push_transformer(api, repo_id: str) -> bool:
    """Push the best transformer checkpoint (latest run with best.pt)."""
    runs = sorted((REPO_ROOT / "models" / "transformer").glob("*/best.pt"), key=lambda p: p.parent.name)
    if not runs:
        print("  No transformer checkpoints found in models/transformer/*/best.pt")
        return False

    best_run = runs[-1].parent   # most recent run
    run_id   = best_run.name
    cfg_path = best_run / "config.json"
    hist_path = best_run / "history.json"

    if not cfg_path.exists():
        print(f"  Missing config.json in {best_run}")
        return False

    cfg     = json.loads(cfg_path.read_text())
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    prefix  = f"checkpoints/transformer_v2"

    print(f"\nPushing transformer checkpoint: {run_id}")
    upload_file(api, best_run / "best.pt",    f"{prefix}/best.pt",       repo_id, f"transformer best.pt [{run_id}]")
    upload_file(api, cfg_path,                 f"{prefix}/config.json",   repo_id, f"transformer config [{run_id}]")
    upload_file(api, hist_path,                f"{prefix}/history.json",  repo_id, f"transformer history [{run_id}]")
    upload_text(api, _transformer_card(cfg, history, run_id),
                f"{prefix}/MODEL_CARD.md", repo_id, f"transformer model card [{run_id}]")
    print(f"  Done → {HF_REPO_TYPE}s/{repo_id}/{prefix}/")
    return True


def push_lgb(api, repo_id: str) -> bool:
    lgb_dir = REPO_ROOT / "models" / "lgb"
    if not (lgb_dir / "lgb_model.txt").exists():
        print("  No LGB model found in models/lgb/lgb_model.txt")
        return False

    meta = json.loads((lgb_dir / "meta.json").read_text()) if (lgb_dir / "meta.json").exists() else {}
    prefix = "checkpoints/lgb_baseline"

    print(f"\nPushing LGB baseline...")
    upload_file(api, lgb_dir / "lgb_model.txt", f"{prefix}/lgb_model.txt", repo_id, "LGB baseline model")
    upload_file(api, lgb_dir / "meta.json",      f"{prefix}/meta.json",    repo_id, "LGB baseline meta")
    upload_text(api, _lgb_card(meta), f"{prefix}/MODEL_CARD.md",           repo_id, "LGB baseline model card")
    print(f"  Done → {HF_REPO_TYPE}s/{repo_id}/{prefix}/")
    return True


def push_gbm_ensemble(api, repo_id: str) -> bool:
    ens_dir = REPO_ROOT / "models" / "gbm_ensemble"
    results_path = ens_dir / "results.json"
    if not results_path.exists():
        print("  GBM ensemble not done yet — models/gbm_ensemble/results.json missing")
        return False

    results = json.loads(results_path.read_text())
    prefix  = "checkpoints/gbm_ensemble"

    print(f"\nPushing GBM ensemble...")
    for fname in ["lgb_gbdt.txt", "lgb_dart.txt", "xgb_model.ubj", "results.json"]:
        upload_file(api, ens_dir / fname, f"{prefix}/{fname}", repo_id, f"GBM ensemble {fname}")
    upload_text(api, _gbm_ensemble_card(results), f"{prefix}/MODEL_CARD.md", repo_id, "GBM ensemble model card")
    print(f"  Done → {HF_REPO_TYPE}s/{repo_id}/{prefix}/")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Push model checkpoints to HuggingFace")
    p.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN", ""),
                   help="HF write token (or set HUGGINGFACE_TOKEN env var)")
    p.add_argument("--repo",  default=HF_REPO_ID, help="HuggingFace repo ID")
    p.add_argument("--only",  default=None, choices=["transformer", "lgb", "gbm_ensemble"],
                   help="Push only one checkpoint type")
    args = p.parse_args()

    if not args.token:
        sys.exit("ERROR: Provide --token or set HUGGINGFACE_TOKEN env var")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Run: uv pip install huggingface_hub")

    api = HfApi(token=args.token)

    # Verify access
    try:
        api.repo_info(args.repo, repo_type=HF_REPO_TYPE)
        print(f"Connected to {HF_REPO_TYPE}s/{args.repo}")
    except Exception as e:
        sys.exit(f"Cannot access repo: {e}")

    pushed = []

    if args.only in (None, "transformer"):
        if push_transformer(api, args.repo):
            pushed.append("transformer")

    if args.only in (None, "lgb"):
        if push_lgb(api, args.repo):
            pushed.append("lgb_baseline")

    if args.only in (None, "gbm_ensemble"):
        if push_gbm_ensemble(api, args.repo):
            pushed.append("gbm_ensemble")

    if pushed:
        print(f"\nAll done. Pushed: {', '.join(pushed)}")
        print(f"View at: https://huggingface.co/datasets/{args.repo}/tree/main/checkpoints/")
    else:
        print("\nNothing pushed — check that training has completed.")


if __name__ == "__main__":
    main()
