# BTC/USDT Intraday Trading System

Probabilistic, regime-aware, multi-agent intraday trading system for BTC/USDT perpetual futures (Binance USDM).
**Target: OOS Sharpe ≥ 1.0 in paper trading before touching live money.**

---

## Current status

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Setup | ✅ Done | |
| 1 Data | ✅ Done | 2,090 days (2020-09-10 → 2026-05-31), on HuggingFace |
| 2 Features | ✅ Done | 601,920 bars × 20 features, on HuggingFace |
| 3 Simulator | ✅ Done | Queue-aware L2 replay, 0.000 bps canary error |
| 4 Forecast | ⏳ Next run | v4 — 2023+ data only, load from HuggingFace (see below) |
| 5 Agents | ⏳ Pending | Code done, needs Phase 4 |
| 6 Aggregator | ⏳ Pending | Code done, needs Phase 4+5 |
| 7 RL Execution | ⏳ Pending | Code done, needs Phase 6 |
| 8 Paper Trading | ❌ Not built | |

---

## Dataset

All raw + feature data is on HuggingFace:
**[ibrahimdaud/binance-btcusdt](https://huggingface.co/datasets/ibrahimdaud/binance-btcusdt)**

```python
from huggingface_hub import snapshot_download

# Feature data only (120 MB — enough for training)
snapshot_download(
    "ibrahimdaud/binance-btcusdt",
    repo_type="dataset",
    local_dir=".",
    ignore_patterns=["raw/*"],
)

# Everything including raw tick data (~17 GB)
snapshot_download("ibrahimdaud/binance-btcusdt", repo_type="dataset", local_dir=".")
```

**Feature schema** — 27 columns per 5-min bar, 20 model input features:

| Group | Features |
|-------|---------|
| Price | `log_ret_1m`, `log_ret_5m`, `log_ret_15m`, `log_ret_60m`, `realized_vol_30m`, `rsi_14` |
| Volume | `vol_5m`, `taker_buy_ratio_5m`, `trade_count_5m`, `avg_trade_size_5m` |
| Depth | `depth_imbalance_1pct` *(available from 2023-01-01 only)* |
| VPIN | `vpin_50`, `vpin_bucket_imbalance` |
| Hawkes | `hawkes_buy_intensity`, `hawkes_sell_intensity`, `hawkes_net` |
| Market | `oi_btc`, `oi_change_1h`, `ls_count_ratio`, `taker_ls_vol_ratio` |
| Targets | `fwd_ret_5m`, `fwd_ret_15m`, `fwd_ret_60m`, `fwd_direction_5m` |

> 5 columns were dropped — always null in Binance bulk data:
> `depth_imbalance_02pct`, `bid_depth_02pct`, `ask_depth_02pct`, `ofi_5m`, `funding_rate`

---

## Architecture

```
Raw data (5 sources: aggTrades, klines_1m/5m, bookDepth, metrics)
    │
    ▼
Feature Store  ·  20 features · 5-min bars
    │
    ├─► ForecastAgent ──────────────────────────────────────────────┐
    │     Kronos-base [frozen backbone, 102M params]                │
    │     + LoRA adapters [last 8 layers, rank=32, 852K params]     │
    │     + SmallTCN  [channels=128, 4-layer dilated causal conv]   │
    │     + ForecastHead [binary: down=0 / up=1]                    │
    │                                                               │
    ├─► OrderflowAgent  (rule-based)                                ├─► Aggregator
    ├─► RegimeAgent     (HMM + LightGBM)                           │   LightGBM meta-learner
    ├─► RiskAgent       (rule-based)                               │   → Decision (side, size)
    └─► StayOutDetector (rule-based) ──────────────────────────────┘        │
                                                                             ▼
                                                                      RL Execution
                                                                      CQL offline policy
```

---

## Phase 4 — Forecast model: what we learned & next run

### Training history

| Run | Key settings | Train→Val loss | Issue / finding |
|-----|-------------|----------------|-----------------|
| v1 | top_k=4, rank=16, smooth=0.05, 11-bin, seq=256 | 0.864→**0.2896** | `label_smoothing=0.05` with n=11 creates a hard CE floor at 0.2896. Model hit it in ep 2, stuck forever. |
| v2 | top_k=8, rank=32, smooth=0.05, 11-bin, seq=256 | same floor | More capacity changed nothing — root cause was the loss function |
| v3 | top_k=8, rank=32, smooth=0.0, binary, seq=512, ch=128 | 0.6912→0.6931 | Floor removed. Train drops steadily but val oscillates ~0.693 (random baseline). **Data split is now the bottleneck**: training on 2020–2025 (mixed regimes, no depth before 2023) doesn't transfer to the val window. |

### Root causes fixed in v3
- `label_smoothing=0.05` → **`0.0`** — was creating a mathematical CE floor of exactly 0.2896 (theoretical min for smooth=0.05, n=11 classes)
- `pt_sl=(1.5, 1.0)` → **`(1.0, 1.0)`** — symmetric barriers; fixed 54%/0.07%/45% → 50%/50% binary split
- `n_bins=11` → **`2`** (binary down/up) — cleaner signal, 131 flat samples dropped
- `seq_klines=256` → **`512`** — 8.5 hours of 1m context for Kronos (was 4.3h)
- `channels=64` → **`128`** in SmallTCN

### Why v3 still plateaued at ~random baseline
Training on 2020–2025 includes **3 years without `depth_imbalance_1pct`** (bookDepth only available from 2023). The model spends most of training on incomplete feature vectors. Additionally, BTC market microstructure in 2020–2022 is structurally different from 2025 — different leverage profiles, exchange mix, ETF flow. These mixed-regime samples add noise without adding relevant signal.

---

## Next run — v4 (2023+ only, load from HuggingFace)

### Why 2023+
- All 20 features fully populated (bookDepth available from 2023-01-01)
- ~1,247 days = 358,848 bars — still large enough for robust training
- Consistent market regime (post-FTX, ETF-era BTC)
- Train/val split stays within the same regime

### Data splits for v4
| Split | Dates | Bars | Purpose |
|-------|-------|------|---------|
| Train | 2023-01-01 → 2025-03-31 | ~207,072 | Learning |
| Val | 2025-04-01 → 2025-09-30 | ~52,128 | Hyperparameter feedback |
| Test (OOS) | 2025-10-01 → 2026-05-31 | ~99,648 | Never touched during training |

### Step 1 — Load feature data from HuggingFace

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ibrahimdaud/binance-btcusdt',
    repo_type='dataset',
    local_dir='.',
    ignore_patterns=['raw/*'],   # skip 17 GB, features only (120 MB)
)
print('Done — data/features/BTCUSDT/ ready')
"
```

### Step 2 — Smoke test

```bash
uv run intraday forecast train \
  --smoke-test \
  --train-end  2025-03-31 --val-start 2025-04-01 --val-end 2025-09-30 \
  --unfreeze-top-k 8 --lora-rank 32 --lora-alpha 64 \
  --batch-size 512 --device cuda
```

### Step 3 — Full training run (v4)

```bash
tmux new-session -d -s train

tmux send-keys -t train "uv run intraday forecast train \
  --train-end      2025-03-31 \
  --val-start      2025-04-01 \
  --val-end        2025-09-30 \
  --epochs         30 \
  --batch-size     512 \
  --grad-accum     4 \
  --unfreeze-top-k 8 \
  --lora-rank      32 \
  --lora-alpha     64 \
  --warmup-steps   100 \
  --lr-lora        2e-5 \
  --lr-head        2e-4 \
  --device         cuda \
  --log-every      50 \
  2>&1 | tee /tmp/forecast_v4.log" Enter

tmux split-window -t train -v -p 25
tmux send-keys -t train "watch -n 2 nvidia-smi" Enter
# Reconnect: tmux attach -t train
```

**What changed vs v3:**
- `--train-end 2025-03-31` — 2023+ only, all features complete
- `--val-start 2025-04-01 --val-end 2025-09-30` — 6-month val, same regime
- `--epochs 30` — more room for cosine decay with better data

**Time estimate on A100 80GB:**
~207K train samples / 512 batch = ~404 batches / 4 grad_accum = ~101 opt steps/epoch → **~11 min/epoch → ~5.5 hours for 30 epochs**

**Resume after interruption:**
```bash
uv run intraday forecast train \
  --train-end 2025-03-31 --val-start 2025-04-01 --val-end 2025-09-30 \
  --epochs 30 --batch-size 512 --grad-accum 4 \
  --unfreeze-top-k 8 --lora-rank 32 --lora-alpha 64 \
  --warmup-steps 100 --lr-lora 2e-5 --lr-head 2e-4 --device cuda \
  --resume-from models/forecast/<timestamp>/checkpoint_epoch10.pt
```

### Step 4 — Push weights to HuggingFace after training

```bash
# Edit scripts/push_weights.py to point RUN_DIR at the new timestamp
HF_TOKEN=hf_xxx uv run python scripts/push_weights.py
```

---

## Hard acceptance gates — do not skip

| Gate | Condition | Action if failed |
|------|-----------|-----------------|
| After Phase 4 | Val loss < 0.680 (below random) AND steadily improving | Fix data split / architecture, re-train |
| After Phase 6 | OOS Sharpe ≥ 1.0 (full costs) | Debug leakage, re-train — **do NOT proceed to Phase 7** |
| After Phase 7 | RL slippage < 0.8× Almgren-Chriss | Ship v5 without RL instead |

---

## Fresh machine setup

### 0. Clone and install

```bash
git clone <your-github-repo-url> quant-hack
cd quant-hack
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.cargo/env
uv sync
git clone https://github.com/shiyu-coder/Kronos.git Kronos
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1. Download Kronos backbone weights

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('NeoQuasar/Kronos-base',           local_dir='models/kronos-base')
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='models/kronos-tokenizer')
"
```

### 2. Get feature data from HuggingFace

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ibrahimdaud/binance-btcusdt', repo_type='dataset',
    local_dir='.', ignore_patterns=['raw/*'],
)
"
```

### 3. Smoke test → train → push weights

Follow the [v4 training steps](#step-1--load-feature-data-from-huggingface) above.

---

## Project layout

```
quant-hack/
├── src/intraday/
│   ├── data/          # Phase 1: download, live capture
│   ├── features/      # Phase 2: 20-feature calculator (VPIN, Hawkes, OFI...)
│   ├── sim/           # Phase 3: queue-aware L2 backtest simulator
│   ├── forecast/      # Phase 4: Kronos + LoRA + SmallTCN + binary head
│   ├── models/        # GRU/LSTM standalone baseline (experimental)
│   ├── agents/        # Phase 5: orderflow, regime, risk, stay_out
│   ├── aggregator/    # Phase 6: MetaLearner + fractional Kelly sizing
│   └── rl/            # Phase 7: ExecutionEnv + CQL offline policy
├── scripts/
│   ├── upload_huggingface.py   # Push dataset to HuggingFace
│   └── push_weights.py         # Push trained model weights to HuggingFace
├── Kronos/            # git clone https://github.com/shiyu-coder/Kronos.git
├── models/            # gitignored — weights live on HuggingFace
├── data/              # gitignored — data lives on HuggingFace
├── idea/
│   ├── PLAN.md        # Master plan — read before starting any phase
│   └── phases/        # Phase specs 00-10
└── pyproject.toml
```

---

## Key design rules

- **Walk-forward only** — purged k-fold + embargo, never random split
- **2023+ data for training** — full feature set (bookDepth), consistent regime
- **RL for execution only** — CQL decides HOW to fill, not WHAT direction
- **No live money until paper trading ≥ 4 weeks, Sharpe ≥ 1.0**
- **Polars not pandas** — all data/feature code
- **UTC always** — all timestamps in milliseconds UTC

---

## Disclaimer

Research software. Cryptocurrency trading carries significant risk of loss. Not financial advice.

---

## Phase 4 — Forecast Models (Completed)

Three models trained, checkpoints on HuggingFace under `checkpoints/`.

### Results

| Model | Val AUC | Train time | Notes |
|-------|---------|-----------|-------|
| LightGBM baseline | 0.5476 | ~2 min | 119 features, early stopped at round 48 |
| CryptoTransformer v1 | 0.5536 (ep5) | ~40 min | d=256, 8L, overfit after ep5 |
| CryptoTransformer v2 | **0.5531 (ep5)** | ~11 min | dropout=0.2, wd=0.05, early stop ✓ |
| GBM Ensemble | pending | ~75 min | GBDT + DART + XGBoost, 200+ features |

### What we learned
- Kronos (102M frozen) + LoRA dominated by irrelevant pretrained representations → replaced with pure CryptoTransformer
- Signal peaks very early (ep 4–6) then overfits sharply → early stopping essential
- LightGBM AUC 0.547 → Transformer AUC 0.553 → ensemble expected 0.56+
- Top predictors: `depth_imbalance_1pct`, `ls_count_ratio`, `vpin_50`, `taker_buy_ratio_5m`, `oi_change_1h`

---

## Model Checkpoints (HuggingFace)

All checkpoints at: **[ibrahimdaud/binance-btcusdt → checkpoints/](https://huggingface.co/datasets/ibrahimdaud/binance-btcusdt/tree/main/checkpoints)**

| Checkpoint folder | What it is | When to use |
|------------------|-----------|------------|
| `transformer_v2/best.pt` | CryptoTransformer, 6.4M params, AUC 0.553 | Real-time inference, sequence-aware |
| `lgb_baseline/lgb_model.txt` | LightGBM GBDT, 119 features, AUC 0.548 | Fast fallback, no GPU needed |
| `gbm_ensemble/` | GBDT + DART + XGBoost blend, 200+ features | Best standalone GBM prediction |

### Download checkpoints
```bash
from huggingface_hub import hf_hub_download, snapshot_download

# Best transformer only
hf_hub_download(
    "ibrahimdaud/binance-btcusdt", repo_type="dataset",
    filename="checkpoints/transformer_v2/best.pt", local_dir="."
)

# All checkpoints
snapshot_download(
    "ibrahimdaud/binance-btcusdt", repo_type="dataset",
    local_dir=".", allow_patterns=["checkpoints/*"]
)
```

### Push new checkpoints after training
```bash
export HUGGINGFACE_TOKEN="hf_..."
python scripts/push_checkpoints_hf.py           # push all
python scripts/push_checkpoints_hf.py --only transformer
python scripts/push_checkpoints_hf.py --only lgb
python scripts/push_checkpoints_hf.py --only gbm_ensemble
```

---

## Training Pipeline

### 1. Download feature data
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ibrahimdaud/binance-btcusdt', repo_type='dataset',
                  local_dir='.', ignore_patterns=['raw/*'])
"
```

### 2. Train LightGBM baseline (~2 min)
```bash
.venv/bin/python scripts/train_lgb.py \
    --features-dir data/features/BTCUSDT \
    --output-dir models/lgb
```

### 3. Train GBM Ensemble (GBDT + DART + XGBoost, ~75 min)
```bash
.venv/bin/python scripts/train_gbm_ensemble.py \
    --features-dir data/features/BTCUSDT \
    --output-dir models/gbm_ensemble
```

### 4. Train CryptoTransformer (~10-15 min with early stopping)
```bash
# Default config (best settings found)
PYTHONUNBUFFERED=1 .venv/bin/python -m intraday.forecast.train_transformer \
    --dropout 0.2 --weight-decay 0.05 --lr 5e-5 \
    --epochs 50 --patience 10 \
    --output-dir models/transformer

# Resume from checkpoint
.venv/bin/python -m intraday.forecast.train_transformer \
    --resume-from models/transformer/<run_id>/latest.pt
```

### 5. Push to HuggingFace
```bash
export HUGGINGFACE_TOKEN="hf_..."
.venv/bin/python scripts/push_checkpoints_hf.py
```

---

## Backtesting

```bash
# LightGBM only (fast, no GPU)
.venv/bin/python scripts/run_backtest.py --lgb-dir models/lgb

# Transformer only
.venv/bin/python scripts/run_backtest.py \
    --transformer-dir models/transformer/<run_id>

# Ensemble: transformer + LGB
.venv/bin/python scripts/run_backtest.py \
    --transformer-dir models/transformer/<run_id> \
    --lgb-dir models/lgb

# Sweep thresholds 0.50–0.65 to find best Sharpe
.venv/bin/python scripts/run_backtest.py \
    --transformer-dir models/transformer/<run_id> \
    --lgb-dir models/lgb \
    --sweep

# GBM ensemble
.venv/bin/python scripts/run_backtest.py \
    --lgb-ensemble-dir models/gbm_ensemble
```

Output: `models/backtest_results/result_<run_id>.json` + `equity_<run_id>.csv`

---

## Phase 8 — Paper Trading & Live Trading

### Paper trading (no real orders, live Binance WebSocket data)
```bash
.venv/bin/python scripts/run_paper_trade.py \
    --transformer-dir models/transformer/<run_id> \
    --lgb-dir models/lgb \
    --capital 10000 \
    --threshold 0.55 \
    --max-position-frac 0.20 \
    --daily-loss-limit 0.02
```

All decisions logged to `logs/trader/trade_log_*.jsonl`.

### Live trading (real Binance USDT-M futures)
> ⚠️ Run paper trading for ≥ 1 week with OOS Sharpe ≥ 1.0 before going live.
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

.venv/bin/python scripts/run_live_trade.py \
    --transformer-dir models/transformer/<run_id> \
    --lgb-dir models/lgb \
    --capital 1000 \
    --threshold 0.57 \
    --max-position-frac 0.10 \
    --leverage 1
```

### Risk defaults
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.55 paper / 0.57 live | Min prob to trade |
| `--max-position-frac` | 0.20 / 0.10 | Max % capital per trade |
| `--daily-loss-limit` | 2% / 1% | Stop trading today if exceeded |
| `--max-drawdown` | 5% / 3% | Halt permanently (resets next UTC day) |
| `--leverage` | 1x | Start with 1x until validated |

---

## Updated Project Layout

```
quant-hack/
├── src/intraday/
│   ├── forecast/
│   │   ├── transformer_model.py     CryptoTransformer (6.4M params)
│   │   ├── train_transformer.py     Training loop with early stopping + checkpoints
│   │   ├── train.py                 Original Kronos training loop
│   │   └── ...
│   ├── backtest/
│   │   └── engine.py                Vectorized backtest (fees, funding, slippage)
│   ├── signal/
│   │   └── combiner.py              Loads transformer + LGB, blends by val AUC
│   ├── risk/
│   │   └── agent.py                 Kelly sizing, daily loss limit, max drawdown halt
│   └── trader/
│       ├── exchange.py              Binance wrapper (paper + live via ccxt)
│       └── loop.py                  Async WebSocket trading loop
├── scripts/
│   ├── train_lgb.py                 LightGBM baseline training
│   ├── train_gbm_ensemble.py        GBDT + DART + XGBoost ensemble
│   ├── run_backtest.py              Backtesting with threshold sweep
│   ├── run_paper_trade.py           Paper trading (live WS, no orders)
│   ├── run_live_trade.py            Live trading (real Binance futures)
│   └── push_checkpoints_hf.py      Push model weights to HuggingFace
├── models/                          gitignored — on HuggingFace
└── logs/trader/                     gitignored — trade logs
```
