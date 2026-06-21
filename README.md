# BTC/USDT Intraday Trading System

Probabilistic, regime-aware, multi-agent intraday trading system for BTC/USDT perpetual futures (Binance USDM).
**Target: OOS Sharpe ≥ 1.0 in paper trading before touching live money.**

---

## Current status

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Setup | ✅ Done | |
| 1 Data | ✅ Done | 2,090 days (2020-09-10 → 2026-05-31), 5 sources, on HuggingFace |
| 2 Features | ✅ Done | 601,920 bars × 20 features, on HuggingFace |
| 3 Simulator | ✅ Done | Queue-aware L2 replay, 0.000 bps canary error |
| 4 Forecast | 🔄 Training | v3 running on A100 — binary classifier, seq=512, LoRA rank=32 |
| 5 Agents | ⏳ Pending | Code done, needs Phase 4 output |
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
local = snapshot_download(
    "ibrahimdaud/binance-btcusdt",
    repo_type="dataset",
    ignore_patterns=["raw/*"],
)

# Everything including raw tick data (~17 GB)
local = snapshot_download("ibrahimdaud/binance-btcusdt", repo_type="dataset")
```

**Feature schema** (27 columns, 20 model input features):

| Group | Features |
|-------|---------|
| Price | `log_ret_1m`, `log_ret_5m`, `log_ret_15m`, `log_ret_60m`, `realized_vol_30m`, `rsi_14` |
| Volume | `vol_5m`, `taker_buy_ratio_5m`, `trade_count_5m`, `avg_trade_size_5m` |
| Depth | `depth_imbalance_1pct` *(from 2023-01-01, bookDepth available)* |
| VPIN | `vpin_50`, `vpin_bucket_imbalance` |
| Hawkes | `hawkes_buy_intensity`, `hawkes_sell_intensity`, `hawkes_net` |
| Market | `oi_btc`, `oi_change_1h`, `ls_count_ratio`, `taker_ls_vol_ratio` |
| Targets | `fwd_ret_5m`, `fwd_ret_15m`, `fwd_ret_60m`, `fwd_direction_5m` |

> Note: 5 columns were removed that are always null in Binance bulk data:
> `depth_imbalance_02pct`, `bid_depth_02pct`, `ask_depth_02pct`, `ofi_5m`, `funding_rate`.

---

## Architecture

```
Raw data (5 sources: aggTrades, klines_1m/5m, bookDepth, metrics)
    │
    ▼
Feature Store  ·  20 features · 5-min bars · 601,920 bars (2020-09 → 2026-05)
    │
    ├─► ForecastAgent ──────────────────────────────────────────────┐
    │     Kronos-base [frozen backbone, 102M params]                │
    │     + LoRA adapters [last 8 layers, rank=32, 852K params]     │
    │     + SmallTCN [channels=128]                                 │
    │     + ForecastHead [binary: down / up]                        │
    │                                                               │
    ├─► OrderflowAgent  (rule-based)                                ├─► Aggregator
    ├─► RegimeAgent     (HMM + LightGBM)                           │   LightGBM meta-learner
    ├─► RiskAgent       (rule-based)                               │   → Decision (side, size)
    └─► StayOutDetector (rule-based) ──────────────────────────────┘        │
                                                                             ▼
                                                                      RL Execution
                                                                      CQL offline policy
                                                                      → HOW to fill
```

---

## Phase 4 — Forecast model training

### What we learned from training experiments

Three training runs were needed to identify and fix the root cause of a stuck loss:

| Run | Config | Val loss | Issue |
|-----|--------|----------|-------|
| v1 | top_k=4, rank=16, smooth=0.05, 11-bin | 0.2896 | **Loss floor** — theoretical min CE with smooth=0.05, n=11 is exactly 0.2896 |
| v2 | top_k=8, rank=32, smooth=0.05, 11-bin | 0.2896 | Same floor — more capacity didn't help |
| v3 | top_k=8, rank=32, **smooth=0.0, binary**, seq=512, ch=128 | 🔄 running | Floor removed, real loss space |

**Root cause:** `label_smoothing=0.05` with 11 output classes creates a mathematical minimum CE of 0.2896. The model hit this ceiling in epoch 2 of every run and couldn't improve — not a model capacity problem, a loss function problem.

**Additional fixes in v3:**
- `pt_sl=(1.5, 1.0)` → `(1.0, 1.0)` — symmetric barriers; fixed 54%/0.07%/45% label skew → 50%/50% binary
- `n_bins=11` → `2` (binary down/up) — 131 flat samples dropped, cleaner signal
- `seq_klines=256` → `512` — 8.5h of 1m context fed to Kronos instead of 4.3h
- `channels=64` → `128` in SmallTCN — wider feature mixing

### Current best training command (v3)

```bash
uv run intraday forecast train \
  --train-end      2025-09-30 \
  --val-start      2025-10-01 \
  --val-end        2026-02-28 \
  --epochs         20 \
  --batch-size     512 \
  --grad-accum     4 \
  --unfreeze-top-k 8 \
  --lora-rank      32 \
  --lora-alpha     64 \
  --warmup-steps   100 \
  --lr-lora        2e-5 \
  --lr-head        2e-4 \
  --device         cuda \
  --log-every      50
```

**What this does:**
- `--unfreeze-top-k 8` — LoRA on last 8 of 12 Kronos layers (851,968 trainable params, 0.83% of backbone)
- `--lora-rank 32` — rank-32 low-rank adapters on q_proj + v_proj
- `--warmup-steps 100` — LR ramp covers ~0.4 epochs (not 2 epochs like before)
- `--grad-accum 4` — effective batch = 512 × 4 = 2048
- `--batch-size 512` — max safe at seq=512 on A100 80GB (82.8% VRAM)

**Run in tmux so it survives disconnects:**
```bash
tmux new-session -d -s train
tmux send-keys -t train "uv run intraday forecast train [args above] 2>&1 | tee /tmp/forecast.log" Enter
tmux split-window -t train -v -p 25
tmux send-keys -t train "watch -n 2 nvidia-smi" Enter
# Reconnect: tmux attach -t train
```

**Resume after interruption:**
```bash
uv run intraday forecast train \
  --train-end 2025-09-30 --val-start 2025-10-01 --val-end 2026-02-28 \
  --epochs 20 --batch-size 512 --grad-accum 4 \
  --unfreeze-top-k 8 --lora-rank 32 --lora-alpha 64 \
  --warmup-steps 100 --lr-lora 2e-5 --lr-head 2e-4 --device cuda \
  --resume-from models/forecast/<timestamp>/checkpoint_epoch05.pt
```

**Time estimates on A100 80GB (seq=512):**

| batch | grad_accum | eff_batch | ~time/epoch | 20 epochs |
|-------|-----------|-----------|-------------|-----------|
| 512 | 4 | 2048 | ~28 min | **~9.5 hours** |

---

## Hard acceptance gates — do not skip

| Gate | Condition | Action if failed |
|------|-----------|-----------------|
| After Phase 4 | OOS accuracy > 52% AND loss improving past ep 5 | Fix labels / architecture, re-train |
| After Phase 6 | OOS Sharpe ≥ 1.0 (full costs) | Debug leakage, re-train — **do NOT proceed to Phase 7** |
| After Phase 7 | RL slippage < 0.8× Almgren-Chriss | Ship v5 without RL instead |

---

## Fresh machine setup — step by step

### 0. Clone and install

```bash
git clone <your-github-repo-url> quant-hack
cd quant-hack

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv sync

# Clone Kronos source (required — loader imports from this repo)
git clone https://github.com/shiyu-coder/Kronos.git Kronos

# Verify GPU
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1. Download Kronos model weights

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('NeoQuasar/Kronos-base',           local_dir='models/kronos-base')
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='models/kronos-tokenizer')
print('Done.')
"
```

Expected: `models/kronos-base/model.safetensors` (~390 MB), `models/kronos-tokenizer/model.safetensors` (~15 MB)

### 2. Get data from HuggingFace (fastest — avoids re-downloading from Binance)

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ibrahimdaud/binance-btcusdt',
    repo_type='dataset',
    local_dir='.',
    ignore_patterns=['raw/aggTrades/*'],  # skip 17 GB tick data if not needed
)
print('Done.')
"
```

Or re-download fresh from Binance:

```bash
uv run intraday data download-bulk --start 2020-09-10 --end 2026-05-31 --concurrency 16
uv run intraday features compute   --start 2020-09-10 --end 2026-05-31 --workers 16
```

### 3. Smoke test

```bash
uv run intraday forecast train --smoke-test --unfreeze-top-k 8 --batch-size 512 --device cuda
```

Expected: `Pipeline verified — all components functional.`

### 4. Train Phase 4

Use the command from the [training section](#current-best-training-command-v3) above.

### 5. After training — push weights to HuggingFace

```bash
HF_TOKEN=hf_xxx uv run python scripts/push_weights.py
```

### 6–10. Remaining phases

See `idea/PLAN.md` for the full phase spec with acceptance gates.

---

## Project layout

```
quant-hack/
├── src/intraday/
│   ├── data/          # Phase 1: download, live capture
│   ├── features/      # Phase 2: 20-feature calculator (VPIN, Hawkes, OFI...)
│   ├── sim/           # Phase 3: queue-aware L2 backtest simulator
│   ├── forecast/      # Phase 4: Kronos + LoRA + SmallTCN + binary head
│   ├── models/        # Standalone GRU/LSTM (experimental baseline)
│   ├── agents/        # Phase 5: orderflow, regime, risk, stay_out
│   ├── aggregator/    # Phase 6: MetaLearner + fractional Kelly sizing
│   └── rl/            # Phase 7: ExecutionEnv + CQL offline policy
├── scripts/
│   ├── upload_huggingface.py   # Push dataset to HuggingFace
│   └── push_weights.py         # Push trained weights to HuggingFace
├── Kronos/            # git clone https://github.com/shiyu-coder/Kronos.git
├── models/            # gitignored — on HuggingFace
│   ├── kronos-base/       # NeoQuasar/Kronos-base
│   ├── kronos-tokenizer/  # NeoQuasar/Kronos-Tokenizer-base
│   └── forecast/          # Trained weights (tcn, head, lora, checkpoints)
├── data/              # gitignored — on HuggingFace
│   ├── raw/binance/       # aggTrades, klines_1m/5m, bookDepth, metrics
│   └── features/BTCUSDT/  # 288 rows/day × 20 features
├── idea/
│   ├── PLAN.md        # Master plan — read before starting any phase
│   └── phases/        # Phase specs 00-10
└── pyproject.toml     # uv project + deps
```

---

## Key design rules

- **Walk-forward only** — purged k-fold + embargo, never random split
- **RL for execution only** — CQL decides HOW to fill, not WHAT direction
- **No live money until paper trading ≥ 4 weeks, Sharpe ≥ 1.0**
- **Polars not pandas** — all data/feature code uses Polars
- **UTC always** — all timestamps in milliseconds UTC
- **Calibration > accuracy** — a 55% calibrated model beats 60% miscalibrated

---

## Disclaimer

Research software. Cryptocurrency trading carries significant risk of loss. Not financial advice.
