# BTC/USDT Intraday Trading System

Probabilistic, regime-aware, multi-agent intraday trading system for BTC/USDT perpetual futures (Binance USDM).
**Target: OOS Sharpe ≥ 1.0 in paper trading before touching live money.**

---

## Current Status

| Phase | What | Status |
|-------|------|--------|
| 0 | Setup — repo, deps, CLI skeleton | ✅ Done |
| 1 | Data — bulk download + live WebSocket capture | ✅ Done (31 days) |
| 2 | Features — 25 features, LazyFeatureStore | ✅ Done (8,928 rows) |
| 3 | Simulator — queue-aware, canary 0.000 bps error | ✅ Done |
| 4 | Forecast — Kronos+TCN+meta-label, smoke-tested | ✅ Code done, needs training |
| 5 | Agents — OrderflowAgent/RiskAgent/StayOut work; RegimeAgent needs fit | ✅ Code done, needs training |
| 6 | Aggregator — MetaLearner + Kelly + CVaR | ✅ Code done, needs Phase 4+5 first |
| 7 | RL Execution — ExecutionEnv + CQL | ✅ Code done, needs Phase 6 first |
| 8 | Paper Trading | ❌ Not built |

**Blocker:** Only 31 days of raw data. Training needs 12 months. The data download below fixes this.

---

## Architecture

```
Feature Store (25 features, 5-min bars)
        │
   ┌────┴────────────────────────────┐
   │                                  │
ForecastAgent (Phase 4)               │
  Kronos-base [frozen, 102M params]   │
  + SmallTCN [trained, 500K params]   │
  → probabilistic 11-bin output       │
                                      │
OrderflowAgent ──┐                    │
RegimeAgent ─────┤→ Aggregator ───────┤→ DecisionEngine → SizingEngine (Kelly)
RiskAgent ───────┤   (Phase 6)        │         │
StayOutDetector ─┘   LightGBM         │         ↓
                                      │   RL Execution (Phase 7)
                                      │   CQL (fills only, not direction)
                                      │         │
                                      └─────────┴─→ Simulator / Live exchange
```

**Hard acceptance gates (do not skip):**
- Phase 4: OOS Brier < baseline AND OOS Sharpe ≥ 0.5
- Phase 6: OOS Sharpe ≥ 1.0 — THE make-or-break gate
- Phase 7: Slippage(RL) < 0.8 × Almgren-Chriss OR Sharpe(v6) ≥ Sharpe(v5) + 0.1

---

## Prerequisites

```bash
# Python 3.11+, uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone this repo
git clone <repo-url>
cd quant-hack

# Install dependencies
uv sync

# Clone Kronos source (required — loader imports from here)
git clone https://github.com/shiyu-coder/Kronos.git Kronos
```

---

## Step-by-Step Pipeline

### Step 1 — Download 12 months of historical data

Downloads all 5 data kinds (aggTrades, klines_1m, klines_5m, bookDepth, metrics)
from `data.binance.vision` (S3 archive — not geo-blocked).
Takes ~20-40 min. Safe to re-run (skips existing files).

```bash
uv run intraday data download-bulk \
  --start 2025-06-01 \
  --end   2026-05-31
```

Check what was downloaded:

```bash
uv run intraday data summary
```

---

### Step 2 — Compute features

Processes one day at a time. Carries rolling state (VPIN, Hawkes) across day boundaries.
Output: `data/features/BTCUSDT/YYYY-MM-DD.parquet` — 288 rows/day, 25 features per row.

```bash
uv run intraday features compute \
  --start 2025-06-01 \
  --end   2026-05-31
```

Verify feature store:

```bash
uv run intraday features summary
```

Expected: ~365 files, ~105,000 rows, 25 features, nulls only in VPIN warmup.

---

### Step 3 — Download Kronos model weights

Only needed once. Already done if `models/kronos-base/` and `models/kronos-tokenizer/` exist.

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('NeoQuasar/Kronos-base',          local_dir='models/kronos-base')
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='models/kronos-tokenizer')
"
```

Model sizes: Kronos-base 102M params (390 MB), tokenizer 4M params (15 MB).

---

### Step 4 — Smoke-test Phase 4 pipeline

Runs 1 batch through the full Kronos → TCN → head → loss → backward path.
Completes in ~5 seconds. Confirms everything is wired correctly before committing to training.

```bash
uv run intraday forecast train --smoke-test
```

Expected output: `[SMOKE TEST DONE] train_loss=X.XXXX  Pipeline verified — all components functional.`

---

### Step 5 — Train Phase 4 forecast model

Uses the first 9 months for training, months 10-11 for validation, month 12 held out for OOS.

```bash
uv run intraday forecast train \
  --train-end  2026-03-31 \
  --val-start  2026-04-01 \
  --val-end    2026-05-31 \
  --epochs     5 \
  --batch-size 4 \
  --device     auto
```

**Time estimates:**

| Hardware | Time/epoch | 5 epochs |
|----------|-----------|----------|
| CPU (this machine) | ~16 min | **~80 min** |
| A100 GPU | ~1.5 min | ~8 min |
| RTX 4090 | ~2 min | ~12 min |

**Resume after interruption** (checkpoints saved after every epoch):

```bash
uv run intraday forecast train \
  --train-end    2026-03-31 \
  --val-start    2026-04-01 \
  --val-end      2026-05-31 \
  --epochs       5 \
  --batch-size   4 \
  --resume-from  models/forecast/<run-timestamp>/checkpoint_epoch02.pt
```

**Gate:** Do not proceed to Step 7 unless OOS Brier < random baseline AND OOS Sharpe ≥ 0.5.

---

### Step 5b — (GPU machine) Transfer data and train

If training on a separate GPU machine:

```bash
# On this machine — copy features to GPU host
rsync -avz data/features/         gpu-host:quant-hack/data/features/
rsync -avz data/raw/binance/klines_1m/  gpu-host:quant-hack/data/raw/binance/klines_1m/
rsync -avz models/kronos-base/    gpu-host:quant-hack/models/kronos-base/
rsync -avz models/kronos-tokenizer/ gpu-host:quant-hack/models/kronos-tokenizer/
rsync -avz Kronos/                gpu-host:quant-hack/Kronos/

# On the GPU machine
uv add torch --index https://download.pytorch.org/whl/cu124   # match your CUDA version

uv run intraday forecast train \
  --train-end  2026-03-31 \
  --val-start  2026-04-01 \
  --val-end    2026-05-31 \
  --epochs     5 \
  --batch-size 32 \
  --device     cuda
```

---

### Step 6 — Train RegimeAgent (Phase 5)

Fits HMM(6 states) + LightGBM on 10 months of feature data.

```bash
uv run intraday agent train regime \
  --start 2025-06-01 \
  --end   2026-03-31
```

Note: `OrderflowAgent`, `RiskAgent`, and `StayOutDetector` are rule-based — no training needed.

---

### Step 7 — Train aggregator (Phase 6)

Trains the LightGBM meta-learner that combines all agent outputs into a final trading decision.
Requires Phase 4 and Phase 5 to be trained first.

```bash
uv run intraday train train \
  --start   2025-06-01 \
  --end     2026-03-31
```

**Hard gate:** OOS Sharpe ≥ 1.0 with full realistic costs (spread + fees + funding).
If this gate fails, debug label/feature leakage before Phase 7 — do not proceed.

---

### Step 8 — Backtest v5 (full pipeline, no RL)

Runs the complete multi-agent pipeline over the held-out month with realistic costs.

```bash
uv run intraday backtest run \
  --strategy v5_full_no_rl \
  --start    2026-04-01 \
  --end      2026-05-31 \
  --capital  10000 \
  --report
```

Compare strategies side-by-side:

```bash
uv run intraday backtest compare <run-id-1> <run-id-2>
```

---

### Step 9 — Train RL execution (Phase 7)

Only do this if Step 8 shows Sharpe ≥ 1.0. RL on a zero-alpha pipeline amplifies noise.

```bash
# Collect offline dataset (~50k episodes)
uv run intraday rl collect-data \
  --start  2026-05-20 \
  --end    2026-06-19

# Train CQL policy (200k steps)
uv run intraday rl train

# Evaluate vs Almgren-Chriss baseline
uv run intraday rl evaluate
```

**Gate:** Realized slippage < 80% of Almgren-Chriss baseline. If not met, ship v5 to paper trading without RL.

---

### Step 10 — Backtest v6 (full pipeline + RL)

```bash
uv run intraday backtest run \
  --strategy v6_full_with_rl \
  --start    2026-04-01 \
  --end      2026-05-31 \
  --report
```

---

## Live Data Capture

For paper trading you need live WebSocket data. Run this continuously in a tmux session:

```bash
tmux new -s data-capture
uv run intraday data live-capture
# Detach: Ctrl+B then D
# Reattach: tmux attach -t data-capture
```

---

## Project Layout

```
quant-hack/
├── src/intraday/
│   ├── data/              # Phase 1 — download, live capture
│   ├── features/          # Phase 2 — 25-feature calculator
│   ├── sim/               # Phase 3 — queue-aware simulator
│   │   └── strategies/    # v0_buy_hold, v1_random, v5_full_no_rl, v6_full_with_rl
│   ├── forecast/          # Phase 4 — Kronos + TCN + meta-label
│   ├── agents/            # Phase 5 — orderflow, regime, risk, stay_out
│   ├── aggregator/        # Phase 6 — MetaLearner + Kelly sizing
│   └── rl/                # Phase 7 — ExecutionEnv + CQL policy
├── Kronos/                # Cloned Kronos source (github.com/shiyu-coder/Kronos)
├── models/
│   ├── kronos-base/       # Kronos-base weights (NeoQuasar/Kronos-base)
│   ├── kronos-tokenizer/  # Tokenizer weights (NeoQuasar/Kronos-Tokenizer-base)
│   └── forecast/          # Trained TCN + head checkpoints (written by train)
├── data/
│   ├── raw/binance/       # aggTrades, klines_1m, klines_5m, bookDepth, metrics
│   └── features/BTCUSDT/  # Computed feature parquets (288 rows/day)
├── runs/                  # Backtest outputs (metrics.json, report.html)
├── idea/phases/           # Phase specs 00-10 — the contract
└── AGENTS.md              # Code style rules — non-negotiable
```

---

## Key Design Rules

- **Polars not pandas** — all feature/data code uses Polars
- **Walk-forward splits only** — purged k-fold + embargo, never random split
- **UTC always** — all timestamps in milliseconds UTC
- **No LSTM** — TCN only for sequence modelling
- **RL for execution only** — CQL fills HOW we execute, never WHAT direction
- **Structlog not print()** — all logging via structlog
- **No live $ until canary passes ≥ 4 weeks paper trading**

---

## Forecast Model Details (Phase 4)

| Component | Details |
|-----------|---------|
| Kronos backbone | NeoQuasar/Kronos-base, 102M params, d_model=832, 12 layers, frozen |
| Tokenizer | NeoQuasar/Kronos-Tokenizer-base, converts OHLCV+amount → discrete tokens |
| SmallTCN | 4 dilated causal conv layers, 64 channels, trained from scratch |
| ForecastHead | MLP: (832+64) → 256 → 128 → 11 logits |
| Labels | Triple-barrier (López de Prado ch.3), pt=1.5σ, sl=1.0σ, horizon=15min |
| Splits | Purged k-fold (López de Prado ch.7), embargo=1% |
| Trainable params | ~500K (only TCN + head; Kronos frozen) |
| Input to Kronos | 256-bar 1m OHLCV window, per-window z-normalized + clipped |
| Input to TCN | 128-bar 5m feature window (25 features) |
| Output | 11-bin probability distribution over σ-normalised forward returns |

---

## Disclaimer

Research/educational software. Cryptocurrency trading carries significant risk of loss.
Not financial advice. Never risk more than you can afford to lose.
