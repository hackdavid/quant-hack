# BTC/USDT Intraday Trading System

Probabilistic, regime-aware, multi-agent intraday trading system for BTC/USDT perpetual futures (Binance USDM).
**Target: OOS Sharpe ≥ 1.0 in paper trading before touching live money.**

---

## What this system is

```
Raw data (5 sources)
    │
    ▼
Feature Store (25 features, 5-min bars)
    │
    ├─► ForecastAgent  ──────────────────────────────────────────────────────┐
    │     Kronos-base [frozen backbone, 102M]                                │
    │     + SmallTCN  [trained head,    500K]                                │
    │     → 11-bin probability over forward returns                          │
    │                                                                        │
    ├─► OrderflowAgent (rule-based, no training)                             │
    │                                                                        ├─► Aggregator
    ├─► RegimeAgent   (HMM + LightGBM, trained)                             │   LightGBM meta-learner
    │                                                                        │   → Decision (side, size)
    ├─► RiskAgent     (rule-based, no training)                             │        │
    │                                                                        │        ▼
    └─► StayOutDetector (rule-based, no training)  ──────────────────────────┘   RL Execution
                                                                                 CQL policy
                                                                                 → HOW to fill
                                                                                      │
                                                                                      ▼
                                                                               Simulator / Exchange
```

**Models trained (in order):**
1. `forecast train` — Kronos+TCN+Head (Phase 4) → `models/forecast/`
2. `agent train regime` — HMM+LightGBM (Phase 5) → `models/regime/`
3. `train train` — Aggregator meta-learner (Phase 6) → `models/aggregator/`
4. `rl collect-data` + `rl train` — CQL execution (Phase 7) → `models/rl/`

---

## Hard acceptance gates — do not skip

| Gate | Condition | Action if failed |
|------|-----------|-----------------|
| After Phase 4 | OOS Brier < random baseline AND OOS Sharpe ≥ 0.5 | Fix forecast, re-train |
| After Phase 6 | OOS Sharpe ≥ 1.0 (full costs) | Debug leakage, re-train — **do NOT proceed to Phase 7** |
| After Phase 7 | RL slippage < 0.8× Almgren-Chriss | Ship v5 without RL instead |

---

## Current status

| Phase | Status |
|-------|--------|
| 0 Setup | ✅ Done |
| 1 Data | ✅ 31 days on this machine |
| 2 Features | ✅ 8,928 feature rows |
| 3 Simulator | ✅ Canary 0.000 bps error |
| 4 Forecast | ✅ Code + smoke-tested (needs GPU training) |
| 5 Agents | ✅ Code done (needs training) |
| 6 Aggregator | ✅ Code done (needs Phase 4+5) |
| 7 RL Execution | ✅ Code done (needs Phase 6) |
| 8 Paper Trading | ❌ Not built |

---

## Complete GPU machine run — step by step

Copy this entire section to the GPU machine and run each block in order.

### 0. Initial setup on GPU machine

```bash
# Clone the repo
git clone <your-github-repo-url> quant-hack
cd quant-hack

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv sync

# Install CUDA-enabled PyTorch (replace cu124 with your CUDA version)
uv add torch --index https://download.pytorch.org/whl/cu124

# Clone the Kronos source (required — loader imports from here)
git clone https://github.com/shiyu-coder/Kronos.git Kronos

# Confirm GPU is visible
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

---

### 1. Download Kronos model weights

```bash
uv run python -c "
from huggingface_hub import snapshot_download
print('Downloading Kronos-base (102M params)...')
snapshot_download('NeoQuasar/Kronos-base',          local_dir='models/kronos-base')
print('Downloading Kronos-Tokenizer-base...')
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='models/kronos-tokenizer')
print('Done.')
"
```

Expected: `models/kronos-base/model.safetensors` (~390 MB), `models/kronos-tokenizer/model.safetensors` (~15 MB)

---

### 2. Download 5.5 years of historical data

Downloads all 5 data kinds from `data.binance.vision` (Binance S3 archive).
Safe to re-run — skips files that already exist.

```bash
uv run intraday data download-bulk \
  --start 2020-09-10 \
  --end   2026-05-31 \
  --concurrency 16
```

**Expected:** ~2,000 days × 5 kinds = ~10,000 files. Takes 30-60 min.
`2020-09-10` is the Binance USDM BTC perpetual launch date — earliest available data.

Verify:
```bash
uv run intraday data summary
```

---

### 3. Compute features (25 features, 5-min bars)

Processes one day at a time. Rolling state (VPIN, Hawkes) carries across day boundaries.

```bash
uv run intraday features compute \
  --start 2020-09-10 \
  --end   2026-05-31
```

**Expected:** ~2,000 files in `data/features/BTCUSDT/`, ~576,000 feature rows. Takes 15-30 min.

Verify:
```bash
uv run intraday features summary
```

---

### 4. Smoke test — verify full pipeline before committing to training

Runs 1 batch through the entire stack in ~10 seconds. Always do this first.

```bash
uv run intraday forecast train \
  --smoke-test \
  --unfreeze-top-k 4 \
  --device cuda
```

Expected: `[SMOKE TEST DONE] train_loss=X.XXXX  Pipeline verified — all components functional.`

---

### 5. Train Phase 4 — Forecast model (~1.5 hours on A100)

Uses 5 years for training (2020-09 to 2025-09), 5 months for validation (2025-10 to 2026-02),
5 months held out for OOS test (never seen during training: 2026-03 to 2026-07 at backtest time).

```bash
uv run intraday forecast train \
  --train-end      2025-09-30 \
  --val-start      2025-10-01 \
  --val-end        2026-02-28 \
  --epochs         10 \
  --batch-size     32 \
  --grad-accum     4 \
  --unfreeze-top-k 4 \
  --lora-rank      16 \
  --warmup-steps   500 \
  --device         cuda \
  --log-every      50
```

**What this does:**
- `--unfreeze-top-k 4` — fine-tunes last 4 of 12 Kronos transformer layers via LoRA (213K params)
- `--grad-accum 4` — effective batch size = 32 × 4 = 128
- `--warmup-steps 500` — LR ramps up linearly for 500 steps, then cosine decays

**If interrupted**, resume from the last saved epoch checkpoint:
```bash
uv run intraday forecast train \
  --train-end      2025-09-30 \
  --val-start      2025-10-01 \
  --val-end        2026-02-28 \
  --epochs         10 \
  --batch-size     32 \
  --grad-accum     4 \
  --unfreeze-top-k 4 \
  --device         cuda \
  --resume-from    models/forecast/<timestamp>/checkpoint_epoch05.pt
```

**Output:** `models/forecast/<timestamp>/` — contains `tcn.safetensors`, `head.safetensors`, `kronos_lora.pt`, `metadata.json`, per-epoch checkpoints.

**Time estimates:**

| GPU | batch | grad_accum | eff_batch | ~time/epoch | 10 epochs |
|-----|-------|-----------|-----------|------------|-----------|
| A100 | 32 | 4 | 128 | ~10 min | **~100 min** |
| RTX 4090 | 32 | 4 | 128 | ~14 min | **~140 min** |
| RTX 3090 | 16 | 4 | 64 | ~20 min | **~200 min** |

---

### 6. Train Phase 5 — RegimeAgent (~10 min)

Fits HMM (6 hidden states) + LightGBM on the feature store. No GPU needed.

```bash
uv run intraday agent train regime \
  --start 2020-09-10 \
  --end   2025-09-30
```

**Output:** saved model path printed at end (default: `models/regime/`)

Note: `OrderflowAgent`, `RiskAgent`, `StayOutDetector` are rule-based — no training needed.

---

### 7. Train Phase 6 — Aggregator meta-learner (~5 min)

Trains LightGBM that combines all agent outputs → trading decision. Requires Phase 4 and 5 done first.

```bash
uv run intraday train train \
  --start   2020-09-10 \
  --end     2025-09-30 \
  --val-end 2026-02-28
```

**Output:** `models/aggregator/` (or path shown at end of run)

**HARD GATE:** Check the printed OOS Sharpe. If Sharpe < 1.0 with full costs, **do not proceed to Phase 7**.
Debug label/feature leakage and re-train before continuing.

---

### 8. Backtest v5 — full pipeline, no RL (~5 min)

Runs the complete multi-agent system over the held-out test period with realistic costs.

```bash
uv run intraday backtest run \
  --strategy v5_full_no_rl \
  --start    2026-03-01 \
  --end      2026-05-31 \
  --capital  10000 \
  --report
```

Review the HTML report in `runs/<run-id>/report.html`. Proceed to Phase 7 only if Sharpe ≥ 1.0.

---

### 9. Train Phase 7 — RL execution policy

**Only run if Step 8 shows Sharpe ≥ 1.0.**

#### 9a. Collect offline dataset (~30 min)

```bash
uv run intraday rl collect-data \
  2020-09-10 2025-09-30 \
  --episodes 50000
```

#### 9b. Train CQL policy (~60 min on GPU)

```bash
uv run intraday rl train \
  --data-from  2020-09-10 \
  --data-to    2025-09-30 \
  --n-steps    200000 \
  --batch-size 512 \
  --cql-alpha  2.0
```

#### 9c. Evaluate vs Almgren-Chriss baseline

```bash
uv run intraday rl evaluate \
  --start 2026-03-01 \
  --end   2026-05-31
```

**Gate:** RL slippage < 0.8 × Almgren-Chriss. If not met, skip RL and go with v5.

---

### 10. Backtest v6 — full pipeline + RL (~5 min)

```bash
uv run intraday backtest run \
  --strategy v6_full_with_rl \
  --start    2026-03-01 \
  --end      2026-05-31 \
  --capital  10000 \
  --report
```

---

## Transfer model weights back to this machine

After all training is done on the GPU machine, copy the trained weights back.

### From the GPU machine — pack everything

```bash
# On the GPU machine, inside the quant-hack directory
tar -czf trained_models_$(date +%Y%m%d).tar.gz \
  models/forecast/ \
  models/regime/ \
  models/aggregator/ \
  models/rl/ \
  models/kronos-base/ \
  models/kronos-tokenizer/
```

### SCP back to this machine

```bash
# Run this on THIS machine (replace gpu-host and paths as needed)
scp user@gpu-host:~/quant-hack/trained_models_*.tar.gz .
tar -xzf trained_models_*.tar.gz
```

### Or rsync individual directories (faster if only some changed)

```bash
# From this machine — pull each model directory
rsync -avz --progress user@gpu-host:~/quant-hack/models/forecast/    models/forecast/
rsync -avz --progress user@gpu-host:~/quant-hack/models/regime/      models/regime/
rsync -avz --progress user@gpu-host:~/quant-hack/models/aggregator/  models/aggregator/
rsync -avz --progress user@gpu-host:~/quant-hack/models/rl/          models/rl/
```

### Verify weights landed correctly

```bash
uv run intraday forecast train --smoke-test   # should load and run instantly
uv run intraday data summary                  # check local data
uv run intraday features summary              # check local features
```

---

## Data you also need on this machine (for live capture + paper trading)

The GPU machine trained on bulk historical data. For live operation you need fresh data.
Run this once you're back on this machine:

```bash
# Catch up any recent days missed during GPU training
uv run intraday data download-bulk --start 2026-06-01 --end 2026-06-21

# Recompute features for new days
uv run intraday features compute --start 2026-06-01 --end 2026-06-21

# Start live capture (in a tmux session)
tmux new -s data-capture
uv run intraday data live-capture
# Detach: Ctrl+B then D    Reattach: tmux attach -t data-capture
```

---

## Project layout

```
quant-hack/
├── src/intraday/
│   ├── data/          # Phase 1 — download, live capture
│   ├── features/      # Phase 2 — 25-feature calculator
│   ├── sim/           # Phase 3 — queue-aware simulator + strategies
│   ├── forecast/      # Phase 4 — Kronos + TCN + meta-label
│   ├── agents/        # Phase 5 — orderflow, regime, risk, stay_out
│   ├── aggregator/    # Phase 6 — MetaLearner + Kelly sizing
│   └── rl/            # Phase 7 — ExecutionEnv + CQL policy
├── Kronos/            # git clone https://github.com/shiyu-coder/Kronos.git
├── models/
│   ├── kronos-base/       # HuggingFace: NeoQuasar/Kronos-base
│   ├── kronos-tokenizer/  # HuggingFace: NeoQuasar/Kronos-Tokenizer-base
│   ├── forecast/          # Phase 4 output (tcn, head, lora, checkpoints)
│   ├── regime/            # Phase 5 output (HMM + LightGBM)
│   ├── aggregator/        # Phase 6 output (meta-learner)
│   └── rl/                # Phase 7 output (CQL policy)
├── data/
│   ├── raw/binance/       # aggTrades, klines_1m, klines_5m, bookDepth, metrics
│   └── features/BTCUSDT/  # 288 rows/day, 25 features per row
├── runs/                  # Backtest outputs (metrics.json, report.html)
├── idea/phases/           # Phase specs 00-10
└── AGENTS.md              # Code style rules
```

---

## Key design rules

- **Walk-forward only** — purged k-fold + embargo, never random split
- **RL for execution only** — CQL decides HOW to fill, not WHAT direction
- **No live money until paper trading ≥ 4 weeks with Sharpe ≥ 1.0**
- **Polars not pandas** — all data/feature code
- **UTC always** — all timestamps in milliseconds UTC

---

## Disclaimer

Research software. Cryptocurrency trading carries significant risk of loss.
Not financial advice. Never risk more than you can afford to lose.
