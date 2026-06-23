# Full Pipeline Documentation (V6)

> **Purpose**: This document explains every component, data column, and training procedure so you can run the pipeline, understand its output, and use LLM reasoning over the results. All model weights required for inference are committed to this repo (including the transformer).

---

## 1. Data Schema — Feature Columns

Every 5-minute bar is represented by **27 columns**. The pipeline needs these to build the feature window for agents.

### Raw input columns (from `data/features/BTCUSDT/YYYY-MM-DD.parquet`)

| Column | Type | Description | Used by |
|--------|------|-------------|---------|
| `bar_time_ms` | int64 | Unix timestamp in milliseconds (UTC) | All agents |
| `symbol` | str | Ticker symbol (e.g. `BTCUSDT`) | — |
| `close` | float64 | Closing price of the 5-min bar | Feature summary |
| `log_ret_1m` | float64 | Log return over the last 1-minute sub-bar | ForecastAgent |
| `log_ret_5m` | float64 | Log return over this 5-min bar | ForecastAgent |
| `log_ret_15m` | float64 | Log return over the last 15 minutes | ForecastAgent |
| `log_ret_60m` | float64 | Log return over the last 60 minutes | ForecastAgent |
| `realized_vol_30m` | float64 | Standard deviation of log returns over last 30 min | RegimeAgent, RiskAgent |
| `rsi_14` | float64 | RSI(14) computed on 1-min closes | ForecastAgent |
| `vol_5m` | float64 | Volume in quote asset (USDT) over this bar | OrderflowAgent |
| `taker_buy_ratio_5m` | float64 | Ratio of taker-buy volume to total volume | OrderflowAgent |
| `trade_count_5m` | int64 | Number of trades in this bar | OrderflowAgent |
| `avg_trade_size_5m` | float64 | Average trade size (quote volume / trade count) | OrderflowAgent |
| `depth_imbalance_1pct` | float64 | Order book depth imbalance at 1% spread | ForecastAgent |
| `vpin_50` | float64 | Volume-synchronized probability of informed trading (50-bar) | OrderflowAgent |
| `vpin_bucket_imbalance` | float64 | Buy/sell imbalance within VPIN buckets | OrderflowAgent |
| `hawkes_buy_intensity` | float64 | Hawkes process buy intensity | OrderflowAgent |
| `hawkes_sell_intensity` | float64 | Hawkes process sell intensity | OrderflowAgent |
| `hawkes_net` | float64 | Hawkes buy − sell intensity | OrderflowAgent |
| `oi_btc` | float64 | Open interest in BTC | RegimeAgent |
| `oi_change_1h` | float64 | 1-hour change in open interest | RegimeAgent |
| `ls_count_ratio` | float64 | Long/short account count ratio | RegimeAgent |
| `taker_ls_vol_ratio` | float64 | Taker buy/sell volume ratio | OrderflowAgent |
| `fwd_ret_5m` | float64 | **Target** — log return 5 minutes forward | Meta-learner training |
| `fwd_ret_15m` | float64 | **Target** — log return 15 minutes forward | — |
| `fwd_ret_60m` | float64 | **Target** — log return 60 minutes forward | — |
| `fwd_direction_5m` | int8 | **Target** — +1 if fwd_ret_5m > 0, else 0 | Meta-learner training |
| `spread_bps` | float64 | Bid-ask spread in basis points | Aggregator, RiskAgent |
| `funding_rate` | float64 | Perpetual funding rate | Aggregator, RiskAgent |

### Notes for streaming / live feed
- **Minimum window**: `ForecastAgent` requires **128 bars** (≈ 10.7 hours) of history.
- **Frequency**: All bars are **5-minute** aligned to UTC (`:00`, `:05`, `:10`, etc.).
- **Missing data**: If `depth_imbalance_1pct` is null (pre-2023), use `0.0`. The model is robust to this.

---

## 1.5 Feature Transformation — How Raw Data Becomes Model Input

The pipeline does **not** expect pre-computed features from any specific provider. It transforms raw OHLCV + order book + funding data into the 20 model features internally. You can feed data from **any exchange** (Binance, Bybit, OKX, Coinbase, etc.) as long as you provide the raw fields below.

### Required raw input (per 5-min bar)

| Raw Field | Source Example | Used to compute |
|-----------|---------------|-----------------|
| `timestamp_ms` | Any exchange timestamp | `bar_time_ms` |
| `open`, `high`, `low`, `close` | OHLC candle | `log_ret_*`, `rsi_14` |
| `volume` | Base or quote volume | `vol_5m`, `avg_trade_size_5m` |
| `taker_buy_volume` | Aggressive buy volume | `taker_buy_ratio_5m` |
| `trade_count` | Number of trades | `trade_count_5m`, `avg_trade_size_5m` |
| `bid_depth_1pct`, `ask_depth_1pct` | Order book depth at 1% | `depth_imbalance_1pct` |
| `funding_rate` | Perpetual funding | `funding_rate` |
| `open_interest` | OI in contracts | `oi_btc`, `oi_change_1h` |
| `long_short_ratio` | Account or volume ratio | `ls_count_ratio`, `taker_ls_vol_ratio` |

### Computed features (formulas)

| Feature | Formula | Rolling Window |
|---------|---------|----------------|
| `log_ret_1m` | `ln(close_t / close_{t-1})` | 1 bar (if 1m sub-bars) |
| `log_ret_5m` | `ln(close_t / close_{t-1})` | 1 bar (5-min) |
| `log_ret_15m` | `mean(log_ret_1m over 3 bars)` | 3 bars |
| `log_ret_60m` | `mean(log_ret_1m over 12 bars)` | 12 bars |
| `realized_vol_30m` | `std(log_ret_1m over 6 bars)` | 6 bars |
| `rsi_14` | Standard RSI on 1-min closes | 14 bars |
| `depth_imbalance_1pct` | `(bid_depth - ask_depth) / (bid_depth + ask_depth)` | Current bar only |
| `vpin_50` | See `intraday/features/vpin.py` | 50-volume buckets |
| `hawkes_buy/sell_intensity` | See `intraday/features/hawkes.py` | Exponential decay kernel |
| `oi_change_1h` | `(oi_t - oi_{t-12}) / oi_{t-12}` | 12 bars |

### Provider swap guide

To use a different data provider (e.g., Bybit, OKX, Coinbase, or your own websocket):

1. **Implement a feature adapter** that converts provider-specific fields to the raw schema above:
```python
# src/intraday/adapters/bybit_adapter.py
import polars as pl

def bybit_bars_to_features(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Convert Bybit 5-min klines to our feature schema."""
    return raw_df.with_columns([
        pl.col("volume").alias("vol_5m"),
        (pl.col("turnover") / pl.col("volume")).alias("avg_trade_size_5m"),
        # ... map remaining fields
    ])
```

2. **Replace the data loader** in `run_pipeline_json.py`:
```python
# Instead of:
# df = pl.read_parquet("data/features/BTCUSDT/2026-01-01.parquet")

# Use:
# raw_df = fetch_bybit_klines("BTCUSDT", "2026-01-01")
# df = bybit_bars_to_features(raw_df)
```

3. **The rest of the pipeline is provider-agnostic**. All agents only consume the 20 standardized features.

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAW BAR DATA (5-min)                                 │
│  bar_time_ms, close, log_ret_*, vol_5m, taker_buy_ratio_5m, ...             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE WINDOW (last 128 bars)                            │
│  Polars DataFrame sorted by bar_time_ms                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│ ForecastAgent │         │OrderflowAgent │         │ RegimeAgent   │
│ (Transformer) │         │ (rule-based)  │         │ (HMM + LGB)   │
│               │         │               │         │               │
│ Input: 128-bar│         │ Input: 128-bar│         │ Input: 128-bar│
│ window of     │         │ window of     │         │ window of     │
│ price/vol/    │         │ volume, taker │         │ returns, vol, │
│ depth features│         │ ratio, vpin,  │         │ OI, LS ratios │
│               │         │ hawkes        │         │               │
│ Output:       │         │ Output:       │         │ Output:       │
│ p_up, p_down, │         │ flow_bias,    │         │ regime,       │
│ confidence,   │         │ flow_strength,│         │ vol_regime,   │
│ meta_act      │         │ step_away,    │         │ is_transition │
│               │         │ vpin          │         │               │
└───────────────┘         └───────────────┘         └───────────────┘
        │                             │                             │
        │         ┌───────────────────┘                             │
        │         │                                                   │
        │         ▼                                                   │
        │   ┌───────────────┐                                   ┌───────────────┐
        │   │   RiskAgent   │                                   │ StayOutDetector│
        │   │ (rule-based)  │                                   │ (rule-based)   │
        │   │               │                                   │                │
        │   │ Input: vol,   │                                   │ Input: vol,    │
        │   │ spread, fund, │                                   │ funding,       │
        │   │ position,     │                                   │ spread, bar    │
        │   │ equity        │                                   │ time           │
        │   │               │                                   │                │
        │   │ Output:       │                                   │ Output: mode,  │
        │   │ allow_trade,  │                                   │ score          │
        │   │ risk_mult,    │                                   │                │
        │   │ stop_trading  │                                   │                │
        │   └───────────────┘                                   └───────────────┘
        │         │                                                   │
        └─────────┼───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGGREGATOR                                      │
│                                                                              │
│  Combines all 5 agent outputs into a single flat row (22 features)         │
│                                                                              │
│  Key fields:                                                                 │
│    fc_p_up, fc_confidence, of_flow_bias, rg_regime, rk_allow_trade, ...     │
│                                                                              │
│  → Passed to MetaLearner (LightGBM ensemble)                                 │
│    Output: p_correct (probability this trade will be profitable)             │
│                                                                              │
│  → Passed to DecisionEngine                                                   │
│    Gates: stop_trading? → flat                                               │
│           stay_out?      → flat                                               │
│           meta_act=False? → flat                                              │
│           p_correct < threshold? → flat                                         │
│    Otherwise: side = long if p_up > p_down else short                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECISION (side, confidence, horizon)                  │
│  side ∈ {long, short, flat}                                                   │
│  confidence = p_correct (0–1)                                                 │
│  horizon = 15 minutes (default)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RL EXECUTION (CQL offline policy)                         │
│                                                                              │
│  If Decision = long/short:                                                   │
│    State = aggregator feature vector (15 dims)                               │
│    Action = [size, aggressiveness, hold_time, stop_pct]                     │
│                                                                              │
│  size: 0 = flat, 1 = full position                                           │
│  aggressiveness: + = market order, − = passive limit                          │
│  hold_time: minutes to hold before force-close                                │
│  stop_pct: stop-loss distance                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORDER EXECUTION (Exchange API)                        │
│  Paper trading: simulated fills with slippage + fees                         │
│  Live trading: Binance USDT-M futures via ccxt                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. How to Run the Full Pipeline

### 3.1 One-shot JSON output (single bar)

```bash
uv run python run_pipeline_json.py \
    --transformer-run models/transformer/20260623T132957Z \
    --date 2026-01-01 \
    --bar-index -1 \
    --output -
```

**Output**: JSON with all agent outputs + English explanations.

### 3.2 Process all bars in a day

```bash
uv run python run_pipeline_json.py \
    --transformer-run models/transformer/20260623T132957Z \
    --date 2026-01-01 \
    --bar-index -2 \
    --output day_output.json
```

### 3.3 From a specific parquet file

```bash
uv run python run_pipeline_json.py \
    --transformer-run models/transformer/20260623T132957Z \
    --features-file data/features/BTCUSDT/2026-01-01.parquet \
    --bar-index -1 \
    --output -
```

### 3.4 Paper trading (live WebSocket)

```bash
# Set your Binance API key (read-only is fine for paper)
export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."

# Run paper trading loop
uv run python scripts/run_paper_trade.py \
    --transformer-run models/transformer/20260623T132957Z \
    --threshold 0.55 \
    --capital 10000
```

All decisions are logged to `logs/trader/trade_log_*.jsonl`.

---

## 4. Agent Details — What Each Takes, Returns, and Means

### 4.1 ForecastAgent (Transformer)

**What it is**: A Kronos-base transformer with LoRA adapters + SmallTCN + binary forecast head.

**Training**:
```bash
uv run intraday forecast train \
    --train-end 2025-03-31 --val-start 2025-04-01 --val-end 2025-09-30 \
    --epochs 30 --batch-size 512 --grad-accum 4 \
    --unfreeze-top-k 8 --lora-rank 32 --lora-alpha 64 \
    --device cuda
```
- **Input**: 128-bar window of 20 features (price, volume, depth, OI, etc.)
- **Output**: `p_up` (probability of up move in next 15 min), `p_down = 1 - p_up`
- **Loss**: Binary cross-entropy with no label smoothing
- **Metrics**: Val AUC, Brier score, ECE
- **Saved to**: `models/transformer/<timestamp>/best.pt`

**Output meaning**:
| Field | Range | Meaning |
|-------|-------|---------|
| `p_up` | 0–1 | Probability of upward move in next 15 min |
| `confidence` | 0–1 | `abs(p_up - 0.5) * 2` — how far from random |
| `meta_act` | bool | `True` if confidence > 0.04 (i.e., p_up > 0.52 or < 0.48) |
| `expected_move_sigma` | float | `(p_up - 0.5) * 2` — directional strength in sigma |

---

### 4.2 OrderflowAgent

**What it is**: Rule-based agent that reads volume, taker ratio, VPIN, and Hawkes intensity.

**Training**: None — purely rule-based.

**Input**: `feat_window` (last 128 bars)  
**Output** (`AgentOpinion`):
| Field | Meaning |
|-------|---------|
| `flow_bias` | +1 = strong buy flow, −1 = strong sell flow, 0 = neutral |
| `flow_strength` | 0–1 intensity of the flow |
| `step_away` | `True` if VPIN > 0.6 (toxic flow — avoid tight spreads) |
| `vpin` | Volume-synchronized probability of informed trading (0–1) |

**Explanation**:
- `flow_bias = 1.0` with `vpin = 0.3` → strong buy flow, low toxicity, normal execution
- `flow_bias = -0.5` with `vpin = 0.8` → sell flow, high toxicity, **step away**

---

### 4.3 RegimeAgent

**What it is**: Hidden Markov Model (HMM) for regime detection + LightGBM classifier.

**Training**:
```bash
# Fit on features in date range
uv run python run_full_pipeline.py \
    --transformer-run models/transformer/20260623T132957Z \
    --start 2026-01-01 --end 2026-05-31 \
    --fit-regime
```
- HMM: `covariance_type="diag"`, 3 hidden states → `bull`, `bear`, `range`
- LightGBM: trained on HMM state + 5 engineered features

**Input**: `feat_window` (last 128 bars)  
**Output** (`AgentOpinion`):
| Field | Meaning |
|-------|---------|
| `regime` | `bull`, `bear`, `range`, or `unknown` |
| `regime_probs` | Dict of probabilities per regime |
| `max_prob` | Highest regime probability |
| `is_transition` | `True` if HMM state changed in last 3 bars |
| `vol_regime` | `low`, `normal`, `high` based on realized vol |

**Explanation**:
- `regime = "bull"`, `is_transition = False` → stable uptrend, trade with trend
- `regime = "range"`, `is_transition = True` → choppy / turning, reduce size or stay flat

**Saved to**: `data/models/regime.pkl`

---

### 4.4 RiskAgent

**What it is**: Rule-based risk gate that limits sizing based on volatility, spread, and funding.

**Training**: None — purely rule-based.

**Input**: `feat_window` (last 128 bars) + current equity (if provided)  
**Output** (`AgentOpinion`):
| Field | Meaning |
|-------|---------|
| `risk_multiplier` | 0–1 scaling factor for position size |
| `allow_trade` | `False` if any hard limit breached |
| `stop_trading` | `True` if daily loss limit hit (requires equity state) |

**Rules**:
| Condition | Multiplier | Action |
|-----------|-----------|--------|
| `realized_vol_30m < 0.005` | 0.35 | Low vol = low conviction, reduce size |
| `spread_bps > 5.0` | 0.50 | Wide spread, half size |
| `funding_rate > 0.001` | 0.70 | High funding, reduce size |
| `vol < 0.005 AND spread > 5.0` | 0.35 | Both bad, aggressive reduction |

---

### 4.5 StayOutDetector

**What it is**: Rule-based filter that blocks trading during specific conditions.

**Training**: None — purely rule-based.

**Input**: `feat_window` (last 128 bars)  
**Output** (`AgentOpinion`):
| Field | Meaning |
|-------|---------|
| `mode` | `normal` or `stay_out` |
| `score` | 0–1 severity score |

**Rules**:
- `mode = "stay_out"` if:
  - `funding_rate > 0.001` (expensive to hold)
  - `spread_bps > 8.0` (too expensive to enter)
  - First 4 bars of UTC day (low liquidity, settlement noise)

---

## 5. Meta-Learner & Decision Engine

### 5.1 MetaLearner (LightGBM ensemble)

**What it is**: A 4-fold ensemble of LightGBM classifiers that learns to predict whether a trade will be profitable.

**Training**:
```bash
# Automatically trained by run_full_pipeline.py
uv run python run_full_pipeline.py \
    --transformer-run models/transformer/20260623T132957Z \
    --start 2026-01-01 --end 2026-05-31 \
    --train-meta
```

**Label**: `1` if `fwd_ret_5m > 0` (5-min forward return was up), else `0`.

**Features**: 22 columns from the aggregator row (forecast + orderflow + regime + risk + stay_out + market context).

**Output**: `p_correct` — probability the trade will be profitable.

**Metrics**:
| Metric | Description | Target |
|--------|-------------|--------|
| AUC | Area under ROC curve | > 0.55 |
| Brier | Mean squared error of probabilities | < 0.25 |
| ECE | Expected calibration error | < 0.10 |

**Calibration**: Threshold is calibrated on the validation set (not fixed at 0.5). Typical threshold: 0.30–0.35.

**Saved to**: `data/models/aggregator/meta_learner.pkl`

### 5.2 DecisionEngine

**Logic** (5 gates, in order):

1. `rk_stop_trading = True` → **FLAT** (reason: risk halt)
2. `so_mode = "stay_out"` → **FLAT** (reason: stay out)
3. `fc_meta_act = False` → **FLAT** (reason: forecast not confident)
4. `p_correct < threshold` (e.g., 0.30) → **FLAT** (reason: meta-learner unsure)
5. Otherwise → **LONG** if `p_up > p_down`, else **SHORT**

**Output** (`Decision`):
| Field | Meaning |
|-------|---------|
| `side` | `long`, `short`, or `flat` |
| `confidence` | `p_correct` from meta-learner |
| `horizon_minutes` | How long to hold (default 15) |
| `reason` | Why the decision was made |

---

## 6. RL Execution (CQL Policy)

### 6.1 What it is

An offline-trained Conservative Q-Learning (CQL) policy that decides **HOW** to execute a trade, not **WHETHER** to trade.

### 6.2 Training

```bash
# 1. Build RL dataset from Almgren-Chriss baseline + perturbations
uv run python run_full_pipeline.py \
    --transformer-run models/transformer/20260623T132957Z \
    --start 2026-01-01 --end 2026-05-31 \
    --save-decisions

# 2. Train CQL policy
uv run intraday rl train \
    --dataset data/rl_dataset.parquet \
    --output-dir data/models/rl/cql_v1 \
    --n-steps 200000 \
    --device cuda
```

**Dataset**: `(state, action, reward, next_state, done)`
- `state`: 15-dim aggregator feature vector
- `action`: 4-dim continuous `[size, aggressiveness, hold_time, stop_pct]`
- `reward`: Implementation shortfall vs. Almgren-Chriss baseline

**Algorithm**: CQL (offline RL) — prevents overestimation of Q-values from out-of-distribution actions.

**Hyperparameters**:
| Param | Value | Description |
|-------|-------|-------------|
| `n_steps` | 200,000 | Total gradient steps |
| `batch_size` | 256 | Mini-batch size |
| `actor_lr` | 1e-4 | Actor learning rate |
| `critic_lr` | 3e-4 | Critic learning rate |
| `cql_alpha` | 2.0 | Conservative penalty weight |
| `seed` | 42 | Reproducibility |

**Saved to**: `data/models/rl/cql_v1/cql_policy/cql.d3`

### 6.3 Output

| Action dim | Range | Meaning |
|------------|-------|---------|
| `size` | −1 to +1 | −1 = full short, 0 = flat, +1 = full long |
| `aggressiveness` | −1 to +1 | +1 = market order, −1 = passive limit |
| `hold_time` | 0 to 15 | Minutes to hold before force-close |
| `stop_pct` | 0 to 0.05 | Stop-loss distance as fraction of price |

---

## 7. LLM Reasoning Interface

The `run_pipeline_json.py` CLI outputs structured JSON that an LLM can reason over. Each block has an `explanation` field in plain English.

### 7.1 Example: LLM prompt for decision review

```
You are a trading risk officer. Review this pipeline output:

{json_output}

Questions:
1. Is the forecast confident enough to trade?
2. Does the orderflow agree with the forecast direction?
3. Is the regime stable?
4. Are risk conditions acceptable?
5. What is the final decision and should we override it?

Answer in 3 sentences. Suggest a position size (0–1).
```

### 7.2 Example: Tool-calling for order execution

The LLM can call the order execution as a tool with arguments derived from the pipeline:

```json
{
  "tool": "place_order",
  "arguments": {
    "side": "long",
    "size": 0.35,
    "order_type": "limit",
    "aggressiveness": 0.2,
    "hold_time": 15,
    "stop_pct": 0.02
  }
}
```

---

## 8. Files Committed to Git

All files required to run inference on a fresh clone are in this repo:

| Path | What | Size |
|------|------|------|
| `data/models/regime.pkl` | Trained HMM + LightGBM regime agent | ~50 KB |
| `data/models/aggregator/meta_learner.pkl` | Trained LightGBM meta-learner | ~200 KB |
| `data/models/rl/cql_v1/cql_policy/cql.d3` | Trained CQL policy | ~2 MB |
| `src/intraday/` | All source code | — |
| `docs/PIPELINE.md` | This documentation | — |
| `run_pipeline_json.py` | CLI tool | — |
| `run_full_pipeline.py` | End-to-end training/inference | — |

**Not in git** (downloaded at runtime):
| Path | Source | Size |
|------|--------|------|
| `models/transformer/*/best.pt` | HuggingFace `checkpoints/transformer_v2/` | ~25 MB |
| `models/kronos-base/` | HuggingFace `NeoQuasar/Kronos-base` | ~400 MB |
| `data/features/BTCUSDT/` | HuggingFace dataset | ~120 MB |

---

## 9. Quick Reference

### Command cheat sheet

```bash
# Run pipeline on one bar and print JSON
uv run python run_pipeline_json.py \
    --transformer-run models/transformer/20260623T132957Z \
    --date 2026-01-01 --bar-index -1 --output -

# Run full pipeline (fit regime, train meta, save decisions)
uv run python run_full_pipeline.py \
    --transformer-run models/transformer/20260623T132957Z \
    --start 2026-01-01 --end 2026-05-31

# Train RL policy
uv run intraday rl train \
    --dataset data/rl_dataset.parquet \
    --output-dir data/models/rl/cql_v2 \
    --n-steps 200000 --device cuda

# Paper trading
uv run python scripts/run_paper_trade.py \
    --transformer-run models/transformer/20260623T132957Z \
    --threshold 0.55 --capital 10000
```

### Minimum data to run inference

```python
# One bar
{
    "bar_time_ms": 1704067200000,
    "close": 42000.0,
    "log_ret_1m": 0.0001,
    "log_ret_5m": 0.0005,
    "log_ret_15m": 0.001,
    "log_ret_60m": 0.003,
    "realized_vol_30m": 0.002,
    "rsi_14": 55.0,
    "vol_5m": 5000000.0,
    "taker_buy_ratio_5m": 0.52,
    "trade_count_5m": 1200,
    "avg_trade_size_5m": 4166.0,
    "depth_imbalance_1pct": 0.1,
    "vpin_50": 0.3,
    "vpin_bucket_imbalance": 0.05,
    "hawkes_buy_intensity": 0.2,
    "hawkes_sell_intensity": 0.15,
    "hawkes_net": 0.05,
    "oi_btc": 100000.0,
    "oi_change_1h": 0.01,
    "ls_count_ratio": 1.2,
    "taker_ls_vol_ratio": 0.9,
    "spread_bps": 1.0,
    "funding_rate": 0.0001,
}
```

---

*Last updated: 2026-06-23*
