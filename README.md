# Kronos LLM Master Trader — Multi-Agent AI Trading System

> **A production-grade, multi-agent AI trading system for BTC/USDT perpetual futures.**
> Combines the Kronos foundation model (12B candlesticks, 45+ exchanges), real-time technical analysis, and Claude LLM for intelligent trade execution via MetaTrader 5.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Primary Model** | Kronos (102M params, 12B candlesticks pre-trained) |
| **LLM Decision Engine** | Claude Sonnet 4.5 (via Symphony AI) |
| **Execution Venue** | MetaTrader 5 (BTCUSDT → BTCUSD mapping) |
| **Data Feed** | Binance Vision API (1m + 5m klines) |
| **Target Asset** | BTC/USDT perpetual futures |
| **Agents** | 6 specialized agents + 1 orchestrator |
| **Latency** | ~7s per prediction cycle (Kronos inference) |
| **Win Rate (Agree-Only)** | 54.5% when Kronos + Trend align |
| **Avg P&L per Trade** | $200–$800 |
| **Risk per Trade** | Max $400 SL / $200 TP |
| **Competition Score** | 9.95 (Target: 75-80 for top 5) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KRONOS LLM MASTER TRADER                         │
│                         Multi-Agent Trading System                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌──────────┐              ┌──────────────┐              ┌────────────┐
│  DATA    │              │   KRONOS     │              │   LLM      │
│  LAYER   │              │   PREDICTOR  │              │   DECISION │
│          │              │              │              │   ENGINE   │
└──────────┘              └──────────────┘              └────────────┘
    │                          │                              │
    │                          │                              │
    ▼                          ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      COUNCIL OF AGENTS                                  │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐  │
│  │ Kronos  │ │ Technical│ │  Trend  │ │ Volume  │ │  Conflict   │  │
│  │Predictor│ │ Analyst  │ │ Detector│ │Profiler │ │  Analyzer   │  │
│  │(102M)   │ │(RSI/MACD)│ │(EMA)    │ │(OrderFlow)│ │(Meta-logic)│  │
│  └─────────┘ └──────────┘ └─────────┘ └─────────┘ └─────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    ORCHESTRATOR / LLM                              │ │
│  │         Claude Sonnet 4.5 — Final Decision Maker                 │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Risk Guard  │───▶│  MT5 Wrapper │───▶│  Position Monitor        │  │
│  │  (SL/TP/Trail)│    │  (Order Mgmt) │    │  (Profit/Loss Tracking) │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Binance Vision API (5m klines)
    │
    ▼
┌──────────────────────────────────┐
│  60-Candle Context Window        │
│  (5 hours of 5m data)            │
└──────────────────────────────────┘
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
┌─────────────┐                      ┌─────────────────────┐
│ Kronos Model │                      │ Technical Indicators │
│ (102M params)│                      │ RSI, MACD, Bollinger │
│ 12B candles │                      │ EMA 5/10/20         │
│ 45 exchanges│                      │ Volume Profile       │
└─────────────┘                      └─────────────────────┘
    │                                          │
    │                                          │
    ▼                                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    COUNCIL VOTING                             │
│  Kronos: bear (95%)  │  Trend: bull (50%)  │  Conflict!    │
│  ┌────────────────────────────────────────────────────────┐│
│  │  Conflict Analyzer:                                     ││
│  │  • Kronos has higher confidence BUT                    ││
│  │  • Historically less reliable in conflicts (25% vs 35%)││
│  │  • RECOMMENDATION: WAIT (preserve capital)             ││
│  └────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  LLM DECISION (Claude Sonnet 4.5)                             │
│  "Agents are in conflict. Kronos bear (95%) vs Trend bull.    │
│   Kronos is historically unreliable in conflicts.             │
│   Technical indicators mildly bullish. Volume neutral.       │
│   DECISION: WAIT — preserve capital until clarity."           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  EXECUTION                                                    │
│  No trade placed. Wait for next cycle (5 minutes).            │
└──────────────────────────────────────────────────────────────┘
```

---

## Agent Specifications

### 1. Kronos Predictor (The Oracle)
| Property | Value |
|----------|-------|
| **Model** | Kronos-small (102M parameters, 12 layers, 16 heads) |
| **Training Data** | 12 billion candlesticks across 45+ global exchanges |
| **Input** | OHLCV (Open, High, Low, Close, Volume) |
| **Output** | Predicted next 12 candlesticks (1 hour forward) |
| **Tokenizer** | Binary Spherical Quantization (BSQuantizer) |
| **Specialization** | First open-source foundation model for financial candlesticks |
| **Confidence Range** | 0% – 95% |
| **Inference Time** | ~7s per prediction on CPU |

**Architecture:**
```
OHLCV Input
    │
    ▼
┌─────────────────┐
│ KronosTokenizer │  ← Binary Spherical Quantization (BSQuantizer)
│   (d_in=5,      │    s1_bits=10, s2_bits=10
│    d_model=832) │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    Kronos       │  ← 12-layer Transformer, 16 heads
│  (102M params)  │    + DualHead output
└─────────────────┘
    │
    ▼
Predicted OHLCV (12 candles)
```

### 2. Technical Analyst (The Scientist)
| Indicator | Window | Purpose | Signal |
|-----------|--------|---------|--------|
| RSI | 14 | Overbought (>70) / Oversold (<30) | Entry timing |
| Stoch RSI | 14 | More sensitive than RSI for crypto | Better entry/exit |
| MACD | 12/26 | Momentum direction | Trend confirmation |
| Bollinger Bands | 20 | Mean reversion | Position < 0.1 = low, > 0.9 = high |
| ADX | 14 | Trend strength | >25 = strong trend, <20 = weak/choppy |
| Supertrend | 10 | Trend following | bull/bear/neutral |
| VWAP | Session | Volume-weighted price | Deviation >1% = entry signal |
| ATR | 14 | Volatility measurement | Stop loss positioning |

### 3. Trend Detector (The Navigator)
| EMA | Period | Signal |
|-----|--------|--------|
| EMA5 | 5 | Short-term direction |
| EMA10 | 10 | Medium-term direction |
| EMA20 | 20 | Long-term direction |

**Trend Classification:**
- `bull`: EMA5 > EMA10 > EMA20 + Higher Highs
- `bear`: EMA5 < EMA10 < EMA20 + Lower Lows
- `ranging`: Mixed signals

### 4. Volume Profiler (The Listener)
| Metric | Threshold | Signal |
|--------|-----------|--------|
| Taker Buy Ratio | > 60% | Strong buying pressure |
| Buy Pressure | > 60% | Volume-weighted bullish |
| Volume Trend | > 1.2x | Increasing activity |

### 5. Conflict Analyzer (The Diplomat)
**Logic:**
```python
if kronos_dir == trend_dir:
    relationship = "AGREE"  # 54.5% historical accuracy
    trust = "high"
elif kronos_dir == "neutral" or trend_dir == "ranging":
    relationship = "NEUTRAL"
    trust = "low"
else:
    relationship = "CONFLICT"  # Trend wins 35.7% vs Kronos 25%
    trust = "medium"
```

### 6. LLM Decision Maker (The Commander)
| Property | Value |
|----------|-------|
| **Model** | Claude Sonnet 4.5 (via Symphony AI) |
| **API** | Read from `.env` (`LLM_BASE_URL`) |
| **Temperature** | 0.2 (low creativity, high precision) |
| **Max Tokens** | 100 (concise decisions) |
| **Context** | Full agent analysis + conflict status + market data + **live competition metrics** |
| **Output Format** | Single word: BUY, SELL, WAIT, CLOSE, or HOLD |
| **Competition Mode** | Tracks live P&L, Win Rate, Sharpe from MT5 |

---

## Decision Matrix

| Kronos | Trend | Conflict | LLM Decision | Historical Accuracy |
|--------|-------|----------|------------|---------------------|
| bull (95%) | bull (50%) | **AGREE** | **BUY** | 54.5% |
| bear (95%) | bear (50%) | **AGREE** | **SELL** | 54.5% |
| bear (95%) | bull (50%) | **CONFLICT** | WAIT or follow higher confidence | 25–35% |
| neutral | ranging | **NEUTRAL** | **WAIT** | N/A |
| bull (95%) | ranging | **NEUTRAL** | BUY (if Kronos > 70%) | 41% |

---

## Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Lot Size** | 8.0 | Fixed position size |
| **Take Profit (TP)** | $200 | Hard profit target |
| **Stop Loss (SL)** | $400 | Max loss per trade |
| **Trailing Stop** | $100 | Close if profit drops $100 from peak |
| **Trailing Activate** | $150 | Trailing stop activates after $150 profit |
| **Max Hold Time** | 600s (10 min) | Force close after 10 minutes |
| **Entry Interval** | 300s (5 min) | Check for new trades every 5 minutes |
| **Monitor Interval** | 60s (1 min) | Check open positions every 1 minute |

---

## Backtest Results

### Conflict Analysis (100 Candles, Walk-Forward)

| Metric | Value |
|--------|-------|
| **Total Analyzed** | 100 candles (2 days of 5m data) |
| **Agreement Rate** | 11% |
| **Conflict Rate** | 28% |
| **Neutral Rate** | 61% |
| **Kronos Accuracy** | 41% |
| **Trend Accuracy** | 41% |
| **Accuracy When Agree** | 54.5% |
| **Accuracy When Conflict** | Kronos 25%, Trend 35.7% |

### Trading Strategies

| Strategy | Trades | Win Rate | P&L |
|----------|--------|----------|-----|
| Trade when AGREE | 11 | 54.5% | **+$3,053** ✅ |
| Trade Kronos only | 66 | 50% | -$1,995 ❌ |
| Kronos high confidence | 66 | 50% | -$1,995 ❌ |

**Conclusion:** The **AGREE-ONLY strategy** is the only profitable approach.

---

## Competition Score Calculator

Real-time score from your MT5 account:

```bash
# Check current score
python scripts/mt5_competition_score.py

# Check today's score only
python scripts/mt5_competition_score.py --today-only

# Specific date range
python scripts/mt5_competition_score.py --from-date 2026-06-20 --to-date 2026-06-25
```

**Output:**
```
========================================
Final Score: 9.95
Win Rate:    40.7%
Sharpe:      -0.1763
P&L:         $-4,480.15
========================================
Target: 75-80 for top 5
========================================
```

The trader uses this script internally to feed live metrics to the LLM.

---

## Competition Strategy

### LLM Prompt Context
The LLM now sees **live competition metrics** every cycle:

| Metric | Current | Target | Weight |
|--------|---------|--------|--------|
| **Final Score** | 9.95 | 75-80 | 100% |
| **Win Rate** | 40.7% | >55% | 70% Return |
| **Sharpe** | -0.18 | >0.5 | 10% Sharpe |
| **P&L** | -$4,480 | +$30K to +$50K | 70% Return |
| **Trades** | 150 | >30 | 5% Risk |

### Strategy Rules
1. **DO NOT WAIT** — if signals are 60%+ aligned, take the trade even if small
2. **AGREE** — trade when Kronos + Trend + Supertrend all agree (highest accuracy)
3. **CONFLICT** — follow Trend only if ADX>25 (strong trend) + Supertrend confirms
4. **NEUTRAL** — if StochRSI extreme (<20/>80) OR VWAP dev >1% OR volume >60%, take trade
5. **FILTER** — avoid trades when ADX<20 (choppy market). Strong trend = ADX>25
6. **Small wins > no trades** — target 3-5 trades per hour
7. **Cut losses fast** — SL=$400, TP=$200, close if loss >$200
8. **Lock in small wins** — close if profit >$50 and time >5min

---

## Live Trading Setup

### Prerequisites

```bash
# 1. Clone the repo
git clone <repo-url> quanthack
cd quanthack

# 2. Install dependencies
pip install openai transformers torch polars httpx structlog

# 3. Clone Kronos model (required for Kronos integration)
git clone https://github.com/shiyu-coder/Kronos.git kronos_module

# 4. Set environment variables (copy .env.example to .env and fill in your values)
set LLM_TOKEN=your_llm_token_here
set LLM_BASE_URL=https://api.fireworks.ai/inference/v1
set LLM_MODEL=accounts/fireworks/routers/kimi-k2p6-turbo

# 5. Test MT5 connection
python scripts/test_mt5.py --account YOUR_ACCOUNT --password "YOUR_PASSWORD" --server "YOUR_SERVER"
```

### Autonomous Trading (Infinite Loop)

```bash
# Start the autonomous trader (runs forever)
python scripts/kronos_llm_master_trader.py \
    --mt5-account YOUR_ACCOUNT \
    --mt5-password "YOUR_PASSWORD" \
    --mt5-server "YOUR_SERVER"
```

**What it does:**
1. Runs in an infinite loop
2. **No position:** Checks for entry every 5 minutes
3. **Position open:** Monitors every 30 seconds via LLM
4. Uses dynamic position sizing (4-20 lots based on signal strength)
5. No trailing stops — LLM watches trend direction
6. Max hold 15 minutes with time-based forced close
7. Logs everything to console

### Trade Monitor (Separate Process)

The monitor communicates with the trader via **SQLite database** (no filesystem locking issues).

```bash
# Open a new terminal and run the monitor

# 1. Check current status
python scripts/trade_state.py --status

# 2. Close all positions immediately
python scripts/trade_state.py --close-all

# 3. Update take profit (e.g., to $5000)
python scripts/trade_state.py --update-tp 5000

# 4. Update stop loss
python scripts/trade_state.py --update-sl 600

# 5. Pause trading (no new entries)
python scripts/trade_state.py --pause

# 6. Resume trading
python scripts/trade_state.py --resume
```

**Monitor Output:**
```
============================================================
  TRADE MONITOR STATUS
============================================================

  Running:        True
  Has Position:   True
  Side:           long
  Lots:           16.0
  Current P&L:    $1,234.00
  TP:             $2,000.00
  SL:             $800.00
  Hold:           420s / 900s
  Signal:         VERY_STRONG (72/100)

  Competition:
    Total P&L:    -$2,100.00
    Win Rate:     42.5%
    Sharpe:       -0.1200
    Final Score:  9.95

  Last Updated: 2026-06-25T14:30:00
============================================================
```

### Architecture

```
┌─────────────────────────────────────────┐
│         Main Trader Process              │
│   (Infinite loop, 5min entry / 30s     │
│    monitor, dynamic position sizing)    │
│                                         │
│   ┌──────────────┐    ┌──────────────┐  │
│   │  Entry Check │    │   Monitor    │  │
│   │  (5 min)     │    │  (30 sec)    │  │
│   └──────┬───────┘    └──────┬───────┘  │
│          │                   │           │
│          ▼                   ▼           │
│   ┌─────────────────────────────────┐  │
│   │      SQLite Database            │  │
│   │   (data/trade_state.db)         │  │
│   └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    │
                    │
┌─────────────────────────────────────────┐
│         Monitor Process                 │
│   (Separate terminal, anytime)          │
│                                         │
│   Commands: --close-all                 │
│             --update-tp 5000            │
│             --update-sl 600             │
│             --pause / --resume          │
│             --status                    │
└─────────────────────────────────────────┘
```

**Why SQLite?**
- Built into Python (no extra dependencies)
- ACID transactions (no data corruption)
- WAL mode (reader doesn't block writer)
- Much more reliable than file-based locking

---

## Model Checkpoints

| Model | Location | Size | Purpose |
|-------|----------|------|---------|
| Kronos Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` (HF) | ~50MB | OHLCV quantization |
| Kronos Small | `NeoQuasar/Kronos-small` (HF) | ~200MB | 12-layer Transformer |
| CryptoTransformer | `models/transformer/20260623T132957Z/best.pt` | 74MB | Local forecast model |
| Regime Agent | `data/models/regime.pkl` | 2MB | HMM + LightGBM |
| Meta-Learner | `data/models/aggregator/meta_learner.pkl` | 1MB | 4-fold ensemble |
| RL Policy | `data/models/rl/cql_v1/cql_policy/cql.d3` | 5MB | CQL offline policy |

---

## Project Structure

```
quanthack/
├── scripts/
│   ├── kronos_llm_master_trader.py      # Main trading bot (Kronos + LLM)
│   ├── kronos_real_predictor.py        # Kronos model test script
│   ├── kronos_conflict_analysis.py     # Walk-forward backtest
│   ├── kronos_full_trader.py           # Full system (5 agents)
│   ├── kronos_live_trader.py           # Live trading with fallback
│   ├── autonomous_trader.py            # V6 pipeline + LLM
│   ├── trade_once.py                   # One-shot trading
│   ├── test_mt5.py                     # MT5 connection test
│   ├── backtest_llm_monitor.py         # Backtest framework
│   ├── mt5_competition_score.py      # Live competition score calculator
│   └── trade_state.py                # IPC state manager (SQLite)
│
├── kronos_module/                      # Kronos model (git submodule)
│   ├── model/
│   │   ├── kronos.py                   # Kronos + KronosTokenizer
│   │   └── module.py                   # Transformer blocks
│   └── examples/
│
├── src/intraday/
│   ├── agents/                         # V6 pipeline agents
│   │   ├── forecast.py                 # Transformer agent
│   │   ├── orderflow.py                # Orderflow agent
│   │   ├── regime.py                   # HMM regime agent
│   │   ├── risk.py                     # Risk gate
│   │   └── stay_out.py                 # Stay-out filter
│   ├── aggregator/
│   │   ├── decision.py                 # DecisionEngine
│   │   ├── features.py                 # Feature builder
│   │   └── meta_learner.py             # LightGBM ensemble
│   ├── trader/
│   │   └── mt5_wrapper.py              # MT5 execution wrapper
│   └── llm/
│       └── review.py                   # LLM review agent
│
├── models/
│   └── transformer/20260623T132957Z/   # Transformer weights
│
├── data/models/
│   ├── regime.pkl                        # Trained regime
│   ├── aggregator/meta_learner.pkl       # Meta-learner
│   └── rl/cql_v1/                        # RL policy
│
└── pyproject.toml
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Foundation Model** | Kronos (102M, 12B candles) | Candlestick prediction |
| **LLM** | Claude Sonnet 4.5 (Symphony AI) | Final decision making |
| **Execution** | MetaTrader 5 | Live order execution |
| **Data** | Binance Vision API | Historical + live klines |
| **Features** | Polars, NumPy | Fast data processing |
| **ML** | PyTorch, Transformers | Model inference |
| **API** | OpenAI SDK (Symphony endpoint) | LLM communication |
| **Risk** | Custom rules | Position management |

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Prediction Latency** | ~7s (Kronos inference on CPU) |
| **Decision Latency** | ~1s (Claude API call) |
| **Cycle Time** | ~10s per analysis cycle |
| **Throughput** | ~360 cycles/hour |
| **Trade Frequency** | ~11 trades per 100 candles (when signals align) |
| **Avg Hold Time** | ~5 minutes (before TP/SL/timeout) |
| **Max Concurrent Positions** | 1 (sequential trading) |

---

## Research Findings

### 1. Kronos is Overconfident in Conflicts
- Kronos shows 95% confidence even when wrong
- In conflicts, Kronos is only 25% accurate vs Trend's 35.7%
- **Lesson:** High confidence ≠ accuracy

### 2. Agreement is the Key Signal
- When Kronos and Trend agree → 54.5% accuracy
- When they conflict → accuracy drops to 25–35%
- **Lesson:** Only trade when agents agree

### 3. Neutral Trend = No Trade
- 61% of candles are neutral/ranging
- System correctly says WAIT during choppy markets
- **Lesson:** Patience preserves capital

### 4. Claude LLM Improves Decision Quality
- Claude correctly interprets conflict data
- Recommends WAIT when signals are unclear
- Respects risk management rules
- **Lesson:** LLM adds meta-cognition layer

---

## Risk Disclaimer

> ⚠️ **Research Software. Not Financial Advice.**
>
> Cryptocurrency trading carries significant risk of loss. This system is for research and educational purposes. Past performance does not guarantee future results. Always use demo/paper trading before risking real capital.

---

## License

MIT License — See `LICENSE` file.

---

## Acknowledgments

- **Kronos Model:** NeoQuasar / shiyu-coder — [GitHub](https://github.com/shiyu-coder/Kronos)
- **Claude API:** Symphony Retail AI
- **Data Source:** Binance Vision API
- **Execution:** MetaTrader 5
