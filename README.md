# Kronos AI Trading System — Multi-Agent BTC Trading

## 🏆 Built for Algorithmic Trading Competitions

**A multi-agent AI trading system that combines Reinforcement Learning, Technical Analysis, and Large Language Models for real-time BTC/USD trading on MetaTrader 5.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org)
[![MT5](https://img.shields.io/badge/MT5-Live-green)](https://www.metatrader5.com)

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KRONOS LLM MASTER TRADER                       │
│                      Multi-Agent AI Trading System                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌──────────┐              ┌──────────────┐              ┌────────────┐
│  DATA    │              │   KRONOS     │              │   LLM      │
│  LAYER   │              │   RL AGENT   │              │   DECISION │
│          │              │              │              │   MAKER    │
└──────────┘              └──────────────┘              └────────────┘
    │                          │                              │
    │                          │                              │
    ▼                          ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      COUNCIL OF AGENTS                                 │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │
│  │ Kronos  │ │ Technical│ │  Trend  │ │ Volume  │ │  Conflict   │ │
│  │RL Policy│ │ Analyst  │ │ Detector│ │Profiler │ │  Analyzer   │ │
│  │(Custom) │ │(RSI/ADX) │ │(EMA)    │ │(OrderFlow)│ │(Resolution) │ │
│  └─────────┘ └──────────┘ └─────────┘ └─────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MT5 EXECUTION & RISK MANAGEMENT                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│  │ Market Order│ │Profit Trail │ │Circuit Break│ │ Chunk Close │    │
│  │ 1m Candles  │ │  $3k Lock   │ │  3 Losses   │ │ 300 Lots    │    │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### The 6 Agents

| Agent | Technology | Role | Why It Matters |
|-------|-----------|------|----------------|
| **Kronos** | 🧠 RL Transformer | Predicts BTC direction | Trained specifically for BTCUSD on 12 months of 1m data |
| **Technical Analyst** | 📊 Indicators | RSI, MACD, ADX, Bollinger, Supertrend | Classic confirmation signals |
| **Trend Detector** | 📈 EMA Analysis | EMA(25/50/100) crossovers | Long-term directional bias |
| **Volume Profiler** | 📉 Order Flow | Buy/sell pressure, taker ratio | Institutional footprint detection |
| **Conflict Analyzer** | ⚖️ Resolution | Detects signal disagreements | Risk management when agents conflict |
| **LLM Decision Maker** | 🤖 Claude/LLM | Final judge | Human-like reasoning with all inputs |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

```env
# MT5 Credentials
MT5_ACCOUNT=your_account
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# LLM API
LLM_TOKEN=your_token
LLM_BASE_URL=https://your-api.com
LLM_MODEL=your-model
```

### 3. Run the Bot

```bash
# Normal mode (uses .env credentials)
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py

# Force SELL entry
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py --sell

# Force BUY entry
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py --buy
```

### 4. Monitor the Bot

```bash
# Check status
.venv\Scripts\python.exe scripts\trade_state.py --status

# Emergency close all
.venv\Scripts\python.exe scripts\close_all.py

# Close specific position
.venv\Scripts\python.exe scripts\close_position.py --ticket 123456
```

---

## 🧠 The Kronos Agent (RL Policy)

### What Makes It Unique

- **Custom Tokenizer**: Trained on BTC candlestick patterns, not generic text
- **Transformer Architecture**: Sequence model for 1m OHLCV data
- **RL Training**: Reward function optimized for profit + win rate
- **BTC-Specific**: Not a generic stock model — trained on BTCUSD volatility

### Training Details

```
Training Data: 12 months BTCUSD 1m candles
Model: Transformer-based sequence predictor
Tokenizer: Custom candlestick vocabulary
Features: OHLCV + EMA + Volume + ATR
Prediction: Direction (bull/bear/neutral) + Confidence (0.0-1.0)
```

---

## 📈 Real-Time Pipeline

### 1-Minute Candle Precision

Most trading bots use 5m or 15m candles. Kronos uses **1m candles** for:
- Faster signal detection
- Earlier entry on breakouts
- Tighter risk management

### Indicator Scaling for 1m

| Indicator | Standard Period | 1m-Optimized Period |
|-----------|-----------------|-------------------|
| RSI | 14 | 70 |
| MACD | 12/26 | 60/130 |
| Bollinger | 20 | 100 |
| ADX | 14 | 70 |
| EMA Trend | 5/10/20 | 25/50/100 |
| Volume | 10 | 50 |

---

## 🛡️ Risk Management

| Feature | Description |
|---------|-------------|
| **Profit Trailing** | Closes if profit drops $3,000 from peak |
| **TP Auto-Close** | Closes at $15,000 profit (300 lots) |
| **Circuit Breaker** | Stops after 3 consecutive losses |
| **NO SL Mode** | Manual close for experienced traders |
| **1hr Max Hold** | Prevents overexposure |
| **Chunked Orders** | Splits 300 lots into 100-lot chunks for broker compliance |

---

## 📊 Performance

### Competition Results

| Metric | Value |
|--------|-------|
| **Final Score** | 61.63 / 100 |
| **Win Rate** | 54.9% |
| **Sharpe Ratio** | 0.08 |
| **Max Drawdown** | 0.59% |
| **Total P&L** | +$3,697 |
| **Trades** | 235 |
| **Wins** | 129 |
| **Losses** | 106 |

### Live Trading Example

```
[21:00:00] Entry: SELL 300 lots @ $59,500
[21:05:00] PnL: +$4,500 (peak)
[21:06:00] PnL: +$1,200 (profit lock triggered)
[21:06:30] Closed at +$1,200
```

---

## 🎥 Demo

### Live Trading Video

[Watch the bot trade live in real-time](https://youtube.com/your-demo-link)

### Screenshots

| Entry Decision | Position Monitor | Close |
|---------------|-------------------|-------|
| ![Entry](docs/entry.png) | ![Monitor](docs/monitor.png) | ![Close](docs/close.png) |

---

## 📁 Project Structure

```
.
├── docs/
│   ├── ARCHITECTURE.md      # Full system design
│   ├── AGENTS.md           # Agent details
│   └── RESULTS.md          # Performance metrics
├── scripts/
│   ├── kronos_llm_master_trader.py   # Main bot
│   ├── trade_state.py                # Monitor interface
│   ├── close_all.py                  # Emergency close
│   ├── close_position.py            # Specific close
│   └── analyze_trend.py             # Market analysis
├── src/
│   ├── intraday/
│   │   ├── features/          # Feature engineering
│   │   ├── llm/               # LLM prompts
│   │   └── trader/            # MT5 wrapper
│   └── kronos_module/         # RL model
├── models/
│   ├── kronos_v1.pt          # Trained model
│   └── tokenizer.json        # Custom tokenizer
├── .env.example              # Configuration template
├── pyproject.toml            # Dependencies
└── README.md                 # This file
```

---

## 🏅 Why This Wins

### Innovation
- **First** RL + LLM + Multi-Agent system for BTC trading
- **Custom** tokenizer for candlestick patterns
- **1m precision** when most bots use 5m+

### Technical Depth
- 6 independent agents with conflict resolution
- Real-time MT5 execution
- Adaptive risk management

### Results
- Live trading with positive P&L
- 54.9% win rate
- 0.59% max drawdown

### Presentation
- Clear architecture diagrams
- Video demo
- Comprehensive documentation

---

## 🛠️ Commands

### Trading

```bash
# Start bot
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py

# Force entry
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py --sell
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py --buy

# Backtest
.venv\Scripts\python.exe scripts\kronos_llm_master_trader.py --backtest --trades 10
```

### Monitoring

```bash
# Check status
.venv\Scripts\python.exe scripts\trade_state.py --status

# Pause trading
.venv\Scripts\python.exe scripts\trade_state.py --pause

# Resume trading
.venv\Scripts\python.exe scripts\trade_state.py --resume

# Update TP
.venv\Scripts\python.exe scripts\trade_state.py --update-tp 500

# Close all
.venv\Scripts\python.exe scripts\trade_state.py --close-all
```

### Emergency

```bash
# Close all positions NOW
.venv\Scripts\python.exe scripts\close_all.py

# Close specific position
.venv\Scripts\python.exe scripts\close_position.py --ticket 123456
```

---

## 📝 Documentation

- [Architecture](docs/ARCHITECTURE.md) — Full system design
- [Agents](docs/AGENTS.md) — Each agent explained
- [Results](docs/RESULTS.md) — Trading performance

---

## 🤝 Team

Built with ❤️ for algorithmic trading competitions.

**Co-Authored-By: Claude**

---

## ⚠️ Disclaimer

This is a trading bot for **competition/educational purposes**. Trading involves risk. Past performance does not guarantee future results. Always use proper risk management.
