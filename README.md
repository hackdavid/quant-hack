# BTC/USD Intraday Trading System

> Institutional-grade multi-agent trading system targeting **Sharpe 1.5+** on BTC/USD perpetual futures using L2 microstructure signals, probabilistic forecasting, and RL-optimized execution.

[![Tests](https://img.shields.io/badge/tests-16%2F16%20passing-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 Project Overview

This system combines cutting-edge quantitative techniques to build a complete intraday trading pipeline:

- **🧠 Kronos Foundation Model**: Pre-trained time-series transformer (like GPT for numbers) - solves "not enough data" problem via transfer learning
- **📊 L2 Microstructure Features**: Order flow imbalance (OFI), microprice, VPIN, Hawkes intensity
- **🤖 Multi-Agent Architecture**: 5 specialized agents (Forecast, Orderflow, Regime, Risk, Stay-out) with meta-learning aggregation
- **⚡ RL Execution**: Conservative Q-Learning (CQL) for slippage-aware execution optimization
- **📈 Probabilistic Forecasting**: Uncertainty-aware predictions with Brier score validation
- **🔄 Continual Learning**: Monthly retraining pipeline to adapt to regime shifts

**Key Innovation:**
- **Transfer Learning**: Uses Kronos (pre-trained on millions of time-series samples) → only 12 months BTC data needed
- **Data Efficiency**: 12mo + Kronos > 5 years without foundation model

**Target Performance:**
- Sharpe Ratio: 1.5+ (sustained over 12+ months)
- Max Drawdown: <15%
- Win Rate: 52-58%
- Avg Trade Duration: 15-120 minutes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline (Phase 1)                  │
│  Historical (12mo) + Live WebSocket (trades, L2 depth, funding) │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   Feature Engine (Phase 2)                      │
│   OFI, Microprice, VPIN, Hawkes, RSI, Funding, Volatility      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                Queue-Aware Simulator (Phase 3)                  │
│      Realistic fills, L2 matching, latency, slippage           │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐    ┌───────────▼──────────┐
│  5 ML Agents   │    │  RL Execution Agent  │
│  (Phase 4-5)   │    │     (Phase 7)        │
│                │    │                      │
│ • Forecast ⭐  │    │  CQL for adaptive    │
│   (Kronos +    │    │  entry/exit timing   │
│    TCN)        │    │                      │
│ • Orderflow    │    └──────────────────────┘
│ • Regime       │
│ • Risk         │
│ • Stay-out     │
└───────┬────────┘
        │
┌───────▼──────────────────────┐
│  Meta-Learner Aggregator     │
│       (Phase 6)              │
│  Kelly-weighted ensemble     │
└───────┬──────────────────────┘
        │
┌───────▼──────────────────────┐
│   Paper Trading (Phase 8)    │
│     1-3 months validation    │
└───────┬──────────────────────┘
        │
┌───────▼──────────────────────┐
│  Continual Learning (Phase 9)│
│   Monthly retraining loop    │
└───────┬──────────────────────┘
        │
┌───────▼──────────────────────┐
│     Live Trading (Phase 10)  │
│  Canary → Scale with safety  │
└──────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- 500GB disk space (for data storage)
- GPU (optional, for Phase 4 & 7 training acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quanthack.git
cd quanthack

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e .
```

### Phase 1: Data Collection

```bash
# 1. Download 12 months historical data (~20-30 minutes)
uv run intraday data download --kind klines_5m --start 2024-01-01 --end 2024-12-31
uv run intraday data download --kind klines_1m --start 2024-01-01 --end 2024-12-31
uv run intraday data download --kind funding --start 2024-01-01 --end 2024-12-31
uv run intraday data download --kind open_interest --start 2024-01-01 --end 2024-12-31

# 2. Start live data capture (runs in background)
tmux new -s data-capture
uv run intraday data live-capture --streams trade,depth,mark_price

# Detach: Ctrl+B, D
# Reattach later: tmux attach -t data-capture

# 3. Check data collection status
uv run intraday data summary
uv run intraday data checkpoint
```

**Note:** Collect at least 4-6 weeks of live tick data before proceeding to Phase 2-7 development.

### Development Workflow

```bash
# Run tests
pytest tests/ -v

# Run specific phase tests
pytest tests/phase_01/ -v  # Data pipeline
pytest tests/phase_02/ -v  # Feature engine (when ready)

# Check code quality
ruff check src/
mypy src/
```

---

## 📊 Development Strategy

**Revised Plan (Optimized for Speed):**

1. **Phase 1 (Days 1-3)**: Collect 4-6 weeks live data + 12 months historical
2. **Phase 2-7 (Days 4-25)**: Build entire pipeline using 4-6 week dataset as MVP
   - Fast iteration on small dataset
   - Validate all code on GPU (training in hours, not days)
   - Ensure end-to-end pipeline works
3. **Retrain on Full Data (Days 26-28)**: Train production models on 12-month dataset
4. **Phase 8 (Days 29-30+)**: Paper trade with production models, optimize

**Key Insight:** Don't wait 4-6 weeks idle. Use small dataset to de-risk all development, then retrain for production.

---

## 🎯 Phase Breakdown

| Phase | Name | Duration | GPU Time | Output |
|-------|------|----------|----------|--------|
| 1 | Data Pipeline | 2-3 days + 4-6 weeks collection | - | Historical + live tick data |
| 2 | Feature Engine | 3-5 days | - | OFI, VPIN, Hawkes, etc. |
| 3 | Simulator | 4-6 days | - | Queue-aware backtester |
| 4 | Forecast Agent | 1-2 days | 2-4 hrs | Probabilistic price forecasts |
| 5 | Other Agents | 3-5 days | - | Orderflow, Regime, Risk, Stay-out |
| 6 | Aggregator | 2-4 days | - | Meta-learner ensemble |
| 7 | RL Execution | 3-5 days | 4-8 hrs | CQL execution policy |
| 8 | Paper Trading | 30-90 days | - | Live validation |
| 9 | Continual Learning | 2-3 days | - | Monthly update pipeline |
| 10 | Live Trading | Ongoing | - | Production deployment |

**Total:** 60-80 days (MVP → Production)

---

## 📁 Project Structure

```
quanthack/
├── idea/                    # Planning & design docs
│   ├── PLAN.md             # Master strategy document
│   ├── AGENTS.md           # Agent design & coding rules
│   ├── CLI.md              # Complete CLI specification
│   └── phases/             # Phase-by-phase implementation specs
│       ├── 01_data.md
│       ├── 02_features.md
│       ├── 04_forecast.md  # ⭐ Kronos foundation model details
│       └── ...
├── ARCHITECTURE.md          # ⭐ Foundation model deep-dive
├── src/intraday/
│   ├── data/               # Phase 1: Data collection
│   │   ├── download.py     # Historical data downloader
│   │   ├── capture.py      # Live WebSocket capture
│   │   ├── schema.py       # Parquet schema validation
│   │   └── cli.py          # CLI commands
│   ├── features/           # Phase 2: Feature engine (coming soon)
│   ├── sim/                # Phase 3: Simulator (coming soon)
│   ├── agents/             # Phase 4-5: ML agents (coming soon)
│   ├── aggregator/         # Phase 6: Meta-learner (coming soon)
│   ├── execution/          # Phase 7: RL execution (coming soon)
│   └── continual/          # Phase 9: Retraining loop (coming soon)
├── tests/
│   ├── phase_01/           # Data pipeline tests (16 passing)
│   └── ...
├── data/                   # Data storage (gitignored)
│   ├── raw/binance/        # Historical + live data
│   ├── processed/          # Feature-engineered data
│   └── checkpoints/        # Download progress tracking
├── config/
│   ├── features.yaml       # Feature engine config
│   ├── models.yaml         # Model hyperparameters
│   └── risk.yaml           # Risk management rules
├── MASTER_INDEX.md         # Project progress tracker
├── SESSION_START.md        # Quick orientation guide
└── QUICKSTART.md           # Phase 1 usage guide
```

---

## 🛠️ CLI Reference

### Data Commands

```bash
# Download historical data
intraday data download \
  --kind klines_5m \
  --start 2024-01-01 \
  --end 2024-12-31

# Start live capture
intraday data live-capture \
  --streams trade,depth,mark_price,liquidations

# Check status
intraday data summary
intraday data checkpoint
```

### Training Commands (Phase 4-7)

```bash
# Train forecast agent (Phase 4)
intraday train forecast \
  --data-start 2024-01-01 \
  --data-end 2024-12-31 \
  --model transformer \
  --device cuda

# Train RL execution (Phase 7)
intraday train execution \
  --episodes 50000 \
  --device cuda \
  --checkpoint-dir models/execution/

# Backtest strategy (Phase 3+)
intraday backtest \
  --start 2024-06-01 \
  --end 2024-12-31 \
  --agents forecast,orderflow,regime \
  --execution rl
```

### Paper Trading (Phase 8)

```bash
# Start paper trading
intraday trade paper \
  --symbol BTCUSDT \
  --strategy ensemble \
  --risk-limit 0.01

# Monitor performance
intraday trade monitor --mode paper

# Generate report
intraday trade report \
  --start 2024-06-01 \
  --end 2024-06-30 \
  --output reports/june_performance.html
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific phase
pytest tests/phase_01/ -v

# Run with coverage
pytest tests/ --cov=src/intraday --cov-report=html

# Run integration tests only
pytest tests/ -m integration
```

**Current Status:** 16/16 Phase 1 tests passing ✅

---

## 📈 Performance Targets

### Backtest Metrics (Phase 3-7)

- **Sharpe Ratio (OOS)**: ≥1.0 (Phase 6), ≥1.5 (Phase 7 with RL)
- **Max Drawdown**: <15%
- **Win Rate**: 52-58%
- **Avg Trade Duration**: 15-120 minutes
- **Feature IC**: >0.1 (key features like OFI, microprice)
- **Brier Score**: <0.5 (probabilistic forecasts)

### Paper Trading Acceptance (Phase 8)

- Sharpe ≥1.0 sustained for ≥4 weeks
- No single-day loss >3%
- Execution slippage <5 bps on average
- Uptime >99.5%

### Live Trading Safety (Phase 10)

- Start with canary (0.1% of target size)
- Scale only after 2 weeks stable performance
- Kill-switch on:
  - Drawdown >5% in 24 hours
  - Sharpe <0.5 over 2 weeks
  - API latency >100ms (p99)

---

## 🔒 Risk Management

**Hard Limits:**
- Max position size: 1% of portfolio per trade
- Max daily loss: 3% of portfolio
- Max drawdown trigger: 15% (pause trading)
- Max leverage: 5x (crypto futures)

**Pre-Trade Checks:**
- Regime agent approval (no high-risk regimes)
- Stay-out agent approval (avoid crowded/toxic flows)
- Risk agent position sizing
- Minimum liquidity requirement (depth >10x position size)

**Real-Time Monitoring:**
- PnL tracking every second
- Latency monitoring (kill if >100ms p99)
- API health checks (reconnect logic)
- Model drift detection (trigger retraining)

---

## 🧠 ML Models & Techniques

### Phase 4: Forecast Agent ⭐ **KEY INNOVATION**
- **Foundation Model**: [Kronos](https://github.com/shiyu-coder/Kronos) (pre-trained time-series transformer)
  - Pre-trained on millions of time-series samples (general temporal patterns)
  - Fine-tuned via LoRA (5% params) on 12-month BTC data
  - **Transfer learning** → data-efficient (like GPT for numbers!)
- **Custom Branch**: Small TCN (4 layers, 64 channels) for crypto microstructure
- **Fusion**: Concat Kronos (256-d) + TCN (64-d) → MLP forecast head
- **Output**: Probability distribution over 11 bins (not point estimate)
- **Target**: 5-min, 15-min, 60-min forward returns
- **Loss**: Cross-entropy + focal loss (tail emphasis)
- **Validation**: OOS Sharpe ≥0.5, Brier <0.5

**Why This Works:**
- Kronos brings pre-trained temporal understanding (trends, seasonality, regimes)
- TCN learns crypto-specific signals (OFI, VPIN, toxic flow)
- **12 months + Kronos > 5 years without foundation model**

### Phase 5: Other Agents
- **Orderflow**: LSTM on L2 depth dynamics
- **Regime**: HMM + volatility clustering
- **Risk**: Gradient boosting (XGBoost)
- **Stay-out**: Binary classifier (toxic flow detection)

### Phase 6: Meta-Learner
- **Aggregation**: Online ridge regression
- **Weighting**: Kelly criterion for agent allocation
- **Updates**: Rolling 7-day window

### Phase 7: RL Execution
- **Algorithm**: Conservative Q-Learning (CQL)
- **State**: Queue position, spread, inventory, urgency
- **Action**: Passive (limit), aggressive (market), cancel
- **Reward**: PnL - λ × slippage

---

## 📚 Key Resources

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - ⭐ Foundation model explanation (why 12 months is enough)
- [Master Plan](idea/PLAN.md) - Overall strategy & data requirements
- [Agent Design](idea/AGENTS.md) - Coding conventions & testing rules
- [Phase 4 Spec](idea/phases/04_forecast.md) - Kronos + TCN implementation details
- [CLI Spec](idea/CLI.md) - Complete command reference
- [Phase Specs](idea/phases/) - Detailed implementation guides
- [MASTER_INDEX.md](MASTER_INDEX.md) - Progress tracking (read this first!)

---

## 🤝 Contributing

This is a personal research project, but feedback is welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

**This is research/educational software for algorithmic trading development.**

- No guarantees of profitability
- Cryptocurrency trading carries significant risk
- Past performance does not guarantee future results
- Use at your own risk
- Not financial advice

**Never risk more than you can afford to lose.**

---

## 🎓 Acknowledgments

Built as part of a quantitative trading research project. Key inspirations:

- **Kronos**: [Time-series foundation model](https://github.com/shiyu-coder/Kronos) (Google Research)
- L2 microstructure papers (Cont, Stoikov, Lehalle)
- RL for execution (Spooner, Vyetrenko)
- Multi-agent systems (Hendrycks, Sutton)
- Continual learning (Kirkpatrick, Rolnick)
- Meta-labeling (López de Prado)

---

## 📊 Current Status

**Phase:** 1 (Data Pipeline)  
**Progress:** Implementation complete (16/16 tests passing)  
**Next Milestone:** Collect 4-6 weeks live data, then build Phase 2-7 pipeline  
**Last Updated:** 2026-06-21

For detailed progress, see [MASTER_INDEX.md](MASTER_INDEX.md)
