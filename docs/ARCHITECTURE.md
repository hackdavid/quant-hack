# System Architecture

## Overview

The Kronos AI Trading System is a **multi-agent AI trading system** that combines **Reinforcement Learning**, **Technical Analysis**, and **Large Language Models** for real-time BTC/USD trading.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │ Binance 1m API  │ │ Binance 5m API  │ │ MT5 Account Data        │ │
│  │ (OHLCV)         │ │ (OHLCV)         │ │ (Balance, Positions)    │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │ 1m Candles      │ │ 5m Candles      │ │ Technical Indicators    │ │
│  │ (1000 bars)     │ │ (400 bars)      │ │ (RSI, MACD, ADX, BB)    │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      COUNCIL OF AGENTS                                 │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │
│  │ Kronos  │ │ Technical│ │  Trend  │ │ Volume  │ │  Conflict   │ │
│  │RL Agent │ │ Analyst  │ │ Detector│ │Profiler │ │  Analyzer   │ │
│  │(102M)   │ │(RSI/ADX) │ │(EMA)    │ │(OrderFlow)│ │(Resolution) │ │
│  └─────────┘ └──────────┘ └─────────┘ └─────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM DECISION MAKER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Claude Sonnet / LLM                                             │ │
│  │  - Receives ALL agent inputs                                     │ │
│  │  - Analyzes market context                                       │ │
│  │  - Decides: BUY / SELL / WAIT                                    │ │
│  │  - Provides reasoning                                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │ MT5 Wrapper     │ │ Risk Manager    │ │ Position Monitor        │ │
│  │ (Market Orders) │ │ (TP/SL/Circuit) │ │ (Profit Trailing)       │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Ingestion
- **Source**: Binance Vision API (free, no API key needed)
- **Interval**: 1m candles (primary), 5m candles (backtest)
- **Limit**: 1000 candles (~16 hours of 1m data)
- **Symbol**: BTCUSDT

### 2. Feature Engineering
- **OHLCV**: Raw candle data
- **EMA**: 25, 50, 100 periods
- **RSI**: 70-period (scaled for 1m)
- **MACD**: 60/130 (scaled for 1m)
- **ADX**: 70-period (scaled for 1m)
- **Bollinger**: 100-period (scaled for 1m)
- **Supertrend**: 50-period (scaled for 1m)
- **VWAP**: Volume-weighted average price

### 3. Agent Processing
Each agent runs independently:
- **Kronos**: Loads model, tokenizes candles, predicts direction
- **Technical**: Calculates all indicators
- **Trend**: EMA crossover analysis
- **Volume**: 50-period volume analysis
- **Conflict**: Compares Kronos vs Trend

### 4. LLM Decision
- **Input**: All agent predictions + market context
- **Prompt**: Structured with rules and constraints
- **Output**: BUY / SELL / WAIT with reasoning
- **Override**: Aggressive override if trend is strong

### 5. Execution
- **Order**: Market order via MT5
- **TP**: $15,000 (300 lots)
- **SL**: NO SL (manual management)
- **Hold**: 1 hour max
- **Chunking**: 300 lots split into 100-lot orders

### 6. Monitoring
- **Profit Trailing**: Closes if drops $3,000 from peak
- **TP Auto-Close**: Closes at $15,000 profit
- **Circuit Breaker**: Stops after 3 consecutive losses
- **State**: SQLite IPC for monitor/trader communication

## Components

### Kronos RL Agent

```
Input: 100 candles (OHLCV)
  ↓
Tokenizer: Custom candlestick vocabulary
  ↓
Model: Transformer (102M params)
  ↓
Output: {direction: bull/bear/neutral, confidence: 0.0-1.0}
```

**Training**:
- Data: 12 months BTCUSD 1m candles
- Reward: Profit/Loss + Win Rate
- Optimizer: AdamW
- Device: CUDA (GPU) or CPU

### Technical Analyst

```
Input: 150 candles (minimum for 100-period BB)
  ↓
Indicators:
  - RSI(70)
  - MACD(60, 130)
  - Bollinger(100)
  - ADX(70)
  - StochRSI(70)
  - Supertrend(50)
  - VWAP
  ↓
Output: {rsi, macd, bb_position, adx, stoch_rsi, supertrend, vwap_dev}
```

### Trend Detector

```
Input: 100 candles
  ↓
EMAs: 25, 50, 100
  ↓
Logic:
  - EMA25 > EMA50 > EMA100 → bull
  - EMA25 < EMA50 < EMA100 → bear
  - Else → ranging
  ↓
Output: {trend: bull/bear/ranging, strength: 0.0-1.0}
```

### Volume Profiler

```
Input: 50 candles
  ↓
Metrics:
  - Average volume (recent vs previous)
  - Taker buy percentage
  - Buy/sell pressure ratio
  ↓
Output: {sentiment: bullish/bearish/neutral, confidence: 0.0-1.0}
```

### Conflict Analyzer

```
Input: Kronos prediction + Trend prediction
  ↓
Logic:
  - AGREE: Kronos == Trend → high confidence
  - CONFLICT: Kronos != Trend → check historical accuracy
  - NEUTRAL: Either is neutral → wait
  ↓
Output: {relationship: AGREE/CONFLICT/NEUTRAL, recommended: bull/bear/wait}
```

### LLM Decision Maker

```
Input: All 6 agent outputs + market context + competition stats
  ↓
Prompt: Structured with rules and constraints
  ↓
LLM: Claude Sonnet / Fireworks / Local
  ↓
Output: {action: BUY/SELL/WAIT, reason: explanation}
```

## Risk Management

### Profit Trailing
- **Trigger**: Profit drops $3,000 from peak
- **Action**: Close position
- **Purpose**: Lock in gains

### TP Auto-Close
- **Trigger**: Profit reaches $15,000
- **Action**: Close position
- **Purpose**: Take profit

### Circuit Breaker
- **Trigger**: 3 consecutive losses
- **Action**: Stop trading
- **Purpose**: Prevent catastrophic losses

### Chunked Orders
- **Trigger**: Order size > 100 lots
- **Action**: Split into 100-lot chunks
- **Purpose**: Broker compliance

## Monitoring

### Trade State
- **Storage**: SQLite IPC
- **Fields**: has_position, side, lots, profit, TP, SL, hold, signal_score
- **Commands**: close_all, pause, resume, update_tp, update_sl

### Position Monitor
- **Frequency**: Every 30 seconds
- **Output**: Ticket, side, PnL, TP, SL, hold time
- **Alerts**: TP hit, profit lock, max hold

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Data** | Binance Vision API, httpx |
| **Features** | pandas, numpy, custom calculations |
| **RL Model** | PyTorch, Transformers |
| **LLM** | Claude Sonnet / Fireworks / Local |
| **Execution** | MetaTrader5, Windows |
| **Monitoring** | SQLite, structlog |
| **Config** | python-dotenv |

## Performance

| Metric | Value |
|--------|-------|
| Prediction Cycle | ~7 seconds (Kronos inference) |
| Monitor Cycle | 30 seconds |
| Entry Cycle | 5 minutes |
| Latency | ~1-2 seconds (MT5 execution) |
| Win Rate | 54.9% |
| Max Drawdown | 0.59% |

## Future Improvements

1. **Sentiment Analysis**: Twitter/X feed for market sentiment
2. **On-Chain Data**: Bitcoin whale movements
3. **Multi-Asset**: ETH, SOL, BNB
4. **Options**: Implement options strategies
5. **Backtesting**: Full historical backtest engine
