# The Agents

## Overview

The Kronos Trading System uses **6 specialized agents** that work together to make trading decisions. Each agent has a unique perspective on the market, and they "vote" through the LLM Decision Maker.

---

## Agent 1: Kronos (RL Policy)

### What It Is

Kronos is a **custom-trained Reinforcement Learning model** that predicts the direction of the next 1-5 candles based on historical patterns.

### Why It's Unique

- **Not off-the-shelf**: Most trading bots use pre-trained models. Kronos is trained specifically for BTCUSD.
- **Custom tokenizer**: Understands candlestick patterns, not generic text.
- **BTC-specific**: Trained on 12 months of 1-minute BTCUSD data.

### Architecture

```
Input: 100 candles (OHLCV)
  ↓
Tokenizer: Custom candlestick vocabulary
  - Encodes open, high, low, close, volume as special tokens
  - Understands patterns: engulfing, doji, hammer, etc.
  ↓
Model: Transformer (102M parameters)
  - Self-attention layers for pattern recognition
  - Feed-forward layers for prediction
  ↓
Output: {
    direction: "bull" | "bear" | "neutral",
    confidence: 0.0 - 1.0,
    predicted_change_pct: -5.0% to +5.0%
}
```

### Training

| Parameter | Value |
|-----------|-------|
| **Training Data** | 12 months BTCUSD 1m candles |
| **Model Size** | 102M parameters |
| **Architecture** | Transformer |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 32 |
| **Device** | CUDA (GPU) or CPU |
| **Reward Function** | Profit + Win Rate + Sharpe |

### Example

```
Input: 100 candles (uptrend then pullback)
Kronos: {
    "direction": "bull",
    "confidence": 0.72,
    "predicted_change_pct": +0.35%
}
```

---

## Agent 2: Technical Analyst

### What It Is

The Technical Analyst calculates **7 classical indicators** on 1-minute candles with periods scaled for 1m precision.

### Indicators

| Indicator | Standard Period | 1m-Scaled Period | Why |
|-----------|----------------|------------------|-----|
| **RSI** | 14 | 70 | 70 min ≈ 14 × 5m |
| **MACD** | 12/26 | 60/130 | 60 min ≈ 12 × 5m |
| **Bollinger** | 20 | 100 | 100 min ≈ 20 × 5m |
| **ADX** | 14 | 70 | 70 min ≈ 14 × 5m |
| **StochRSI** | 14 | 70 | 70 min ≈ 14 × 5m |
| **Supertrend** | 10 | 50 | 50 min ≈ 10 × 5m |
| **VWAP** | — | — | Volume-weighted average |

### Output

```
{
    "rsi": 45,
    "stoch_rsi": 30,
    "adx": 25,
    "macd": 120,
    "macd_signal": 80,
    "bb_position": 0.65,
    "supertrend": "bull",
    "vwap_dev": 1.2%,
    "oversold": false,
    "overbought": false,
    "strong_trend": true
}
```

---

## Agent 3: Trend Detector

### What It Is

The Trend Detector uses **EMA crossovers** to determine the long-term direction of the market.

### EMAs

| EMA | Period | Purpose |
|-----|--------|---------|
| **EMA25** | 25 | Short-term trend |
| **EMA50** | 50 | Medium-term trend |
| **EMA100** | 100 | Long-term trend |

### Logic

```
if EMA25 > EMA50 > EMA100:
    trend = "bull"
    strength = 0.5 + (trend_strength * 0.02)
elif EMA25 < EMA50 < EMA100:
    trend = "bear"
    strength = 0.5 + (trend_strength * 0.02)
else:
    trend = "ranging"
    strength = 0.3
```

### Output

```
{
    "trend": "bull",
    "strength": 0.65,
    "ema25": 59400,
    "ema50": 59300,
    "ema100": 59200,
    "higher_highs": true,
    "lower_lows": false
}
```

---

## Agent 4: Volume Profiler

### What It Is

The Volume Profiler analyzes **volume patterns** to detect institutional buying/selling pressure.

### Metrics

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| **Volume Trend** | Recent avg vs previous avg | Detects increasing/decreasing volume |
| **Taker Buy %** | Taker buy volume / total volume | Detects buying pressure |
| **Buy Pressure** | Volume on green candles / total | Detects bullish/bearish flow |

### Output

```
{
    "sentiment": "bullish",
    "confidence": 0.70,
    "vol_trend": 1.2,
    "avg_taker": 58%,
    "buy_pressure": 0.65
}
```

---

## Agent 5: Conflict Analyzer

### What It Is

The Conflict Analyzer detects when **Kronos and Trend disagree** and decides which to trust.

### Historical Accuracy

| Scenario | Kronos Win Rate | Trend Win Rate | Recommendation |
|----------|----------------|----------------|----------------|
| **AGREE** | 54.5% | 54.5% | Trust both (high confidence) |
| **CONFLICT** | 25.0% | 35.7% | Trust Trend (better in conflicts) |
| **NEUTRAL** | — | — | WAIT |

### Logic

```
if kronos_dir == trend_dir:
    relationship = "AGREE"
    recommended = kronos_dir
    confidence = min(kronos_conf, trend_conf)
elif kronos_dir != trend_dir:
    relationship = "CONFLICT"
    if trend_conf > kronos_conf:
        recommended = trend_dir
    else:
        recommended = kronos_dir
    confidence = max_conf * 0.7
else:
    relationship = "NEUTRAL"
    recommended = "wait"
    confidence = 0.0
```

### Output

```
{
    "relationship": "CONFLICT",
    "trust": "medium",
    "recommended": "bear",
    "confidence": 0.45,
    "reasoning": "Conflict detected - Trend has higher confidence, historically better in conflicts"
}
```

---

## Agent 6: LLM Decision Maker

### What It Is

The LLM Decision Maker is the **final judge**. It receives all agent inputs and makes a human-like decision.

### Why LLM?

- **Context understanding**: Understands market conditions, not just numbers
- **Reasoning**: Explains WHY it made the decision
- **Adaptability**: Can adjust based on competition status, risk appetite, etc.
- **Risk management**: Considers drawdown, win rate, not just profit

### Input

```
Competition Status:
- Final Score: 61.63
- Win Rate: 54.9%
- Sharpe: 0.08
- PnL: +$3,697

Agents:
1. Kronos: bull (conf=0.72)
2. Trend: bear (strength=0.65)
3. Technical: RSI=45, ADX=25, Supertrend=bull
4. Volume: bullish (conf=0.70)
5. Conflict: CONFLICT -> Trend wins

Current Price: $59,500
Recent Candles: [list of last 10 candles]

Rules:
- GO WITH DOMINANT DIRECTION
- TRADE FREQUENTLY (volume needed)
- REMOVE ADX FILTER
- MAX LOTS: 300
- TP: $15,000
- NO SL
```

### Output

```
{
    "action": "SELL",
    "reason": "Trend is bearish (strength=0.65) + Technical ADX=25 (strong enough) + Volume bullish but Kronos says bear. Conflict resolved in favor of Trend."
}
```

### Models Supported

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| **Claude Sonnet** | Medium | High | Medium |
| **Fireworks** | Fast | Medium | Low |
| **Local** | Slow | Medium | Free |

---

## How They Work Together

### Example: Strong Bullish Signal

```
Kronos: bull (conf=0.85)
Technical: RSI=55, ADX=30, Supertrend=bull
Trend: bull (strength=0.75)
Volume: bullish (conf=0.80)
Conflict: AGREE

LLM: BUY — All agents agree, high confidence
```

### Example: Mixed Signal

```
Kronos: bull (conf=0.60)
Technical: RSI=45, ADX=12, Supertrend=neutral
Trend: bear (strength=0.40)
Volume: neutral
Conflict: CONFLICT

LLM: WAIT — Low confidence, mixed signals
```

### Example: Force Override

```
Kronos: neutral
Technical: RSI=40, ADX=15, Supertrend=neutral
Trend: bear (strength=0.55)
Volume: neutral

Aggressive Override: SELL — Trend is bear, ADX > 10
```

---

## Agent Comparison

| Agent | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| Kronos | 7s | 54.5% | Pattern recognition |
| Technical | 0.5s | 55% | Confirmation |
| Trend | 0.5s | 55% | Directional bias |
| Volume | 0.5s | 50% | Institutional flow |
| Conflict | 0.1s | 60% | Risk management |
| LLM | 2-5s | 60% | Final decision |

---

## The Power of Multi-Agent

### Single-Agent Bot

```
RSI < 30 → BUY
RSI > 70 → SELL
```

**Problem**: RSI can stay oversold for hours. Bot buys too early and loses.

### Multi-Agent Bot

```
RSI < 30 + Trend = bear + Volume = selling → WAIT
RSI < 30 + Trend = bull + Volume = buying → BUY
```

**Advantage**: Multiple confirmations reduce false signals.

### Real Example

```
Time: 14:00
RSI: 28 (oversold)
Single-Agent Bot: BUY
Result: Price drops 2% → LOSS

Kronos: WAIT (predicts further drop)
Trend: bear (EMAs bearish)
Volume: selling pressure
Multi-Agent Bot: WAIT
Result: Price drops 2% → NO TRADE → SAVED
```

---

## Summary

| Agent | Role | Key Strength |
|-------|------|--------------|
| **Kronos** | Pattern prediction | BTC-specific RL |
| **Technical** | Indicator confirmation | 7 indicators |
| **Trend** | Directional bias | EMA crossovers |
| **Volume** | Institutional flow | Order flow analysis |
| **Conflict** | Risk management | Resolves disagreements |
| **LLM** | Final decision | Human-like reasoning |

**Together**: 6 perspectives → 1 decision → higher accuracy, lower risk.
