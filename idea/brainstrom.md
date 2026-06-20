\# BTC/USDT Intraday Multi-Agent Trading System



\## Objective



Build a real-time intraday trading system for BTC/USDT using:



\* Binance websocket streaming

\* Time-series foundation models (Kronos)

\* Multi-agent architecture

\* Probabilistic forecasting

\* RL-based execution

\* Online adaptation

\* Regime-aware trading



Primary target:



\* Predict short-term directional movement

\* Estimate movement probability and volatility

\* Execute trades dynamically

\* Continuously adapt to market regime changes



\---



\# Core Philosophy



The system should NOT attempt:



```text

Predict exact BTC price

```



Instead:



```text

Model market state transitions

```



This means:



\* probabilistic forecasting

\* volatility understanding

\* liquidity understanding

\* orderflow pressure analysis

\* regime detection

\* adaptive execution



The alpha comes from:



\* market state understanding

\* short-term inefficiencies

\* liquidation events

\* volatility expansion

\* leverage imbalance

\* execution optimization



NOT from exact price prediction.



\---



\# High-Level Architecture



```text

&#x20;               Binance Websocket Streams

&#x20;                           |

&#x20;                           v

&#x20;               --------------------------------

&#x20;               |      Stream Ingestion        |

&#x20;               --------------------------------

&#x20;                           |

&#x20;                           v

&#x20;               --------------------------------

&#x20;               |       Feature Engine         |

&#x20;               --------------------------------

&#x20;                           |

&#x20;                           v

&#x20;               --------------------------------

&#x20;               |       State Builder          |

&#x20;               --------------------------------

&#x20;                           |

&#x20;       -------------------------------------------------

&#x20;       |                Shared State Store             |

&#x20;       -------------------------------------------------

&#x20;         |           |            |            |

&#x20;         v           v            v            v

&#x20;   Forecast      Orderflow      Regime       Risk

&#x20;     Agent         Agent         Agent       Agent

&#x20;         \\           |            |           /

&#x20;          \\          |            |          /

&#x20;           ----------------------------------

&#x20;                   Decision Aggregator

&#x20;                           |

&#x20;                           v

&#x20;                 RL Execution Agent

&#x20;                           |

&#x20;                           v

&#x20;                     Binance Orders

```



\---



\# Binance Websocket Data Sources



\## 1. Trade Stream



Endpoint:



```text

wss://stream.binance.com:9443/ws/btcusdt@trade

```



Provides:



\* price

\* quantity

\* aggressive buyer/seller

\* timestamp



Used for:



\* trade flow

\* CVD

\* momentum pressure

\* microstructure analysis



\---



\## 2. Depth Stream



Endpoint:



```text

btcusdt@depth@100ms

```



Provides:



\* bid levels

\* ask levels

\* liquidity movement



Used for:



\* orderbook imbalance

\* liquidity voids

\* spread analysis

\* spoofing detection



\---



\## 3. Kline Stream



Endpoint:



```text

btcusdt@kline\_1m

```



Provides:



\* OHLCV candles



Used for:



\* Kronos forecasting

\* volatility modeling

\* trend analysis



\---



\## 4. Liquidation Stream



Can be sourced externally.



Used for:



\* liquidation cascades

\* leverage pressure

\* squeeze detection



\---



\## 5. Funding + Open Interest



Important for futures market state.



Used for:



\* leverage imbalance

\* directional crowding

\* regime identification



\---



\# Data Pipeline



\## Step 1 — Raw Stream Ingestion



All websocket streams are normalized into:



```python

{

&#x20;   "timestamp": ...,

&#x20;   "price": ...,

&#x20;   "size": ...,

&#x20;   "side": ...,

&#x20;   "bid": ...,

&#x20;   "ask": ...

}

```



Store in:



\* Kafka

\* Redis Streams

\* NATS

\* Pulsar



Recommended:



```text

Kafka + Redis

```



Kafka:



\* replay

\* persistence

\* offline training



Redis:



\* low-latency state serving



\---



\# Feature Engineering Layer



This is the MOST IMPORTANT layer.



The system should NOT feed raw candles directly into models.



Instead:



```text

Raw Data

&#x20;   ↓

Statistical Transform

&#x20;   ↓

Microstructure Features

&#x20;   ↓

Latent State

```



\---



\# Mathematical Feature Engineering



\## 1. Log Returns



Purpose:



\* normalize price movement

\* remove scale dependence



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"r\_t = \\log\\left(\\frac{P\_t}{P\_{t-1}}\\right)"}}



Used in:



\* Kronos forecast agent

\* volatility estimation

\* regime detection



\---



\## 2. Realized Volatility



Purpose:



\* estimate current market instability



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"\\sigma = \\sqrt{\\frac{1}{N}\\sum\_{i=1}^{N}(r\_i-\\bar r)^2}"}}



Used in:



\* volatility agent

\* risk agent

\* execution sizing



\---



\## 3. Bid/Ask Imbalance



Purpose:



\* detect directional liquidity pressure



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"Imbalance = \\frac{\\sum BidVolume - \\sum AskVolume}{\\sum BidVolume + \\sum AskVolume}"}}



Used in:



\* orderflow agent

\* execution agent



\---



\## 4. Cumulative Volume Delta (CVD)



Purpose:



\* track aggressive buying/selling



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"CVD\_t = CVD\_{t-1} + (BuyVolume - SellVolume)"}}



Used in:



\* orderflow analysis

\* momentum detection

\* liquidation pressure



\---



\## 5. Average True Range (ATR)



Purpose:



\* estimate expected movement range



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"ATR = \\frac{1}{N}\\sum TR\_t"}}



Where:



```text

TR = max(

&#x20;   high - low,

&#x20;   abs(high - prev\_close),

&#x20;   abs(low - prev\_close)

)

```



Used in:



\* stop placement

\* volatility sizing

\* risk management



\---



\## 6. Sharpe-Like Reward



Purpose:



\* prevent RL reward hacking



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"Reward = \\frac{E\[R]}{\\sigma\_R} - Cost - DrawdownPenalty"}}



Used in:



\* RL training

\* policy optimization



\---



\## 7. Entropy



Purpose:



\* measure randomness vs structured movement



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"H(X) = -\\sum p(x)\\log p(x)"}}



Used in:



\* regime detection

\* market state analysis



\---



\# Shared State Builder



The feature engine outputs compressed structured state.



Example:



```python

market\_state = {

&#x20;   "returns": ...,

&#x20;   "realized\_volatility": ...,

&#x20;   "funding\_rate": ...,

&#x20;   "oi\_delta": ...,

&#x20;   "cvd": ...,

&#x20;   "imbalance": ...,

&#x20;   "spread": ...,

&#x20;   "liquidation\_pressure": ...,

&#x20;   "entropy": ...,

&#x20;   "atr": ...,

&#x20;   "trend\_strength": ...,

&#x20;   "kronos\_embedding": ...

}

```



This becomes the central state store.



\---



\# Multi-Agent Architecture



\# 1. Forecast Agent (Kronos)



\## Purpose



Predict:



\* directional probability

\* expected move magnitude

\* volatility expansion



NOT exact prices.



\---



\## Input State



```python

\[

&#x20;   returns,

&#x20;   volatility,

&#x20;   funding\_rate,

&#x20;   oi\_delta,

&#x20;   liquidation\_delta,

&#x20;   volume\_delta

]

```



Tensor format:



```python

\[batch, time, features]

```



\---



\## Feature Type



\* sequential

\* normalized

\* time-series embeddings



\---



\## Output



```python

{

&#x20;   "p\_up\_0\_5": 0.71,

&#x20;   "p\_down\_0\_5": 0.18,

&#x20;   "expected\_move": 0.42,

&#x20;   "confidence": 0.82,

&#x20;   "expected\_volatility": "high"

}

```



\---



\## Model



```text

Frozen Kronos

&#x20;   ↓

LoRA Adapter

&#x20;   ↓

Probabilistic Head

```



\---



\# 2. Orderflow Agent



\## Purpose



Understand:



\* short-term pressure

\* liquidity imbalance

\* microstructure dynamics



\---



\## Input State



```python

{

&#x20;   "bid\_ask\_imbalance": ...,

&#x20;   "spread": ...,

&#x20;   "cvd": ...,

&#x20;   "trade\_aggression": ...,

&#x20;   "liquidity\_gap": ...,

&#x20;   "depth\_pressure": ...,

&#x20;   "large\_trade\_density": ...

}

```



\---



\## Core Signals



\### Bid/Ask Imbalance



Measures:



\* directional liquidity pressure



\---



\### CVD



Measures:



\* aggressive buying/selling



\---



\### Liquidity Gaps



Measures:



\* thin zones causing rapid movement



\---



\## Output



```python

{

&#x20;   "flow\_bias": "bullish",

&#x20;   "flow\_strength": 0.77,

&#x20;   "liquidity\_risk": 0.33,

&#x20;   "short\_term\_pressure": "up"

}

```



\---



\# 3. Regime Agent



\## Purpose



Detect market condition.



\---



\## Input State



```python

{

&#x20;   "volatility": ...,

&#x20;   "entropy": ...,

&#x20;   "trend\_strength": ...,

&#x20;   "funding": ...,

&#x20;   "volume\_profile": ...,

&#x20;   "liquidation\_pressure": ...

}

```



\---



\## Possible Regimes



```python

\[

&#x20;   "trend\_up",

&#x20;   "trend\_down",

&#x20;   "mean\_reverting",

&#x20;   "breakout",

&#x20;   "high\_volatility",

&#x20;   "low\_liquidity",

&#x20;   "liquidation\_cascade"

]

```



\---



\## Output



```python

{

&#x20;   "regime": "high\_volatility\_breakout",

&#x20;   "confidence": 0.84

}

```



\---



\# 4. Risk Agent



\## Purpose



Global system safety.



This agent can override all other agents.



\---



\## Input State



```python

{

&#x20;   "rolling\_pnl": ...,

&#x20;   "drawdown": ...,

&#x20;   "volatility": ...,

&#x20;   "leverage": ...,

&#x20;   "position\_exposure": ...,

&#x20;   "market\_stress": ...

}

```



\---



\## Responsibilities



\* leverage reduction

\* kill-switch

\* no-trade zone

\* drawdown control

\* exposure management



\---



\## Output



```python

{

&#x20;   "max\_position\_size": 0.3,

&#x20;   "allow\_trade": True,

&#x20;   "risk\_multiplier": 0.7,

&#x20;   "stop\_trading": False

}

```



\---



\# 5. Decision Aggregator



\## Purpose



Combine outputs from all agents.



Agents should NOT trade independently.



\---



\## Input



```python

forecast\_output

orderflow\_output

regime\_output

risk\_output

```



\---



\## Weighted Aggregation



Equation:



genui{"math\_block\_widget\_always\_prefetch\_v2":{"content":"Score = w\_1F + w\_2O + w\_3R - w\_4Risk"}}



Where:



```text

F = forecast score

O = orderflow score

R = regime alignment

Risk = risk penalty

```



\---



\## Output



```python

{

&#x20;   "trade\_direction": "long",

&#x20;   "trade\_confidence": 0.81,

&#x20;   "position\_size": 0.22

}

```



\---



\# 6. RL Execution Agent



\## Purpose



RL should optimize:



\* entry timing

\* execution

\* scaling

\* slippage reduction

\* trade management



NOT direction prediction.



\---



\## Input State



```python

{

&#x20;   "forecast\_probability": ...,

&#x20;   "flow\_bias": ...,

&#x20;   "volatility": ...,

&#x20;   "spread": ...,

&#x20;   "position\_size": ...,

&#x20;   "unrealized\_pnl": ...,

&#x20;   "risk\_multiplier": ...

}

```



\---



\## Actions



```python

\[

&#x20;   "enter\_long",

&#x20;   "enter\_short",

&#x20;   "exit",

&#x20;   "scale\_in",

&#x20;   "scale\_out",

&#x20;   "hold"

]

```



\---



\## RL Algorithms



Recommended:



\* PPO

\* SAC



Avoid:



\* DQN for continuous execution



\---



\## Reward Function



Use risk-adjusted reward.



NOT raw PnL.



\---



\# Online Learning Strategy



DO NOT train continuously every tick.



That becomes unstable.



\---



\## Recommended Schedule



\### Every Few Minutes



\* update state

\* infer signals

\* infer probabilities



\### Every Few Hours



\* retrain LoRA adapter

\* retrain forecasting head



\### Daily



\* evaluate regime drift

\* recalibrate probabilities



\### Weekly



\* RL policy refresh

\* replay training



\---



\# Storage Architecture



\## Hot Storage



Use:



\* Redis

\* Feature cache

\* in-memory state



\---



\## Historical Storage



Use:



\* Parquet

\* DuckDB

\* ClickHouse



For:



\* backtesting

\* offline RL

\* replay simulation



\---



\# Backtesting Requirements



Must support:



\* realistic slippage

\* spread simulation

\* latency simulation

\* partial fills

\* liquidation events

\* funding fees



Without this:



```text

Backtest is invalid.

```



\---



\# Biggest Failure Modes



\## 1. Data Leakage



Common leakage:



\* future candles

\* improper normalization

\* future volatility

\* delayed timestamps



\---



\## 2. RL Reward Hacking



Model learns:



\* overtrading

\* leverage abuse

\* volatility gambling



Need strict risk boundaries.



\---



\## 3. Regime Collapse



Model trained on:



```text

trend market

```



Fails in:



```text

mean-reverting market

```



Need regime awareness.



\---



\## 4. Overfitting



Crypto data is noisy.



Most alpha disappears live.



Need:



\* walk-forward testing

\* out-of-sample validation

\* realistic execution simulation



\---



\# Suggested V1 Stack



\## Streaming



\* Binance Websocket

\* Kafka

\* Redis



\---



\## ML Stack



\* PyTorch

\* Kronos

\* PPO/SAC

\* Ray RLlib



\---



\## Feature Engine



\* NumPy

\* Polars

\* Numba



\---



\## Storage



\* DuckDB

\* Parquet

\* ClickHouse



\---



\## Monitoring



\* Prometheus

\* Grafana



\---



\# Recommended Development Phases



\# Phase 1



Build:



\* websocket ingestion

\* feature engine

\* replay engine

\* offline backtesting



No RL yet.



\---



\# Phase 2



Build:



\* Kronos probability forecasting

\* volatility prediction

\* regime detection



\---



\# Phase 3



Build:



\* orderflow agent

\* liquidity analysis

\* state aggregation



\---



\# Phase 4



Build:



\* RL execution layer

\* adaptive sizing

\* online adaptation



\---



\# Phase 5



Build:



\* continual learning

\* model drift detection

\* adaptive retraining

\* ensemble coordination



\---



\# Final System Philosophy



This system should behave like:



```text

A probabilistic adaptive market-state machine

```



NOT:



```text

A naive price predictor

```



The edge comes from:



\* state understanding

\* volatility adaptation

\* orderflow awareness

\* execution optimization

\* risk control

\* regime transition detection



rather than simple directional forecasting.



