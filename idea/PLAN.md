# Master Plan

This document is the contract between you, me, and any coding agent.
Read it fully before starting any phase.

---

## 1. Goal

Build a probabilistic, regime-aware, multi-agent intraday trading system
for BTC/USDT that:

1. Models **market state**, not raw price (no naive direction prediction).
2. Combines a **time-series foundation model (Kronos)** for forecasts with
   **microstructure signals** (OFI, microprice, Hawkes, VPIN, cross-venue
   basis, funding/OI/liquidation) for state.
3. Uses **supervised learning + classical sizing** for direction, and
   **RL only for execution** (where it actually adds value).
4. Has a **realistic backtest simulator** (queue-aware L2 replay) so
   reported numbers are not fiction.
5. Captures live data continuously and **updates the policy monthly**
   via drift-triggered fine-tuning + canary deploy (never blind retrain).

---

## 2. Non-negotiable principles

- **Realistic costs always.** No backtest result without spread, fees,
  funding, queue position, partial fills.
- **Walk-forward, never random split.** Use López de Prado purged k-fold
  with embargo.
- **Calibration > accuracy.** A 55% calibrated model beats 60% miscalibrated.
- **One change at a time.** Phases ship independently and verifiably.
- **Every action logged.** Decisions, trades, slippage, drift events — all
  to JSONL with ISO-8601 timestamps.
- **No live $ until canary passes for ≥4 weeks at paper level.**

---

## 3. Data strategy — how much history do we need?

### Short answer

- **Pre-training (Phase 4 forecast head):** 12–18 months of BTC 1m+5m
  klines + funding + OI history is enough for the first lightweight pass.
  Use 12 months train / 3 months val / 3 months OOS test (purged + embargoed).
- **Microstructure features (Phase 2):** L2 + tick data is **not freely
  available historically**. Capture from day 1 of Phase 1; you need
  ≥4–6 weeks of live tick history before microstructure agents can be
  trained meaningfully.
- **RL execution (Phase 7):** generate ~50k execution episodes from the
  realistic simulator using accumulated tick data (4+ weeks captured live).
- **Paper trading (Phase 8):** **1 month minimum**, ideally 3 months to
  span at least one regime shift.
- **Monthly continual update (Phase 9):** ~30 days of live+paper data
  per cycle, stored in canonical schema.

### Long answer — why not "more years"

> *"More data = better model"* is mostly false in non-stationary markets.
> Pre-2022 BTC is a different regime (low ETF flow, different exchange
> mix, different leverage profile). 12–24 months of recent data >
> 5 years of mixed-regime data for an intraday system.

### Concrete data sourcing plan

| Need | Source | Cost | Phase |
|---|---|---|---|
| Historical klines 1m, 5m, 15m | `data.binance.vision` (free) | $0 | Phase 1 |
| Historical funding rate | Binance Futures API (free) | $0 | Phase 1 |
| Historical open interest | Binance Futures API (free) | $0 | Phase 1 |
| Historical aggTrades | `data.binance.vision` (free, partial) | $0 | Phase 1 |
| **Live trade stream** | Binance WS `@trade`, `@aggTrade` | $0 | Phase 1 |
| **Live L2 depth** | Binance WS `@depth@100ms`, `@depth20@100ms` | $0 | Phase 1 |
| Coinbase trade lead-lag | Coinbase Pro WS | $0 | Phase 1 |
| Deribit IV / skew | Deribit WS public ticker | $0 | Phase 1 |
| Historical L2 (optional, costs) | Tardis.dev | ~$50–200/mo | Phase 1 *(deferred)* |

**Key insight:** the L2/microstructure features cannot be backfilled for
free. The system must capture live from day 1 and accumulate.

### Data layout (canonical)

```
data/
  raw/
    binance/
      klines_5m/BTCUSDT/{year}/{year}-{month}.parquet
      klines_1m/BTCUSDT/{year}/{year}-{month}.parquet
      funding/BTCUSDT/{year}/{year}-{month}.parquet
      open_interest/BTCUSDT/{year}/{year}-{month}.parquet
      trades/BTCUSDT/{date}.parquet     # live captured from Phase 1
      depth/BTCUSDT/{date}.parquet      # live captured from Phase 1
    coinbase/
      trades/BTC-USD/{date}.parquet
    deribit/
      ticker/BTC-PERPETUAL/{date}.parquet
  features/
    state_5m/BTCUSDT/{date}.parquet     # canonical state vector @ 5m grid
    micro_event/BTCUSDT/{date}.parquet  # event-time microstructure features
runs/
  {RUN_ID}/                              # one folder per backtest / paper run
    config.yaml
    decisions.jsonl
    trades.jsonl
    pnl.parquet
    metrics.json
    log.jsonl
models/
  forecast/v{N}/...
  policy/v{N}/...
  aggregator/v{N}/...
```

`RUN_ID` format: `{mode}-{YYYYMMDD-HHMMSS}-{shortuuid}`,
e.g. `backtest-20250127-103045-a8f3`.

---

## 4. Training cadence

```
┌─ One-time pre-training ────────────────────────────────────────────┐
│ Phase 4: Train forecast head on 12 mo historical klines            │
│ Phase 6: Train aggregator + sizing on 12 mo historical             │
│ Phase 7: Train RL execution policy offline (CQL) on accumulated    │
│          tick data + simulated episodes                            │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ Paper trading loop (Phase 8) ─────────────────────────────────────┐
│ Daily: capture all WS streams, compute features, run agents,       │
│        store decisions + simulated fills + PnL                     │
│ Weekly: monitor drift detector (KSWIN/ADWIN); alert if firing      │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ Monthly continual update (Phase 9) ───────────────────────────────┐
│ 1. Snapshot last 30 days of live+paper data                        │
│ 2. Recompute features on snapshot                                  │
│ 3. Retrain forecast LoRA adapter on snapshot+anchor (EWC)          │
│ 4. Recalibrate (isotonic) on most-recent 7 days                    │
│ 5. RL adapter fine-tune on new episodes (conservative LR)          │
│ 6. Spawn shadow/canary policy v(N+1)                               │
│ 7. Run canary for 7 days; promote only if Sharpe ≥ floor and DD    │
│    ≤ live policy's recent DD                                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase index

Each phase has a dedicated spec under `phases/`. Order is mandatory.

| # | Phase | What ships | CLI added |
|---|---|---|---|
| 0 | Setup | Repo skeleton, deps, logging, CI | — |
| 1 | Data | Historical download + live WS capture | `intraday data ...` |
| 2 | Features | Microstructure feature engine | `intraday features ...` |
| 3 | Simulator | Queue-aware L2 backtest simulator | `intraday backtest run` (with stub strategy) |
| 4 | Forecast | Kronos + TCN + meta-label + calibration | `intraday train forecast`, `intraday predict ...` |
| 5 | Other agents | Orderflow / Regime / Risk / Stay-out | `intraday agent ...` |
| 6 | Aggregator + sizing | Meta-learner + fractional Kelly | `intraday train aggregator`, end-to-end backtest works |
| 7 | RL execution | CQL offline policy for execution | `intraday train policy` |
| 8 | Paper trading | Live decisions, simulated fills, capture | `intraday paper run` |
| 9 | Continual | Drift detection + monthly update + canary | `intraday update run`, `intraday canary ...` |
| 10 | Live | Tiny-size live with kill-switch | `intraday live run` |

---

## 6. Definition of "done" for a phase

Every phase ships only when **all four** of these are true:

1. **All unit tests pass** (`uv run pytest tests/phase_NN/ -v`).
2. **Integration test passes** (the phase's smoke-test command in its spec).
3. **Acceptance criteria met** (objective numbers, defined per phase).
4. **Logs clean** (no ERRORs, every action has a structured event).

If a phase fails its acceptance criteria, **do not move forward**; either
fix the phase or document why the acceptance bar should be lowered.

---

## 7. Risk profile defaults (modifiable in `config/`)

These are the safe defaults for the conservative track. Adjust later.

```yaml
instrument: BTCUSDT_PERP            # Binance USDM Perp
max_leverage: 1.0                   # 1x to start
direction_modes: [long, short, flat]
max_position_usd: 200               # absolute cap, paper
max_daily_drawdown_pct: 2.0         # hard kill-switch
max_open_positions: 1
maker_fee_bps: 2.0                  # 0.02% (Binance VIP 0)
taker_fee_bps: 5.0                  # 0.05%
funding_charge_per_8h_pct: 0.01     # placeholder; real-time loaded
slippage_model: queue_aware         # never use naive
kill_switch_triggers:
  - daily_drawdown_exceeded
  - drift_detector_alarm
  - basis_dispersion_zscore_gt: 3.0
  - latency_p99_gt_ms: 500
```

---

## 8. Open decisions deferred (to be resolved in their phase)

- Coinbase / Deribit subscription levels (Phase 1).
- Kronos checkpoint size (small vs base) — Phase 4 will benchmark both.
- Whether to include on-chain whale flow (Phase 5, optional).
- Tardis.dev historical L2 buy decision (Phase 1; defer until live capture
  budget is exhausted).

---

## 9. What this plan deliberately does NOT do (yet)

- Multi-asset (only BTC).
- Cross-margin / portfolio optimization.
- Options trading on Deribit (only IV as a feature).
- Distributed training (single machine, single GPU is enough).
- Web UI / dashboard (CLI only; one read-only HTML report per run).

These are explicitly out of scope for v1. Resist scope creep.
