# Phase 5 — Orderflow / Regime / Risk / Stay-Out Agents

**Goal:** complete the agent layer with four specialist agents that consume
Phase 2 features and emit structured opinions for the aggregator (Phase 6).

**Why fifth:** the forecast agent alone is necessary but not sufficient.
These four specialist agents add the regime + microstructure + safety
signals that turn forecast into a robust system.

**Estimated effort:** 5–7 days.

**Activates dep group:** `phase5` (hmmlearn).

---

## 1. Inputs / outputs

- **Inputs:** Phase 2 `state_5m` and `micro_event` features.
- **Outputs:**
  - 4 agent classes, each with `predict()` returning a typed opinion.
  - `intraday agent <name> --at <ts>` for debug inference.
  - 4 trained sub-models (HMM, GBM, etc.) saved under `models/agents/`.
  - All four plug into the simulator via the same `Strategy` interface
    so they can be backtested individually.

---

## 2. Files to create

```
src/intraday/agents/
  __init__.py
  base.py                    # Agent ABC + AgentOpinion base class
  orderflow.py               # OrderflowAgent
  regime.py                  # RegimeAgent (HMM + GBM)
  risk.py                    # RiskAgent (rule-based)
  stay_out.py                # StayOutDetector
  registry.py
  cli.py
tests/phase_05/
  test_orderflow.py
  test_regime.py
  test_risk.py
  test_stay_out.py
```

---

## 3. Common opinion contract

```python
class AgentOpinion(BaseModel):
    agent: str                   # "orderflow" | "regime" | "risk" | "stay_out"
    ts_ms: int
    payload: dict[str, Any]      # agent-specific
    confidence: float            # 0..1
    inference_ms: float
```

Concrete payloads below.

---

## 4. OrderflowAgent

### Purpose
Read microstructure features (OFI, microprice drift, Hawkes, VPIN, spread)
and emit a short-term **flow bias**.

### Implementation
Pure functional, no learned model — these features are already engineered.
The agent computes a weighted aggregate score:

```
flow_bias = w1 * sign(ofi_5m_l5) * |ofi_5m_l5|
          + w2 * sign(hawkes_imbalance) * |hawkes_imbalance|
          + w3 * sign(microprice_drift) * |microprice_drift|
          - w4 * vpin                    # high VPIN = step away
```

Weights `w_i` are not learned; they are normalized so each term
contributes equally on its rolling-z-scored variant. The OrderflowAgent
is intentionally simple — its job is to **pre-compute** signals; the
**weighting** that actually goes into a trade decision happens in the
aggregator (Phase 6) which learns regime-conditional weights.

### Output
```python
{
    "agent": "orderflow",
    "ts_ms": ...,
    "payload": {
        "flow_bias": float,            # signed, roughly [-1, +1]
        "flow_strength": float,        # |flow_bias|
        "vpin": float,
        "step_away": bool,             # True if vpin > 0.7 OR spread > 5×median
        "ofi_l5_z": float,
        "hawkes_imb": float,
    },
    "confidence": float,
    ...
}
```

### Test cases
- All-zero microstructure features → flow_bias = 0, confidence = 0.
- Strong positive OFI + Hawkes → flow_bias > 0.5.
- VPIN = 0.85 → step_away = True regardless of other features.

---

## 5. RegimeAgent

### Purpose
Classify the current market into a regime label, with a learned
transition matrix so we know when transitions are *imminent*.

### Implementation
**Two-stage** ensemble:

1. **HMM with 6 hidden states** trained on a feature subset:
   `(z_return_5m, realized_vol_30m, hurst, trend_strength, funding_z, jump_z_score)`.
   - Use `hmmlearn.GaussianHMM`.
   - Fit on 12 months of historical state_5m.
   - States are unlabeled; cluster them post-hoc into named regimes
     based on mean feature values.

2. **GBM (LightGBM) classifier** trained with **manual labels** for the
   regimes we actually care about. Target labels:
   - `trend_up`, `trend_down`, `mean_revert`, `breakout`,
     `high_volatility`, `low_liquidity`, `liquidation_cascade`.
   - Manual labeling: use rule-based heuristics on ground truth (e.g.
     `liquidation_cascade` if `liq_pressure > 0.7 AND realized_vol_30m
     > 95th percentile`).

3. **Combine:** the HMM gives a posterior over hidden states; the GBM
   maps that posterior + raw features → labeled regime. The HMM is
   used for **transition forecasting** (next-step regime probabilities).

### Output
```python
{
    "agent": "regime",
    "ts_ms": ...,
    "payload": {
        "regime": "trend_up",
        "regime_probs": {"trend_up": 0.62, "trend_down": 0.04, ...},
        "next_regime_probs": {"trend_up": 0.55, "mean_revert": 0.18, ...},
        "is_transition": bool,         # True if max(next) < 0.5
        "vol_regime": "high" | "normal" | "low",
    },
    "confidence": float,               # max regime prob
    ...
}
```

### Persistence
`models/agents/regime/v1/`:
- `hmm.pkl` (hmmlearn dump)
- `gbm.lgbm`
- `metadata.json`

### Test cases
- HMM trained on synthetic 2-regime data recovers both regimes.
- GBM AUC > 0.8 on held-out month for each regime label.
- `is_transition` = True when regime probabilities are near uniform.

---

## 6. RiskAgent

### Purpose
**Rule-based, no learning.** Track account state and emit hard caps.

### Inputs (from sim/account state, not from Phase 2 directly)
- Rolling 1d, 1w PnL
- Current drawdown
- Realized volatility
- Current position exposure
- Total trades today

### Output
```python
{
    "agent": "risk",
    "ts_ms": ...,
    "payload": {
        "max_position_size_btc": float,       # cap from this agent
        "risk_multiplier": float,             # 0..1, scales aggregator's size
        "allow_trade": bool,
        "stop_trading": bool,                 # hard halt; aggregator must respect
        "reason": str,                         # human-readable
    },
    "confidence": 1.0,
    ...
}
```

### Logic
```python
if drawdown_today > config.max_daily_drawdown_pct:
    stop_trading = True
    reason = "daily DD limit"
elif vol_regime == "high":
    risk_multiplier = 0.5
    reason = "high vol — half size"
elif n_trades_today > config.max_trades_per_day:
    allow_trade = False
    reason = "trade count cap"
elif account.position_exposure_usd > config.hard_caps.max_total_exposure_usd:
    allow_trade = False
    reason = "exposure cap"
else:
    risk_multiplier = 1.0
    allow_trade = True
```

The aggregator must multiply its sizing by `risk_multiplier`. The
simulator must hard-block any order when `stop_trading=True` regardless
of strategy intent (defense-in-depth).

### Test cases
- DD > limit → stop_trading=True.
- Vol regime = high → risk_multiplier = 0.5.
- Position cap reached → allow_trade=False.

---

## 7. StayOutDetector — the news-shock killer

### Purpose
Replace the brainstorm doc's wrong "treat news as outliers" with a
**stay-out detector** that recognizes regimes in which we should not
trade at all.

### Inputs
- `realized_vol_5m` z-score over rolling 1h
- `spread_bps_avg_5m` z-score over rolling 1h
- `basis_bn_cb_z`
- `iv_atm` change rate
- `n_jumps_last_5m` (count of |r| > 3σ events)
- Optional: `liq_pressure`

### Logic
Joint-z-score detector:

```
score = max(
    z(realized_vol_5m),
    z(spread_bps_avg_5m),
    abs(basis_bn_cb_z),
    z(iv_atm_change),
    z(n_jumps_last_5m)
)

if score > 3.0:        STAY_OUT (kill all open orders, no new orders)
elif score > 2.0:      DEFENSIVE (mean-revert only, half size)
else:                  NORMAL
```

The thresholds are calibrated on history so the detector fires for
known events (e.g. on Bitcoin: ETF approval day Jan 11 2024, Trump
tweets, FOMC days). Document each calibration date in
`docs/stay_out_calibration.md`.

### Output
```python
{
    "agent": "stay_out",
    "ts_ms": ...,
    "payload": {
        "mode": "normal" | "defensive" | "stay_out",
        "score": float,
        "drivers": {"realized_vol_z": ..., "spread_z": ..., ...},
    },
    "confidence": float,
    ...
}
```

The aggregator + simulator must respect `mode == "stay_out"` as a
**hard halt** equivalent to RiskAgent's `stop_trading=True`.

### Test cases
- All inputs at z=0 → mode=normal.
- One driver at z=3.5 → mode=stay_out.
- Two drivers at z=2.5 each (max=2.5) → mode=defensive.
- Calibration test: known event days produce mode=stay_out.

---

## 8. CLI

| Command | Behavior |
|---|---|
| `intraday agent train regime --start ... --end ...` | trains HMM + GBM |
| `intraday agent inspect orderflow --start ... --end ...` | distribution stats |
| `intraday agent predict <name> --at <ts>` | single-call inference |
| `intraday backtest run --strategy v3_orderflow_only ...` | smoke-test using just OrderflowAgent |
| `intraday backtest run --strategy v4_regime_only ...` | smoke-test using just RegimeAgent (long when trend_up, short when trend_down) |

---

## 9. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | HMM converges on 12 months data, log-likelihood improves over baseline GMM by ≥ 5% | training log |
| 2 | GBM regime classifier macro-F1 > 0.6 on held-out month | training report |
| 3 | StayOut detector triggers `mode=stay_out` on at least 80% of pre-labeled "shock" days | calibration test |
| 4 | StayOut detector false-positive rate < 5% on quiet days | calibration test |
| 5 | OrderflowAgent + RegimeAgent + StayOut + Risk all run within 5ms total inference at 5m boundary | benchmark |
| 6 | v3_orderflow_only OOS Sharpe > 0 (positive expectation) | backtest |
| 7 | v4_regime_only OOS Sharpe > 0 | backtest |

Acceptance #6 and #7 are **lower bars than Phase 4's #6** because
specialist signals are noisier alone. The aggregator (Phase 6) is what
combines them into a positive-Sharpe system.

---

## 10. Common mistakes to avoid

- **Don't fit HMM on raw returns.** Always on standardized residual series
  (z-scored over rolling vol).
- **Don't use too many HMM states.** 6 is enough; more = overfit.
- **Don't make StayOutDetector "smart".** Simple thresholds beat ML for
  this safety-critical role.
- **Don't let agents share state via globals.** Every agent must be a
  pure function of its declared inputs.
- **Don't skip the v3 / v4 backtests.** They surface bugs that don't
  show up in unit tests but do show up in PnL.
- **Funding rate is published every 8h, not continuous.** Don't compute
  `funding_z` from a constant.

---

## 11. Done ⇒ proceed to `phases/06_aggregator_sizing.md`. **You finally
have all the building blocks for an end-to-end strategy.**
