# Phase 6 — Aggregator + Sizing (no RL yet)

**Goal:** combine all four agent opinions + forecast into a single
trade decision, with a learned regime-conditional gate, then size with
**fractional Kelly + CVaR cap**. **No RL yet** — this phase must produce
a strategy with **OOS Sharpe ≥ 1.0 with realistic costs**, otherwise
the system has no alpha to give RL.

**Why sixth:** RL is amplification, not creation. If the supervised
+ classical pipeline doesn't beat its baseline here, RL will not save it.

**Estimated effort:** 4–6 days.

**Activates dep group:** `phase6` (lightgbm).

---

## 1. Inputs / outputs

- **Inputs:** Phase 4 forecast + Phase 5 agent opinions + Phase 2 features.
- **Outputs:**
  - `models/aggregator/v{N}/`:
    - `meta_learner.lgbm`           # stacked GBM
    - `metadata.json`
  - `intraday train aggregator` CLI.
  - End-to-end strategy `v5_full_no_rl` that uses every agent and ships
    real PnL through the simulator.
  - **Acceptance: OOS Sharpe ≥ 1.0** on Dec 2024 with full realistic costs.

---

## 2. Files to create

```
src/intraday/aggregator/
  __init__.py
  features.py                # build aggregator feature row from agent outputs
  meta_learner.py            # LightGBM stacked classifier
  decision.py                # final decision logic (gate + side + horizon)
  sizing.py                  # fractional Kelly + CVaR cap
  cli.py
src/intraday/sim/strategies/
  v5_full_no_rl.py           # the integrated strategy
tests/phase_06/
  test_features.py
  test_meta_learner.py
  test_decision.py
  test_sizing.py
  test_v5_smoke.py
```

---

## 3. The aggregator's input feature row

For each 5-minute boundary, build a feature row with:

```python
{
    # Forecast agent outputs:
    "fc_p_up": float,
    "fc_p_down": float,
    "fc_expected_move_sigma": float,
    "fc_confidence": float,
    "fc_meta_act": int,            # 0/1
    "fc_meta_p_correct": float,

    # Orderflow:
    "of_flow_bias": float,
    "of_flow_strength": float,
    "of_step_away": int,
    "of_vpin": float,

    # Regime:
    "rg_regime": str,              # categorical, will be one-hot
    "rg_max_prob": float,
    "rg_is_transition": int,
    "rg_vol_regime": str,

    # Risk:
    "rk_risk_multiplier": float,
    "rk_allow_trade": int,
    "rk_stop_trading": int,

    # Stay-out:
    "so_mode": str,                # categorical
    "so_score": float,

    # Raw context:
    "spread_bps": float,
    "realized_vol_30m": float,
    "funding_z": float,
    "basis_bn_cb_z": float,
    "iv_25d_rr": float,
    "hour_of_day_utc": int,
    "minute_of_hour": int,
    "day_of_week": int,
}
```

This is the X for the meta-learner.

---

## 4. The label for the meta-learner

**Triple-barrier label over 15-minute horizon** (from Phase 4 labels).
Convert to:
- `y = 1` if barrier-touch is on profit side (matching forecast direction).
- `y = 0` otherwise.

This is **meta-labeling at the system level** (not just at the forecast
level): it asks "given everything we know right now, will trading the
forecast direction be profitable?"

---

## 5. Meta-learner training

### `src/intraday/aggregator/meta_learner.py`

```python
class MetaLearner:
    def __init__(self, model_dir: Path) -> None: ...

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        *,
        ts: pl.Series,
        n_folds: int = 5,
        embargo_pct: float = 0.01,
    ) -> dict[str, float]:
        """LightGBM, purged k-fold CV.
        Returns OOF metrics: AUC, Brier, calibration ECE.
        """

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> pl.DataFrame: ...
```

Hyper-params (locked):

```yaml
lgbm:
  num_leaves: 31
  learning_rate: 0.03
  n_estimators: 500
  min_data_in_leaf: 100
  reg_alpha: 0.1
  reg_lambda: 0.1
  early_stopping_rounds: 25
```

Output OOF metrics MUST be reported in `metadata.json`. If OOF AUC
< 0.55, the system has no alpha — stop and investigate.

---

## 6. Decision module

### `src/intraday/aggregator/decision.py`

```python
class Decision(BaseModel):
    ts_ms: int
    side: Literal["long", "short", "flat"]
    confidence: float                 # meta-learner probability
    horizon_minutes: int = 15

class DecisionEngine:
    def __init__(
        self,
        meta_learner: MetaLearner,
        *,
        threshold: float = 0.55,       # learned from OOF curve
    ) -> None: ...

    def decide(
        self,
        agent_features: dict,
        forecast_output: ForecastOutput,
    ) -> Decision: ...
```

Logic:

```python
if rk_stop_trading or so_mode == "stay_out":
    return Decision(side="flat", confidence=0.0)

if not fc_meta_act:
    return Decision(side="flat", confidence=0.0)

p_correct = meta_learner.predict_proba(X)[0, 1]

if p_correct < threshold:
    return Decision(side="flat", confidence=p_correct)

side = "long" if fc_p_up > fc_p_down else "short"
return Decision(side=side, confidence=p_correct)
```

The threshold is selected on OOF predictions to maximize OOS Sharpe via
grid search over `[0.50, 0.52, ..., 0.70]`.

---

## 7. Sizing module — fractional Kelly + CVaR cap

### `src/intraday/aggregator/sizing.py`

```python
class SizingEngine:
    def __init__(
        self,
        *,
        kelly_fraction: float = 0.25,    # quarter-Kelly
        cvar_alpha: float = 0.05,
        cvar_cap_usd: float,             # from config
        max_position_usd: float,
    ) -> None: ...

    def size_usd(
        self,
        decision: Decision,
        *,
        expected_edge_bps: float,        # from forecast.expected_move_sigma * vol
        vol_30m_bps: float,
        risk_multiplier: float,
    ) -> float:
        """Returns USD notional to allocate (signed).
        """
```

Closed-form fractional Kelly, then clamp by CVaR:

```python
edge = expected_edge_bps / 10_000      # in returns
variance = (vol_30m_bps / 10_000) ** 2
kelly_f = edge / variance              # full-Kelly fraction
target_f = kelly_fraction * kelly_f * confidence

usd = target_f * account_equity_usd
usd = clamp(usd, -max_position_usd, +max_position_usd)
usd = usd * risk_multiplier

# CVaR cap: ensure that worst-5% loss on this position size is bounded
expected_cvar = abs(usd) * vol_30m_bps / 10_000 * cvar_multiplier_5pct
if expected_cvar > cvar_cap_usd:
    usd = usd * (cvar_cap_usd / expected_cvar)

return signed_usd
```

`cvar_multiplier_5pct` for normal returns ≈ 2.06; for fat-tailed crypto
returns we calibrate empirically to ≈ 2.5–3.0.

This is **not RL.** It's closed-form, transparent, audit-able. Most of
your sizing alpha lives here.

---

## 8. Strategy `v5_full_no_rl`

```python
class V5FullNoRL(Strategy):
    def on_event(self, event, ctx):
        if event.kind != "bar_5m":
            return []

        forecast = self.forecast_model.predict(...)
        opinions = {
            "orderflow": self.orderflow.predict(...),
            "regime": self.regime.predict(...),
            "risk": self.risk.predict(...),
            "stay_out": self.stay_out.predict(...),
        }
        agent_features = build_aggregator_row(forecast, opinions, ctx)
        decision = self.decision_engine.decide(agent_features, forecast)

        if decision.side == "flat":
            # if currently in a position and we said flat, exit
            return self._exit_orders(ctx)

        size_usd = self.sizing.size_usd(decision, ...)
        return self._execute_with_simple_post_only(decision, size_usd, ctx)
```

Execution in `v5` is intentionally simple: post-only at microprice with
3-tick offset, cancel-and-replace if not filled in 30s, fall back to IOC.
**Phase 7 will replace this with RL.**

---

## 9. CLI

| Command | Behavior |
|---|---|
| `intraday train aggregator --start ... --end ... --val-end ...` | full training pipeline |
| `intraday inspect aggregator --version v1` | feature importance, calibration plot |
| `intraday backtest run --strategy v5_full_no_rl --start 2024-12-01 --end 2024-12-31` | OOS run |

---

## 10. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | OOF AUC of meta-learner ≥ 0.58 | training metadata |
| 2 | OOF Brier ≤ baseline (always-predict-base-rate) − 0.02 | training metadata |
| 3 | Top-5 feature importances contain ≥ 2 from forecast and ≥ 1 from orderflow | feature importance table |
| 4 | **OOS Sharpe (Dec 2024) ≥ 1.0** with realistic costs | backtest report |
| 5 | OOS max drawdown ≤ 8% | backtest report |
| 6 | OOS hit rate when `confidence > 0.6` strictly higher than baseline | inspect log |
| 7 | OOS turnover ≤ 50× / month (no overtrading) | metrics.json |
| 8 | Re-running with same seed gives identical run metrics | seeded test |

**Acceptance #4 is the make-or-break gate for the entire system.**
If you can't reach OOS Sharpe ≥ 1.0 here:
- Check label leakage (Phase 4 acceptance #8).
- Check feature leakage (Phase 2 features must not look forward).
- Check sim realism (Phase 3 acceptance #1).
- Re-examine the threshold tuning grid.
- **Do not proceed to Phase 7.** RL on top of a no-alpha pipeline is
  pure noise amplification.

---

## 11. Common mistakes to avoid

- **Don't tune the threshold on the OOS month.** Use OOF predictions only.
- **Don't include features that include future information.** Audit
  every feature for `t+something` references.
- **Don't use raw probability without calibration.** The meta-learner's
  raw output must be passed through isotonic calibration if its
  reliability plot deviates from the diagonal.
- **Don't full-Kelly.** Quarter-Kelly is the empirically robust choice;
  half-Kelly only if you have very high confidence in your variance
  estimate (you don't).
- **Don't skip CVaR cap.** Without it, one bad regime-shift trade can
  exceed your daily loss limit.
- **Don't include `risk_multiplier` twice.** It's already in sizing —
  do not also gate the decision on it.
- **Don't try to "fix" low Sharpe by lowering threshold.** Lower
  threshold = more trades = worse PnL. Fix the upstream signal instead.

---

## 12. Done ⇒ proceed to `phases/07_rl_execution.md`.

**Important:** at this point the system can paper-trade as v5 without
ever running RL. If your goal is "fastest path to a live thin slice",
you can skip to Phase 8 with v5 and add RL later. **I recommend doing
that** — collect 1 month of paper data with v5, then add RL in Phase 7
with that real data instead of synthetic episodes.
