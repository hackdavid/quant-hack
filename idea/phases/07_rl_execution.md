# Phase 7 — RL Execution Policy (offline CQL)

**Goal:** an RL policy that **decides only how to fill a given decision**
— `{post-only, IOC, market}` × `{tick-offset, urgency}`. Direction and
size are already decided by Phase 6.

**This is the only place RL belongs in the system.** Direction-RL
overfits; execution-RL has a clean reward signal (slippage vs
benchmark) and a tractable action space.

**Why seventh:** by now the supervised pipeline (Phase 6) has shown
≥ 1.0 OOS Sharpe. The only remaining alpha is execution efficiency.

**Estimated effort:** 7–10 days.

**Activates dep group:** `phase7` (d3rlpy, gymnasium).

**Prerequisite:** Phase 6 acceptance #4 passed AND ≥ 4 weeks of
captured live tick data.

---

## 1. Inputs / outputs

- **Inputs:**
  - Phase 6 decisions (`v5_full_no_rl` produces a stream of
    `Decision` + target USD).
  - Phase 1 captured tick + depth data for realistic sim.
- **Outputs:**
  - `models/policy/v{N}/{cql.safetensors, metadata.json}`.
  - Strategy `v6_full_with_rl` that uses the RL execution policy.
  - **Acceptance: realized slippage reduced by ≥ 20% vs the
    Almgren-Chriss baseline at equal fill rate**, OR strategy OOS Sharpe
    increases by ≥ 0.1 vs `v5_full_no_rl`. Either is sufficient.

---

## 2. Files to create

```
src/intraday/rl/
  __init__.py
  env.py                       # gymnasium env: ExecutionEnv
  state.py                     # build state vector from sim ctx
  action.py                    # action space + decoding
  reward.py                    # differential Sharpe + slippage + cost
  baseline.py                  # Almgren-Chriss benchmark execution
  data_collection.py           # generate offline dataset from sim
  train.py                     # CQL training loop
  predict.py                   # inference path
  cli.py
src/intraday/sim/strategies/
  v6_full_with_rl.py
tests/phase_07/
  test_env.py
  test_state.py
  test_action.py
  test_reward.py
  test_baseline.py
  test_data_collection.py
  test_train_smoke.py
```

---

## 3. The RL problem — sized correctly

### State (15-dim)

```
- ts_normalized                    # how far into the 5m window we are
- target_usd_normalized            # remaining target (signed) / equity
- already_filled_usd               # / target
- microprice_drift_5m_z
- spread_bps
- ofi_5m_l5_z
- queue_imbalance_l5
- vpin
- volatility_regime_id (one-hot 3-dim)
- forecast_confidence
- recent_fill_slippage_bps_avg     # last 10 fills
- time_in_window_remaining_s
- recent_cancel_rate
```

### Action (4-dim continuous, decoded to 6 discrete intents)

The policy outputs a 4-dim continuous vector that decodes to:

```
a[0]: order_type      (sigmoid → 0=post-only, 0.33=limit-IOC, 0.66=market, 1.0=cancel-all)
a[1]: tick_offset     (tanh × 5 → [-5..+5] ticks from microprice)
a[2]: child_size_pct  (sigmoid → fraction of remaining target to send now)
a[3]: urgency         (sigmoid → if > 0.7, cancel any resting and re-send)
```

Decoding is deterministic. This continuous-then-decode pattern is more
sample-efficient than direct discrete action.

### Reward

Per-step reward (each step = one event in the execution window):

```python
reward_t = (
    -slippage_bps_t                       # primary cost (negative reward)
    - alpha * spread_bps_t * is_taker     # taker-fee cost
    + beta * is_filled                    # small bonus for fills (avoid loitering)
    - gamma * cancel_count_t              # cancel penalty
    - delta * (window_overshoot_indicator) # heavy penalty for missing window
)
```

Episode-level shaping (added at terminal):

```
+ E_episode_pnl - E_baseline_pnl     # vs Almgren-Chriss baseline run on same data
```

This makes the RL **compete against AC**, which is the right benchmark.

### Episode

- Starts when an aggregator decision arrives.
- Ends when target is filled OR 5-minute window elapses OR direction
  changes.

---

## 4. The Almgren-Chriss baseline (must work first)

```python
class AlmgrenChrissBaseline:
    """Optimal execution baseline.

    Splits target into N child orders along an AC trajectory (cosine).
    Each child is post-only at microprice for first half of slot, then
    IOC for the remainder.
    """
    def step(self, state) -> Action: ...
```

This is **not optional**. The RL is judged against it. Implement it
first; benchmark its slippage; only then train the RL.

---

## 5. Offline data collection

```python
def collect_offline_dataset(
    *,
    start: dt.datetime,
    end: dt.datetime,
    strategy: Literal["baseline", "perturbed_baseline"] = "perturbed_baseline",
    n_episodes: int = 50_000,
) -> pl.DataFrame:
    """Run the simulator with the baseline (with action noise) and
    record (s, a, r, s', done) tuples.
    """
```

**Critical**: include action perturbations during data collection —
random ε ~ Normal(0, 0.2) added to baseline actions — so the offline
dataset covers actions outside the baseline's narrow distribution.
Without this, CQL will fail to extrapolate.

---

## 6. CQL training

Use `d3rlpy`:

```python
from d3rlpy.algos import CQLConfig

cfg = CQLConfig(
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    batch_size=256,
    n_action_samples=10,
    alpha=2.0,                 # CQL conservatism — tune in [0.5, 5.0]
)
algo = cfg.create()
algo.fit(dataset, n_steps=200_000, n_steps_per_epoch=10_000, ...)
algo.save_model("models/policy/v1/cql.safetensors")
```

Validation during training:
- After every epoch, run policy on a held-out month with the realistic
  simulator; record realized slippage vs baseline.
- Track OPE (off-policy evaluation) score (FQE) as a sanity check.

---

## 7. Strategy `v6_full_with_rl`

Identical to `v5_full_no_rl` but:
- The `_execute_with_simple_post_only` method is replaced by the RL
  policy.
- The aggregator still produces direction + size; RL only decides how
  to execute.

---

## 8. CLI

| Command | Behavior |
|---|---|
| `intraday rl collect-data --start ... --end ... --episodes 50000` | offline dataset |
| `intraday train policy --algo cql --mode offline --data-from ... --data-to ...` | CQL training |
| `intraday rl evaluate --version v1 --start 2024-12-01 --end 2024-12-31` | OOS slippage report |
| `intraday backtest run --strategy v6_full_with_rl --start ... --end ...` | full strategy with RL |
| `intraday train policy --mode finetune --from-version v3 --since 30d` | monthly fine-tune (Phase 9) |

---

## 9. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | AC baseline implemented and produces fills with median slippage < 3 bps on liquid hours | `rl evaluate --version baseline` |
| 2 | Offline dataset has ≥ 50k transitions, with action coverage measured by std(a)/range(a) > 0.15 in each dim | data_collection report |
| 3 | CQL training converges (FQE plateau) within 200k steps | training log |
| 4 | OOS realized slippage of v1 policy < 0.8 × AC baseline slippage at equal fill rate | rl evaluate report |
| 5 | OR: v6_full_with_rl OOS Sharpe ≥ v5_full_no_rl OOS Sharpe + 0.1 | backtest comparison |
| 6 | No catastrophic actions in OOS run (no orders > 10 ticks from microprice; no cancel-storms) | log audit |
| 7 | Policy inference latency p99 < 5 ms | benchmark |
| 8 | Determinism: same seed + same dataset → same checkpoint hash | seeded test |

**If acceptance #4 OR #5 fails**, the RL adds no value over Almgren-
Chriss. **Ship v5_full_no_rl** to paper trading without RL. Don't force
RL where it doesn't help.

---

## 10. Common mistakes to avoid

- **Don't try to learn direction with RL.** Direction is decided by
  Phase 6. RL only executes.
- **Don't train online (live).** Online RL on real markets is unsafe
  and inefficient. Always offline → canary → carefully gated online
  fine-tune.
- **Don't underweight the cancel penalty.** Without it, the policy
  learns cancel-storms that destroy the order book and your latency.
- **Don't compare RL to "no execution" baseline.** Compare to AC.
  The goal is to beat the optimal classical baseline.
- **Don't forget perturbed-baseline data collection.** Pure baseline
  data → CQL has no out-of-distribution information → useless policy.
- **Don't tune CQL alpha by OOS reward.** Use OPE during training; reserve
  OOS for the final evaluation.
- **Don't deploy without a hard kill-switch on the RL policy.** Phase 8
  must be able to override it.
- **Don't skip latency in the env.** RL trained without latency models
  will overfit to instant-execution assumptions and break in paper.

---

## 11. Done ⇒ proceed to `phases/08_paper_trading.md`.
