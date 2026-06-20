# Phase 9 — Continual Update (drift-triggered + monthly cadence)

**Goal:** safely update the deployed models with the most recent
month's data **without catastrophic forgetting** and **without blind
deployment**. This is what makes the system adapt to current patterns
instead of yesterday's.

**Why ninth:** Phase 8 has been running for 30+ days, capturing
canonical data. Now we use it to update — but every update is
EWC-anchored, validated on a held-out canary slice, and shadow-tested
before promotion.

**Estimated effort:** 4–6 days (one-time build) then ~30 min/month
(running it).

**Activates dep group:** `phase9` (river).

---

## 1. Inputs / outputs

- **Inputs:**
  - Last 30 days of `runs/paper-{date}/` data.
  - Last 30 days of `data/raw/.../live/...` raw captures.
  - Current model versions: forecast `vN`, aggregator `vN`, policy `vN`.
- **Outputs:**
  - `intraday update run` workflow CLI.
  - Drift report.
  - Updated model versions `v(N+1)` (not yet promoted).
  - Canary run results.
  - **Promotion is manual.** No auto-promote unless explicit flag.

---

## 2. Files to create

```
src/intraday/update/
  __init__.py
  workflow.py                  # the orchestration of the 7-step update
  snapshot.py                  # snapshot last-30d data from runs/ + data/raw/
  drift_report.py              # comprehensive drift summary
  retrain_forecast.py          # LoRA finetune + EWC anchoring + recalibrate
  retrain_aggregator.py        # incremental fit on new month
  retrain_policy.py            # CQL fine-tune with conservative LR
  canary.py                    # run shadow policy alongside live
  promote.py                   # version promotion gate
  ewc.py                       # Elastic Weight Consolidation utilities
  cli.py
tests/phase_09/
  test_snapshot.py
  test_drift_report.py
  test_ewc.py
  test_retrain_forecast.py
  test_retrain_policy.py
  test_canary.py
  test_promote.py
```

---

## 3. The 7-step workflow

```python
def monthly_update(
    *,
    since: dt.timedelta = dt.timedelta(days=30),
    components: list[str] = ["forecast", "aggregator", "policy"],
    auto_promote: bool = False,
) -> UpdateReport:
    # 1. Snapshot
    snap = snapshot_last_n_days(since)

    # 2. Recompute features over snapshot (idempotent)
    features = compute_features(snap)

    # 3. Drift report
    drift = generate_drift_report(features, current_models)
    if drift.severity == "critical":
        log.warning("update.drift_critical", ...)
        # don't auto-skip; user must decide

    # 4. Retrain forecast (LoRA + EWC) + recalibrate
    if "forecast" in components:
        new_forecast = retrain_forecast(
            snapshot=snap,
            base_version=latest_forecast_version(),
            ewc_lambda=0.5,
        )

    # 5. Retrain aggregator (warm-start from previous, append new fold)
    if "aggregator" in components:
        new_agg = retrain_aggregator(snap, base_version=latest_agg_version())

    # 6. Retrain policy (offline CQL fine-tune, conservative LR)
    if "policy" in components:
        new_policy = retrain_policy(
            snapshot=snap,
            base_version=latest_policy_version(),
            learning_rate=3e-5,        # 1/10 of pretrain
            n_steps=20_000,             # 1/10 of pretrain
            ewc_lambda=1.0,
        )

    # 7. Run canary
    canary_result = run_canary(
        new_versions={
            "forecast": new_forecast,
            "aggregator": new_agg,
            "policy": new_policy,
        },
        duration=dt.timedelta(days=7),
        live_versions=current_models,
    )

    return UpdateReport(...)
```

---

## 4. EWC — Elastic Weight Consolidation

Even though the brainstorm doc said "EWC may not work great", we use a
**lightweight EWC** restricted to the LoRA adapters and forecast head
(not the whole model). On these small subsets, EWC is well-behaved and
adds cheap regularization that prevents the new month's data from
catastrophically forgetting general patterns.

### `src/intraday/update/ewc.py`

```python
class EWCRegularizer:
    """Compute Fisher information on previous data; apply quadratic
    penalty around previous params during new training.
    """
    def __init__(
        self,
        old_params: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        lambda_: float = 0.5,
    ) -> None: ...

    def penalty(self, current_params: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.lambda_ * sum(
            (self.fisher[k] * (current_params[k] - self.old_params[k]) ** 2).sum()
            for k in self.fisher
        )
```

Fisher matrix is approximated with squared gradients on a sample of the
previous month's data, computed once at the start of each update cycle.

---

## 5. Drift report

Comprehensive comparison of distributions and decision quality between
the previous training window and the new month:

```
DRIFT REPORT — 2025-02-15  vs  2025-01-15..2025-02-14

  Feature                       KS  ADWIN  KL    Action
  log_return_5m                 0.03   no   0.01  ok
  realized_vol_30m              0.18   YES  0.34  refit
  ofi_5m_l5                     0.07   no   0.04  monitor
  basis_bn_cb_z                 0.22   YES  0.41  refit (cross-venue)
  funding_z                     0.04   no   0.02  ok
  iv_25d_rr                     0.31   YES  0.58  refit (sentiment shift)
  ...

  Decision-quality drift:
    Forecast hit-rate (last 7d):     53.1%   (window avg: 56.2%)   YELLOW
    Aggregator OOF AUC drop:         0.605 → 0.581                 YELLOW
    Realized slippage avg:           2.4 bps (avg: 1.9 bps)        YELLOW
    Stay-out activations:            ↑ 28% MoM                     RED — investigate

  Recommended actions:
    - Refit forecast LoRA + recalibrate (3 features fired)
    - Recompute basis features carefully
    - Aggregator AUC drop is small; warm-update only
    - Investigate stay-out activations before any promotion
```

The user must read this BEFORE running canary. Critical drift =
investigate first; do not push update.

---

## 6. Canary deployment

```python
async def run_canary(
    *,
    new_versions: dict[str, str],
    duration: dt.timedelta,
    live_versions: dict[str, str],
) -> CanaryReport:
    """Spawn a shadow paper run with new versions alongside the live
    paper run. Both consume the same live events. Compare:
      - Decision agreement rate
      - Sharpe ratio over canary window
      - Max drawdown
      - Realized slippage
      - Trade frequency

    Promotion criteria:
      - new Sharpe ≥ 0.9 × live Sharpe (allow small downgrade for fresh fit)
      - new max DD ≤ 1.1 × live max DD
      - new turnover within ±25% of live
      - no canary kill-switch fires
      - drift report severity ≤ "moderate"
    """
```

The canary is **a second strategy instance running on the same data**,
but its decisions go to the simulator only (no overlap with live trades).
This is cheap and safe.

---

## 7. Promotion

```python
def promote_version(
    *,
    component: Literal["forecast", "aggregator", "policy"],
    new_version: str,
    confirm: bool = False,
) -> None:
    """Atomically swap the 'latest' symlink. If confirm=False, prompt
    interactively with diff vs current.
    """
```

After promotion, the next strategy load picks up the new version on
its next 5-minute boundary. Old version files are kept; rollback is
just re-pointing the symlink.

---

## 8. CLI

| Command | Behavior |
|---|---|
| `intraday update run --since 30d --components forecast,aggregator,policy` | full workflow |
| `intraday update drift-report --since 30d` | drift report only, no retraining |
| `intraday canary run --new-version v5 --duration 7d` | run canary |
| `intraday canary report --new-version v5` | summary |
| `intraday canary promote --component policy --new-version v5 --confirm` | promote |
| `intraday update rollback --component policy` | revert to previous |

---

## 9. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | `update run` completes end-to-end on a synthetic 30-day fixture in < 30 min | smoke test |
| 2 | EWC penalty correctly anchors to previous params (zero penalty when params unchanged) | unit test |
| 3 | Drift report flags engineered shift (synthetic injection) | unit test |
| 4 | Canary report compares decisions and PnL with statistically valid summary | report inspection |
| 5 | Promotion is atomic and rollback works | manual test |
| 6 | After update + canary + promote, the next live decision uses new model versions | integration test |
| 7 | All update artifacts (snapshots, retrained models, reports) are versioned and reproducible | audit |

---

## 10. Common mistakes to avoid

- **Don't promote without canary.** Even if drift is mild. Even if you're
  in a hurry. Canary is cheap insurance.
- **Don't retrain on a single day.** Always at least 30 days. The "every
  night" pattern from the original brainstorm is exactly what causes
  catastrophic overfitting.
- **Don't skip the drift report.** It tells you *whether* updating is
  safe, not just whether it's possible.
- **Don't tune EWC `lambda_` aggressively.** Stick to 0.3–1.0. Higher
  values prevent any learning; lower values defeat the purpose.
- **Don't promote all three components at once unless drift is uniform.**
  If only forecast features drifted, only retrain + promote forecast.
- **Don't forget to back up `models/` before promotion.** Atomic
  symlink swap is reversible only if you keep the old files.
- **Don't run canary on a different data window than live.** They must
  see the same events.

---

## 11. Done ⇒ after one successful monthly cycle (update → canary →
promote → next month no regression), proceed to `phases/10_live.md`.

This is also the **steady-state operating loop** of the system. From
here on, you do this once a month forever.
