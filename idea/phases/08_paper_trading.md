# Phase 8 — Paper Trading

**Goal:** run the system end-to-end on **live data** with **simulated
fills** (via Phase 3's simulator). All decisions, fills, PnL, drift
signals, and incoming raw data are captured to disk for monthly retraining.

**Why eighth:** real markets are non-stationary in ways your historical
backtests cannot reveal. Paper trading is the only honest validation
short of real money.

**Estimated effort:** continuous (≥ 30 days per cycle).

**Activates dep group:** none new.

**Prerequisite:** Phase 6 (or 7) acceptance passed AND live capture
running stably for ≥ 1 week.

---

## 1. Inputs / outputs

- **Inputs:**
  - Live WS streams (Phase 1 captures already running).
  - Trained models (Phase 4, 5, 6, optionally 7).
- **Outputs:**
  - One long-running process: `intraday paper run`.
  - Per-day rotating run dir `runs/paper-{YYYY-MM-DD}/` with:
    - `decisions.jsonl`, `trades.jsonl`, `pnl.parquet`, `log.jsonl`,
      `metrics.json`, drift events.
  - Continuous tick + depth captured to `data/raw/.../live/...`
    (already happening from Phase 1; paper run just adds annotations).
  - Daily report email/file (optional).
  - Drift-detector alerts written to `runs/paper-{date}/drift.jsonl`.

---

## 2. Files to create

```
src/intraday/paper/
  __init__.py
  runner.py                      # main paper-trading loop
  router.py                      # routes events from live capture → strategy
  state_clock.py                 # 5-minute boundary scheduler
  drift.py                       # KSWIN/ADWIN per key feature + decision quality
  daily_report.py                # builds runs/paper-{date}/report.html
  cli.py
src/intraday/paper/health/
  watchdog.py                    # process supervisor, restart on crash
  heartbeat.py                   # write to runs/paper-{date}/heartbeat.txt every 5s
tests/phase_08/
  test_runner.py
  test_router.py
  test_drift.py
  test_state_clock.py
  test_daily_report.py
```

---

## 3. Architecture

```
       Live WS captures (already running from Phase 1)
                         │
                         ▼
            ┌───────────────────────────┐
            │   In-memory event router  │
            │   (subset re-broadcast    │
            │    from disk tail-reading │
            │    OR direct WS taps)     │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │  Online feature engine    │
            │  (Phase 2 functions but   │
            │  with rolling windows in  │
            │  memory; no disk reads)   │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │  Strategy: v5_full_no_rl  │
            │           (or v6+rl)      │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │  Realistic simulator      │
            │  (Phase 3, fed by same    │
            │  live events)             │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │  Logger / persister       │
            │  → runs/paper-{date}/...  │
            └───────────────────────────┘
```

Key design choices:
- The paper runner does **not** spawn its own WS connections; it reads
  from the running capture process's tail or shares a queue. This
  guarantees paper sees the same data that gets persisted.
- The feature engine reuses Phase 2 functions but with **stateful
  rolling windows** kept in memory; no recomputation per event.

---

## 4. Function / class signatures

### `src/intraday/paper/runner.py`

```python
class PaperRunner:
    def __init__(
        self,
        *,
        config: PaperConfig,
        strategy: Strategy,
        sim: SimulatorLoop,
        feature_engine: OnlineFeatureEngine,
        capture_tap: CaptureTap,
        drift_monitor: DriftMonitor,
        run_dir: Path,
    ) -> None: ...

    async def run(self, duration: dt.timedelta | None = None) -> None:
        """Main loop. Forever, or until duration elapsed, or until
        kill-switch.
        """
```

### `src/intraday/paper/state_clock.py`

```python
class StateClock:
    """Schedules 5-minute boundary callbacks aligned to UTC."""

    async def each_boundary(
        self,
        callback: Callable[[int], Awaitable[None]],
    ) -> None: ...
```

### `src/intraday/paper/drift.py`

```python
class DriftMonitor:
    """Per-feature KSWIN drift detector + decision-quality monitor."""

    def __init__(
        self,
        *,
        watch_features: list[str],         # e.g. ['ofi_5m_l5', 'realized_vol_30m', ...]
        watch_quality: bool = True,
        out_path: Path,
    ) -> None: ...

    def update_feature(self, name: str, value: float, ts_ms: int) -> bool:
        """Returns True if drift detected on this feature."""

    def update_decision_quality(
        self,
        forecast_p: float,
        realized_y: int,
        ts_ms: int,
    ) -> bool: ...

    def status(self) -> dict[str, Any]: ...
```

Uses `river.drift.KSWIN(alpha=0.001, window_size=200)`. On drift event:
- Append to `runs/paper-{date}/drift.jsonl`.
- Emit log event `drift.detected`.
- Optionally trigger an early monthly update (Phase 9).

---

## 5. Daily lifecycle

The paper runner is a long-running process. Each UTC midnight:

1. Close current run directory (`runs/paper-{YYYY-MM-DD}/`).
2. Generate `report.html` for the day.
3. Open new run directory `runs/paper-{tomorrow}/`.
4. Continue without interrupting strategy state.

This means **strategy state (open positions, rolling windows) persists
across day boundaries**; only logging artefacts roll.

---

## 6. CLI

| Command | Behavior |
|---|---|
| `intraday paper run --capital 10000 --max-leverage 1.0 --policy-version latest` | start runner |
| `intraday paper status` | live status from heartbeat file |
| `intraday paper stop --reason "..."` | graceful kill-switch |
| `intraday paper report --date 2025-02-15` | rebuild HTML report |

Recommended `paper run` invocation:

```bash
nohup uv run intraday paper run \
    --capital 10000 \
    --max-leverage 1.0 \
    --policy-version latest \
    --duration 30d \
    --log-level info \
    > runs/paper-stdout.log 2>&1 &
```

---

## 7. What gets stored (very important — Phase 9 depends on it)

For each day, in `runs/paper-{date}/`:

| File | Content |
|---|---|
| `decisions.jsonl` | Every aggregator decision with full feature row |
| `trades.jsonl` | Every fill with pre-trade state + post-trade PnL |
| `forecasts.jsonl` | Every ForecastOutput emitted |
| `agent_opinions.jsonl` | Every AgentOpinion from each agent |
| `pnl.parquet` | Per-second equity / position / unrealized PnL |
| `drift.jsonl` | Drift events (feature + decision quality) |
| `metrics.json` | End-of-day summary |
| `log.jsonl` | All structured logs |
| `config.yaml` | Resolved config used for this day |

This is the **canonical data** that Phase 9's monthly update reads.
Schema for these files is locked in `intraday.paper.schemas` and
validated on write — same discipline as Phase 1 raw data.

---

## 8. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | Process runs continuously for ≥ 7 days without crash | uptime log |
| 2 | Heartbeat file updated every ≤ 10s; watchdog restarts on stale | manual disconnect test |
| 3 | All 9 daily files written for every UTC date | inspect runs/ |
| 4 | Drift monitor emits event when fed synthetic distribution shift | unit test |
| 5 | Realized paper Sharpe over a calendar month is **within 1σ of backtest** Sharpe over the same period (compute σ from purged-fold variance) | comparison |
| 6 | Paper realized turnover within ±20% of backtest turnover | comparison |
| 7 | All decisions reproducible from `decisions.jsonl` (re-running with stored features re-produces same `Decision`) | seeded test |
| 8 | When kill-switch fires (config or manual), all open orders cancel within 5s | manual test |

**Acceptance #5 is the truth-telling gate.** If paper Sharpe is
significantly worse than backtest Sharpe (> 1σ below), your sim is
optimistic somewhere. Common causes:
- Latency model under-estimating real RTT.
- Queue position model too generous.
- Slippage model missing impact.
- Look-ahead in features (one of them peeks forward).

Investigate before continuing.

---

## 9. Common mistakes to avoid

- **Don't paper-trade with a different code path than backtest.** Same
  `Strategy`, same simulator, same feature engine — just live event
  source. The bug-finding power is in this equivalence.
- **Don't sleep through gaps.** When live capture hits a gap, the
  paper runner must enter `stay_out` mode for `gap_duration + buffer`
  rather than trading on stale features.
- **Don't restart a paper run with stale position state.** On restart,
  reconcile from `runs/paper-{date}/positions.jsonl` (write it!) or
  start flat.
- **Don't mix paper and live in the same process.** Different processes,
  different config files, different data dirs.
- **Don't forget timezone.** All UTC. The "day" boundary is UTC midnight,
  not local.
- **Don't skip the daily report.** Looking at the report every day is
  how you catch slow degradation (drift, regime change, bug).

---

## 10. Operating routine (the actual daily ritual)

This is what *you* do each day during paper trading:

1. **Morning:** open `runs/paper-{yesterday}/report.html`. Skim:
   - Equity curve
   - Drift events count
   - Top winners / losers
   - Did anything weird happen overnight?
2. **Note any anomaly** in `docs/observations/{date}.md`.
3. **Continue** — do not retrain on impulse. Monthly cadence only,
   unless drift count > threshold.
4. **End of week:** check cumulative Sharpe vs backtest expectation.
5. **End of month:** trigger Phase 9 update (`intraday update run`).

---

## 11. Done ⇒ after **at least 30 days** of clean paper trading with
acceptance #5 met, proceed to `phases/09_continual.md`.
