# AGENTS.md — rules for any coding agent working in this repo

These are non-negotiable. If you can't satisfy a rule, **stop and ask**;
do not improvise.

---

## 1. Working agreement

- **Read the active phase spec in full before writing any code.**
- **Never skip a phase.** Phase N depends on Phase N-1's invariants.
- **One coherent change per commit.** Commit message format below.
- **No silent fallbacks.** If something fails (network, disk, bad config),
  raise a typed error; do not return zeros, NaNs, or empty dicts.
- **No hidden state.** Every agent / model has explicit inputs and outputs;
  no reading from globals, no writing to a shared dict from arbitrary code.
- **Deterministic by default.** Seed every RNG (numpy, torch, python `random`).
  Provide a `--seed` flag on every CLI command that has stochasticity.

---

## 2. Code style

- Python 3.11+, type hints on every public function.
- Use `pydantic` v2 for all configs, dataclasses for internal records.
- Use `polars` (not pandas) for any tabular work.
- Use `pathlib.Path` (not `os.path`).
- No `print()` in library code; use the project logger.
- Functions ≤ 50 lines unless there's a clear reason; classes ≤ 300 lines.
- No "convenience" notebooks committed; all reproducible work is a CLI command.

---

## 3. Logging — every action has a structured event

Use `structlog` configured with JSON output. Every important action emits
one log line with at minimum:

```json
{
  "ts": "2025-01-27T10:30:45.123456+00:00",
  "level": "info",
  "event": "decision",
  "phase": "live|paper|backtest",
  "run_id": "backtest-20250127-103045-a8f3",
  "module": "intraday.aggregator",
  "elapsed_ms": 12.4,
  "...payload...": "..."
}
```

Required event names (use exactly these strings; downstream tooling
filters by event):

| event | when emitted |
|---|---|
| `data.bar_received` | New bar / event arrives |
| `feature.computed` | Feature vector built for one timestamp |
| `forecast.predicted` | Forecast agent produced output |
| `agent.opinion` | Any sub-agent (orderflow / regime / risk / stay-out) emits its output |
| `aggregator.decision` | Decision aggregator outputs final score |
| `sizing.computed` | Position size decided |
| `order.submitted` | Order sent to sim or exchange |
| `order.filled` | Fill received (full or partial) |
| `order.cancelled` | Order cancelled |
| `pnl.update` | PnL ticker update |
| `drift.detected` | Drift detector fires |
| `policy.updated` | Policy weights swapped |
| `kill_switch.engaged` | Trading halted |

**Every log line carries `run_id` so all events can be replayed by run.**

Logs are written to **both** console (human-readable) and file
(`runs/{RUN_ID}/log.jsonl`). The JSONL file is the source of truth.

---

## 4. Time discipline

- **All timestamps in UTC.** Never local time. Never naive datetimes.
- **All timestamps in nanosecond-precision** when from market data;
  microsecond minimum.
- **Log `elapsed_ms` for every operation > 1ms.** Use a context manager:
  ```python
  with timed("forecast.inference") as t:
      ...
  log.info("forecast.predicted", elapsed_ms=t.elapsed_ms, ...)
  ```
- **No wall-clock-based bar boundaries.** Sample on event time
  (volume bars / dollar bars / imbalance bars) per López de Prado.

---

## 5. Tests

Folder layout:

```
tests/
  phase_00/
  phase_01/
  ...
  fixtures/
    sample_klines_1m.parquet
    sample_trades_1s.parquet
    sample_depth_100ms.parquet
```

Rules:

- Every public function in `intraday.*` has at least one unit test.
- Every CLI command has a smoke test (`pytest tests/cli/test_smoke.py`).
- Use `pytest.mark.slow` for tests > 5 seconds.
- Use `pytest.mark.network` for tests that hit live APIs (default skip).
- Property-based tests (`hypothesis`) for any pure numerical function
  (microprice, OFI, Hurst, etc.).

Required acceptance for each phase:

```
uv run pytest tests/phase_NN/ -v
uv run pytest tests/integration/test_phase_NN.py -v
```

---

## 6. Errors and validation

- Use a single `IntradayError` base class; subclass per error type
  (`DataIntegrityError`, `ForecastError`, `SimulatorError`, ...).
- Every config file passes through a `pydantic` model on load.
- Every Parquet read goes through a schema validator
  (`intraday.io.schema.validate(df, kind="kline_1m")`).
- Reject corrupt data **at ingestion**; never propagate NaNs through
  the system.

---

## 7. Determinism + reproducibility

- Every run emits `runs/{RUN_ID}/config.yaml` with the **fully resolved**
  config (defaults expanded). Re-running with that config = same numbers.
- Model checkpoints carry sidecar `metadata.json`:
  ```json
  {
    "version": "v3",
    "git_sha": "...",
    "trained_at": "2025-01-27T10:30:45Z",
    "data_window": {"start": "2024-01-01", "end": "2024-12-31"},
    "metrics_at_train": {"val_brier": 0.21, "val_sharpe": 1.12},
    "parent_version": "v2",
    "training_seed": 1729,
    "config_hash": "..."
  }
  ```
- Seeds: numpy, torch (CPU + CUDA), python `random`, env `PYTHONHASHSEED`.

---

## 8. Money safety

- **No live order can be sent unless** `config.mode == "live"` AND
  `config.live.armed == True`. The arming flag is set explicitly per session,
  not stored in YAML.
- **Hard cap** on max single order size, hard cap on max daily loss,
  hard cap on max position. All three checked in `intraday.risk.hard_caps`,
  not just in the soft risk agent.
- **Kill-switch is a process-level signal.** When engaged, all open orders
  cancelled, all open positions closed, no new orders accepted until
  manual rearm.
- **Dry-run flag (`--dry-run`)** on every command that touches live state.

---

## 9. Commit message format

```
<phase>: <imperative summary, ≤ 72 chars>

<optional body explaining why>

- specific change 1
- specific change 2

Tests: <which tests added/modified>
Acceptance: <which acceptance criterion this satisfies>
```

Example:

```
phase-2: implement OFI feature with rolling window

- intraday/features/ofi.py: vectorized OFI using polars
- tests/phase_02/test_ofi.py: 4 unit tests + 1 hypothesis test
- benchmarks: 1.2ms / 10k events on M2 (acceptance criterion: < 5ms)

Tests: tests/phase_02/test_ofi.py
Acceptance: phase-02 acceptance #3 (OFI throughput)
```

---

## 10. When stuck

If you cannot satisfy a rule or acceptance criterion, write a short
investigation note to `docs/investigations/{date}-{topic}.md` and stop.
Do not work around the rule. Surface the issue.
