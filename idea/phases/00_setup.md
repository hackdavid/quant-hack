# Phase 0 — Project Setup

**Goal:** establish the repo skeleton, install core deps, wire up logging,
testing, CLI scaffolding, and CI gates. **No business logic yet.**

**Estimated effort:** 0.5–1 day.

---

## 1. Inputs / outputs

- **Inputs:** this `phases/00_setup.md`, `pyproject.toml`, `AGENTS.md`.
- **Outputs:**
  - Initialized `src/intraday/` package importable as `intraday`.
  - `intraday --help` runs and lists empty command groups.
  - `uv run pytest` runs and passes (one trivial test).
  - Logging configured (structlog → JSON file + console).
  - Pre-commit + CI smoke check working.

---

## 2. Files to create

```
src/intraday/
  __init__.py                 # version, package metadata
  cli.py                      # typer app — empty groups stubbed
  config.py                   # pydantic Settings model + loader
  logging_setup.py            # structlog configuration
  errors.py                   # IntradayError + subclasses
  io/
    __init__.py
    paths.py                  # project paths (data/, runs/, models/)
    schema.py                 # parquet schema validators (stubs for now)
  utils/
    __init__.py
    timing.py                 # `timed` context manager for elapsed_ms
    seeding.py                # deterministic seeding helper
    runs.py                   # RUN_ID generator
config/
  default.yaml                # default config, fully expanded
  example_live.yaml           # example live config (with armed=false)
tests/
  __init__.py
  conftest.py                 # shared fixtures, seed control
  test_smoke.py               # imports + CLI --help
.github/
  workflows/
    ci.yml                    # ruff + mypy + pytest
.pre-commit-config.yaml       # ruff + mypy
```

---

## 3. Function / class signatures (essential ones)

### `src/intraday/logging_setup.py`

```python
def configure_logging(
    *,
    run_id: str,
    log_file: Path | None = None,
    level: str = "info",
    quiet_console: bool = False,
) -> None:
    """Configure structlog with JSON file sink + human console.

    Must be called once at process start. All subsequent log lines
    automatically include run_id.
    """
```

### `src/intraday/utils/timing.py`

```python
class timed:
    """Context manager that records elapsed milliseconds.

    Usage:
        with timed("operation") as t:
            ...
        log.info("operation.done", elapsed_ms=t.elapsed_ms)
    """
    name: str
    elapsed_ms: float
```

### `src/intraday/utils/runs.py`

```python
def new_run_id(mode: str) -> str:
    """Return e.g. 'backtest-20250127-103045-a8f3' (UTC)."""

def run_dir(run_id: str) -> Path:
    """Return absolute Path to runs/{run_id}, creating it if needed."""
```

### `src/intraday/io/paths.py`

```python
class Paths:
    """Canonical project paths, anchored at repo root."""
    repo_root: Path
    data_raw: Path
    data_features: Path
    runs: Path
    models: Path

def project_paths() -> Paths:
    """Return Paths instance with directories created if missing."""
```

### `src/intraday/cli.py`

```python
import typer

app = typer.Typer(no_args_is_help=True)

# Stubbed groups — populated by later phases
data_app = typer.Typer(help="Data acquisition + capture")
features_app = typer.Typer(help="Feature computation")
backtest_app = typer.Typer(help="Backtesting")
train_app = typer.Typer(help="Model training")
paper_app = typer.Typer(help="Paper trading")
update_app = typer.Typer(help="Continual update / drift")
canary_app = typer.Typer(help="Canary deployment")
live_app = typer.Typer(help="Live trading (gated)")
inspect_app = typer.Typer(help="Read-only inspection")

app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(backtest_app, name="backtest")
app.add_typer(train_app, name="train")
app.add_typer(paper_app, name="paper")
app.add_typer(update_app, name="update")
app.add_typer(canary_app, name="canary")
app.add_typer(live_app, name="live")
app.add_typer(inspect_app, name="inspect")
```

### `src/intraday/errors.py`

```python
class IntradayError(Exception):
    """Base exception for all intraday errors."""

class ConfigError(IntradayError): ...
class DataIntegrityError(IntradayError): ...
class ForecastError(IntradayError): ...
class SimulatorError(IntradayError): ...
class RiskError(IntradayError): ...
```

---

## 4. `config/default.yaml` — minimal but complete

```yaml
run:
  log_level: info
  seed: 1729

instrument:
  symbol: BTCUSDT
  venue: binance
  market: perp        # perp | spot
  base: BTC
  quote: USDT

risk:
  max_leverage: 1.0
  max_position_usd: 200
  max_daily_drawdown_pct: 2.0
  max_open_positions: 1
  hard_caps:
    max_single_order_usd: 50
    max_total_exposure_usd: 500
    max_daily_loss_usd: 20

fees:
  maker_bps: 2.0
  taker_bps: 5.0

storage:
  data_dir: ./data
  runs_dir: ./runs
  models_dir: ./models

# Phase-specific configs are added by their respective phases under
# `features:`, `forecast:`, `aggregator:`, `policy:`, `paper:`, `live:`.
```

---

## 5. CLI commands added by this phase

None functional. `intraday --help` must show the empty group skeleton.

---

## 6. Unit tests

`tests/test_smoke.py`:

1. `test_package_importable` — `import intraday` works.
2. `test_cli_help_runs` — `subprocess.run(["uv", "run", "intraday", "--help"])`
   exits 0.
3. `test_logging_emits_json_to_file` — call `configure_logging`, log one
   event, parse the JSONL file, assert keys: `ts`, `level`, `event`, `run_id`.
4. `test_run_id_format` — `new_run_id("backtest")` matches regex
   `^backtest-\d{8}-\d{6}-[a-z0-9]{4}$`.
5. `test_seeding_is_deterministic` — seeding with same seed gives
   identical numpy + torch random sequences.
6. `test_paths_directories_exist_after_call` — `project_paths()` creates
   `data/`, `runs/`, `models/`.

Use `tmp_path` fixture for everything that touches the filesystem.

---

## 7. Integration / smoke test

`tests/integration/test_phase_00.py`:

```python
def test_phase_00_smoke():
    """End-to-end: install, import, cli --help, configure logger, log one event."""
    # No exceptions allowed.
```

Manual smoke:

```bash
uv venv
uv sync
uv run intraday --help
uv run pytest -v
```

---

## 8. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | `uv run intraday --help` exits 0 and lists 9 command groups | manual |
| 2 | All phase-0 unit tests pass | `uv run pytest tests/ -v` |
| 3 | `ruff check` and `mypy --strict` pass on `src/` | `uv run ruff check src && uv run mypy src` |
| 4 | First log line in any run is `{"event": "run.start", "run_id": "..."}` | inspect log file |
| 5 | CI workflow runs on push and passes | GitHub Actions tab |

---

## 9. Common mistakes to avoid

- **Don't mix logging libraries.** Only `structlog`. No `logging.basicConfig`,
  no `print`.
- **Don't import `pandas`.** Use `polars` from day 1.
- **Don't hardcode paths.** Always go through `project_paths()`.
- **Don't make `configure_logging` import-time.** It must be explicit so
  tests can re-configure per run.
- **Don't enable `mypy --strict` without typing `pydantic` models** —
  use `model_config = ConfigDict(strict=True)`.

---

## 10. Done ⇒ proceed to `phases/01_data.md`.
