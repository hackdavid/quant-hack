# Intraday — BTC/USDT Multi-Agent Trading System

A probabilistic, regime-aware, multi-agent trading system for BTC/USDT.
**Philosophy:** model market state, do not predict price.

## What this repo is right now

This is a **planning workspace**. There is no production code yet.
The implementation is laid out as 11 phases that a coding agent (or you)
will build sequentially. Every phase has its own spec with concrete files,
function signatures, tests, and acceptance criteria.

## How to use this repo

1. Read `PLAN.md` — master plan, data/training strategy, phase index.
2. Read `AGENTS.md` — non-negotiable conventions for any coding agent.
3. Read `CLI.md` — the complete CLI surface designed up-front so all
   phases align to a single user-facing interface.
4. Open `phases/00_setup.md` and execute it.
5. Move to phase 1, 2, 3 ... only after the previous phase passes its
   acceptance criteria.

## Hard rules

- **No phase is "done" without passing tests + acceptance criteria.**
- **No skipping phases.** The system is designed so that each phase
  *de-risks* the next one. Skipping = future debugging hell.
- **Every action is logged with timestamp + context.** No silent code paths.
- **Every backtest result must use the realistic simulator** (Phase 3),
  never naive shift-and-multiply.
- **No live trading until Phase 8 (paper) has passed for ≥4 weeks.**

## Quick map

```
README.md           → you are here
PLAN.md             → master plan, data strategy, training cadence
AGENTS.md           → conventions for any coding agent (must-read)
CLI.md              → complete `intraday ...` command reference
pyproject.toml      → package skeleton (deps grouped per phase)
.gitignore
phases/
  00_setup.md       → project setup, deps, folder layout
  01_data.md        → historical download + live WS capture
  02_features.md    → microstructure feature engine
  03_simulator.md   → queue-aware L2 backtest simulator
  04_forecast.md    → Kronos + custom TCN + meta-labeling + calibration
  05_other_agents.md→ orderflow, regime, risk, stay-out detector
  06_aggregator_sizing.md → stacked meta-learner + fractional Kelly
  07_rl_execution.md→ CQL offline RL — execution only
  08_paper_trading.md → paper trading with continuous data capture
  09_continual.md   → drift-triggered monthly update
  10_live.md        → canary deploy + tiny-size live trading
```

## Tech stack (locked)

- Python 3.11+ via `uv`
- PyTorch (forecast + RL)
- Polars (data wrangling)
- DuckDB (queryable storage) + Parquet (cold storage)
- websockets / aiohttp (ingestion)
- d3rlpy (offline RL: CQL/IQL)
- typer (CLI)
- structlog (structured JSON logging)
- pytest (tests)
- river (online drift detection)

See `phases/00_setup.md` for the exact `pyproject.toml`.
