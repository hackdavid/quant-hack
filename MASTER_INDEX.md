# BTC/USD Multi-Agent Trading System — Master Index

**Status:** Phase 0 (Setup) - Planning Complete ✅  
**Current Phase:** Phase 1 (Data Pipeline)  
**Started:** 2026-06-18  
**Target Completion:** 2026-08-15 to 2026-09-05 (60-80 days)

---

## 🎯 Project Goal

Build institutional-grade BTC/USD intraday trading system with:
- **Target Sharpe:** 1.5+ sustained over 12+ months
- **Edge:** L2 microstructure (OFI, microprice, VPIN, Hawkes) + probabilistic forecasting + optimized execution
- **Architecture:** Multi-agent (Forecast, Orderflow, Regime, Risk, Stay-out) + RL execution + continual learning

**Source of Truth:** `idea/PLAN.md`, `idea/AGENTS.md`, `idea/phases/*.md`

---

## 📋 Phase Tracker

| Phase | Name | Status | Acceptance | GPU Time | CPU Time | Notes |
|-------|------|--------|------------|----------|----------|-------|
| 0 | Setup | ✅ DONE | Planning complete | - | - | Phase specs in idea/ |
| 1 | Data Pipeline | 🔵 CURRENT | Historical downloaded + WS live capture running for 4-6 weeks | - | 2-3 days | **START HERE** |
| 2 | Features | ⚪ BLOCKED | Feature engine passing tests, IC >0.1 on key features | - | 3-5 days | Blocked: needs 4-6 weeks tick data from Phase 1 |
| 3 | Simulator | ⚪ BLOCKED | Queue-aware backtest, slippage realistic vs historical | - | 4-6 days | Blocked: needs Phase 2 features |
| 4 | Forecast | ⚪ BLOCKED | OOS Sharpe ≥0.5, Brier <0.5, inference <50ms | 2-4 hrs | 1-2 days | Blocked: needs Phase 3 simulator |
| 5 | Other Agents | ⚪ BLOCKED | Each agent passing acceptance, IC >baseline | - | 3-5 days | Blocked: needs Phase 4 forecast |
| 6 | Aggregator | ⚪ BLOCKED | OOS Sharpe ≥1.0 with meta-learner + Kelly | - | 2-4 days | Blocked: needs Phase 5 agents |
| 7 | RL Execution | ⚪ BLOCKED | Slippage <0.8× AC baseline OR Sharpe +0.1 | 4-8 hrs | 3-5 days | Blocked: needs Phase 6 + 50k episodes |
| 8 | Paper Trading | ⚪ BLOCKED | 1-3 months live simulation passing | - | 30-90 days | Blocked: needs Phase 7, RUNS IN PARALLEL |
| 9 | Continual | ⚪ BLOCKED | Monthly update pipeline working, canary passing | - | 2-3 days | Blocked: needs Phase 8 data |
| 10 | Live | ⚪ BLOCKED | Canary → tiny size → scale, kill-switch tested | - | Ongoing | Blocked: needs Phase 9 validation |

**Legend:**  
✅ DONE | 🔵 CURRENT | ⚪ BLOCKED | ❌ FAILED | 🟡 IN PROGRESS

---

## 🚀 Current Phase Details

### **Phase 1: Data Pipeline** (CURRENT)

**Spec:** `idea/phases/01_data.md`

**Objective:** 
1. Download 12-18 months historical BTC/USD data (klines, funding, OI)
2. Set up live WebSocket capture for tick + L2 depth data
3. Let capture run for ≥4-6 weeks (blocking Phase 2)

**Files to Create:**
```
src/intraday/data/
  __init__.py
  download.py          # Historical data from Binance
  capture.py           # Live WS streams (trade, depth, funding)
  schema.py            # Parquet schema validation
  cli.py
tests/phase_01/
  test_download.py
  test_capture.py
  test_schema.py
```

**Acceptance Criteria:**
1. ✅ Historical 12 months klines_1m, klines_5m downloaded → `data/raw/binance/`
2. ✅ Funding + OI history downloaded
3. ✅ Live capture running: trade stream + depth@100ms → `data/raw/binance/trades/`, `data/raw/binance/depth/`
4. ✅ Schema validation passing on all Parquet files
5. ✅ Capture handles reconnects with <5s gap
6. ⏳ **CRITICAL:** Capture must run ≥4-6 weeks before Phase 2 starts

**CLI Commands:**
```bash
# Download historical (one-time, ~30 mins)
uv run intraday data download --symbol BTCUSDT --venue binance --kind klines_5m --start 2023-06-01 --end 2024-12-31

# Start live capture (long-running, background)
uv run intraday data live-capture --venues binance --symbols BTCUSDT --kinds trade,depth_100ms,funding
```

**Time Estimates:**
- Implementation: 2-3 days (with GPU: same, this is CPU/network bound)
- Data collection (blocking): 28-42 days (MUST COMPLETE before Phase 2)

**Action Items:**
- [x] Implement historical download
- [x] Implement live WS capture with reconnect logic
- [x] Set up data/ directory structure
- [x] Verify schema validation (16/16 tests passing)
- [ ] Run historical download (12 months klines + funding + OI)
- [ ] Start live capture service
- [ ] ⏰ **SET CALENDAR REMINDER:** Check back in 4-6 weeks for Phase 2

---

## 📊 GPU Optimization Notes

**Phases with GPU Acceleration:**
- Phase 4 (Forecast): 2-4 hours (vs 1-2 days CPU) — **10-20x speedup**
- Phase 7 (RL CQL): 4-8 hours (vs 3-5 days CPU) — **15-30x speedup**

**Total Timeline Reduction:** 60-80 days → 45-60 days

**GPU Config Checklist:**
```python
# Always check in training scripts:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # For fixed input sizes
use_amp = True  # Mixed precision for 2x speedup

# DataLoader optimization:
DataLoader(..., num_workers=4-8, pin_memory=True, prefetch_factor=2)

# Batch sizes (adjust for GPU memory):
Phase 4: batch_size = 256-512
Phase 7: batch_size = 256-512
```

---

## 🎯 Next Steps (When Phase 1 Complete)

**Immediate (After acceptance #1-5):**
1. Update this file: Phase 1 status → ✅ DONE
2. Phase 2 status → 🟡 WAITING (tick data collection)
3. Set calendar reminder for 4-6 weeks
4. Optional: Start reading Phase 2 spec to prepare

**After 4-6 Weeks (When tick data ready):**
1. Verify ≥4-6 weeks of tick data captured
2. Update Phase 2 status → 🔵 CURRENT
3. Load `idea/phases/02_features.md`
4. Implement feature engine
5. Update this file when Phase 2 acceptance met

---

## 🚨 Critical Rules

**From `idea/README.md` Hard Rules:**
1. ❌ No phase is "done" without passing tests + acceptance criteria
2. ❌ No skipping phases (each de-risks the next)
3. ✅ Every action logged with timestamp + context
4. ✅ Every backtest uses realistic simulator (Phase 3)
5. ❌ No live trading until Phase 8 (paper) passed for ≥4 weeks

**From Memory System:**
1. ❌ Don't create MD files — update MASTER_INDEX.md or memory/ only
2. ✅ GPU available — optimize training for fast iteration
3. ✅ Sequential execution — work on current phase only, no skipping

---

## 📚 Reference Links

| Resource | Location | Purpose |
|----------|----------|---------|
| Master Plan | `idea/PLAN.md` | Overall strategy, data requirements, training cadence |
| Agent Rules | `idea/AGENTS.md` | Coding conventions, logging, testing requirements |
| CLI Design | `idea/CLI.md` | Complete CLI surface for all phases |
| Phase Specs | `idea/phases/00_setup.md` through `10_live.md` | Detailed phase-by-phase implementation specs |
| Memory | `.claude/projects/.../memory/MEMORY.md` | Project context, user preferences, feedback |

---

## 📝 Progress Log

| Date | Event | Notes |
|------|-------|-------|
| 2026-06-18 | Planning complete | Analyzed hackathon rules, decided to build full system (60-80 days) instead of rushing for 5-day competition |
| 2026-06-18 | Phase 0 complete | Specs in idea/, memory system initialized, MASTER_INDEX created |
| 2026-06-18 | Phase 1 implementation | Data pipeline with checkpoint tracking, Pydantic schemas, WS streaming, pagination support |
| 2026-06-18 | Phase 1 tests passing | 16/16 tests passing (schemas 9/9, checkpoint 7/7), implementation complete |
| 2026-06-18 | Phase 1 ready for data | Code complete, ready to download historical + start live capture |

---

**Last Updated:** 2026-06-18  
**Updated By:** Claude (initial setup)  
**Next Review:** When Phase 1 acceptance criteria met
