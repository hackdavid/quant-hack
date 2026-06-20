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
| 1 | Data Pipeline | 🔵 CURRENT | 12mo historical + 4-6wk live tick data | - | 2-3 days | **START HERE** |
| 2 | Features (MVP) | ⚪ WAITING | Feature engine on 4-6wk dataset, IC >0.05 | - | 3-5 days | Use small dataset for dev |
| 3 | Simulator (MVP) | ⚪ WAITING | Queue-aware backtest working on 4-6wk data | - | 4-6 days | Validate on small dataset |
| 4 | Forecast (MVP) | ⚪ WAITING | Training pipeline working on GPU | 0.5-1 hr | 1-2 days | Test with 4-6wk data |
| 5 | Other Agents (MVP) | ⚪ WAITING | All 4 agents training on GPU | 0.5-1 hr | 3-5 days | Fast iteration on small data |
| 6 | Aggregator (MVP) | ⚪ WAITING | Meta-learner working on 4-6wk data | - | 2-4 days | End-to-end pipeline validated |
| 7 | RL Execution (MVP) | ⚪ WAITING | CQL training on GPU (10k episodes) | 1-2 hrs | 3-5 days | Prove training works |
| **PROD** | **Retrain Full** | ⚪ WAITING | All models retrained on 12mo dataset | **6-12 hrs** | - | **Production models** |
| 8 | Paper Trading | ⚪ WAITING | 1-3 months with production models | - | 30-90 days | Use full-data checkpoints |
| 9 | Continual | ⚪ WAITING | Monthly update pipeline working | 2-4 hrs | 2-3 days | Retraining automation |
| 10 | Live | ⚪ WAITING | Canary → scale, kill-switch tested | - | Ongoing | Production deployment |

**Legend:**  
✅ DONE | 🔵 CURRENT | ⚪ BLOCKED | ❌ FAILED | 🟡 IN PROGRESS

---

## 🚀 Current Phase Details

### **Phase 1: Data Pipeline** (CURRENT)

**Spec:** `idea/phases/01_data.md`

**Objective:** 
1. Download 12 months historical BTC/USD data (klines, funding, OI)
2. Set up live WebSocket capture for tick + L2 depth data
3. Collect ≥4-6 weeks tick data for MVP development

**REVISED STRATEGY (2026-06-21):**
- Use 4-6 weeks data as **development dataset** (fast iteration)
- Build entire Phase 2-7 pipeline on small dataset first
- Validate all code works end-to-end on GPU
- **Then retrain on full 12-month dataset** for production models
- Deploy production models to paper trading

**Why:** Don't wait idle for 4-6 weeks. Use small dataset to de-risk development, then retrain for production.

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
- [ ] **Deploy CPU machine** (8 CPU / 32GB RAM / 500GB disk)
- [ ] Run historical download (12 months klines + funding + OI)
- [ ] Start live capture service (background for 4-6 weeks)
- [ ] ⏰ **WAIT 4-6 weeks** while working on Phase 2-7 MVP
- [ ] After 4-6 weeks: Start Phase 2-7 development on small dataset
- [ ] After Phase 7 MVP complete: Retrain all models on 12-month data
- [ ] Deploy production checkpoints to paper trading

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

## 🎯 Next Steps (Revised Strategy)

**Week 1 (Current - Data Collection):**
1. ✅ Provision CPU machine (8 CPU / 32GB RAM / 500GB)
2. Download 12 months historical data (klines, funding, OI)
3. Start live capture in tmux (runs for 4-6 weeks)
4. Update Phase 1 status → ✅ DONE

**Weeks 2-6 (MVP Development on 4-6 Week Dataset):**
1. Wait for 4-6 weeks tick data to accumulate
2. Implement Phase 2-7 using small dataset:
   - Phase 2: Feature engine (3-5 days)
   - Phase 3: Simulator (4-6 days)
   - Phase 4: Forecast agent MVP (1-2 days + 0.5-1 hr GPU)
   - Phase 5: Other agents MVP (3-5 days + 0.5-1 hr GPU)
   - Phase 6: Aggregator MVP (2-4 days)
   - Phase 7: RL execution MVP (3-5 days + 1-2 hrs GPU)
3. Validate entire pipeline works end-to-end
4. Fix bugs, optimize code

**Week 7 (Production Training):**
1. Retrain all models on full 12-month dataset
2. GPU training: 6-12 hours total (Phase 4 + 5 + 7 full-scale)
3. Export production checkpoints
4. Validate on holdout set (OOS metrics)

**Week 8+ (Paper Trading):**
1. Deploy production models to CPU machine
2. Run paper trading for 1-3 months
3. Monitor performance, optimize as needed
4. Scale to live trading when ready

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
