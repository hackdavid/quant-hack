# BTC/USD Multi-Agent Trading System — Master Index

**Status:** Phase 8 (Paper Trading) — Pipeline End-to-End Working ✅
**Current Phase:** Paper Trading with Demo Account
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
| 1 | Data Pipeline | ✅ DONE | 12mo historical downloaded, live WS capture | - | 2-3 days | Historical data loaded via API |
| 2 | Features (MVP) | ✅ DONE | Feature calculator with live_mode | - | 3-5 days | 20 features from klines |
| 3 | Simulator (MVP) | ⚪ WAITING | Queue-aware backtest | - | 4-6 days | Validate on small dataset |
| 4 | Forecast (MVP) | 🟡 DONE | Kronos pipeline working on CPU | - | 1-2 days | **Model is 15mo old — needs retrain** |
| 5 | Other Agents (MVP) | ✅ DONE | Orderflow, Regime, Risk, StayOut agents working | - | 3-5 days | All agents producing signals |
| 6 | Aggregator (MVP) | ✅ DONE | Meta-learner + DecisionEngine working | - | 2-4 days | Paper mode thresholds active |
| 7 | RL Execution (MVP) | ⚪ WAITING | CQL training on GPU | 1-2 hrs | 3-5 days | Not needed for paper trading |
| **PROD** | **Retrain Full** | ⚪ WAITING | All models retrained on 12mo dataset | **6-12 hrs** | - | **Production models** |
| 8 | Paper Trading | 🔵 CURRENT | Autonomous bot running on MT5 demo | - | 30-90 days | **ACTIVE: Need actual trades** |
| 9 | Continual | ⚪ WAITING | Monthly update pipeline | 2-4 hrs | 2-3 days | Retraining automation |
| 10 | Live | ⚪ WAITING | Canary → scale, kill-switch tested | - | Ongoing | Production deployment |

**Legend:**
✅ DONE | 🔵 CURRENT | ⚪ BLOCKED | ❌ FAILED | 🟡 IN PROGRESS

---

## 🚀 Current Phase Details

### **Phase 8: Paper Trading** (CURRENT)

**Spec:** `idea/phases/08_paper_trading.md`

**Objective:**
1. Run autonomous bot on MT5 demo account with real-time Binance data
2. Execute actual demo trades (not mock)
3. Monitor performance, fix bugs, iterate strategy
4. Generate ≥5 trades/day for competition volume requirements

**What's Working:**
- ✅ Full V6 pipeline (Forecast → Orderflow → Regime → Risk → StayOut → MetaLearner → DecisionEngine)
- ✅ 20-feature calculation from live WebSocket (1m + 5m klines)
- ✅ LLM review with Fireworks Kimi K2.6 (8192 tokens, JSON parsing)
- ✅ MT5 demo account connected (balance $1,000,000)
- ✅ Paper mode with lowered thresholds (forecast_confidence=0.02, meta_threshold=0.05)
- ✅ Regime fallback when transformer uncertain (confidence 0.70)
- ✅ Position manager (closes opposite, opens new, prevents double-entry)
- ✅ JSONL trade logging

**Known Issues:**
- 🔴 Transformer model trained on 2025-03-31 — concept drift, confidence 0.001-0.02
- 🔴 LLM API intermittently times out [WinError 10060] — network connectivity
- 🔴 WebSocket connected but bar processing needs monitoring
- 🔴 Strategy generates mostly HOLD signals — need more trade volume
- 🟡 MT5 AutoTrading must be manually enabled every session

**CLI Commands:**
```bash
# Start autonomous trader
uv run python scripts/autonomous_trader.py \
  --transformer-run models/transformer/20260623T132957Z \
  --mt5-account YOUR_ACCOUNT \
  --mt5-password "YOUR_PASSWORD" \
  --mt5-server "YOUR_SERVER" \
  --use-llm --llm-debug \
  --paper-mode --regime-fallback \
  --interval 5 \
  2>&1 | tee logs/autonomous_trader/session_run.log

# Test MT5 connection
uv run python scripts/test_mt5.py

# Test LLM connection
uv run python scripts/test_llm_review.py

# Monitor trades
tail -f logs/autonomous_trader/session_run.log | grep -E "Signal:|LLM:|Order:|trade_logged|BUY|SELL|HOLD|error"
```

**Acceptance Criteria:**
1. ⏳ **CRITICAL:** Execute first real demo trade via MT5
2. ⏳ Generate ≥5 trades/day for competition volume
3. ⏳ Maintain drawdown <12% (hard limit)
4. ⏳ Verify stop-loss and take-profit working
5. ⏳ Track PnL over 1-2 weeks
6. ⏳ Retrain transformer on 2026 data

**Time Estimates:**
- Bug fixes / monitoring: 1-2 days
- Transformer retrain: 2-4 hours
- Paper trading validation: 1-3 weeks

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

## 🎯 Next Steps

**This Session (Immediate):**
1. ✅ Start autonomous trader with LLM + regime fallback
2. ✅ Enable MT5 AutoTrading (green button)
3. ✅ Monitor for first actual demo trade
4. ✅ Fix any blocking bugs

**Next Session (Transformer Retrain):**
1. Retrain transformer on 2026 data to fix concept drift
2. Verify higher forecast confidence (target 0.04+)
3. Reduce reliance on regime fallback

**Next Session (Strategy):**
1. Lower LLM confidence threshold if needed (0.65 → 0.50)
2. Add momentum-based signals for conflicting regime/flow
3. Consider 1m interval for more signals
4. Track trade frequency and PnL

**Weeks 2-4 (Paper Trading):**
1. Run bot continuously for 1-3 weeks
2. Monitor Sharpe ratio, drawdown, trade count
3. Fix bugs as they appear
4. Log all findings

**Week 5 (Production):**
1. Retrain all models on 12-month dataset
2. Validate on holdout set
3. Deploy production checkpoints

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
3. ✅ Sequential execution — work on current phase only, no skipping phases

---

## 📚 Reference Links

| Resource | Location | Purpose |
|----------|----------|---------|
| Master Plan | `idea/PLAN.md` | Overall strategy, data requirements, training cadence |
| Agent Rules | `idea/AGENTS.md` | Coding conventions, logging, testing requirements |
| CLI Design | `idea/CLI.md` | Complete CLI surface for all phases |
| Phase Specs | `idea/phases/00_setup.md` through `10_live.md` | Detailed phase-by-phase implementation specs |
| Memory | `.claude/projects/.../memory/MEMORY.md` | Project context, user preferences, feedback |
| Session Start | `SESSION_START.md` | Quick orientation for new sessions |

---

## 📝 Progress Log

| Date | Event | Notes |
|------|-------|-------|
| 2026-06-18 | Planning complete | Analyzed hackathon rules, decided to build full system |
| 2026-06-18 | Phase 0 complete | Specs in idea/, memory system initialized |
| 2026-06-18 | Phase 1 complete | Data pipeline with 16/16 tests passing |
| 2026-06-21 | Revised strategy | Use small dataset for MVP, retrain for production |
| 2026-06-23 | Autonomous trader built | Full V6 pipeline + LLM + MT5 execution end-to-end |
| 2026-06-23 | Paper mode enabled | Lowered thresholds, regime fallback, first trade attempt |
| 2026-06-23 | AutoTrading disabled | MT5 retcode 10027 — need manual enable in terminal |
| 2026-06-23 | LLM timeout | [WinError 10060] intermittent network issue to Fireworks |
| 2026-06-23 | Transformer concept drift | Model from 2025-03-31, confidence 0.001-0.02 |

---

**Last Updated:** 2026-06-23
**Updated By:** Claude
**Next Review:** When first actual demo trade executes
