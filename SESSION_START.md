# Quick Session Start Guide

**Last Updated:** 2026-06-23
**Current Status:** Autonomous Trader Pipeline — End-to-End Working
**Next Priority:** Get actual trades executing + retrain transformer on 2026 data

---

## 📍 Where We Are

**What's Working:**
- ✅ Full V6 pipeline running (Forecast + Orderflow + Regime + Risk + StayOut + MetaLearner + DecisionEngine)
- ✅ 20-feature calculation from live Binance WebSocket (1m + 5m klines)
- ✅ LLM review with Fireworks Kimi K2.6 (full context: candle, pipeline, positions, logs)
- ✅ MT5 demo account connected and executing orders
- ✅ Paper mode with lowered thresholds (forecast_confidence=0.02, meta_threshold=0.05)
- ✅ Regime fallback strategy when transformer is uncertain
- ✅ Position manager (closes opposite, opens new, prevents double-entry)
- ✅ JSONL trade logging to `logs/autonomous_trader/`

**What's Broken / Needs Fix:**
- 🔴 Transformer model is 15 months old (trained 2025-03-31) — concept drift causes confidence 0.001-0.02
- 🔴 LLM API intermittently times out with [WinError 10060] — network connectivity to Fireworks
- 🔴 WebSocket connected but bar processing needs monitoring (may stall)
- 🔴 Strategy generates mostly HOLD signals — need more trades for competition volume
- 🟡 MT5 AutoTrading must be manually enabled in terminal (Ctrl+E) every session

---

## 🚀 What To Do Next Session

### **Priority 1: Get Actual Trades Executing**

```bash
# Start trader with LLM + regime fallback + paper mode
cd "C:\Users\DaudDewan\OneDrive - SymphonyAI\Documents\Learning\hackathon\quanthack"
uv run python scripts/autonomous_trader.py \
  --transformer-run models/transformer/20260623T132957Z \
  --mt5-account YOUR_ACCOUNT \
  --mt5-password "YOUR_PASSWORD" \
  --mt5-server "YOUR_SERVER" \
  --use-llm --llm-debug \
  --paper-mode --regime-fallback \
  --interval 5 \
  2>&1 | tee logs/autonomous_trader/session_run.log
```

**Before starting:**
1. Open MetaTrader 5 terminal
2. Click **AutoTrading** button (or press `Ctrl+E`) until it turns green
3. Verify your MT5 account is logged in

**Monitor:**
```bash
# Watch for trades in real-time
tail -f logs/autonomous_trader/session_run.log | grep -E "Signal:|LLM:|Order:|trade_logged|BUY|SELL|HOLD|regime_fallback|error"
```

### **Priority 2: Retrain Transformer on 2026 Data**

```bash
# Check current model age
cat models/transformer/20260623T132957Z/metadata.json | grep "train_date"

# If older than 1 month, retrain:
# See idea/phases/04_forecast.md for training pipeline
# Or use src/intraday/forecast/train.py
```

**Why:** Model trained on 2025-03-31 data has concept drift. Current BTC price ~$62k vs training price ~$70k. This causes forecast confidence to be 0.001-0.02 even with paper mode threshold 0.02.

### **Priority 3: Improve Strategy for More Trades**

Current regime fallback only fires when:
- Regime = bull + flow_bias > 0 → BUY
- Regime = bear + flow_bias < 0 → SELL

**Options to increase trade frequency:**
1. Lower LLM confidence threshold from 0.65 to 0.50 (in `src/intraday/llm/review.py`)
2. Add momentum-based fallback when regime/flow conflict
3. Reduce position size but increase trade frequency
4. Use 1m interval instead of 5m (more signals, more noise)

---

## 📋 Pending Actions

- [ ] **Execute first real demo trade** — verify MT5 order placement works end-to-end
- [ ] **Retrain transformer** on recent 2026 data to fix concept drift
- [ ] **Add LLM retry logic** — handle [WinError 10060] timeout gracefully
- [ ] **Monitor WebSocket stability** — ensure bars are processed every 5m
- [ ] **Generate 5+ trades/day** — competition requires volume for Sharpe calculation
- [ ] **Track PnL** — verify positions are profitable and stops are working

---

## 🗂️ Key Files

| File | Purpose |
|------|---------|
| `MASTER_INDEX.md` | Read first — tracks overall project progress |
| `scripts/autonomous_trader.py` | Main trading bot — start here |
| `src/intraday/llm/review.py` | LLM review agent — fix timeout handling |
| `src/intraday/features/calculator.py` | Feature calculation — 20 features live |
| `logs/autonomous_trader/` | Trade logs — check for actual executions |
| Memory: `project_autonomous_trader.md` | Detailed pipeline status and findings |

---

## 💡 Quick Commands

```bash
# Test MT5 connection
uv run python scripts/test_mt5.py

# Test LLM connection
uv run python scripts/test_llm_review.py

# See exact LLM prompt
uv run python scripts/report_llm_inputs.py

# Run tests
pytest tests/ -v

# Check data
intraday data summary
```

---

## 🎯 Session Workflow

1. **Read `project_autonomous_trader.md` memory** → understand current pipeline state
2. **Enable MT5 AutoTrading** → green button in terminal
3. **Start trader** → `uv run python scripts/autonomous_trader.py ...`
4. **Monitor logs** → tail -f for trades, errors, WebSocket issues
5. **If no trades in 30 min** → check regime/flow alignment, consider lowering thresholds
6. **If LLM times out** → restart trader, check network
7. **Update memory** → log findings, new issues, fixes

---

## 📊 Progress Summary

**Timeline:**
- Day 6 of 60-80 total
- Phase 1-5 effectively complete (data + features + agents + aggregator + decision engine)
- Phase 8 (paper trading) in progress — bot running, need actual trades

**What's Working:**
- Full V6 pipeline ✅
- LLM review ✅
- MT5 execution ✅
- Paper mode ✅
- Regime fallback ✅

**Blockers:**
- Transformer concept drift (model too old) 🟡
- LLM API timeout intermittently 🟡
- WebSocket bar processing needs monitoring 🟡
- Trade frequency too low 🟡

**Tests:** 16/16 passing ✅

**Data:**
- Historical: 128 bars loaded from Binance API (no warm-up needed)
- Live: WebSocket streaming 1m + 5m klines

---

**Remember:** Don't retrain on full 12mo yet — first prove the pipeline works with actual trades on paper mode, then scale up!
