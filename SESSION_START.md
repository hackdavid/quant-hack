# Quick Session Start Guide

**Last Updated:** 2026-06-18  
**Current Phase:** Phase 1 (Data Collection)  
**Status:** ✅ Implementation Complete, ⏳ Waiting for Data

---

## 📍 Where We Are

**Phase 1 Implementation:** ✅ **COMPLETE** (2026-06-18)
- All code working
- 16/16 tests passing
- CLI functional
- Ready to collect data

**What's Blocking Phase 2:**
- Need ≥4-6 weeks of live tick data (trades + depth)
- Historical tick data NOT freely available
- Must capture from day 1 and accumulate

---

## 🚀 What To Do Next Session

### **If <4 weeks have passed:**

Check live capture status:
```bash
# Check if capture is still running
ps aux | grep "intraday data live-capture"

# Check captured data
intraday data summary

# Verify checkpoint
intraday data checkpoint
```

If capture stopped, restart it:
```bash
# In tmux/screen
intraday data live-capture --streams trade,depth,mark_price
```

### **If ≥4 weeks have passed:**

1. Verify data collected:
   ```bash
   intraday data summary
   # Should show 4-6 weeks of trade/depth data
   ```

2. Update MASTER_INDEX.md:
   - Phase 1 status → ✅ DONE
   - Phase 2 status → 🔵 CURRENT

3. Start Phase 2:
   - Read `idea/phases/02_features.md`
   - Implement feature engine (OFI, microprice, VPIN, Hawkes, etc.)

---

## 📋 Pending Actions (Phase 1)

- [ ] **Download 12 months historical data** (~10-30 mins total):
  ```bash
  intraday data download --kind klines_5m --start 2024-01-01 --end 2024-12-31
  intraday data download --kind klines_1m --start 2024-01-01 --end 2024-12-31
  intraday data download --kind funding --start 2024-01-01 --end 2024-12-31
  intraday data download --kind open_interest --start 2024-01-01 --end 2024-12-31
  ```

- [ ] **Start live capture** (run in background for 4-6 weeks):
  ```bash
  # In tmux/screen session
  intraday data live-capture --streams trade,depth,mark_price
  
  # Detach: Ctrl+B, D (tmux) or Ctrl+A, D (screen)
  ```

- [ ] **Set calendar reminder** for 4-6 weeks:
  - Date: ~2026-07-23 to 2026-08-01
  - Action: Check back, verify data, start Phase 2

---

## 🗂️ Key Files

| File | Purpose |
|------|---------|
| `MASTER_INDEX.md` | **Read this first every session** - tracks progress |
| `SESSION_START.md` | This file - quick orientation |
| `QUICKSTART.md` | Phase 1 usage guide |
| `idea/PLAN.md` | Master plan for all 11 phases |
| `idea/phases/02_features.md` | Next phase spec (when ready) |
| Memory: `~/.claude/projects/.../memory/MEMORY.md` | Project memory index |

---

## 💡 Quick Commands

```bash
# Check what's downloaded
intraday data summary

# Resume download from checkpoint
intraday data download --kind klines_5m

# Download specific range
intraday data download --kind klines_5m --start 2024-06-01 --end 2024-06-30

# Start live capture (all streams)
intraday data live-capture --streams trade,depth,mark_price,liquidations

# View checkpoint details
intraday data checkpoint

# Run tests
pytest tests/phase_01/ -v

# CLI help
intraday --help
intraday data --help
```

---

## 🎯 Session Workflow

1. **Read MASTER_INDEX.md** → see current phase
2. **Check if blocked** → Phase 2 blocked until 4-6 weeks data
3. **Work on current phase** → Phase 1 = data collection
4. **Update MASTER_INDEX** → when milestones hit
5. **Update memory** → key learnings/decisions

---

## 📊 Progress Summary

**Timeline:**
- Day 1 of 60-80 total
- Next milestone: 4-6 weeks (tick data collection)

**Phases:**
- ✅ Phase 0: Planning
- ✅ Phase 1: Implementation (code done)
- ⏳ Phase 1: Data collection (4-6 weeks)
- ⚪ Phase 2: Blocked (waiting for data)
- ⚪ Phase 3-10: Blocked

**Tests:** 16/16 passing ✅

**Data Collected:**
- Historical: TBD (run downloads)
- Live: TBD (start capture, accumulate 4-6 weeks)

---

**Remember:** Don't skip to Phase 2 until ≥4-6 weeks of live tick data is captured!
