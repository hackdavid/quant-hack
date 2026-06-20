# Development Strategy - MVP First, Production Second

**Last Updated:** 2026-06-21  
**Rationale:** De-risk development with small dataset, then scale to production

---

## 🎯 Core Strategy

**Problem:** Don't want to wait 4-6 weeks idle during data collection.

**Solution:** Use 4-6 week dataset as **MVP development data**, build entire pipeline to prove code works, then retrain on full 12-month data for production.

---

## 📊 Two-Tier Approach

### Tier 1: MVP Development (4-6 Week Dataset)

**Purpose:** Validate code, catch bugs, iterate fast

**Timeline:** Weeks 2-6 (while data collecting)

**What to Build:**
- Phase 2: Feature engine (OFI, microprice, VPIN, etc.)
- Phase 3: Queue-aware simulator
- Phase 4: Forecast agent training pipeline
- Phase 5: Other 4 agents (Orderflow, Regime, Risk, Stay-out)
- Phase 6: Meta-learner aggregator
- Phase 7: RL execution (CQL)

**Training Time (GPU):**
- Phase 4: ~0.5-1 hour (small dataset)
- Phase 5: ~0.5-1 hour (4 agents on small data)
- Phase 7: ~1-2 hours (10k episodes, reduced from 50k)
- **Total GPU:** ~2-4 hours

**Acceptance Criteria (Relaxed for MVP):**
- Code runs without errors
- Feature IC >0.05 (vs >0.1 for production)
- Models train and produce predictions
- Simulator produces realistic fills
- End-to-end pipeline works

**Output:** 
- ✅ Validated code (no bugs)
- ✅ Proven architecture works
- ✅ Fast iteration cycles
- ❌ NOT for trading (models undertrained)

---

### Tier 2: Production Training (12-Month Dataset)

**Purpose:** Train production-quality models for paper/live trading

**Timeline:** Week 7 (after MVP validated)

**What to Train:**
- Phase 4: Forecast agent (full dataset)
- Phase 5: All 4 agents (full dataset)
- Phase 7: RL execution (50k episodes)

**Training Time (GPU):**
- Phase 4: 2-4 hours (vs 0.5-1 hr MVP)
- Phase 5: 2-4 hours (4 agents full-scale)
- Phase 7: 4-8 hours (50k episodes)
- **Total GPU:** ~8-16 hours

**Acceptance Criteria (Full Production):**
- OOS Sharpe ≥0.5 (Forecast)
- Feature IC >0.1 (key features)
- Brier score <0.5
- OOS Sharpe ≥1.0 (Aggregator)
- Slippage <0.8× baseline OR Sharpe +0.1 (RL)

**Output:**
- Production model checkpoints
- Ready for paper trading
- Meets all acceptance criteria from `idea/phases/`

---

## 🚀 Timeline Breakdown

| Week | Phase | Activity | Dataset | GPU Time | Output |
|------|-------|----------|---------|----------|--------|
| 1 | Phase 1 | Deploy CPU, download 12mo, start live capture | - | - | Data collection running |
| 2-3 | Phase 2-3 | Features + Simulator | 4-6wk (when ready) | - | Feature engine, backtest working |
| 3-4 | Phase 4-5 | ML Agents MVP | 4-6wk | ~1-2 hrs | Training pipelines validated |
| 4-5 | Phase 6-7 | Aggregator + RL MVP | 4-6wk | ~1-2 hrs | End-to-end pipeline working |
| 6 | Testing | Fix bugs, optimize code | 4-6wk | - | Production-ready code |
| 7 | **PROD** | **Retrain on 12mo data** | **12 months** | **8-16 hrs** | **Production models** |
| 8+ | Phase 8 | Paper trading | Live | - | Performance validation |

**Total Development:** ~7 weeks  
**Total GPU:** ~12-20 hours (~$12-60 on cloud GPU)

---

## 💡 Why This Works

### Benefits of MVP-First Approach

1. **Early Bug Detection**
   - Catch feature engineering bugs on small data (fast debug)
   - Find simulator issues before expensive training
   - Validate data pipeline edge cases

2. **Fast Iteration**
   - Train in 1-2 hours (vs 8-16 hours)
   - Test hyperparameters quickly
   - Experiment with architectures

3. **Risk Reduction**
   - Prove entire pipeline works before committing to full training
   - No wasted GPU time on broken code
   - No 8-hour training runs that fail

4. **Cost Savings**
   - MVP training: ~$2-5 GPU cost
   - Find bugs early, not after $50 production training run
   - Only pay for full training once code validated

5. **Learning**
   - Understand data characteristics on small sample
   - Calibrate expectations for production metrics
   - Identify bottlenecks before scale

### What MVP Won't Give You

- ❌ Production-quality models (undertrained)
- ❌ Realistic Sharpe ratios (small sample size)
- ❌ Robust regime detection (need full market cycles)
- ❌ Trading-ready systems (use production models for that)

**MVP = Code Validation, NOT Trading**

---

## 🎯 Key Distinctions

### MVP Models vs Production Models

| Aspect | MVP Models | Production Models |
|--------|------------|-------------------|
| **Data** | 4-6 weeks | 12 months |
| **Purpose** | Validate code | Trade with |
| **Training** | 2-4 hrs GPU | 8-16 hrs GPU |
| **Acceptance** | Relaxed (IC >0.05) | Strict (IC >0.1, Sharpe ≥1.0) |
| **Use Case** | Development, testing | Paper/live trading |
| **Checkpoints** | `models/mvp/` | `models/production/` |

### Dataset Size Impact

**4-6 Week Dataset:**
- ~40,320 - 60,480 5-minute bars
- ~201,600 - 302,400 1-minute bars
- Good for: Feature engineering, code validation
- Bad for: Regime detection, robust statistics

**12-Month Dataset:**
- ~525,600 5-minute bars
- ~2,628,000 1-minute bars
- Covers: Multiple market regimes, full volatility cycles
- Good for: Production training, robust metrics
- Required for: Trading with confidence

---

## 🛠️ Implementation Details

### MVP Development Workflow

```bash
# Week 2-3: Features + Simulator
cd ~/quanthack
git checkout -b phase-2-features

# Use 4-6 week dataset (small, fast)
uv run intraday features compute \
  --start 2024-11-01 \
  --end 2024-12-15 \
  --output data/processed/features_mvp.parquet

# Test simulator
uv run intraday backtest \
  --data data/processed/features_mvp.parquet \
  --start 2024-11-01 \
  --end 2024-12-15

# Week 3-4: Train ML agents (MVP)
uv run intraday train forecast \
  --data-start 2024-11-01 \
  --data-end 2024-12-15 \
  --device cuda \
  --output models/mvp/forecast/

# Week 4-5: RL execution (MVP, 10k episodes)
uv run intraday train execution \
  --episodes 10000 \
  --device cuda \
  --output models/mvp/execution/
```

### Production Training Workflow

```bash
# Week 7: Retrain on full dataset
uv run intraday train forecast \
  --data-start 2024-01-01 \
  --data-end 2024-12-31 \
  --device cuda \
  --output models/production/forecast/

# Full-scale RL (50k episodes)
uv run intraday train execution \
  --episodes 50000 \
  --device cuda \
  --output models/production/execution/

# Validate OOS metrics
uv run intraday validate \
  --model-dir models/production/ \
  --oos-start 2024-10-01 \
  --oos-end 2024-12-31

# Expected output:
# ✅ Forecast Sharpe (OOS): 0.52
# ✅ Brier score: 0.47
# ✅ Aggregator Sharpe (OOS): 1.12
# ✅ RL slippage reduction: 15%
```

---

## 📋 Acceptance Checklist

### MVP Acceptance (Week 6)

- [ ] Feature engine runs without errors on 4-6wk data
- [ ] Simulator produces realistic fills
- [ ] Forecast agent trains successfully on GPU
- [ ] All 4 other agents train successfully
- [ ] Meta-learner aggregates predictions
- [ ] RL execution trains successfully (10k episodes)
- [ ] End-to-end backtest completes
- [ ] Code pushed to Git, tests passing

### Production Acceptance (Week 7)

- [ ] All models retrained on 12-month dataset
- [ ] Forecast OOS Sharpe ≥0.5
- [ ] Brier score <0.5
- [ ] Feature IC >0.1 (OFI, microprice, VPIN)
- [ ] Aggregator OOS Sharpe ≥1.0
- [ ] RL slippage <0.8× baseline OR Sharpe +0.1
- [ ] All tests passing (100+ tests)
- [ ] Checkpoints saved to `models/production/`
- [ ] Ready for Phase 8 (paper trading)

---

## 🚨 Common Pitfalls to Avoid

### Don't Confuse MVP with Production

❌ **Wrong:** Use MVP models for paper trading  
✅ **Right:** Use production models (12mo data) for paper trading

❌ **Wrong:** Judge strategy viability from MVP Sharpe  
✅ **Right:** MVP validates code, production validates strategy

❌ **Wrong:** Skip MVP, train on 12mo immediately  
✅ **Right:** Always validate with MVP first (save time/money)

### Don't Skip Full Retraining

❌ **Wrong:** "MVP Sharpe is 0.8, that's good enough"  
✅ **Right:** Always retrain on full 12mo before trading

❌ **Wrong:** "Just fine-tune MVP models on more data"  
✅ **Right:** Full retrain from scratch (avoid overfitting artifacts)

### Don't Over-Optimize on MVP Data

❌ **Wrong:** Tune hyperparameters until MVP Sharpe = 1.5  
✅ **Right:** Use MVP to validate architecture, tune on full data

---

## 🎓 Key Takeaways

1. **MVP = Code Validation** (not strategy validation)
2. **Production = Trading** (always use 12mo models)
3. **Save Time:** Find bugs on 4-6wk data (fast iteration)
4. **Save Money:** Only pay for full training once code works
5. **De-Risk:** Prove pipeline before expensive GPU runs
6. **Never Trade with MVP Models** (undertrained, unreliable)

---

## 📊 Cost Comparison

### Without MVP (Naive Approach)

```
Week 1: Train on 12mo → Bug found → 8hr GPU wasted
Week 2: Fix bug, retrain → Another bug → 8hr GPU wasted
Week 3: Fix bug, retrain → Finally works → 8hr GPU
Total: 24 hours GPU, 3 weeks, $72-240
```

### With MVP (Smart Approach)

```
Week 1: Train on 4-6wk → Bug found → 1hr GPU
Week 2: Fix bug, retrain MVP → Another bug → 1hr GPU
Week 3: Fix bugs, validate MVP → Works → 1hr GPU
Week 4: Retrain on 12mo → Success first try → 8hr GPU
Total: 11 hours GPU, 4 weeks, $33-110 (60% savings)
```

**MVP approach saves 50-60% GPU cost by catching bugs early.**

---

## 📈 Next Steps

1. ✅ Read this strategy document
2. ✅ Deploy CPU machine (see `DEPLOYMENT.md`)
3. ✅ Start data collection (12mo + live 4-6wk)
4. ⏰ Wait 4-6 weeks for tick data
5. 📖 Read `idea/phases/02_features.md`
6. 🛠️ Build Phase 2-7 MVP on small dataset
7. ✅ Validate entire pipeline works
8. 🚀 Retrain on 12-month data (production)
9. 📊 Deploy to paper trading
10. 💰 Scale to live trading

---

**Remember:** MVP proves the code works. Production proves the strategy works. Never confuse the two.
