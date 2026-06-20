# Phase 2 — Feature Engine

**Goal:** turn raw data into the microstructure features the agents need.
This is the layer that contains 70% of the alpha — every feature here
must be implemented carefully and unit-tested.

**Why second:** features are the foundation for both the simulator
(Phase 3) and every agent (Phases 4–7). Bugs here are silent killers.

**Estimated effort:** 5–8 days.

**Activates dep group:** `phase2` (numba, scipy).

---

## 1. Inputs / outputs

- **Inputs:** `data/raw/...` (Phase 1 output).
- **Outputs:**
  - Pure functions for every feature, vectorized + `numba`-jitted where
    hot.
  - Two pipelines:
    - **Slow / time-grid pipeline:** outputs `data/features/state_5m/` —
      canonical state vector aligned to 5-minute event boundaries.
    - **Fast / event-time pipeline:** outputs `data/features/micro_event/`
      — features sampled on volume bars / imbalance bars / CUSUM events.
  - `intraday features compute` CLI to run either pipeline over a date range.
  - `intraday features inspect` for IC and signal-to-noise diagnostics.

---

## 2. Files to create

```
src/intraday/features/
  __init__.py
  registry.py                 # decorator @feature(name=...) + lookup
  bars.py                     # event-time bar construction (volume/dollar/imbalance)
  cusum.py                    # CUSUM filter (Lopez de Prado)
  returns.py                  # log returns, standardized returns, vol scaling
  microprice.py               # Stoikov microprice
  ofi.py                      # Cont-Kukanov-Stoikov OFI (multi-level)
  hawkes.py                   # univariate self-exciting intensity
  vpin.py                     # volume-synchronized PIN
  bipower.py                  # bipower variation, jump test
  spread.py                   # Roll, Corwin-Schultz spread estimators
  hurst.py                    # rescaled-range Hurst, DFA variant
  entropy.py                  # Shannon entropy of return signs
  trend_strength.py           # linear-fit R^2 on log returns
  funding_signal.py           # composite from funding + OI + liq
  basis.py                    # cross-venue basis & lead-lag (Binance vs Coinbase)
  iv_skew.py                  # Deribit 25-delta risk-reversal
  pipelines/
    __init__.py
    state_5m.py               # builds canonical 5m state vector
    micro_event.py            # builds event-time microstructure frame
  cli.py
tests/phase_02/
  test_bars.py
  test_cusum.py
  test_returns.py
  test_microprice.py
  test_ofi.py
  test_hawkes.py
  test_vpin.py
  test_bipower.py
  test_spread.py
  test_hurst.py
  test_entropy.py
  test_basis.py
  test_iv_skew.py
  test_pipelines.py
  test_inspect.py
  fixtures/
    synthetic_trades.parquet  # known-answer test data
    synthetic_depth.parquet
```

---

## 3. Feature registry pattern

Every feature is a pure function decorated with `@feature(name=...)`:

```python
@feature(
    name="microprice",
    inputs=["best_bid", "best_ask", "best_bid_size", "best_ask_size"],
    horizon="event",
    description="Stoikov 2017 weighted mid-price.",
)
def microprice(
    best_bid: pl.Series,
    best_ask: pl.Series,
    best_bid_size: pl.Series,
    best_ask_size: pl.Series,
) -> pl.Series:
    return (best_bid * best_ask_size + best_ask * best_bid_size) / (
        best_bid_size + best_ask_size
    )
```

This lets the CLI list/select features by name and lets the inspect
command auto-generate IC tables.

---

## 4. Feature specs (one section per feature)

For each feature, the spec must include: equation, inputs, outputs, edge
cases, test cases.

### 4.1 Log returns

`r_t = log(P_t / P_{t-1})`. Use `close` for kline-derived returns.
Edge cases: clamp `P_{t-1} > 0` else raise.

### 4.2 Standardized log returns

`z_t = (r_t - μ_t) / σ_t`, where `μ_t, σ_t` are rolling EW-stats with
half-life 60 minutes. The model never sees raw returns; it sees `z_t`.

### 4.3 Microprice (Stoikov 2017)

```
μ = (bid * V_ask + ask * V_bid) / (V_bid + V_ask)
```

Use `@bookTicker` as input source. Two variants:
- **L1 microprice** (best level only).
- **L5 microprice** — sum sizes across top 5 levels (more stable).

### 4.4 Order Flow Imbalance — Cont-Kukanov-Stoikov 2014

For each event, compute:

```
e_n = 1[Δp_b > 0] q_b - 1[Δp_b < 0] q_b_prev
    - 1[Δp_a > 0] q_a_prev + 1[Δp_a < 0] q_a
```

where `p_b, q_b` are best bid price/size, similarly ask, and `Δ` is
change vs previous depth update. Sum `e_n` over a rolling window
(default 1s, 5s, 30s, 1m). Output normalized OFI = `Σ e_n / volume`.

**Multi-level OFI:** repeat for the top 5 levels, sum with depth-weighting
(weight `w_k = exp(-k/2)` for level `k`). This is the version that
actually has alpha.

### 4.5 Hawkes self-exciting intensity

Univariate Hawkes process for trade arrivals:

```
λ(t) = μ + Σ α exp(-β(t - t_i))
```

Implementation: maintain running intensity per side (buyer-initiated,
seller-initiated). Use exponential-kernel recursive update — O(1) per
event. Default `β = 1/0.5s` (decay half-life ~350ms), `α` learned
from L-BFGS once daily on the previous day's data.

Output features:
- `hawkes_buy_intensity`, `hawkes_sell_intensity`
- `hawkes_imbalance = (buy - sell) / (buy + sell)`
- `hawkes_branching_ratio = α/β` (regime indicator)

### 4.6 VPIN

Volume-synchronized buckets. Default bucket = 50 BTC. Window = 50 buckets.

```
VPIN = mean over last N buckets of |buy_vol - sell_vol| / bucket_volume
```

Trade-side classification: tick-rule (price up = buy) is too noisy for
crypto. Use **bulk volume classification (BVC)** by Easley et al. 2012
which uses standardized return CDF.

### 4.7 Bipower variation + jump test

```
RV = Σ r_i²        (realized variance)
BV = (π/2) Σ |r_i| |r_{i-1}|   (bipower variation)
J = max(0, RV - BV)            (jump component)
```

Output: `rv_5m`, `bv_5m`, `jump_5m`, `jump_z_score` (BNS test statistic).

### 4.8 Roll / Corwin-Schultz spread

Roll: `s ≈ 2√(-Cov(Δp_t, Δp_{t-1}))` (use only when Cov < 0).
Corwin-Schultz: from rolling 2-day high-low pairs.

### 4.9 Hurst exponent (rescaled range)

`R/S(n) = C n^H`. Estimate H over a rolling window of 500 5m bars.

Output: `hurst`, `hurst_state` ∈ {`mean_revert` (H<0.45), `random` (0.45-0.55), `trend` (>0.55)}.

### 4.10 Entropy

Shannon entropy of return signs (`+/−/0`) over rolling window.

```
H(X) = -Σ p(x) log p(x)
```

Output `entropy_5m`.

### 4.11 Funding signal

```
funding_z = (funding_rate - μ_funding) / σ_funding   (rolling 30d)
oi_change_pct_1h
liq_pressure = sum_long_liq_volume_last_5m / open_interest
funding_basis_dispersion = funding_perp - implied_basis_from_spot_perp
```

### 4.12 Cross-venue basis + lead-lag

```
basis_bn_cb = mid_binance - mid_coinbase   (USD)
basis_z = (basis - μ) / σ                  (rolling 1h)
lead_lag_corr_lag_k = corr(returns_cb, returns_bn.shift(-k))
```

`lead_lag_corr` peak indicates which venue leads at current moment.
Strong predictive feature during US hours.

### 4.13 Deribit IV skew

`iv_25d_rr = iv_call_25d - iv_put_25d` (risk-reversal, sentiment proxy).
`iv_atm` for vol regime.

### 4.14 Volume / dollar / imbalance bars (López de Prado Ch. 2)

- **Volume bar:** emit when cumulative base volume ≥ threshold T_vol.
- **Dollar bar:** emit when cumulative quote volume ≥ T_dollar.
- **Imbalance bar:** emit when cumulative |buy - sell| volume ≥
  E[T] (expected) using the imbalance formula in the book.

Default thresholds: targeted to give ~250 bars/day each.

### 4.15 CUSUM filter for sampling

Sample only when cumulative absolute return crosses a vol-scaled
threshold `h_t = k σ_t`. Halves data size, doubles SNR.

---

## 5. Two pipelines

### `pipelines/state_5m.py` — canonical 5-minute state

For each 5-minute boundary `t`, build a `dict[str, float]`:

```python
{
    "ts_ms": ...,
    "log_return_5m": ...,
    "log_return_15m": ...,
    "z_return_5m": ...,
    "realized_vol_30m": ...,
    "bv_30m": ...,
    "jump_z_score": ...,
    "hurst": ...,
    "entropy": ...,
    "trend_strength": ...,
    "funding_z": ...,
    "oi_change_pct_1h": ...,
    "liq_pressure": ...,
    "iv_atm": ...,
    "iv_25d_rr": ...,
    "basis_bn_cb_z": ...,
    "lead_lag_argmax_lag_ms": ...,
    # microstructure aggregated at the 5m boundary:
    "ofi_5m_l1": ...,
    "ofi_5m_l5": ...,
    "vpin_50": ...,
    "hawkes_imbalance_avg_5m": ...,
    "spread_bps_avg_5m": ...,
    "microprice_drift_5m": ...,
    # housekeeping:
    "regime_hint_label": ...,    # set by Phase 5; placeholder here
}
```

Output: one parquet per UTC date in `data/features/state_5m/{symbol}/{date}.parquet`.

### `pipelines/micro_event.py` — event-time microstructure frame

For each tick or bar emitted by the imbalance-bar sampler, produce a
shorter-horizon feature row used by the orderflow agent and the
execution policy. Schema is similar but at event resolution.

---

## 6. CLI commands wired

| Command | Behavior |
|---|---|
| `intraday features compute --start ... --end ... --features all` | run both pipelines |
| `intraday features compute --pipeline state_5m --features ofi,hawkes` | partial recompute |
| `intraday features inspect --feature ofi --target return_15m --start ... --end ...` | IC table |

`features compute` is **idempotent** at file level (skip files already
written) and **partial** at column level (`--features` selects which
columns to (re)compute).

---

## 7. Unit tests

For each feature, tests must include:

1. **Closed-form verification** on a synthetic input where the result
   is known analytically (e.g. constant price → 0 returns; symmetric
   book → microprice = mid).
2. **Property test** with `hypothesis` for continuity / monotonicity
   where applicable.
3. **Edge cases**: zero volume, zero spread, NaN inputs (must raise
   `DataIntegrityError`, not propagate).
4. **Performance**: must process 1 day of synthetic data within a
   per-feature budget (table below).

| Feature | 1-day budget |
|---|---|
| log_return | 50 ms |
| microprice | 50 ms |
| ofi (multi-level) | 500 ms |
| hawkes | 800 ms |
| vpin | 200 ms |
| bipower | 100 ms |
| hurst | 300 ms |
| basis | 200 ms |

Use `pytest-benchmark`. Fail the test if budget exceeded.

---

## 8. Integration test

`tests/integration/test_phase_02.py`:

```python
def test_full_pipeline_one_week(tmp_path, prepared_raw_data):
    """Given 1 week of synthetic raw data:
       - run state_5m pipeline → produces 7 parquets, no NaNs.
       - run micro_event pipeline → produces 7 parquets, no NaNs.
       - features inspect on 'ofi' returns IC dataframe with > 50 rows.
    """
```

Manual smoke:

```bash
# After Phase 1 has at least 1 month of klines + 1 week of live ticks:
uv run intraday features compute --start 2025-01-01 --end 2025-01-31
uv run intraday features inspect --feature ofi_5m_l5 --target log_return_15m \
    --start 2025-01-01 --end 2025-01-31
```

---

## 9. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | Every feature passes closed-form unit test | `pytest tests/phase_02/ -v` |
| 2 | Every feature meets its perf budget | `pytest tests/phase_02/ -v --benchmark-only` |
| 3 | `features compute` for 1 month (5m grid) finishes < 60s on 16-core CPU | benchmark |
| 4 | Output parquets validate against canonical schema | `intraday data verify --kind state_5m` |
| 5 | IC of `ofi_5m_l5` against `log_return_15m` over 1 month is > 0.02 | `features inspect` |
| 6 | IC of `funding_z` against `log_return_24h` over 30d is > 0.05 | `features inspect` |
| 7 | No feature ever emits NaN given valid inputs | unit + property tests |

If acceptance #5 or #6 fail, **stop**. Either the implementation is wrong
or the data is bad. Investigate before moving on — these are the
"alpha-bearing" features and zero IC means the rest of the system has
nothing to work with.

---

## 10. Common mistakes to avoid

- **Don't compute features lazy-row-by-row.** Always vectorize with
  polars; jit hot loops with numba (Hawkes, OFI multi-level).
- **Don't forget `ts_local_ns`** — it's needed to recover event ordering
  when multiple WS streams arrive within the same ms.
- **Don't classify trades by tick rule on crypto** — too noisy. Use BVC.
- **Don't compute Hurst on < 500 samples** — extremely unstable. Output
  `null` until enough data.
- **Don't use the same vol horizon for everything.** Microprice drift at
  5m needs a 30m vol; funding z-score needs a 30d vol.
- **Don't merge state_5m with micro_event in storage.** They have
  different time semantics and should not share a parquet file.

---

## 11. Done ⇒ proceed to `phases/03_simulator.md`.
