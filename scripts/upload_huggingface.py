#!/usr/bin/env python3
"""Upload the BTCUSDT futures dataset to Hugging Face Hub.

Uploads two datasets under one repo:
  features/BTCUSDT/YYYY-MM-DD.parquet  — ML-ready 27-column feature bars (120 MB)
  raw/klines_1m/BTCUSDT/               — 1-minute OHLCV bars               (132 MB)
  raw/klines_5m/BTCUSDT/               — 5-minute OHLCV bars                (41 MB)
  raw/bookDepth/BTCUSDT/               — order-book depth snapshots         (158 MB)
  raw/metrics/BTCUSDT/                 — OI / L-S ratios                     (32 MB)
  raw/aggTrades/BTCUSDT/               — tick-level aggTrades               (~17 GB, optional)

Run:
    uv run python scripts/upload_huggingface.py
"""

import getpass
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ── Dataset card ──────────────────────────────────────────────────────────────

DATASET_CARD = """\
---
license: mit
language:
  - en
tags:
  - finance
  - crypto
  - bitcoin
  - quantitative-finance
  - time-series
  - algorithmic-trading
size_categories:
  - 1M<n<10M
---

# BTCUSDT Perpetual Futures — 5-Minute Feature Dataset

Complete historical dataset for **Binance BTCUSDT USDT-Margined Perpetual Futures**,
covering **2020-09-10 → 2026-05-31** (~5.7 years, 601,920 five-minute bars).

Built for quantitative research and ML model training. All raw data is sourced from
[data.binance.vision](https://data.binance.vision) (Binance's official public archive)
and processed with a deterministic, event-driven feature pipeline.

---

## Repository structure

```
features/
  BTCUSDT/
    2020-09-10.parquet   # 288 rows — one per 5-min bar
    2020-09-11.parquet
    ...
    2026-05-31.parquet

raw/
  klines_1m/BTCUSDT/    # 1-minute OHLCV bars
  klines_5m/BTCUSDT/    # 5-minute OHLCV bars
  bookDepth/BTCUSDT/    # L2 order-book depth snapshots (from 2023-01-01)
  metrics/BTCUSDT/      # Open interest, long/short ratios
  aggTrades/BTCUSDT/    # Tick-level aggregated trades (~17 GB, optional)
```

---

## Quick start

```python
import polars as pl
from huggingface_hub import snapshot_download

# Download only the feature files (120 MB) — skip raw data
local_dir = snapshot_download(
    repo_id="YOUR_USERNAME/btcusdt-futures-features",
    repo_type="dataset",
    ignore_patterns=["raw/*"],
)

# Load all feature bars into a single DataFrame
df = pl.read_parquet(f"{local_dir}/features/BTCUSDT/*.parquet")
print(df.shape)          # (~601920, 27)
print(df.dtypes)
```

Or load a single day:

```python
df = pl.read_parquet(f"{local_dir}/features/BTCUSDT/2024-01-15.parquet")
```

---

## Feature schema (27 columns)

### Identity

| Column | Type | Description |
|--------|------|-------------|
| `bar_time_ms` | int64 | Bar **open** time in milliseconds UTC |
| `symbol` | str | Always `"BTCUSDT"` |

### Price features

| Column | Type | Formula / Notes |
|--------|------|-----------------|
| `close` | float64 | 5m bar close price (USDT) |
| `log_ret_1m` | float64 | `ln(close_1m[t] / close_1m[t-1])` — log return of the most recent 1m bar |
| `log_ret_5m` | float64 | `ln(close_5m[t] / close_5m[t-1])` — log return of this 5m bar |
| `log_ret_15m` | float64 | `ln(close_5m[t] / close_5m[t-3])` — log return over the last 3 × 5m bars |
| `log_ret_60m` | float64 | `ln(close_5m[t] / close_5m[t-12])` — log return over the last 12 × 5m bars |
| `realized_vol_30m` | float64 | Sample std-dev of the last 30 one-minute log returns: `sqrt( Var( ln(c[i]/c[i-1]) ) )` |
| `rsi_14` | float64 | Wilder RSI(14) on 5m close prices: `100 - 100/(1 + avg_gain/avg_loss)` over the last 14 bars |

> **Null policy:** `log_ret_15m` / `log_ret_60m` are `null` for the first 3 / 12 bars of the dataset
> (insufficient history). All other price features are available from the first bar.

### Volume / taker-flow features

| Column | Type | Formula / Notes |
|--------|------|-----------------|
| `vol_5m` | float64 | Total BTC volume traded in this 5m bar (from klines_5m) |
| `taker_buy_ratio_5m` | float64 | `taker_buy_volume / vol_5m` ∈ [0, 1]. Values > 0.5 indicate net taker buying. |
| `trade_count_5m` | int64 | Number of aggregated trades in this 5m bar |
| `avg_trade_size_5m` | float64 | `vol_5m / trade_count_5m` — mean aggTrade size in BTC |

### Order book depth

Sourced from Binance `bookDepth` snapshots (available from **2023-01-01** onward).
Each snapshot covers cumulative depth in percentage-price bands around mid.

| Column | Type | Formula / Notes |
|--------|------|-----------------|
| `depth_imbalance_1pct` | float64 | `(bid_depth_1pct − ask_depth_1pct) / (bid_depth_1pct + ask_depth_1pct)` ∈ [−1, 1]. Positive = more bid-side depth within 1% of mid. **Null before 2023-01-01** (no bookDepth data in Binance bulk archive). |

> **Note on 0.2% band:** Binance's bulk bookDepth export does **not** populate the ±0.2% band
> (`bid_02pct` / `ask_02pct` are always null in the source files). Those columns were therefore
> excluded from this dataset entirely.

### VPIN (Volume-Synchronized Probability of Informed Trading)

Implementation follows [Easley et al. (2012)](https://doi.org/10.1093/rfs/hhs053).
Trade-flow is classified using the bulk-volume method (no tick test needed).

| Column | Type | Formula / Notes |
|--------|------|-----------------|
| `vpin_50` | float64 | `(1/50) × Σ|V_buy − V_sell| / V_bucket` over the last 50 buckets of 100 BTC each. Measures the fraction of volume driven by informed traders. Higher = more toxic flow. |
| `vpin_bucket_imbalance` | float64 | Buy-volume fraction in the **current open** bucket: `V_buy / (V_buy + V_sell)` ∈ [0, 1] |

Parameters used: `bucket_btc = 100`, `window = 50` (≈65 minutes of flow at average volume).

### Hawkes process intensities

Trades are modelled as a bivariate Hawkes process. Each buy or sell trade excites future
arrivals of its own kind. The intensity at time *t* is:

```
λ_buy(t)  = μ  +  α × Σ_{t_i < t, buy}   exp(−β × (t − t_i))
λ_sell(t) = μ  +  α × Σ_{t_j < t, sell}  exp(−β × (t − t_j))
```

Parameters used: `α = 1.0`, `β = 10.0 /s` (decay half-life ≈ 70 ms), `μ = 6.0 trades/s`.

| Column | Type | Description |
|--------|------|-------------|
| `hawkes_buy_intensity` | float64 | `λ_buy(t)` at bar close — buy-side arrival rate (trades/s) |
| `hawkes_sell_intensity` | float64 | `λ_sell(t)` at bar close — sell-side arrival rate (trades/s) |
| `hawkes_net` | float64 | `(λ_buy − λ_sell) / (λ_buy + λ_sell)` ∈ [−1, 1] — directional imbalance of trade flow |

### Market structure (from Binance futures metrics endpoint)

5-minute snapshots of open interest and long/short positioning.
Small data gaps exist around **2022 Q1** (128 days for `taker_ls_vol_ratio`, 19 days for `ls_count_ratio`).

| Column | Type | Description |
|--------|------|-------------|
| `oi_btc` | float64 | Open interest in BTC. Fully populated from 2020-09-10. |
| `oi_change_1h` | float64 | Fractional OI change vs 60 minutes ago: `(OI[t] − OI[t−12]) / OI[t−12]` |
| `ls_count_ratio` | float64 | Long-account count / short-account count (all accounts on the exchange) |
| `taker_ls_vol_ratio` | float64 | Taker buy volume / taker sell volume over the last 5 minutes |

### Forward targets (ML labels)

Filled **post-hoc** from future bars. The last few bars of the dataset have `null` targets
(no future data to look forward to).

| Column | Type | Description |
|--------|------|-------------|
| `fwd_ret_5m` | float64 | `ln(close[t+1] / close[t])` — log return of the NEXT 5m bar |
| `fwd_ret_15m` | float64 | `ln(close[t+3] / close[t])` — 15-minute forward return |
| `fwd_ret_60m` | float64 | `ln(close[t+12] / close[t])` — 60-minute forward return |
| `fwd_direction_5m` | int64 | `+1` if `fwd_ret_5m > 0.05%`, `−1` if `< −0.05%`, `0` otherwise |

---

## Raw data schemas

### `raw/klines_1m/` and `raw/klines_5m/`

Standard Binance OHLCV kline format.

| Column | Type |
|--------|------|
| `open_time_ms` | int64 |
| `open`, `high`, `low`, `close` | float64 |
| `volume` | float64 (BTC) |
| `close_time_ms` | int64 |
| `quote_volume` | float64 (USDT) |
| `trade_count` | int64 |
| `taker_buy_volume` | float64 (BTC) |
| `taker_buy_quote_volume` | float64 (USDT) |

### `raw/aggTrades/`

Each row is one aggregated trade (all fills of a single taker order).

| Column | Type |
|--------|------|
| `time_ms` | int64 |
| `price` | float64 |
| `quantity` | float64 (BTC) |
| `is_buyer_maker` | bool — `True` means the buyer was the maker (i.e., a taker sell) |

### `raw/bookDepth/`

L2 depth snapshots at ±1%, ±2%, ±3%, ±4%, ±5% price bands from mid. Available from **2023-01-01**.

| Column | Type |
|--------|------|
| `snapshot_time_ms` | int64 |
| `bid_1pct` … `bid_5pct` | float64 — cumulative BTC depth on the bid side |
| `ask_1pct` … `ask_5pct` | float64 — cumulative BTC depth on the ask side |

### `raw/metrics/`

Binance futures 5-minute metrics snapshot.

| Column | Type |
|--------|------|
| `create_time_ms` | int64 |
| `oi_btc`, `oi_usd` | float64 |
| `ls_count_ratio` | float64 |
| `taker_ls_vol_ratio` | float64 |
| `top_ls_count`, `top_ls_value` | float64 — top-trader L/S ratios |

---

## Data coverage summary

| Source | Coverage | Notes |
|--------|----------|-------|
| klines (1m, 5m) | 2020-09-10 → 2026-05-31 | Complete, no gaps |
| aggTrades | 2020-09-10 → 2026-05-31 | Complete, ~17 GB |
| metrics (OI / L/S) | 2020-09-10 → 2026-05-31 | Two small gaps in 2022 Q1 |
| bookDepth | 2023-01-01 → 2026-05-31 | Binance bulk archive starts here |

---

## Reproducing this dataset

All feature computation code is open-source:

```bash
git clone https://github.com/YOUR_USERNAME/quant-hack
cd quant-hack
uv sync

# 1. Download raw data from data.binance.vision
uv run intraday data download --start 2020-09-10 --end 2026-05-31

# 2. Compute features (16 parallel workers, ~20 min on 16-core machine)
uv run intraday features compute --start 2020-09-10 --end 2026-05-31 --workers 16
```

---

## Citation

If you use this dataset in research, please cite:

```bibtex
@dataset{btcusdt_futures_features_2026,
  title        = {BTCUSDT Perpetual Futures 5-Minute Feature Dataset},
  author       = {YOUR_USERNAME},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/YOUR_USERNAME/btcusdt-futures-features},
  note         = {2020-09-10 to 2026-05-31, sourced from data.binance.vision}
}
```

---

## License

MIT — free to use for research and commercial purposes. Data originally sourced from
Binance's public archive ([terms](https://www.binance.com/en/terms)).
"""


# ── Upload helpers ─────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _upload_folder(api: HfApi, src: Path, dest: str, repo_id: str) -> None:
    if not src.exists():
        print(f"  [skip] {src} does not exist")
        return
    n = sum(1 for _ in src.glob("*.parquet"))
    print(f"  {src.relative_to(ROOT)}  →  {dest}/  ({n} files)")
    api.upload_folder(
        folder_path=str(src),
        path_in_repo=dest,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {dest}",
    )
    print(f"  ✓  {dest}/  done")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Hugging Face Dataset Upload — BTCUSDT Futures")
    print("=" * 60)

    username = input("\nHugging Face username: ").strip()
    if not username:
        sys.exit("Username cannot be empty.")

    token = getpass.getpass("Hugging Face token (write access): ").strip()
    if not token:
        sys.exit("Token cannot be empty.")

    default_name = "btcusdt-futures-features"
    repo_name = input(f"Dataset repo name [{default_name}]: ").strip() or default_name
    repo_id = f"{username}/{repo_name}"

    print(f"\nTarget repo: https://huggingface.co/datasets/{repo_id}")
    confirm = input("Proceed? [Y/n]: ").strip().lower()
    if confirm == "n":
        sys.exit("Aborted.")

    api = HfApi(token=token)

    # ── Create repo ───────────────────────────────────────────────────────────
    _section("1 / 7  Creating dataset repo")
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"  ✓  {repo_id}")
    except HfHubHTTPError as e:
        sys.exit(f"Failed to create repo: {e}")

    # ── Dataset card ──────────────────────────────────────────────────────────
    _section("2 / 7  Uploading dataset card (README.md)")
    card = DATASET_CARD.replace("YOUR_USERNAME", username)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print("  ✓  README.md uploaded")

    # ── Features ──────────────────────────────────────────────────────────────
    _section("3 / 7  Uploading features (~120 MB)")
    _upload_folder(
        api,
        src=DATA_DIR / "features" / "BTCUSDT",
        dest="features/BTCUSDT",
        repo_id=repo_id,
    )

    # ── Raw: klines ───────────────────────────────────────────────────────────
    _section("4 / 7  Uploading raw/klines_1m (~132 MB)")
    _upload_folder(api, DATA_DIR / "raw/binance/klines_1m/BTCUSDT", "raw/klines_1m/BTCUSDT", repo_id)

    _section("5 / 7  Uploading raw/klines_5m (~41 MB)")
    _upload_folder(api, DATA_DIR / "raw/binance/klines_5m/BTCUSDT", "raw/klines_5m/BTCUSDT", repo_id)

    # ── Raw: bookDepth + metrics ───────────────────────────────────────────────
    _section("6 / 7  Uploading raw/bookDepth (~158 MB) + raw/metrics (~32 MB)")
    _upload_folder(api, DATA_DIR / "raw/binance/bookDepth/BTCUSDT", "raw/bookDepth/BTCUSDT", repo_id)
    _upload_folder(api, DATA_DIR / "raw/binance/metrics/BTCUSDT",   "raw/metrics/BTCUSDT",   repo_id)

    # ── Raw: aggTrades (large, optional) ─────────────────────────────────────
    _section("7 / 7  raw/aggTrades (~17 GB) — optional")
    agg_src = DATA_DIR / "raw/binance/aggTrades/BTCUSDT"
    if agg_src.exists():
        print("  aggTrades is ~17 GB (tick-level data). This upload may take 30–60 min.")
        ans = input("  Upload aggTrades? [y/N]: ").strip().lower()
        if ans == "y":
            _upload_folder(api, agg_src, "raw/aggTrades/BTCUSDT", repo_id)
        else:
            print("  [skipped]")
    else:
        print("  [skip] aggTrades directory not found")

    # ── Done ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Done!  https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
