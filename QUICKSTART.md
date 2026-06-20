# Phase 1 Quick Start Guide

## Installation

```bash
# Install package in editable mode with phase1 dependencies
pip install -e ".[phase1,dev]"
```

## Usage

### 1. Download Historical Data

**Download 12 months of 5-minute klines:**
```bash
intraday data download \
    --kind klines_5m \
    --start 2024-01-01 \
    --end 2024-12-31
```

**Download 1-minute klines (for Kronos):**
```bash
intraday data download \
    --kind klines_1m \
    --start 2024-01-01 \
    --end 2024-12-31
```

**Download funding rate history:**
```bash
intraday data download \
    --kind funding \
    --start 2024-01-01 \
    --end 2024-12-31
```

**Download open interest history:**
```bash
intraday data download \
    --kind open_interest \
    --start 2024-01-01 \
    --end 2024-12-31
```

### 2. Resume from Checkpoint (Pagination)

The system automatically tracks what's been downloaded:

```bash
# Continue from where you left off
intraday data download --kind klines_5m

# Start from checkpoint + 1 week (604800000 ms)
intraday data download --kind klines_5m --offset 604800000

# Force re-download (ignores checkpoint)
intraday data download --kind klines_5m --start 2024-01-01 --end 2024-01-31 --force
```

### 3. Start Live Data Capture

**Capture trades and depth (recommended for Phase 2):**
```bash
intraday data live-capture --streams trade,depth
```

**Capture all streams:**
```bash
intraday data live-capture --streams trade,depth,mark_price,liquidations
```

**Custom flush interval (default 60s):**
```bash
intraday data live-capture --streams trade,depth --flush-interval 30
```

**Background capture (run in tmux/screen):**
```bash
# In tmux/screen session
intraday data live-capture --streams trade,depth,mark_price

# Detach (Ctrl+B, D in tmux)
# Data will keep capturing in background
```

### 4. Check Progress

**View checkpoint summary:**
```bash
intraday data summary
```

**View detailed checkpoint:**
```bash
intraday data checkpoint
```

## Data Structure

All data is saved as Parquet files:

```
data/
  raw/
    binance/
      klines_5m/BTCUSDT/2024/2024-01.parquet
      klines_1m/BTCUSDT/2024/2024-01.parquet
      funding/BTCUSDT/2024/2024-01.parquet
      open_interest/BTCUSDT/2024/2024-01.parquet
      trades/BTCUSDT/2024-06-18.parquet    # Live capture (daily)
      depth/BTCUSDT/2024-06-18.parquet     # Live capture (daily)
      mark_price/BTCUSDT/2024-06-18.parquet
      liquidations/BTCUSDT/2024-06-18.parquet
  checkpoints/
    data_checkpoint.json                    # Tracks what's been downloaded
```

## Pydantic Schemas

All data is validated with Pydantic schemas:

```python
from intraday.data import Kline, Trade, Depth, FundingRate

# Data is automatically validated on load
import polars as pl

df = pl.read_parquet("data/raw/binance/klines_5m/BTCUSDT/2024/2024-01.parquet")
# All fields match Kline schema
```

## Tips

**Incremental downloads:**
- Run `intraday data download` periodically to stay up-to-date
- Checkpoint system prevents re-downloading existing data
- Use `--offset` to skip forward in time

**Live capture for Phase 2:**
- Start capture NOW and let it run for 4-6 weeks
- You need tick-level data (trades, depth) for microstructure features
- Historical tick data is not freely available from Binance
- Set a calendar reminder to check back in 4-6 weeks

**Data size estimates:**
- 1 year klines_5m: ~50-100 MB
- 1 year klines_1m: ~250-500 MB
- 1 day live trades: ~500 MB - 2 GB (depends on volatility)
- 1 day live depth@100ms: ~1-3 GB

**Rate limiting:**
- Default: 5 requests/second (200ms between requests)
- Binance limits: 1200 req/min (20 req/s)
- Download will auto-retry on 429 (rate limit)

## Next Steps

Once you have ≥4-6 weeks of live tick data:
1. Check MASTER_INDEX.md to verify Phase 1 complete
2. Update Phase 2 status to CURRENT
3. Load `idea/phases/02_features.md`
4. Start building feature engine

## Testing

Run Phase 1 tests:
```bash
pytest tests/phase_01/ -v
```

Run all tests:
```bash
pytest -v
```
