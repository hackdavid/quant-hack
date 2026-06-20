# Phase 1 — Data Layer

**Goal:** historical download (free Binance APIs) + live WebSocket capture
of all streams the system needs, with strict schema validation and gap
tracking.

**Why first:** every subsequent phase reads from the data layer. Schema
mistakes here propagate everywhere. **Get this right before anything else.**

**Estimated effort:** 4–6 days.

**Activates dep group:** `phase1`.

---

## 1. Inputs / outputs

- **Inputs:** Phase 0 skeleton + canonical config.
- **Outputs:**
  - `intraday data download` — fetches historical klines / funding / OI
    for any date range; idempotent.
  - `intraday data live-capture` — long-running WS capture for
    Binance + Coinbase + Deribit; gap-tracked.
  - `intraday data summary` — print local data inventory.
  - `intraday data verify` — schema + integrity check.
  - All data lands at the canonical paths defined in `PLAN.md` §3.

---

## 2. Files to create

```
src/intraday/data/
  __init__.py
  schemas.py            # canonical pydantic + polars schemas
  storage.py            # parquet read/write with schema enforcement
  partitioning.py       # path -> (symbol, kind, year, month, date)
  download/
    __init__.py
    base.py             # abstract Downloader
    binance_klines.py
    binance_funding.py
    binance_oi.py
    binance_aggtrades.py
    coinbase_klines.py  # optional, only if needed for lead-lag history
  capture/
    __init__.py
    base.py             # abstract LiveCapture
    binance_ws.py       # trade + depth_100ms + book_ticker + mark_price
    coinbase_ws.py      # match channel
    deribit_ws.py       # ticker channel for BTC-PERPETUAL
    runner.py           # supervises N captures, handles reconnects
    gap_tracker.py
  cli.py                # data commands wired to typer
tests/phase_01/
  test_schemas.py
  test_partitioning.py
  test_storage_roundtrip.py
  test_binance_klines_download.py
  test_capture_runner.py
  test_gap_tracker.py
  fixtures/
    sample_kline_response.json
    sample_trade_event.json
    sample_depth_event.json
```

---

## 3. Canonical schemas

Define one polars schema per data kind. All readers/writers must use these.

### Klines (`klines_1m`, `klines_5m`, `klines_15m`, `klines_1h`)

| column | dtype | notes |
|---|---|---|
| `ts_open_ms` | `Int64` | UTC ms, primary key |
| `ts_close_ms` | `Int64` | always `ts_open_ms + interval_ms - 1` |
| `open` | `Float64` | |
| `high` | `Float64` | |
| `low` | `Float64` | |
| `close` | `Float64` | |
| `volume_base` | `Float64` | BTC |
| `volume_quote` | `Float64` | USDT |
| `n_trades` | `Int32` | |
| `taker_buy_volume_base` | `Float64` | |
| `taker_buy_volume_quote` | `Float64` | |

### Funding rate

| column | dtype | notes |
|---|---|---|
| `ts_ms` | `Int64` | UTC funding event |
| `funding_rate` | `Float64` | per 8h, decimal (e.g. 0.0001) |
| `mark_price` | `Float64` | |

### Open interest

| column | dtype | notes |
|---|---|---|
| `ts_ms` | `Int64` | sample time |
| `open_interest_base` | `Float64` | BTC |
| `open_interest_quote` | `Float64` | USDT |

### Trades (live)

| column | dtype | notes |
|---|---|---|
| `ts_ms` | `Int64` | event time |
| `ts_local_ns` | `Int64` | local receive monotonic ns |
| `trade_id` | `Int64` | |
| `price` | `Float64` | |
| `size` | `Float64` | |
| `is_buyer_maker` | `Boolean` | (Binance convention) |

### Depth (live, `@depth20@100ms`)

| column | dtype | notes |
|---|---|---|
| `ts_ms` | `Int64` | event time |
| `ts_local_ns` | `Int64` | local recv |
| `bid_prices` | `List[Float64]` | length 20 |
| `bid_sizes` | `List[Float64]` | length 20 |
| `ask_prices` | `List[Float64]` | length 20 |
| `ask_sizes` | `List[Float64]` | length 20 |

### Book ticker (live, `@bookTicker`)

| column | dtype | notes |
|---|---|---|
| `ts_ms` | `Int64` | |
| `ts_local_ns` | `Int64` | |
| `best_bid` | `Float64` | |
| `best_bid_size` | `Float64` | |
| `best_ask` | `Float64` | |
| `best_ask_size` | `Float64` | |

### Mark price (live, `@markPrice@1s`, perp only)

| column | dtype | notes |
|---|---|---|
| `ts_ms` | `Int64` | |
| `mark_price` | `Float64` | |
| `index_price` | `Float64` | |
| `next_funding_ms` | `Int64` | |
| `funding_rate` | `Float64` | most recent |

---

## 4. Function / class signatures

### `src/intraday/data/storage.py`

```python
class ParquetStore:
    """Schema-validated parquet writer/reader with idempotent writes."""

    def __init__(self, root: Path) -> None: ...

    def write_month(
        self,
        df: pl.DataFrame,
        *,
        venue: str,
        symbol: str,
        kind: str,           # 'klines_5m', 'funding', etc.
        year: int,
        month: int,
        force: bool = False,
    ) -> Path:
        """Atomic write to {root}/{venue}/{kind}/{symbol}/{year}/{year}-{month:02d}.parquet."""

    def write_day(
        self,
        df: pl.DataFrame,
        *,
        venue: str,
        symbol: str,
        kind: str,
        date: dt.date,
    ) -> Path: ...

    def read_range(
        self,
        *,
        venue: str,
        symbol: str,
        kind: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pl.LazyFrame:
        """Lazy concatenation of all monthly/daily files overlapping [start, end]."""
```

### `src/intraday/data/download/binance_klines.py`

```python
class BinanceKlinesDownloader(Downloader):
    base_url = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/"

    async def download_month(
        self,
        symbol: str,
        interval: str,        # '1m', '5m', '15m', '1h'
        year: int,
        month: int,
        out_path: Path,
    ) -> int:
        """Download monthly zip from data.binance.vision, validate schema,
        write parquet. Returns row count.
        Idempotent: skips if file exists and rows match expectation.
        """
```

Use `data.binance.vision` for monthly zips (free, no rate limit). For
gaps in the most recent month, fall back to the REST `/api/v3/klines`
endpoint. Both go through the same schema validator.

### `src/intraday/data/capture/binance_ws.py`

```python
class BinanceCapture(LiveCapture):
    """Captures trade + depth20@100ms + bookTicker + markPrice@1s."""

    def __init__(
        self,
        *,
        symbols: list[str],
        out_root: Path,
        flush_interval_s: float = 60.0,
    ) -> None: ...

    async def run(self) -> None:
        """Main loop. SIGINT triggers graceful flush + exit."""

    async def _on_trade(self, msg: dict) -> None: ...
    async def _on_depth(self, msg: dict) -> None: ...
    async def _on_book_ticker(self, msg: dict) -> None: ...
    async def _on_mark_price(self, msg: dict) -> None: ...
```

### `src/intraday/data/capture/runner.py`

```python
class CaptureRunner:
    """Supervises multiple LiveCapture instances with reconnection.

    On any capture's WS disconnect:
      - Log gap event (start, end, venue, stream).
      - Reconnect with exponential backoff (1s -> 60s cap).
      - Resync: fetch missing depth snapshot via REST.
    """

    async def run(self, captures: list[LiveCapture]) -> None: ...
```

### `src/intraday/data/capture/gap_tracker.py`

```python
class GapTracker:
    """Records every WS disconnect / message gap > threshold."""

    def __init__(self, out_path: Path) -> None: ...

    def record(
        self,
        *,
        venue: str,
        stream: str,
        gap_start_ms: int,
        gap_end_ms: int,
        reason: str,
    ) -> None:
        """Append a JSONL gap record to runs/_capture/{date}/gaps.jsonl."""
```

---

## 5. CLI commands wired

| Command | Behavior |
|---|---|
| `intraday data download --kind klines_5m --start ... --end ...` | downloader loop |
| `intraday data download --kind funding --start ... --end ...` | funding history |
| `intraday data download --kind open_interest --start ... --end ...` | OI history (note: Binance OI API is rate-limited and only returns last 30 days; use 5m sampling on a rolling capture for older data) |
| `intraday data live-capture --venues binance,coinbase,deribit ...` | runs `CaptureRunner` |
| `intraday data summary` | walk `data/raw/`, print rows + date ranges per (venue, kind, symbol) |
| `intraday data verify --kind ... --start ... --end ...` | schema check + integrity |

---

## 6. Unit tests

### `test_schemas.py`
- For each schema, `validate()` accepts a sample, rejects bad dtype / missing column / NaN in non-nullable.
- Round-trip: write parquet via store, read back, schema preserved.

### `test_partitioning.py`
- `partition_path("klines_5m", "BTCUSDT", year=2024, month=3)` → expected path.
- Year boundary handling (Dec → Jan).

### `test_storage_roundtrip.py`
- Write → read → equal.
- Idempotency: writing same month twice is a no-op, except `force=True`.
- Concurrent writes to different (year, month) do not corrupt each other.

### `test_binance_klines_download.py` (uses `pytest.mark.network`)
- Download Jan 2024 BTCUSDT 5m, assert row count = 12 × 24 × 31 = 8928.
- Re-run is no-op (idempotent).

### `test_capture_runner.py`
- Mock WS that emits 100 sample events, then closes; runner reconnects and
  records a gap event with correct timestamps.

### `test_gap_tracker.py`
- After 3 record calls, JSONL file has 3 lines with required fields.

---

## 7. Integration / smoke test

`tests/integration/test_phase_01.py`:

```python
@pytest.mark.network
async def test_download_then_summary_then_verify(tmp_path):
    # 1. Download 1 month of BTCUSDT 5m klines.
    # 2. summary command lists it with correct row count.
    # 3. verify command passes.
    # 4. Read range with ParquetStore returns expected rows.
```

```python
async def test_live_capture_short_run(tmp_path):
    # Replace WS with mock fixture that emits 60s of synthetic data.
    # Run capture for 70s, assert daily parquet exists, no gaps recorded.
```

Manual smoke (requires internet):

```bash
uv run intraday data download \
    --venue binance --kind klines_5m \
    --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31

uv run intraday data summary

uv run intraday data verify \
    --kind klines_5m --start 2024-01-01 --end 2024-01-31

uv run intraday data live-capture \
    --venues binance --symbols BTCUSDT --kinds trade,depth_100ms &
# wait 5 minutes, Ctrl-C
ls -la data/raw/binance/trades/BTCUSDT/  # 1 daily parquet
```

---

## 8. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | Schema validators reject 100% of malformed inputs in fuzz tests | `pytest tests/phase_01/test_schemas.py -v` |
| 2 | Round-trip write→read preserves all rows + dtypes for all 7 schemas | `test_storage_roundtrip.py` |
| 3 | Downloading 12 months of klines for BTCUSDT completes < 5 minutes | manual benchmark |
| 4 | Idempotent download: re-run is < 1s and writes no bytes | manual |
| 5 | Live capture survives a forced WS disconnect with gap recorded and zero data loss after reconnect | manual disconnect test |
| 6 | `intraday data verify` flags any month with > 60s gap or > 0 NaNs in OHLCV | seeded corruption test |
| 7 | `data summary` prints inventory in < 1s for ≤ 24 months | benchmark |

---

## 9. Common mistakes to avoid

- **Don't trust Binance timestamps blindly.** Always also stamp `ts_local_ns`
  on receive — needed later for latency analysis.
- **Don't write parquet directly.** Always go through `ParquetStore` so
  schema validation happens.
- **Don't silently drop bad messages.** Log + record gap. A silent drop in
  capture is worse than a crash.
- **Don't store WS payloads as strings.** Parse to typed columns at receive.
- **Don't open one WS per stream when multiplex is supported** — Binance
  combined stream `wss://stream.binance.com:9443/stream?streams=...`
  is much more reliable.
- **Don't use REST polling for trades.** Always WS. REST has lag + rate limits.
- **OneDrive sync warning:** `data/` will become large quickly. Move
  `data/` and `runs/` outside OneDrive (or add to OneDrive ignore list)
  before running serious capture.

---

## 10. Done ⇒ at least 4 weeks of live data captured AND ≥ 12 months of historical klines available locally ⇒ proceed to `phases/02_features.md`.

**Important:** you can start Phase 2 in parallel with continuing Phase 1
live capture; the historical-only features can be developed and tested
without waiting for tick data to accumulate.
