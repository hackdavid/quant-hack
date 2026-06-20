# Phase 3 — Realistic Backtest Simulator

**Goal:** a queue-aware L2 replay simulator that turns "decisions"
(direction + size + execution intent) into realistic fills, accounting
for spread, fees, funding, partial fills, and queue position.

**Why third (and BEFORE any agents):** every later phase reports numbers.
If those numbers come from a naive `next_close * size` simulator they are
**lies**. Build the truth-teller first.

**Estimated effort:** 5–7 days.

**Activates dep group:** none new.

---

## 1. Inputs / outputs

- **Inputs:** Phase 1 raw data (depth, trades, klines, funding) +
  Phase 2 features (for strategies to consume).
- **Outputs:**
  - `intraday backtest run --strategy v0_buy_hold` works end-to-end.
  - `intraday backtest replay --run-id ...` works.
  - PnL, slippage, fee accruals, funding accruals all match independent
    accounting on a known synthetic case to within 0.01 bps.
  - The simulator is the **only** code path through which strategies
    can submit orders — there is no shortcut.

---

## 2. Files to create

```
src/intraday/sim/
  __init__.py
  events.py                   # Event types: BarEvent, TradeEvent, DepthEvent, FundingEvent
  loop.py                     # event-driven simulator main loop
  book.py                     # local L2 order book reconstruction from snapshots + diffs
  matching.py                 # queue-aware fill engine
  costs.py                    # fees + funding accrual
  market_impact.py            # square-root impact for orders that walk the book
  latency.py                  # network/processing latency injection
  account.py                  # position, PnL, equity tracking
  reports.py                  # metrics.json + report.html generator
  strategies/
    __init__.py
    base.py                   # Strategy ABC
    registry.py
    v0_buy_hold.py            # smoke-test strategy
    v1_random.py              # smoke-test strategy with random trades
    # v2_+ live in their own phases
  cli.py
tests/phase_03/
  test_events.py
  test_book.py
  test_matching.py
  test_costs.py
  test_account.py
  test_loop_smoke.py
  test_reports.py
  fixtures/
    one_day_synthetic.parquet  # known OHLCV+L2 path
```

---

## 3. Core abstractions

### Event-driven loop

The simulator consumes a **time-ordered iterator of events** and feeds
them to: (a) the local book, (b) the strategy's `on_event` hook, and
(c) the matching engine for any open orders.

```python
class Event(Protocol):
    ts_ms: int
    ts_local_ns: int
    kind: Literal["bar", "trade", "depth", "funding", "mark"]

class SimulatorLoop:
    def __init__(
        self,
        *,
        events: Iterator[Event],
        book: LocalOrderBook,
        matching: MatchingEngine,
        account: Account,
        strategy: Strategy,
        latency: LatencyModel,
        clock_speedup: float = float("inf"),
    ) -> None: ...

    def run(self) -> RunResult: ...
```

### Strategy interface

```python
class Strategy(ABC):
    @abstractmethod
    def on_event(
        self,
        event: Event,
        ctx: StrategyContext,
    ) -> list[OrderRequest]:
        """Called once per event. Return zero or more orders."""

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None: ...
    def on_cancel(self, order_id: str, ctx: StrategyContext) -> None: ...
```

`StrategyContext` exposes: read-only book view, current position,
unrealized + realized PnL, time, recent feature snapshot, latency budget.

### Order types

```python
class OrderRequest(BaseModel):
    side: Literal["buy", "sell"]
    qty_base: float
    type: Literal["market", "limit", "post_only", "ioc"]
    limit_price: float | None = None
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"
    reduce_only: bool = False
    client_order_id: str
```

### Fill

```python
class Fill(BaseModel):
    ts_ms: int
    order_id: str
    side: Literal["buy", "sell"]
    qty_base: float
    price: float
    is_maker: bool
    fee_quote: float
```

---

## 4. Queue-aware matching engine — the heart of this phase

When a limit order is placed at price `p`:

1. Compute its **queue position** = sum of resting size at `p` ahead
   of it on the same side, taken from the current book snapshot.
2. As subsequent depth diffs and trade events arrive:
   - Trades that hit the same price level **eat the queue ahead** by
     trade size (capped at queue position).
   - Cancellations on the same level that occur ahead in queue are
     not directly observable; approximate with **proportional ageing**:
     reduce queue position by a fraction of any size decrease at the
     level not explained by trades.
3. The order fills (fully or partially) when its queue position reaches 0
   AND a same-side trade occurs of remaining size or larger.

This is the single feature that wipes out 60–80% of the typical retail
backtest illusion.

### Matching for market / IOC orders

Walk the book on the appropriate side, consuming sizes level by level,
applying **square-root impact** for sizes above a threshold:

```
impact_bps(size) = c * sqrt(size / V_daily) * 10_000
```

Calibrate `c` per asset from historical aggressive trade prints
(default `c = 0.4` for BTCUSDT perp).

### Latency

Every `OrderRequest` is delayed by a sampled latency before reaching the
matching engine. Default model: `Lognormal(μ=log(80ms), σ=0.4)`.
Calibrate from your own ping data later.

---

## 5. Costs

```python
class Costs:
    maker_bps: float = 2.0
    taker_bps: float = 5.0

    def apply(self, fill: Fill, account: Account) -> None: ...
```

### Funding accrual

For perp positions: at each funding interval (every 8h on Binance),
apply `funding_rate * notional * sign(position)`. Inject `FundingEvent`
into the event stream from the funding parquet.

```python
def apply_funding(account: Account, funding_event: FundingEvent) -> None:
    notional = account.position_base * funding_event.mark_price
    payment = -account.position_base.sign() * funding_event.funding_rate * notional
    account.realized_pnl_quote += payment
```

---

## 6. Account / PnL

```python
class Account:
    position_base: float = 0.0      # signed (negative = short)
    avg_entry_price: float = 0.0    # cost basis
    cash_quote: float                # USDT
    realized_pnl_quote: float = 0.0
    funding_paid_quote: float = 0.0

    def equity(self, mark_price: float) -> float:
        unrealized = self.position_base * (mark_price - self.avg_entry_price)
        return self.cash_quote + unrealized + self.realized_pnl_quote

    def update_on_fill(self, fill: Fill) -> None: ...
```

---

## 7. Reports

`reports.py` generates `metrics.json`:

```json
{
  "run_id": "...",
  "config_hash": "...",
  "period": {"start": "2024-01-01", "end": "2024-12-31"},
  "n_decisions": 87234,
  "n_orders": 12341,
  "n_fills": 11829,
  "fill_rate": 0.958,
  "gross_pnl_quote": 412.30,
  "net_pnl_quote": 287.15,
  "fees_paid_quote": 89.12,
  "funding_paid_quote": 36.03,
  "max_drawdown_pct": 4.2,
  "sharpe": 1.18,
  "sortino": 1.74,
  "calmar": 0.81,
  "avg_slippage_bps": 1.93,
  "median_latency_ms": 81.0,
  "p99_latency_ms": 178.0,
  "regime_distribution": {"trend_up": 0.31, "trend_down": 0.18, ...},
  "agent_attribution": {"forecast": 0.42, "orderflow": 0.18, ...}
}
```

Optional `--report` emits an HTML page with:
- Equity curve + drawdown
- Daily PnL bars
- Slippage vs decision-confidence scatter
- Per-regime PnL breakdown
- Top 20 winning + losing trades with full context

Use plain HTML + a single small JS chart library (e.g. uPlot embedded as
a string). **No webpack.** **No npm.**

---

## 8. Replay rendering

`intraday backtest replay --render console` uses `rich` to render a
scrolling 5-line panel updated per decision:

```
[2024-01-15 10:32:00 UTC]  microprice=42_512.30  spread=0.6bp  regime=trend_up
forecast: p_up=0.62 p_down=0.21 expected_move_bps=14
flow: bias=+0.27  vpin=0.18  hawkes_imb=+0.41
risk: mult=0.8 allow_trade=True
DECISION: long  size=0.03 BTC  (urgency=low, post-only @ 42_511.50)
```

`--speed Nx` uses `time.sleep` between events to throttle.

---

## 9. CLI commands wired

| Command | Behavior |
|---|---|
| `intraday backtest run --strategy v0_buy_hold ...` | full sim, writes run dir |
| `intraday backtest run --strategy v1_random ...` | smoke-test stochastic strategy |
| `intraday backtest replay --run-id ... --speed 10x --render console` | replay rendering |
| `intraday backtest compare --runs A,B,C` | metrics diff table |

`v0_buy_hold` strategy: enter long at first bar, hold, exit at last bar.
**Used as the canary.** Its result must match `(end_close - start_close)
/ start_close * leverage - fees` to within 1 bp.

---

## 10. Unit tests

### `test_book.py`
- Reconstruct book from snapshot + 100 diffs; final state matches expected.
- Out-of-order diffs raise `SimulatorError`.

### `test_matching.py`
- **Maker fill at front of queue:** order placed when queue=0 fills on
  next same-side trade exactly.
- **Maker fill mid-queue:** queue=10, trades of 3, 4, 6 happen; order
  fills on the third trade with size = 5 (10 - 3 - 4 = 3 ahead;
  6 - 3 = 3 fills the order's 5 short, partial fill).
- **No fill if no trades at level:** queue=0, no trades, no fill.
- **Walk-the-book market order:** buy 1.5 BTC against asks
  `[0.5@100, 0.5@101, 1.0@102]` → average price = (0.5*100 + 0.5*101 + 0.5*102) / 1.5 = 100.83.
- **Square-root impact:** large market order has price worse than
  walk-the-book by impact_bps.

### `test_costs.py`
- Maker fill of 1 BTC at 42500 → fee = 42500 * 0.0002 = 8.5 USDT.
- Taker fill same → fee = 21.25 USDT.
- Funding event with rate 0.0001, position +1 BTC, mark 42500 →
  payment = -4.25 USDT.

### `test_account.py`
- Long 0.5 @ 42000, then long 0.5 @ 43000 → avg = 42500.
- Long 1 @ 42000, then short 0.5 @ 43000 → realized PnL = +500;
  position 0.5 long; avg unchanged at 42000.
- Short 1 @ 43000 → close at 42000 → realized PnL = +1000.

### `test_loop_smoke.py`
- Run v0_buy_hold on 1 day of synthetic data.
- Run v1_random with seed=1 → deterministic fingerprint.

### `test_reports.py`
- Given a synthetic series of fills, metrics.json fields all match
  hand-computed values.

---

## 11. Integration test

`tests/integration/test_phase_03.py`:

```python
def test_buy_hold_matches_naive_calc(tmp_path):
    # Run v0_buy_hold for 1 month on real klines.
    # Compute expected PnL = (close_end - close_start) / close_start
    #                        - 2 * taker_fee_bps / 10_000
    # Assert |sim_net_pnl_pct - expected| < 0.01 bps.
```

```python
def test_random_strategy_seeded_determinism(tmp_path):
    # Run v1_random twice with same seed → identical run_id-stripped output.
```

Manual smoke:

```bash
uv run intraday backtest run \
    --strategy v0_buy_hold \
    --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31 \
    --capital 10000 --report

uv run intraday backtest replay \
    --run-id $(ls runs/ | tail -1) \
    --speed 100x --render console
```

---

## 12. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | v0_buy_hold matches independent calc to ≤ 0.01 bp | integration test |
| 2 | All matching unit tests pass | `pytest tests/phase_03/ -v` |
| 3 | Sim throughput ≥ 100k events/sec on M2-class CPU | benchmark |
| 4 | Replay rendering shows decision panel updating in real time | manual |
| 5 | Funding accrual exactly matches sum of (rate * notional * sign) over period | seeded test |
| 6 | Square-root impact behaves correctly on book-walking benchmark | unit test |
| 7 | report.html opens and displays all sections | manual |
| 8 | Determinism: same seed + config + data → same metrics.json | seeded test |

---

## 13. Common mistakes to avoid

- **Don't fill instantly at mid-price.** Even market orders have spread
  cost. Even limit orders have queue cost.
- **Don't forget to age the queue** when book size at level decreases
  without an explained trade.
- **Don't skip funding.** It's a major cost on perp; for a 5x leveraged
  long held through one funding event in a hot market, funding can
  exceed gross PnL.
- **Don't mix fills and bars.** Fills are events; PnL marks at the bar
  close use the most recent best-bid/best-ask, not the bar close price.
- **Don't allow strategies to read from the future.** The
  `StrategyContext` must be a read-only view as of the **last completed
  event**. Add a unit test that asserts strategies cannot see events
  with `ts_ms` ≥ current.
- **Don't write the report HTML using a templating engine you don't
  vendor.** Inline the chart lib so the file opens with no network.

---

## 14. Done ⇒ proceed to `phases/04_forecast.md`. **Forecast agent will
plug into this simulator from day 1.**
