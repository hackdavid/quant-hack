# CLI Reference

The complete `intraday` command surface, designed up-front so every phase
implements its slice consistently.

Entry point: `uv run intraday <group> <command> [options]`

Implemented incrementally — see each phase's spec for which commands ship
in that phase.

---

## Global flags (apply to every command)

```
--config PATH        # YAML config file (default: ./config/default.yaml)
--seed INT           # RNG seed (default: 1729)
--log-level LEVEL    # debug | info | warning | error (default: info)
--log-file PATH      # extra file sink for logs (default: runs/{RUN_ID}/log.jsonl)
--dry-run            # do not write any state-changing side effects
--quiet              # suppress console logs (file logs unaffected)
```

---

## `intraday data` — data acquisition + capture

### `intraday data download`

Download historical data from public APIs. Idempotent.

```
intraday data download \
    --symbol BTCUSDT \
    --venue binance \
    --kind klines_5m \
    --start 2024-01-01 \
    --end 2024-12-31 \
    [--out-dir data/raw]
```

Supported `--kind`:
- `klines_1m`, `klines_5m`, `klines_15m`, `klines_1h`
- `funding` (Binance perp funding history)
- `open_interest` (Binance perp OI history)
- `agg_trades` (where available)

Behavior: writes `data/raw/{venue}/{kind}/{symbol}/{year}/{year}-{month}.parquet`,
skipping months already present unless `--force`.

### `intraday data live-capture`

Start live WebSocket capture. Long-running. Writes one parquet per UTC date.

```
intraday data live-capture \
    --venues binance,coinbase,deribit \
    --symbols BTCUSDT,BTC-USD,BTC-PERPETUAL \
    [--out-dir data/raw] \
    [--kinds trade,depth_100ms,funding,ticker]
```

Behavior:
- Opens one WS per (venue, stream).
- Buffers in memory, flushes to parquet every 60 seconds.
- Reconnects on disconnect with exponential backoff.
- Records gaps (`runs/_capture/{date}/gaps.jsonl`).
- SIGINT → graceful flush + exit.

### `intraday data summary`

Show what data is available locally, with date ranges and row counts.

```
intraday data summary [--symbol BTCUSDT]
```

### `intraday data verify`

Run schema + integrity checks (no NaNs in critical columns, no duplicate
timestamps, no big gaps).

```
intraday data verify --kind klines_5m --start 2024-01-01 --end 2024-12-31
```

---

## `intraday features` — feature computation

### `intraday features compute`

Compute the canonical feature set for a date range. Output goes to
`data/features/state_5m/...` and `data/features/micro_event/...`.

```
intraday features compute \
    --symbol BTCUSDT \
    --start 2024-01-01 \
    --end 2024-06-30 \
    [--features ofi,microprice,hawkes,vpin,bpv,hurst,...]
```

Default `--features all`. Reads raw, writes features, idempotent.

### `intraday features inspect`

Print summary stats / IC / signal-to-noise for a feature.

```
intraday features inspect \
    --feature ofi \
    --window 5m \
    --target return_15m \
    --start 2024-01-01 --end 2024-06-30
```

Output: distribution stats, autocorrelation, Information Coefficient
against forward returns at multiple horizons.

---

## `intraday backtest` — backtesting

### `intraday backtest run`

Run a backtest. Strategy is a registered name from
`intraday.strategies.registry`.

```
intraday backtest run \
    --strategy v0_buy_hold \
    --symbol BTCUSDT \
    --start 2024-01-01 \
    --end 2024-12-31 \
    [--capital 10000] \
    [--max-leverage 1.0] \
    [--report]
```

Output:
- `runs/{RUN_ID}/config.yaml`
- `runs/{RUN_ID}/decisions.jsonl`
- `runs/{RUN_ID}/trades.jsonl`
- `runs/{RUN_ID}/pnl.parquet`
- `runs/{RUN_ID}/metrics.json`
- `runs/{RUN_ID}/log.jsonl`
- `runs/{RUN_ID}/report.html` (if `--report`)

### `intraday backtest replay`

Replay a previously completed run, optionally with playback speed
controls and live console rendering of decisions.

```
intraday backtest replay \
    --run-id backtest-20250127-103045-a8f3 \
    [--speed 10x] \
    [--render console|html]
```

`--render console` prints a colored, scrolling view of every decision
with the agent opinions, regime, microprice, sizing, and PnL — useful
for **understanding what the system did at every step**.

### `intraday backtest compare`

Compare metrics across multiple runs.

```
intraday backtest compare --runs RUN1,RUN2,RUN3 [--metric sharpe,max_dd,calmar]
```

---

## `intraday train` — model training

### `intraday train forecast`

Train (or fine-tune) the forecast agent.

```
intraday train forecast \
    --symbol BTCUSDT \
    --train-start 2023-01-01 --train-end 2024-09-30 \
    --val-start 2024-10-01 --val-end 2024-12-31 \
    [--mode pretrain|finetune] \
    [--lora-rank 8] \
    [--epochs 5] \
    [--from-version v2]   # for finetune
```

Output: `models/forecast/v{N+1}/` with `.safetensors` + `metadata.json`.

### `intraday train aggregator`

Train the stacked meta-learner (Phase 6).

```
intraday train aggregator \
    --start 2024-01-01 --end 2024-12-31 \
    [--folds 5]
```

### `intraday train policy`

Train (or update) the RL execution policy.

```
intraday train policy \
    --algo cql|iql|sac \
    --mode offline|finetune \
    --data-from 2024-06-01 --data-to 2024-12-31 \
    [--episodes 50000] \
    [--from-version v2]
```

---

## `intraday paper` — paper trading

### `intraday paper run`

Live data → live decisions → simulated fills (queue-aware).

```
intraday paper run \
    --capital 10000 \
    --max-leverage 1.0 \
    [--policy-version latest] \
    [--duration 30d]
```

Behavior:
- Subscribes to all required WS streams.
- Computes features in real time.
- Runs all agents and aggregator.
- Sends decisions to the realistic simulator (Phase 3).
- Captures every tick / depth update to `data/raw/.../live/...` so
  monthly retraining can use it.
- All actions logged with timestamps to JSONL.

### `intraday paper status`

Show current paper run status, open positions, today's PnL.

```
intraday paper status [--run-id RUN_ID]
```

---

## `intraday update` — monthly continual update

### `intraday update run`

Run the monthly update workflow on accumulated paper/live data.

```
intraday update run \
    --since 30d \
    [--components forecast,aggregator,policy] \
    [--auto-promote=false]
```

Steps performed:
1. Snapshot last 30d of data.
2. Recompute features.
3. Drift report (KSWIN/ADWIN).
4. Fine-tune forecast LoRA + recalibrate.
5. Fine-tune RL policy (conservative LR, EWC anchored).
6. Generate canary version.
7. Print summary; do **not** auto-promote unless flag set.

### `intraday canary run`

Run the new policy in shadow mode alongside the live one.

```
intraday canary run --new-version v5 --duration 7d
```

### `intraday canary promote`

Promote canary to live (manual gate).

```
intraday canary promote --new-version v5 --confirm
```

---

## `intraday live` — live trading (gated)

### `intraday live run`

```
intraday live run \
    --capital 100 \
    --max-leverage 1.0 \
    [--policy-version latest] \
    [--armed]                # explicit arming required
```

Without `--armed`, **no real orders are sent**. With `--armed`, the
process must additionally pass an interactive confirmation prompt
showing the config diff vs last live run.

### `intraday live kill`

Engage the kill-switch. Cancels open orders, closes positions, halts.

```
intraday live kill --reason "manual"
```

---

## `intraday inspect` — read-only inspection

### `intraday inspect run`

```
intraday inspect run --run-id RUN_ID
```

Prints: config, summary metrics, regime distribution, top decisions
by PnL contribution, drift events.

### `intraday inspect policy`

```
intraday inspect policy --version v3
```

Prints: training metadata, parent version, calibration plot path,
performance on canary set.

---

## Exit codes

```
0  success
1  user error (bad args / config)
2  data error (missing / corrupt)
3  model error (load / inference)
4  network error (after retries)
5  acceptance failure (e.g. drift detected, kill-switch fired)
```
