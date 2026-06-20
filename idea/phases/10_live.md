# Phase 10 — Live Trading (gated, tiny-size)

**Goal:** transition from paper to **real money** in a controlled,
reversible way. Tiny capital, tiny size, hard kill-switch, one-way
arming, and a clear escalation ladder.

**This is the last phase. Treat it as the most dangerous.**

**Estimated effort:** 3–4 days (one-time build) then continuous
operation.

**Activates dep group:** none new.

**Prerequisite:**
- ≥ 30 days of paper trading completed.
- Phase 8 acceptance #5 met (paper Sharpe within 1σ of backtest).
- ≥ 1 monthly update cycle completed (Phase 9).
- Exchange API keys created with **trade-only**, **no withdrawal**
  permissions.

---

## 1. Inputs / outputs

- **Inputs:**
  - All phases 1–9 working in paper mode.
  - Live exchange API credentials (Binance for spot/perp).
- **Outputs:**
  - `intraday live run` — the only command that places real orders.
  - Hard caps enforced at three layers (config, RiskAgent, broker).
  - Kill-switch at process and exchange level.
  - Same daily capture and reporting as paper.

---

## 2. Files to create

```
src/intraday/live/
  __init__.py
  broker.py                  # exchange client (REST + WS user data)
  reconciliation.py          # match local position state vs exchange truth
  arming.py                  # explicit arming workflow
  kill_switch.py             # process kill + exchange-level cancel-all
  runner.py                  # live runner (parallels PaperRunner)
  ratchet.py                 # capital ratchet — only increase after stable weeks
  cli.py
src/intraday/risk/
  hard_caps.py               # the *third* defense layer (after config + RiskAgent)
tests/phase_10/
  test_broker.py             # against testnet
  test_reconciliation.py
  test_arming.py
  test_kill_switch.py
  test_hard_caps.py
```

---

## 3. The arming workflow — one-way, explicit, audit-logged

Live mode requires THREE positive confirmations:

1. **Config flag:** `live.armed: true` in the config YAML used.
2. **Environment variable:** `INTRADAY_LIVE_CONFIRM=1`.
3. **Interactive prompt** at process start showing:
   - Current strategy version
   - Capital cap, leverage cap, position cap, daily loss cap
   - Diff from last live run config (if any)
   - User must type the exact phrase:
     `I confirm live trading with these caps`
     in stdin to proceed.

If any of three is missing, exit 1 with a clear message.

```python
class Arming:
    def confirm(self, config: LiveConfig) -> None:
        """Raise if not all three confirmations present."""
```

Logs `live.armed` event with the config hash + caps + timestamp.

---

## 4. Three-layer hard caps

Defense in depth. **The same cap is enforced at three independent layers**;
any single layer can stop a bad order.

### Layer 1: config validation

```yaml
live:
  capital_usd: 100
  max_position_usd: 50           # < capital, sanity
  max_single_order_usd: 25
  max_daily_loss_usd: 10
  max_orders_per_minute: 5
```

The config loader rejects nonsensical caps (e.g. position > capital).

### Layer 2: `RiskAgent` (Phase 5)

Reads same caps and refuses decisions that exceed them.

### Layer 3: `intraday.risk.hard_caps`

Inserted **between strategy and broker** as a final filter. Sees every
`OrderRequest` immediately before submission:

```python
def enforce_hard_caps(
    order: OrderRequest,
    *,
    current_position_usd: float,
    today_realized_loss_usd: float,
    orders_in_last_minute: int,
    caps: HardCaps,
) -> OrderRequest:
    """Returns the order unchanged if all caps satisfied; otherwise
    raises HardCapViolation. NEVER silently shrinks the order."""
```

If layer 1 or 2 has a bug and lets a too-large order through, layer 3
catches it. Layer 3's logic is **deliberately simple and auditable**.

---

## 5. Reconciliation

### The problem
Local state can drift from exchange state (network blip, restart, user
manual order via the website, exchange-side fill we missed).

### The solution
Reconciler runs every 30 seconds:
1. Fetch from exchange: open orders, position, balance.
2. Compare to local Account state.
3. If mismatch:
   - Log `reconciliation.mismatch` with full diff.
   - **Engage kill-switch** (cancel all, halt) if mismatch exceeds
     tolerance.
   - Otherwise, sync local to exchange truth and continue.

```python
class Reconciler:
    async def reconcile(self, broker: Broker, account: Account) -> ReconcileResult: ...
```

Tolerance: position diff > 0.001 BTC OR balance diff > 1 USDT triggers
kill-switch. Smaller diffs auto-sync.

---

## 6. Kill-switch

Two levels.

### Process-level
Engaged by:
- Daily loss > cap.
- Reconciliation mismatch beyond tolerance.
- StayOutDetector fires `mode=stay_out` for > 10 minutes consecutively.
- Drift detector RED severity for any feature.
- Manual `intraday live kill --reason "..."` invocation.
- SIGTERM / SIGINT.

When engaged:
1. Send mass-cancel to broker (all symbols).
2. If `flat_on_kill: true`, send IOC to close any open position at market.
3. Set `armed=false`, refuse all new orders.
4. Log `kill_switch.engaged` with reason.

### Exchange-level (defense in depth)
Use exchange's "Listen Key" + cancel-all on disconnect feature where
available. On Binance USDM-Perp, use the REST `/fapi/v1/allOpenOrders`
mass-cancel endpoint. Bound your client to a "kill token" so a runaway
process can be terminated externally.

---

## 7. Capital ratchet — escalation ladder

Don't go from $0 to $1000 overnight. Use a **ratchet** that auto-suggests
capital increases based on weeks-of-stable-Sharpe:

| Capital | Min weeks of stable live Sharpe | Notes |
|---|---|---|
| $100 | 0 (initial) | smallest meaningful |
| $300 | 2 weeks ≥ 0.7 | first ratchet |
| $1000 | 4 weeks ≥ 0.9 | meaningful |
| $3000 | 8 weeks ≥ 1.0 | with at least one regime shift survived |
| $10000+ | Manual review only | requires explicit human gate |

The ratchet is **suggestion only**, surfaced in the daily report.
Promotion is always manual.

---

## 8. Live runner (parallel to PaperRunner)

```python
class LiveRunner(PaperRunner):
    """Same skeleton as paper, but:
       - Strategy emits orders that go through HardCaps → Broker (real).
       - Reconciler runs every 30s.
       - User-data WS for fill events (instant truth, no latency).
       - Kill-switch wired at every layer.
    """
```

Key difference from paper: the simulator **still runs in parallel** and
its decisions are logged for comparison. Live fills are the truth, but
sim provides counterfactual ("what would the queue-aware sim have
expected?") for slippage analysis.

---

## 9. CLI

| Command | Behavior |
|---|---|
| `intraday live run --capital 100 --max-leverage 1.0 --policy-version latest --armed` | start, after triple confirmation |
| `intraday live status` | open orders, position, today's PnL |
| `intraday live kill --reason "..."` | engage process kill-switch |
| `intraday live reconcile` | one-shot reconciliation report |
| `intraday live ratchet-check` | suggest next capital tier |
| `intraday live disarm` | explicit disarm without killing process |

The `live` group is hidden from `--help` unless `INTRADAY_LIVE=1` is
set. This is to prevent accidental `intraday live run` typos when
someone is exploring the CLI for the first time.

---

## 10. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | Triple-arming refuses to start without all three confirmations | unit + manual test |
| 2 | Hard cap layer 3 rejects orders > cap with `HardCapViolation` (not silent clamp) | unit test |
| 3 | Kill-switch cancels all open orders within 5 seconds of trigger | testnet test |
| 4 | Reconciler detects engineered mismatch and engages kill | testnet test |
| 5 | First $100 live week realized Sharpe ≥ 0.5 × paper Sharpe | end-of-week review |
| 6 | No silent fallbacks anywhere on order path (audit) | code review |
| 7 | Live data captured into the same canonical paths as paper (so monthly update sees it) | inspection |
| 8 | After SIGTERM, exit cleanly with no orphan orders on exchange | shutdown drill |

---

## 11. Common mistakes to avoid

- **Don't disable the kill-switch "temporarily".** This is how blow-ups
  happen.
- **Don't trade the same model in paper and live simultaneously.** They
  consume the same data; their decisions overlap; confusing PnL
  attribution.
- **Don't silently shrink orders that exceed caps.** Reject them
  loudly. A silenced cap is an absent cap.
- **Don't run live without reconciliation.** Even one missed fill
  drifts state forever.
- **Don't trust local time for anything financial.** UTC. Always.
  Exchange clocks may also drift; trust their event timestamps over
  local ones for ordering, but local for monitoring latency.
- **Don't store API keys in YAML.** Use environment variables and
  `.env` files (which `.gitignore` already excludes).
- **Don't trade in low-liquidity hours your system wasn't trained on.**
  If you trained on UTC 13:00–22:00 (US session), the model is
  systematically miscalibrated for Asian session.
- **Don't ratchet capital after one good week.** Variance is huge; one
  week proves nothing. The ladder above is conservative on purpose.
- **Don't combine the first live deployment with new code.** Run live
  with **exactly** the model + code that passed paper acceptance.
  Bug fixes can wait one cycle.

---

## 12. Final state — what success looks like

After Phase 10 with one full year of operation:

- Daily report comes in every morning.
- Monthly update + canary + promote happens like clockwork.
- Drift events get investigated, not ignored.
- Capital has ratcheted (or not) per the ladder.
- The system has survived ≥ 2 regime shifts without manual intervention
  (other than monthly updates).
- You can hand the system to a knowledgeable stranger and the docs +
  logs let them understand any decision in < 10 minutes.

That's the bar. Ship it.
