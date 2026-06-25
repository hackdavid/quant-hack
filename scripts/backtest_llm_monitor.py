#!/usr/bin/env python3
"""Backtest the autonomous LLM monitor bot logic on historical data.

Simulates trades without MT5 to verify strategy logic.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl

from intraday.agents.forecast import ForecastAgent
from intraday.agents.orderflow import OrderflowAgent
from intraday.agents.regime import RegimeAgent
from intraday.agents.risk import RiskAgent
from intraday.agents.stay_out import StayOutDetector
from intraday.aggregator.decision import Decision, DecisionEngine
from intraday.aggregator.features import build_aggregator_row
from intraday.aggregator.meta_learner import MetaLearner
from intraday.features.calculator import FeatureCalculator, KlineBar
from intraday.features.schema import FEATURE_ROW_SCHEMA
from intraday.forecast.output import ForecastOutput

# ── Settings ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
LOT_SIZE = 8.0
MAX_SL = 400.0        # Hard stop: never lose more than $400
MAX_TP = 200.0        # Hard target: close at $200 profit
MAX_HOLD_SECONDS = 600  # 10 minutes max
TRAIL_DROP = 100.0    # Close if profit drops $100 from peak
TRAIL_ACTIVATE = 150.0  # Trailing stop activates after $150 profit


def fetch_and_build_features(symbol: str, bars: int = 200) -> tuple[pl.DataFrame, list[dict]]:
    """Fetch 1m + 5m klines and build features."""
    url = "https://data-api.binance.vision/api/v3/klines"
    calc = FeatureCalculator(symbol=symbol, live_mode=True)
    raw_candles = []

    # 1m bars for feature calculator state
    m1_limit = bars * 5 + 60
    r1 = httpx.get(url, params={"symbol": symbol.upper(), "interval": "1m", "limit": m1_limit}, timeout=30.0)
    r1.raise_for_status()
    for row in r1.json():
        bar = KlineBar(
            open_time_ms=int(row[0]), close_time_ms=int(row[6]),
            open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]),
            volume=float(row[5]), trade_count=int(row[8]), taker_buy_volume=float(row[9]),
            interval="1m",
        )
        calc.dispatch(bar)
        raw_candles.append({
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
            "volume": bar.volume, "trades": bar.trade_count,
            "taker_buy_pct": (bar.taker_buy_volume / bar.volume * 100) if bar.volume > 0 else 50.0,
        })

    # 5m bars for feature rows
    rows = []
    r5 = httpx.get(url, params={"symbol": symbol.upper(), "interval": "5m", "limit": bars}, timeout=30.0)
    r5.raise_for_status()
    for row in r5.json():
        bar = KlineBar(
            open_time_ms=int(row[0]), close_time_ms=int(row[6]),
            open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]),
            volume=float(row[5]), trade_count=int(row[8]), taker_buy_volume=float(row[9]),
            interval="5m",
        )
        result = calc.dispatch(bar)
        if result is not None:
            rows.append(result)

    df = pl.DataFrame(
        [r.model_dump() for r in rows],
        schema=FEATURE_ROW_SCHEMA,
    ).fill_null(0)

    return df, raw_candles[-48:]


def run_pipeline(df: pl.DataFrame, run_dir: Path, data_dir: Path) -> dict:
    """Run the full V6 pipeline on the latest bar."""
    forecast_agent = ForecastAgent(run_dir=run_dir, device="cpu")
    orderflow_agent = OrderflowAgent()
    regime_agent = RegimeAgent.load(data_dir / "models" / "regime.pkl")
    risk_agent = RiskAgent()
    stay_out = StayOutDetector()
    meta_learner = MetaLearner.load(data_dir / "models" / "aggregator" / "meta_learner.pkl")
    decision_engine = DecisionEngine(meta_learner=meta_learner, threshold=0.05)

    i = len(df) - 1
    row = df.row(i, named=True)
    ts_ms = int(row["bar_time_ms"])
    feat_window = df.slice(max(0, i - 127), min(128, i + 1))

    forecast_opinion = forecast_agent.predict(feat_window)
    opinions = {
        "orderflow": orderflow_agent.predict(feat_window),
        "regime": regime_agent.predict(feat_window),
        "risk": risk_agent.predict(feat_window),
        "stay_out": stay_out.predict(feat_window),
    }

    fc_payload = forecast_opinion.payload
    prob_up = fc_payload.get("forecast_prob_up", 0.5)
    p_up = max(0.0, min(1.0, prob_up))
    p_down = 1.0 - p_up
    forecast = ForecastOutput(
        ts_ms=ts_ms,
        horizon_minutes=15,
        p_bins=[p_down, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_up],
        p_up_05sigma=p_up,
        p_down_05sigma=p_down,
        expected_move_sigma=(p_up - 0.5) * 2.0,
        confidence=abs(p_up - 0.5) * 2.0,
        meta_act=p_up > 0.5 + 0.02 or p_up < 0.5 - 0.02,
        meta_p_correct=abs(p_up - 0.5) * 2.0,
        model_version="forecast_agent",
        inference_ms=forecast_opinion.inference_ms,
    )

    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    realized_vol = row.get("realized_vol_30m", 0.0) or 0.0
    spread_bps = row.get("spread_bps", 0.0) or 0.0
    funding_rate = row.get("funding_rate", 0.0) or 0.0

    agg_row = build_aggregator_row(
        forecast=forecast,
        opinions=opinions,
        spread_bps=spread_bps,
        realized_vol_30m=realized_vol,
        funding_rate=funding_rate,
        hour_utc=dt.hour,
        minute_of_hour=dt.minute,
        day_of_week=dt.weekday(),
    )

    decision = decision_engine.decide(agg_row, forecast)

    # Regime fallback
    rg = opinions["regime"].payload.get("regime", "unknown")
    of_bias = opinions["orderflow"].payload.get("flow_bias", 0.0)
    if rg == "bull" and of_bias > 0:
        decision = Decision(ts_ms=ts_ms, side="long", confidence=0.70, horizon_minutes=15,
                            reason=f"regime_fallback: {rg} + flow_bias={of_bias}")
    elif rg == "bear" and of_bias < 0:
        decision = Decision(ts_ms=ts_ms, side="short", confidence=0.70, horizon_minutes=15,
                            reason=f"regime_fallback: {rg} + flow_bias={of_bias}")

    return {
        "decision": {"side": decision.side, "confidence": decision.confidence, "reason": decision.reason},
        "regime": {"regime": opinions["regime"].payload.get("regime", "unknown")},
        "orderflow": {"flow_bias": opinions["orderflow"].payload.get("flow_bias", 0.0)},
        "forecast": {"p_up": round(p_up, 4), "confidence": round(forecast.confidence, 4)},
    }


def simulate_trade(entry_price: float, side: str, lot: float, current_price: float) -> float:
    """Simulate P&L for a trade."""
    if side == "buy":
        profit = (current_price - entry_price) * lot
    else:
        profit = (entry_price - current_price) * lot
    return profit


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=["entry", "monitor", "full"], default="full")
    parser.add_argument("--trades", type=int, default=3, help="Number of trades to simulate")
    args = parser.parse_args()

    print("=" * 60)
    print("BACKTEST LLM MONITOR BOT")
    print("=" * 60)

    total_pnl = 0.0
    wins = 0
    losses = 0
    n_trades = 0

    for trade_num in range(args.trades):
        print(f"\n{'='*60}")
        print(f"TRADE #{trade_num + 1}")
        print(f"{'='*60}")

        # 1. Fetch data and get pipeline decision
        df, raw_candles = fetch_and_build_features(SYMBOL, bars=200)
        pipeline = run_pipeline(df, args.transformer_run, args.data_dir)
        side = pipeline["decision"]["side"]
        regime = pipeline["regime"]["regime"]
        conf = pipeline["decision"]["confidence"]

        print(f"Pipeline: {side} (conf={conf:.2f}) | Regime: {regime}")
        print(f"Reason: {pipeline['decision']['reason']}")

        # 2. Regime filter
        regime = pipeline["regime"]["regime"]
        if side == "flat":
            print("HOLD — no trade signal")
            continue

        if regime == "bull" and side == "short":
            print("[BLOCKED] Regime filter: bullish trend, short rejected")
            continue
        elif regime == "bear" and side == "long":
            print("[BLOCKED] Regime filter: bearish trend, long rejected")
            continue

        entry_price = df.row(len(df) - 1, named=True).get("close", 0.0)
        desired_side = "buy" if side == "long" else "sell"
        sl = entry_price - (MAX_SL / LOT_SIZE) if desired_side == "buy" else entry_price + (MAX_SL / LOT_SIZE)
        tp = entry_price + (MAX_TP / LOT_SIZE) if desired_side == "buy" else entry_price - (MAX_TP / LOT_SIZE)

        print(f"Entry: {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

        # 3. Simulate monitoring
        max_profit = 0.0
        start_time = time.time()
        trade_pnl = 0.0
        closed = False
        final_price = entry_price
        exit_reason = ""

        for i in range(10):  # Simulate 10 minutes
            time.sleep(0.5)  # Fast simulation

            # Get next price from recent candles (simulated)
            idx = min(i, len(raw_candles) - 1)
            current_price = raw_candles[idx]["close"]
            final_price = current_price
            elapsed = (i + 1) * 60  # Each iteration = 1 minute

            profit = simulate_trade(entry_price, desired_side, LOT_SIZE, current_price)
            if profit > max_profit:
                max_profit = profit

            print(f"  T+{elapsed}s: Price={current_price:.2f} PnL=${profit:.2f} (max=${max_profit:.2f})")

            # Check hard limits
            if profit >= MAX_TP:
                print(f"  HARD TP: ${profit:.2f} >= ${MAX_TP:.0f}")
                trade_pnl = profit
                closed = True
                exit_reason = "TP"
                break

            if profit < -MAX_SL:
                print(f"  HARD SL: ${profit:.2f} < -${MAX_SL:.0f}")
                trade_pnl = profit
                closed = True
                exit_reason = "SL"
                break

            if max_profit > TRAIL_ACTIVATE and profit <= max_profit - TRAIL_DROP:
                print(f"  TRAILING STOP: dropped from ${max_profit:.2f} to ${profit:.2f}")
                trade_pnl = profit
                closed = True
                exit_reason = "TRAILING"
                break

            if elapsed >= MAX_HOLD_SECONDS:
                print(f"  MAX HOLD: {elapsed}s")
                trade_pnl = profit
                closed = True
                exit_reason = "TIMEOUT"
                break

            # Simulate LLM decision: close if profit drops from peak by $50
            if i >= 2 and profit > 0 and max_profit > 50:
                if profit <= max_profit - 50:
                    print(f"  LLM CLOSE: Profit dropped from ${max_profit:.2f} to ${profit:.2f}")
                    trade_pnl = profit
                    closed = True
                    exit_reason = "LLM"
                    break

        if not closed:
            # Close at final price
            trade_pnl = simulate_trade(entry_price, desired_side, LOT_SIZE, final_price)
            exit_reason = "FINAL"
            print(f"  Final close at ${trade_pnl:.2f}")

        print(f"\nResult: ${trade_pnl:.2f} ({exit_reason})")
        total_pnl += trade_pnl
        n_trades += 1
        if trade_pnl > 0:
            wins += 1
        else:
            losses += 1

    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Trades: {n_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {wins/n_trades*100:.1f}%" if n_trades > 0 else "N/A")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Avg P&L per trade: ${total_pnl/n_trades:.2f}" if n_trades > 0 else "N/A")


if __name__ == "__main__":
    main()
