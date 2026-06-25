#!/usr/bin/env python3
"""Autonomous LLM-driven trading bot with real-time monitoring.

Runs 24/7. Asks LLM every 5 min for entry decisions, every 1 min for exit decisions.

Usage:
    .venv/Scripts/python.exe scripts/autonomous_llm_monitor.py
        --transformer-run models/transformer/20260623T132957Z
        --mt5-account YOUR_ACCOUNT --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER"
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl
import structlog
from dotenv import load_dotenv

load_dotenv()

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
from intraday.llm.review import LLMReviewAgent, LLMReview
from intraday.trader.mt5_wrapper import MT5TradingWrapper

log = structlog.get_logger(__name__)

# ── Settings ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
LOT_SIZE = 8.0
MAX_SL = 800.0        # Hard stop: never lose more than $800
MAX_TP = 800.0        # Hard target: close at $800 profit
MAX_HOLD_SECONDS = 600  # 10 minutes max
ENTRY_INTERVAL = 300   # 5 minutes between entry checks
MONITOR_INTERVAL = 60  # 1 minute between monitor checks


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
            "volume": bar.volume, "trades": bar.trade_count, "taker_buy_pct": (bar.taker_buy_volume / bar.volume * 100) if bar.volume > 0 else 50.0,
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

    # Regime fallback: ALWAYS use regime when clear
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


def get_llm_entry_decision(candles: list, indicators: dict, pipeline: dict, account: dict) -> LLMReview:
    """Ask LLM: should we enter a trade now?"""
    if not os.getenv("LLM_TOKEN"):
        side = pipeline["decision"]["side"]
        if side == "long":
            return LLMReview("BUY", 0.70, "Pipeline fallback", 0.25, 0.0, 0.0, True)
        elif side == "short":
            return LLMReview("SELL", 0.70, "Pipeline fallback", 0.25, 0.0, 0.0, True)
        return LLMReview("HOLD", 0.0, "No signal", 0.0, 0.0, 0.0, False)

    llm = LLMReviewAgent(api_key=os.getenv("LLM_TOKEN"), timeout=120.0)
    review = llm.analyze_chart(
        candles=candles,
        indicators=indicators,
        positions=[],
        account=account,
        risk_state={},
        recent_logs=[],
        competition_rules={},
    )
    return review


def get_llm_monitor_decision(candles: list, position: dict, account: dict, max_profit: float, elapsed: float) -> str:
    """Ask LLM: should we close the trade or hold? Returns 'CLOSE' or 'HOLD'."""
    if not os.getenv("LLM_TOKEN"):
        # Fallback: close if profit is negative and dropping, or if profit > 400 and dropping
        profit = position.get("profit", 0.0)
        if profit < -MAX_SL:
            return "CLOSE"
        if profit >= MAX_TP:
            return "CLOSE"
        if max_profit > 400 and profit <= max_profit - 200:
            return "CLOSE"
        if elapsed >= MAX_HOLD_SECONDS:
            return "CLOSE"
        return "HOLD"

    side = position.get("side", "unknown")
    profit = position.get("profit", 0.0)
    entry_price = position.get("open_price", 0.0)
    current_price = position.get("current_price", 0.0)

    prompt = f"""You are monitoring an open trade.

TRADE STATUS:
- Side: {side}
- Entry Price: {entry_price:.2f}
- Current Price: {current_price:.2f}
- Current PnL: ${profit:.2f}
- Peak Profit: ${max_profit:.2f}
- Time Open: {elapsed:.0f} seconds

LAST 10 CANDLES:
"""
    for i, c in enumerate(candles[-10:]):
        prompt += f"  {i+1}. O:{c['open']:.2f} H:{c['high']:.2f} L:{c['low']:.2f} C:{c['close']:.2f}\n"

    prompt += f"""
RULES:
- Close at ${MAX_SL:.0f} loss (hard stop)
- Close at ${MAX_TP:.0f} profit (hard target)
- Trailing stop: if profit drops $200 from peak, close
- Max hold: {MAX_HOLD_SECONDS} seconds

CURRENT SITUATION:
- Profit is ${profit:.2f}
- Peak was ${max_profit:.2f}
- Time elapsed: {elapsed:.0f}s

DECISION: Say ONLY one word: "CLOSE" or "HOLD".
"""

    llm = LLMReviewAgent(api_key=os.getenv("LLM_TOKEN"), timeout=60.0)
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("LLM_TOKEN"), base_url=os.getenv("LLM_BASE_URL", "https://api.fireworks.ai/inference/v1"))
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "accounts/fireworks/routers/kimi-k2p6-turbo"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip().upper()
        if "CLOSE" in text:
            return "CLOSE"
        return "HOLD"
    except Exception as exc:
        log.error("llm_monitor_error", error=str(exc))
        return "HOLD"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mt5-account", type=int, required=True)
    parser.add_argument("--mt5-password", type=str, required=True)
    parser.add_argument("--mt5-server", type=str, required=True)
    parser.add_argument("--no-regime-filter", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTONOMOUS LLM MONITOR BOT")
    print("=" * 60)
    print(f"Settings:")
    print(f"  Lot: {LOT_SIZE}")
    print(f"  Max Loss: ${MAX_SL:.0f}")
    print(f"  Max Profit: ${MAX_TP:.0f}")
    print(f"  Entry Check: every {ENTRY_INTERVAL}s")
    print(f"  Monitor Check: every {MONITOR_INTERVAL}s")
    print(f"  Max Hold: {MAX_HOLD_SECONDS}s")
    print("=" * 60)

    # Connect to MT5
    wrapper = MT5TradingWrapper(
        account_id=args.mt5_account,
        password=args.mt5_password,
        server=args.mt5_server,
        magic=999999,
    )
    if not wrapper.connect():
        print("[red]Failed to connect to MT5[/red]")
        sys.exit(1)

    print("[green]MT5 Connected[/green]")
    print("[cyan]Press Ctrl+C to stop[/cyan]\n")

    try:
        while True:
            # Check if any position is open
            positions = wrapper.get_positions(SYMBOL)

            if not positions:
                # ── NO TRADE OPEN: Ask LLM for entry every 5 minutes ──
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No position. Checking entry...")

                # Fetch data
                df, raw_candles = fetch_and_build_features(SYMBOL, bars=200)
                pipeline = run_pipeline(df, args.transformer_run, args.data_dir)
                side = pipeline["decision"]["side"]
                regime = pipeline["regime"]["regime"]

                indicators = {
                    "realized_vol_30m": df.row(len(df)-1, named=True).get("realized_vol_30m", None),
                    "taker_buy_ratio_5m": df.row(len(df)-1, named=True).get("taker_buy_ratio_5m", None),
                }
                account = wrapper.state_snapshot()

                review = get_llm_entry_decision(raw_candles, indicators, pipeline, account)
                print(f"  LLM: {review.action} (conf={review.confidence:.2f})")
                print(f"  Reason: {review.reason}")

                # Regime filter
                if not args.no_regime_filter and review.action in ("BUY", "SELL"):
                    trend_direction = side if side != "flat" else regime
                    if trend_direction in ("long", "bull") and review.action == "SELL":
                        print("  [BLOCKED] Regime filter: bullish trend")
                        review = LLMReview("HOLD", 0.0, "Regime filter", 0.0, 0.0, 0.0, False)
                    elif trend_direction in ("short", "bear") and review.action == "BUY":
                        print("  [BLOCKED] Regime filter: bearish trend")
                        review = LLMReview("HOLD", 0.0, "Regime filter", 0.0, 0.0, 0.0, False)

                if review.action in ("BUY", "SELL") and review.confidence >= 0.65:
                    desired_side = "buy" if review.action == "BUY" else "sell"
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = df.row(len(df)-1, named=True).get("close", 0.0)

                    # Calculate SL/TP
                    if desired_side == "buy":
                        sl = price - (MAX_SL / LOT_SIZE)
                        tp = price + (MAX_TP / LOT_SIZE)
                    else:
                        sl = price + (MAX_SL / LOT_SIZE)
                        tp = price - (MAX_TP / LOT_SIZE)

                    result = wrapper.market_order(SYMBOL, desired_side, LOT_SIZE, 0.0, 0.0)
                    if result.success:
                        print(f"  [green]Trade opened: {review.action} at {price:.2f}[/green]")
                        print(f"  Script SL: {sl:.2f} | TP: {tp:.2f}")
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                else:
                    print(f"  [yellow]HOLD — waiting {ENTRY_INTERVAL}s[/yellow]")
                    time.sleep(ENTRY_INTERVAL)
                    continue

            else:
                # ── TRADE OPEN: Monitor every 1 minute ──
                position = positions[0]
                ticket = position.ticket
                side = position.side
                profit = position.profit
                open_price = position.open_price
                current_price = position.current_price
                volume = position.volume

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring #{ticket} | {side} | PnL=${profit:.2f}")

                # Fetch latest data
                df, raw_candles = fetch_and_build_features(SYMBOL, bars=200)
                account = wrapper.state_snapshot()

                # Track max profit
                max_profit = getattr(main, "_max_profit", 0.0)
                if profit > max_profit:
                    max_profit = profit
                    main._max_profit = max_profit

                # Track elapsed time
                start_time = getattr(main, "_start_time", time.time())
                elapsed = time.time() - start_time

                # Check hard limits first
                if profit >= MAX_TP:
                    print(f"  [green]HARD TP: ${profit:.2f} >= ${MAX_TP:.0f} — closing![/green]")
                    wrapper.close_position(ticket)
                    main._max_profit = 0.0
                    time.sleep(ENTRY_INTERVAL)
                    continue

                if profit < -MAX_SL:
                    print(f"  [red]HARD SL: ${profit:.2f} < -${MAX_SL:.0f} — closing![/red]")
                    wrapper.close_position(ticket)
                    main._max_profit = 0.0
                    time.sleep(ENTRY_INTERVAL)
                    continue

                if max_profit > 400 and profit <= max_profit - 200:
                    print(f"  [yellow]TRAILING STOP: dropped from ${max_profit:.2f} to ${profit:.2f} — closing![/yellow]")
                    wrapper.close_position(ticket)
                    main._max_profit = 0.0
                    time.sleep(ENTRY_INTERVAL)
                    continue

                if elapsed >= MAX_HOLD_SECONDS:
                    print(f"  [yellow]MAX HOLD: {elapsed:.0f}s — closing at ${profit:.2f}![/yellow]")
                    wrapper.close_position(ticket)
                    main._max_profit = 0.0
                    time.sleep(ENTRY_INTERVAL)
                    continue

                # Ask LLM for monitor decision
                decision = get_llm_monitor_decision(raw_candles, {
                    "side": side,
                    "profit": profit,
                    "open_price": open_price,
                    "current_price": current_price,
                    "volume": volume,
                }, account, max_profit, elapsed)

                if decision == "CLOSE":
                    print(f"  [cyan]LLM says CLOSE at ${profit:.2f}[/cyan]")
                    wrapper.close_position(ticket)
                    main._max_profit = 0.0
                    time.sleep(ENTRY_INTERVAL)
                    continue
                else:
                    print(f"  [green]LLM says HOLD — PnL=${profit:.2f} (max=${max_profit:.2f}) | {elapsed:.0f}s[/green]")

                time.sleep(MONITOR_INTERVAL)

    except KeyboardInterrupt:
        print("\n[yellow]Stopping bot...[/yellow]")
        # Close any open position
        positions = wrapper.get_positions(SYMBOL)
        for p in positions:
            wrapper.close_position(p.ticket)
        wrapper.shutdown()
        print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
