#!/usr/bin/env python3
"""One-shot trading script.

Pulls latest data, runs pipeline + LLM, executes ONE trade with SL/TP, and exits.
No cooldown, no continuous monitoring.

Usage:
    uv run python scripts/trade_once.py \
        --transformer-run models/transformer/20260623T132957Z \
        --mt5-account YOUR_ACCOUNT --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER" \
        --use-llm
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
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
PROFIT_CAP = 800.0      # Close at $800 profit
STOP_LOSS = 800.0       # Close at $800 loss
MAX_HOLD_SECONDS = 600  # 10 minutes max hold time
COOLDOWN_SECONDS = 120  # 2 minutes between trades
MAX_EXPOSURE_PCT = 50.0
MIN_CONFIDENCE = 0.65
MIN_RISK_REWARD = 2.0


def fetch_and_build_features(symbol: str, bars: int = 200) -> tuple[pl.DataFrame, list[dict]]:
    """Fetch 1m + 5m klines and build features via FeatureCalculator."""
    url = "https://data-api.binance.vision/api/v3/klines"
    calc = FeatureCalculator(symbol=symbol, live_mode=True)
    raw_candles = []

    # 1. Load 1m bars first (needed for feature calculator state)
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
        taker_pct = (bar.taker_buy_volume / bar.volume * 100) if bar.volume > 0 else 50.0
        raw_candles.append({
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
            "volume": bar.volume, "trades": bar.trade_count, "taker_buy_pct": taker_pct,
        })

    # 2. Load 5m bars for feature rows
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

    return df, raw_candles[-40:]


def run_pipeline(df: pl.DataFrame, run_dir: Path, data_dir: Path, regime_fallback: bool = True) -> dict:
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
    if regime_fallback:
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


def get_llm_decision(candles: list, indicators: dict, pipeline: dict, account: dict, use_llm: bool) -> LLMReview:
    """Get LLM decision for the trade."""
    if not use_llm or not os.getenv("LLM_TOKEN"):
        # Fallback to pipeline signal
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


def validate_stops(side: str, price: float, sl: float, tp: float) -> tuple[float, float]:
    """Validate and fix SL/TP for MT5. BTCUSD needs 3 decimal places."""
    MIN_STOP_DISTANCE = 0.05
    if side == "buy":
        if sl >= price - MIN_STOP_DISTANCE:
            sl = price - MIN_STOP_DISTANCE
        if tp <= price + MIN_STOP_DISTANCE:
            tp = price + MIN_STOP_DISTANCE
    else:
        if sl <= price + MIN_STOP_DISTANCE:
            sl = price + MIN_STOP_DISTANCE
        if tp >= price - MIN_STOP_DISTANCE:
            tp = price - MIN_STOP_DISTANCE
    return round(sl, 3), round(tp, 3)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mt5-account", type=int, required=True)
    parser.add_argument("--mt5-password", type=str, required=True)
    parser.add_argument("--mt5-server", type=str, required=True)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-regime-fallback", action="store_true")
    parser.add_argument("--no-regime-filter", action="store_true", help="Allow LLM to trade against trend")
    parser.add_argument("--side", choices=["buy", "sell"], default=None, help="Manually set trade direction (buy/sell). Overrides pipeline/LLM decision.")
    args = parser.parse_args()

    print("=" * 60)
    print("ONE-SHOT TRADER")
    print("=" * 60)

    # 1. Fetch data and build features
    print("\n[1] Fetching 5m data and building features...")
    df, raw_candles = fetch_and_build_features(SYMBOL, bars=200)
    print(f"   Feature rows: {len(df)}")
    print(f"   Raw candles: {len(raw_candles)}")

    # 2. Run pipeline
    print("\n[2] Running pipeline...")
    pipeline = run_pipeline(df, args.transformer_run, args.data_dir,
                            regime_fallback=not args.no_regime_fallback)
    side = pipeline["decision"]["side"]
    conf = pipeline["decision"]["confidence"]
    reason = pipeline["decision"]["reason"]
    regime = pipeline["regime"]["regime"]
    print(f"   Pipeline: {side} (conf={conf:.2f}) | Regime: {regime}")
    print(f"   Reason: {reason}")

    # 3. Connect to MT5
    print("\n[3] Connecting to MT5...")
    wrapper = MT5TradingWrapper(
        account_id=args.mt5_account,
        password=args.mt5_password,
        server=args.mt5_server,
        magic=999999,
    )
    if not wrapper.connect():
        print("   FAILED to connect")
        sys.exit(1)
    account = wrapper.state_snapshot()
    balance = account.get("balance", 0.0)
    print(f"   Balance: ${balance:,.2f}")

    # 4. Close any existing positions
    positions = wrapper.get_positions(SYMBOL)
    if positions:
        print(f"   Closing {len(positions)} existing position(s)...")
        for p in positions:
            wrapper.close_position(p.ticket)

    # 5. Get trading decision
    print("\n[4] Trading decision...")
    # Feed last 48 candles (4 hours of 5m data) to LLM for trend analysis
    recent_candles = raw_candles[-48:]
    indicators = {
        "realized_vol_30m": df.row(len(df)-1, named=True).get("realized_vol_30m", None),
        "taker_buy_ratio_5m": df.row(len(df)-1, named=True).get("taker_buy_ratio_5m", None),
    }

    if args.side:
        # Manual override
        action = "BUY" if args.side == "buy" else "SELL"
        review = LLMReview(action, 1.0, f"Manual override: {args.side}", 0.25, 0.0, 0.0, True)
        print(f"   Manual: {review.action} (conf=1.00)")
        print(f"   Reason: {review.reason}")
    elif args.use_llm:
        review = get_llm_decision(recent_candles, indicators, pipeline, account, args.use_llm)
        print(f"   LLM: {review.action} (conf={review.confidence:.2f})")
        print(f"   Reason: {review.reason}")
    else:
        # Pipeline-only mode
        side = pipeline["decision"]["side"]
        if side == "long":
            review = LLMReview("BUY", 0.70, "Pipeline signal", 0.25, 0.0, 0.0, True)
        elif side == "short":
            review = LLMReview("SELL", 0.70, "Pipeline signal", 0.25, 0.0, 0.0, True)
        else:
            review = LLMReview("HOLD", 0.0, "No signal", 0.0, 0.0, 0.0, False)
        print(f"   Pipeline: {review.action} (conf={review.confidence:.2f})")
        print(f"   Reason: {review.reason}")

    # 6. Regime filter: block trades against trend (LLM + Pipeline alignment)
    if not args.no_regime_filter and review.action in ("BUY", "SELL"):
        trend_direction = side if side != "flat" else regime
        if trend_direction in ("long", "bull") and review.action == "SELL":
            print("   [BLOCKED] Regime filter: bullish trend, SELL rejected")
            review = LLMReview("HOLD", 0.0, "Regime filter blocked", 0.0, 0.0, 0.0, False)
        elif trend_direction in ("short", "bear") and review.action == "BUY":
            print("   [BLOCKED] Regime filter: bearish trend, BUY rejected")
            review = LLMReview("HOLD", 0.0, "Regime filter blocked", 0.0, 0.0, 0.0, False)

    # 7. Risk check
    if review.action in ("BUY", "SELL") and review.confidence < MIN_CONFIDENCE:
        print(f"   [BLOCKED] Confidence too low: {review.confidence:.2f} < {MIN_CONFIDENCE}")
        review = LLMReview("HOLD", 0.0, "Low confidence", 0.0, 0.0, 0.0, False)

    # 8. Execute
    if review.action in ("BUY", "SELL"):
        desired_side = "buy" if review.action == "BUY" else "sell"
        candle_close = df.row(len(df)-1, named=True).get("close", 0.0)

        # Get ACTUAL tick price from MT5 (not candle close)
        tick_price = wrapper.get_current_price(SYMBOL, desired_side)
        if tick_price is None or tick_price <= 0:
            tick_price = candle_close
        price = tick_price

        # Get minimum stop distance from MT5
        min_distance = wrapper.get_min_stop_distance(SYMBOL)
        # MT5 wrapper adjusts by min_distance * 2, so we need at least that
        min_offset = min_distance * 2.5

        # Calculate SL/TP based on DOLLAR profit/loss, not LLM price levels
        # For BTCUSD: 1 price unit = $1 per lot = $8 for 8 lot
        sl_dollar = STOP_LOSS      # $100 loss
        tp_dollar = PROFIT_CAP     # $200 profit
        price_offset_sl = max(sl_dollar / LOT_SIZE, min_offset)
        price_offset_tp = max(tp_dollar / LOT_SIZE, min_offset)

        if desired_side == "buy":
            sl = price - price_offset_sl
            tp = price + price_offset_tp
        else:
            sl = price + price_offset_sl
            tp = price - price_offset_tp

        # Validate SL/TP for MT5 minimum distance
        sl, tp = validate_stops(desired_side, price, sl, tp)

        actual_sl_dollar = abs(sl - price) * LOT_SIZE
        actual_tp_dollar = abs(tp - price) * LOT_SIZE

        print(f"\n[5] EXECUTING {review.action}")
        print(f"   Candle close: {candle_close:.2f}")
        print(f"   Tick price: {price:.2f}")
        print(f"   Min distance: {min_distance:.3f} ({min_distance * LOT_SIZE:.0f}$)")
        print(f"   SL: {sl:.3f} (${actual_sl_dollar:.0f} loss)")
        print(f"   TP: {tp:.3f} (${actual_tp_dollar:.0f} profit)")
        print(f"   Lot: {LOT_SIZE}")

        result = wrapper.market_order(SYMBOL, desired_side, LOT_SIZE, 0.0, 0.0)
        print(f"   Result: {result}")
        print(f"   Script SL: {sl:.3f} | Script TP: {tp:.3f}")

        if result.success:
            ticket = result.ticket
            print(f"\n[6] Monitoring position {ticket}...")
            print(f"   Profit cap: ${PROFIT_CAP:.2f}")
            print(f"   Stop loss: ${STOP_LOSS:.2f}")
            print(f"   Press Ctrl+C to stop monitoring")

            max_profit = 0.0
            start_time = time.time()
            try:
                while True:
                    time.sleep(1.0)
                    elapsed = time.time() - start_time
                    positions = wrapper.get_positions(SYMBOL)
                    if not positions:
                        print("   Position closed externally")
                        break

                    p = positions[0]
                    profit = p.profit
                    if profit > max_profit:
                        max_profit = profit

                    # 1. Profit cap: close at $800
                    if profit >= PROFIT_CAP:
                        print(f"\n   PROFIT CAP: ${profit:.2f} >= ${PROFIT_CAP:.2f} — closing!")
                        wrapper.close_position(p.ticket)
                        break

                    # 2. Trailing stop: after $400 profit, close if profit drops $200 from peak
                    if max_profit > 400 and profit <= max_profit - 200:
                        print(f"\n   TRAILING STOP: Profit dropped from ${max_profit:.2f} to ${profit:.2f} — closing!")
                        wrapper.close_position(p.ticket)
                        break

                    # 3. Stop loss: close at $800 loss
                    if profit < -STOP_LOSS:
                        print(f"\n   STOP LOSS: ${profit:.2f} — closing!")
                        wrapper.close_position(p.ticket)
                        break

                    # 4. Max hold time: close after 10 minutes
                    if elapsed >= MAX_HOLD_SECONDS:
                        print(f"\n   MAX HOLD TIME: {elapsed:.0f}s — closing at ${profit:.2f}!")
                        wrapper.close_position(p.ticket)
                        break

                    if profit != 0:
                        print(f"   PnL: ${profit:.2f} (max: ${max_profit:.2f}) | {elapsed:.0f}s", end="\r")
            except KeyboardInterrupt:
                print("\n   Manual stop — closing position...")
                for p in wrapper.get_positions(SYMBOL):
                    wrapper.close_position(p.ticket)
        else:
            print(f"   FAILED: {result}")
    else:
        print("\n[5] HOLD — no trade")

    # 9. Exit
    print("\n[6] Disconnecting...")
    wrapper.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
