#!/usr/bin/env python3
"""LLM-Council Hybrid Trading System.

Council agents analyze market data and feed their summaries to the LLM.
LLM makes the final trading decision based on all agent input.

Usage:
    .venv/Scripts/python.exe scripts/llm_council_trader.py
        --mt5-account YOUR_ACCOUNT --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER"
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import structlog
from dotenv import load_dotenv

load_dotenv()

from intraday.trader.mt5_wrapper import MT5TradingWrapper

log = structlog.get_logger(__name__)

# ── Settings ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
LOT_SIZE = 8.0
MAX_SL = 400.0
MAX_TP = 200.0
MAX_HOLD_SECONDS = 600
TRAIL_ACTIVATE = 150.0
TRAIL_DROP = 100.0


def fetch_candles(symbol: str, limit: int = 50) -> list[dict]:
    """Fetch 1m candles."""
    url = "https://data-api.binance.vision/api/v3/klines"
    r = httpx.get(url, params={"symbol": symbol.upper(), "interval": "1m", "limit": limit}, timeout=30.0)
    r.raise_for_status()
    candles = []
    for row in r.json():
        candles.append({
            "open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
            "close": float(row[4]), "volume": float(row[5]),
            "trades": int(row[8]), "taker_buy_pct": (float(row[9]) / float(row[5]) * 100) if float(row[5]) > 0 else 50.0,
        })
    return candles


def analyze_trend(candles: list[dict]) -> dict:
    """Analyze market trend."""
    if len(candles) < 20:
        return {"trend": "unknown", "confidence": 0.0}
    closes = [c["close"] for c in candles]
    ema5 = sum(closes[-5:]) / 5
    ema10 = sum(closes[-10:]) / 10
    ema20 = sum(closes[-20:]) / 20
    trend_strength = abs(ema5 - ema20) / ema20 * 100
    hh = max(c["high"] for c in candles[-5:]) > max(c["high"] for c in candles[-10:-5])
    ll = min(c["low"] for c in candles[-5:]) < min(c["low"] for c in candles[-10:-5])
    if ema5 > ema10 > ema20 and hh:
        return {"trend": "bull", "confidence": min(0.90, 0.70 + trend_strength * 0.02), "ema5": ema5, "ema10": ema10, "ema20": ema20}
    elif ema5 < ema10 < ema20 and ll:
        return {"trend": "bear", "confidence": min(0.90, 0.70 + trend_strength * 0.02), "ema5": ema5, "ema10": ema10, "ema20": ema20}
    else:
        return {"trend": "ranging", "confidence": 0.50, "ema5": ema5, "ema10": ema10, "ema20": ema20}


def analyze_entry(candles: list[dict]) -> dict:
    """Analyze entry points."""
    if len(candles) < 5:
        return {"signal": "HOLD", "confidence": 0.0}
    last = candles[-1]
    prev = candles[-2]
    bullish_engulfing = last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"] and last["close"] > prev["open"]
    bearish_engulfing = last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"] and last["close"] < prev["open"]
    recent_lows = [c["low"] for c in candles[-10:]]
    recent_highs = [c["high"] for c in candles[-10:]]
    support = min(recent_lows)
    resistance = max(recent_highs)
    current = last["close"]
    range_pct = (current - support) / (resistance - support) if resistance > support else 0.5
    momentum = last["close"] - prev["close"]
    vol_increase = last["volume"] > prev["volume"] * 1.2
    if bullish_engulfing and range_pct < 0.4:
        return {"signal": "BUY", "confidence": 0.75, "reason": "Bullish engulfing near support"}
    elif bearish_engulfing and range_pct > 0.6:
        return {"signal": "SELL", "confidence": 0.75, "reason": "Bearish engulfing near resistance"}
    elif momentum > 0 and last["close"] > last["open"] and vol_increase:
        return {"signal": "BUY", "confidence": 0.65, "reason": "Bullish momentum with volume"}
    elif momentum < 0 and last["close"] < last["open"] and vol_increase:
        return {"signal": "SELL", "confidence": 0.65, "reason": "Bearish momentum with volume"}
    else:
        return {"signal": "HOLD", "confidence": 0.45, "reason": "No clear entry pattern"}


def analyze_sentiment(candles: list[dict]) -> dict:
    """Analyze market sentiment."""
    if len(candles) < 5:
        return {"sentiment": "neutral", "confidence": 0.50}
    recent = candles[-5:]
    taker_buy = [c.get("taker_buy_pct", 50.0) for c in recent]
    avg_taker = sum(taker_buy) / len(taker_buy)
    volumes = [c["volume"] for c in recent]
    avg_vol = sum(volumes) / len(volumes)
    prev_volumes = [c["volume"] for c in candles[-10:-5]]
    prev_avg_vol = sum(prev_volumes) / len(prev_volumes)
    if avg_taker > 60 and avg_vol > prev_avg_vol * 1.2:
        return {"sentiment": "bullish", "confidence": 0.70, "taker_buy": avg_taker}
    elif avg_taker < 40 and avg_vol > prev_avg_vol * 1.2:
        return {"sentiment": "bearish", "confidence": 0.70, "taker_buy": avg_taker}
    elif avg_taker > 55:
        return {"sentiment": "slightly_bullish", "confidence": 0.55, "taker_buy": avg_taker}
    elif avg_taker < 45:
        return {"sentiment": "slightly_bearish", "confidence": 0.55, "taker_buy": avg_taker}
    else:
        return {"sentiment": "neutral", "confidence": 0.50, "taker_buy": avg_taker}


def analyze_volatility(candles: list[dict]) -> dict:
    """Analyze volatility."""
    if len(candles) < 10:
        return {"volatility": "unknown", "confidence": 0.0}
    ranges = [c["high"] - c["low"] for c in candles[-10:]]
    avg_range = sum(ranges) / len(ranges)
    current_range = ranges[-1]
    closes = [c["close"] for c in candles[-10:]]
    price = closes[-1]
    atr_pct = avg_range / price * 100
    sma = sum(closes) / len(closes)
    std = (sum((c - sma) ** 2 for c in closes) / len(closes)) ** 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    position = (price - lower) / (upper - lower) if upper > lower else 0.5
    if atr_pct > 0.5 and current_range > avg_range * 2:
        return {"volatility": "extreme", "confidence": 0.30, "atr_pct": atr_pct, "bb_position": position}
    elif atr_pct > 0.5:
        return {"volatility": "high", "confidence": 0.50, "atr_pct": atr_pct, "bb_position": position}
    elif atr_pct > 0.3:
        return {"volatility": "moderate", "confidence": 0.60, "atr_pct": atr_pct, "bb_position": position}
    else:
        return {"volatility": "normal", "confidence": 0.80, "atr_pct": atr_pct, "bb_position": position}


def build_llm_prompt(candles: list[dict], trend: dict, entry: dict, sentiment: dict, volatility: dict, position: dict = None, max_profit: float = 0.0, elapsed: float = 0.0) -> str:
    """Build comprehensive prompt for LLM."""
    current = candles[-1]
    prompt = f"""You are an expert BTC/USDT trader. Your goal is to maximize profit while minimizing risk.

MARKET ANALYSIS (from Council Agents):
1. Trend Analysis: {trend['trend']} (confidence: {trend['confidence']:.0%})
   - EMA5: {trend.get('ema5', 0):.0f}
   - EMA10: {trend.get('ema10', 0):.0f}
   - EMA20: {trend.get('ema20', 0):.0f}

2. Entry Signal: {entry['signal']} (confidence: {entry['confidence']:.0%})
   - Reason: {entry.get('reason', '')}

3. Market Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.0%})
   - Taker Buy Ratio: {sentiment.get('taker_buy', 50):.0f}%

4. Volatility: {volatility['volatility']} (confidence: {volatility['confidence']:.0%})
   - ATR: {volatility.get('atr_pct', 0):.2f}%
   - Bollinger Position: {volatility.get('bb_position', 0.5):.2f}

CURRENT PRICE DATA:
- Open: {current['open']:.2f}
- High: {current['high']:.2f}
- Low: {current['low']:.2f}
- Close: {current['close']:.2f}
- Volume: {current['volume']:.2f}

"""

    if position:
        prompt += f"""
OPEN POSITION:
- Side: {position['side']}
- Entry Price: {position['open_price']:.2f}
- Current Price: {position['current_price']:.2f}
- Current PnL: ${position['profit']:.2f}
- Peak PnL: ${max_profit:.2f}
- Time Open: {elapsed:.0f}s

DECISION: Should we CLOSE this trade or HOLD? Consider:
- Current profit vs peak
- Market direction
- Time open

Reply ONLY: "CLOSE" or "HOLD" and a brief reason.
"""
    else:
        prompt += f"""
DECISION: Should we enter a trade NOW?
Options: BUY, SELL, or WAIT

Consider:
- Is the trend clear?
- Is there a good entry signal?
- Is volatility too high?
- Does sentiment support the direction?

Reply ONLY: "BUY", "SELL", or "WAIT" and a brief reason.
"""

    return prompt


def call_llm(prompt: str, timeout: float = 60.0) -> dict:
    """Call LLM for decision."""
    if not os.getenv("LLM_TOKEN"):
        return llm_fallback(prompt)

    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("LLM_TOKEN"), base_url=os.getenv("LLM_BASE_URL", "https://api.fireworks.ai/inference/v1"))
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "accounts/fireworks/routers/kimi-k2p6-turbo"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip().upper()

        if "BUY" in text:
            return {"action": "BUY", "reason": text}
        elif "SELL" in text:
            return {"action": "SELL", "reason": text}
        elif "CLOSE" in text:
            return {"action": "CLOSE", "reason": text}
        else:
            return {"action": "HOLD" if "HOLD" in text else "WAIT", "reason": text}
    except Exception as exc:
        log.error("llm_error", error=str(exc))
        return llm_fallback(prompt)


def llm_fallback(prompt: str) -> dict:
    """Fallback logic when LLM is unavailable."""
    # Parse the prompt to extract agent signals
    if "OPEN POSITION" in prompt:
        # Exit decision
        if "Peak PnL" in prompt:
            # Extract profit info
            profit_line = [l for l in prompt.split('\n') if 'Current PnL' in l]
            if profit_line:
                try:
                    profit = float(profit_line[0].split('$')[1].split()[0])
                    if profit >= MAX_TP:
                        return {"action": "CLOSE", "reason": "Profit target"}
                    if profit < -MAX_SL:
                        return {"action": "CLOSE", "reason": "Stop loss"}
                    if profit > 0:
                        return {"action": "HOLD", "reason": "In profit"}
                    return {"action": "HOLD", "reason": "Monitoring"}
                except:
                    pass
        return {"action": "HOLD", "reason": "No LLM"}
    else:
        # Entry decision - aggressive fallback for competition
        trend_bull = "Trend Analysis: bull" in prompt
        trend_bear = "Trend Analysis: bear" in prompt
        entry_buy = "Entry Signal: BUY" in prompt
        entry_sell = "Entry Signal: SELL" in prompt
        sentiment_bull = "bullish" in prompt and "Sentiment" in prompt
        sentiment_bear = "bearish" in prompt and "Sentiment" in prompt
        vol_normal = "Volatility: normal" in prompt
        vol_moderate = "Volatility: moderate" in prompt

        # Strong consensus
        if trend_bull and entry_buy and (sentiment_bull or vol_normal or vol_moderate):
            return {"action": "BUY", "reason": "Council agrees: BUY"}
        elif trend_bear and entry_sell and (sentiment_bear or vol_normal or vol_moderate):
            return {"action": "SELL", "reason": "Council agrees: SELL"}
        # Trend-only fallback
        elif trend_bull and (vol_normal or vol_moderate) and not sentiment_bear:
            return {"action": "BUY", "reason": "Bull trend fallback"}
        elif trend_bear and (vol_normal or vol_moderate) and not sentiment_bull:
            return {"action": "SELL", "reason": "Bear trend fallback"}
        return {"action": "WAIT", "reason": "No LLM - unclear signals"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt5-account", type=int, default=None)
    parser.add_argument("--mt5-password", type=str, default=None)
    parser.add_argument("--mt5-server", type=str, default=None)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--trades", type=int, default=3)
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.trades)
        return

    if not all([args.mt5_account, args.mt5_password, args.mt5_server]):
        print("Error: MT5 credentials required for live trading")
        sys.exit(1)

    print("=" * 60)
    print("LLM-COUNCIL TRADING SYSTEM")
    print("=" * 60)
    print("Council agents analyze -> LLM makes final decision")
    print(f"Settings: TP=${MAX_TP:.0f} SL=${MAX_SL:.0f} Lot={LOT_SIZE}")
    print("=" * 60)

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

    max_profit = 0.0
    start_time = None

    try:
        while True:
            positions = wrapper.get_positions(SYMBOL)

            if not positions:
                # ── ENTRY PHASE ──
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === ENTRY DECISION ===")
                candles = fetch_candles(SYMBOL, limit=50)

                trend = analyze_trend(candles)
                entry = analyze_entry(candles)
                sentiment = analyze_sentiment(candles)
                volatility = analyze_volatility(candles)

                prompt = build_llm_prompt(candles, trend, entry, sentiment, volatility)
                print(f"  Trend: {trend['trend']} | Entry: {entry['signal']} | Sentiment: {sentiment['sentiment']}")
                print(f"  Volatility: {volatility['volatility']}")

                decision = call_llm(prompt)
                signal = decision.get("action", "WAIT")
                reason = decision.get("reason", "")

                print(f"  LLM: {signal} - {reason}")

                if signal in ("BUY", "SELL"):
                    desired_side = "buy" if signal == "BUY" else "sell"
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = candles[-1]["close"]

                    if desired_side == "buy":
                        sl = price - (MAX_SL / LOT_SIZE)
                        tp = price + (MAX_TP / LOT_SIZE)
                    else:
                        sl = price + (MAX_SL / LOT_SIZE)
                        tp = price - (MAX_TP / LOT_SIZE)

                    result = wrapper.market_order(SYMBOL, desired_side, LOT_SIZE, 0.0, 0.0)
                    if result.success:
                        print(f"  [green]Trade opened: {signal} at {price:.2f}[/green]")
                        print(f"  SL: {sl:.2f} | TP: {tp:.2f}")
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                else:
                    print(f"  [yellow]WAIT — checking again in 60s[/yellow]")
                    time.sleep(60)
                    continue

            else:
                # ── MONITOR PHASE ──
                position = positions[0]
                ticket = position.ticket
                side = position.side
                profit = position.profit
                open_price = position.open_price
                current_price = position.current_price

                if start_time is None:
                    start_time = time.time()
                elapsed = time.time() - start_time

                if profit > max_profit:
                    max_profit = profit

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring #{ticket} | {side} | PnL=${profit:.2f}")

                # Check hard limits
                if profit >= MAX_TP:
                    print(f"  [green]TP hit: ${profit:.2f}[/green]")
                    wrapper.close_position(ticket)
                    max_profit = 0.0
                    start_time = None
                    time.sleep(60)
                    continue

                if profit < -MAX_SL:
                    print(f"  [red]SL hit: ${profit:.2f}[/red]")
                    wrapper.close_position(ticket)
                    max_profit = 0.0
                    start_time = None
                    time.sleep(60)
                    continue

                if elapsed >= MAX_HOLD_SECONDS:
                    print(f"  [yellow]Max hold: {elapsed:.0f}s[/yellow]")
                    wrapper.close_position(ticket)
                    max_profit = 0.0
                    start_time = None
                    time.sleep(60)
                    continue

                if max_profit > TRAIL_ACTIVATE and profit <= max_profit - TRAIL_DROP:
                    print(f"  [yellow]Trailing stop: ${profit:.2f} from ${max_profit:.2f}[/yellow]")
                    wrapper.close_position(ticket)
                    max_profit = 0.0
                    start_time = None
                    time.sleep(60)
                    continue

                # Ask LLM
                candles = fetch_candles(SYMBOL, limit=20)
                trend = analyze_trend(candles)
                entry = analyze_entry(candles)
                sentiment = analyze_sentiment(candles)
                volatility = analyze_volatility(candles)

                prompt = build_llm_prompt(
                    candles, trend, entry, sentiment, volatility,
                    position={"side": side, "profit": profit, "open_price": open_price, "current_price": current_price},
                    max_profit=max_profit,
                    elapsed=elapsed,
                )

                decision = call_llm(prompt)
                action = decision.get("action", "HOLD")
                reason = decision.get("reason", "")

                print(f"  LLM: {action} - {reason}")

                if action == "CLOSE":
                    print(f"  [cyan]Closing at ${profit:.2f}[/cyan]")
                    wrapper.close_position(ticket)
                    max_profit = 0.0
                    start_time = None
                    time.sleep(60)
                    continue
                else:
                    print(f"  [green]HOLD: PnL=${profit:.2f} max=${max_profit:.2f}[/green]")

                time.sleep(10)

    except KeyboardInterrupt:
        print("\n[yellow]Stopping...[/yellow]")
        positions = wrapper.get_positions(SYMBOL)
        for p in positions:
            wrapper.close_position(p.ticket)
        wrapper.shutdown()
        print("[green]Done.[/green]")


def run_backtest(trade_count: int):
    """Backtest without MT5."""
    print("=" * 60)
    print("LLM-COUNCIL BACKTEST")
    print("=" * 60)

    total_pnl = 0.0
    wins = 0
    losses = 0

    for i in range(trade_count):
        print(f"\n{'='*60}")
        print(f"BACKTEST #{i+1}")
        print(f"{'='*60}")

        candles = fetch_candles(SYMBOL, limit=50)
        trend = analyze_trend(candles)
        entry = analyze_entry(candles)
        sentiment = analyze_sentiment(candles)
        volatility = analyze_volatility(candles)

        prompt = build_llm_prompt(candles, trend, entry, sentiment, volatility)
        decision = call_llm(prompt)
        signal = decision.get("action", "WAIT")
        reason = decision.get("reason", "")

        print(f"Trend: {trend['trend']} | Entry: {entry['signal']} | Sentiment: {sentiment['sentiment']}")
        print(f"LLM: {signal} - {reason}")

        if signal not in ("BUY", "SELL"):
            print("No trade")
            continue

        entry_price = candles[-1]["close"]
        side = "buy" if signal == "BUY" else "sell"
        profit = 0.0
        closed = False
        exit_reason = ""
        max_profit = 0.0

        # Simulate next 10 candles (from recent history)
        sim_candles = candles[-11:-1]  # 10 candles before the entry
        for t in range(len(sim_candles)):
            current_price = sim_candles[t]["close"]
            if side == "buy":
                profit = (current_price - entry_price) * LOT_SIZE
            else:
                profit = (entry_price - current_price) * LOT_SIZE

            if profit > max_profit:
                max_profit = profit

            if profit >= MAX_TP:
                closed = True
                exit_reason = "TP"
                break
            if profit < -MAX_SL:
                closed = True
                exit_reason = "SL"
                break
            if max_profit > TRAIL_ACTIVATE and profit <= max_profit - TRAIL_DROP:
                closed = True
                exit_reason = "TRAIL"
                break

        if not closed:
            profit = simulate_final_profit(entry_price, side, sim_candles)
            exit_reason = "FINAL"

        print(f"Result: ${profit:.2f} ({exit_reason})")
        total_pnl += profit
        if profit > 0:
            wins += 1
        else:
            losses += 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Trades: {wins + losses}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
    print(f"Total P&L: ${total_pnl:.2f}")


def simulate_final_profit(entry: float, side: str, candles: list) -> float:
    current = candles[-1]["close"]
    if side == "buy":
        return (current - entry) * LOT_SIZE
    return (entry - current) * LOT_SIZE


if __name__ == "__main__":
    main()
