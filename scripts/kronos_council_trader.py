#!/usr/bin/env python3
"""Kronos-Enhanced Council Trading System.

Uses Kronos (or similar) for candlestick prediction, combined with:
- Multi-agent council analysis
- LLM for final decision making
- Real-time profit monitoring

Architecture:
  1. Kronos Predictor: Predicts next candlestick direction
  2. Technical Analyst: RSI, MACD, Bollinger Bands
  3. Trend Detector: EMA crossovers
  4. Volume Profiler: Order flow analysis
  5. LLM: Final decision maker with all agent input

Usage:
    .venv/Scripts/python.exe scripts/kronos_council_trader.py
        --mt5-account YOUR_ACCOUNT --mt5-password "..." --mt5-server "..."
"""
from __future__ import annotations

import os
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any

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


def fetch_candles(symbol: str, interval: str = "1m", limit: int = 50) -> list[dict]:
    """Fetch candles from Binance."""
    url = "https://data-api.binance.vision/api/v3/klines"
    r = httpx.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=30.0)
    r.raise_for_status()
    candles = []
    for row in r.json():
        candles.append({
            "open_time": int(row[0]),
            "open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
            "close": float(row[4]), "volume": float(row[5]),
            "trades": int(row[8]), "taker_buy_pct": (float(row[9]) / float(row[5]) * 100) if float(row[5]) > 0 else 50.0,
        })
    return candles


# ── Kronos-Inspired Predictor ──────────────────────────────────────────────
class KronosPredictor:
    """Predicts next candlestick direction using OHLCV patterns.

    Simplified Kronos-inspired approach: autoregressive candlestick prediction.
    """
    def __init__(self):
        self.name = "KronosPredictor"
        self.history: list[dict] = []

    def predict(self, candles: list[dict]) -> dict:
        """Predict next candlestick direction and confidence."""
        if len(candles) < 20:
            return {"direction": "unknown", "confidence": 0.0, "target": 0.0}

        # Candlestick pattern analysis (Kronos-like)
        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        # Pattern detection
        patterns = self._detect_patterns(candles[-10:])

        # Momentum analysis
        momentum_5m = (last["close"] - candles[-5]["close"]) / candles[-5]["close"] * 100
        momentum_10m = (last["close"] - candles[-10]["close"]) / candles[-10]["close"] * 100

        # Volume profile
        avg_vol = sum(c["volume"] for c in candles[-10:]) / 10
        vol_trend = last["volume"] / avg_vol if avg_vol > 0 else 1.0

        # Autoregressive prediction (next candle)
        body = last["close"] - last["open"]
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]

        # Predict next candle direction
        bull_score = 0.0
        bear_score = 0.0

        # Pattern score
        if "hammer" in patterns:
            bull_score += 0.30
        if "shooting_star" in patterns:
            bear_score += 0.30
        if "engulfing_bull" in patterns:
            bull_score += 0.40
        if "engulfing_bear" in patterns:
            bear_score += 0.40
        if "three_white_soldiers" in patterns:
            bull_score += 0.50
        if "three_black_crows" in patterns:
            bear_score += 0.50

        # Momentum score
        if momentum_5m > 0.1:
            bull_score += 0.20
        elif momentum_5m < -0.1:
            bear_score += 0.20

        if momentum_10m > 0.2:
            bull_score += 0.15
        elif momentum_10m < -0.2:
            bear_score += 0.15

        # Volume confirmation
        if vol_trend > 1.2:
            if body > 0:
                bull_score += 0.15
            else:
                bear_score += 0.15

        # Wick analysis (rejection signals)
        if lower_wick > abs(body) * 2 and body > 0:
            bull_score += 0.20  # Hammer
        if upper_wick > abs(body) * 2 and body < 0:
            bear_score += 0.20  # Shooting star

        # Determine direction
        total = bull_score + bear_score
        if total < 0.2:
            direction = "neutral"
            confidence = 0.0
        elif bull_score > bear_score:
            direction = "bull"
            confidence = min(0.95, bull_score)
        else:
            direction = "bear"
            confidence = min(0.95, bear_score)

        # Predict target price
        atr = self._calculate_atr(candles[-10:])
        if direction == "bull":
            target = last["close"] + atr * 0.5
        elif direction == "bear":
            target = last["close"] - atr * 0.5
        else:
            target = last["close"]

        return {
            "direction": direction,
            "confidence": confidence,
            "target": target,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "patterns": patterns,
            "momentum_5m": momentum_5m,
            "momentum_10m": momentum_10m,
            "vol_trend": vol_trend,
        }

    def _detect_patterns(self, candles: list[dict]) -> list[str]:
        """Detect candlestick patterns."""
        patterns = []
        if len(candles) < 3:
            return patterns

        c1, c2, c3 = candles[-1], candles[-2], candles[-3]

        # Hammer
        body1 = abs(c1["close"] - c1["open"])
        lower_wick1 = min(c1["close"], c1["open"]) - c1["low"]
        upper_wick1 = c1["high"] - max(c1["close"], c1["open"])
        if lower_wick1 > body1 * 2 and upper_wick1 < body1 * 0.5 and c1["close"] > c1["open"]:
            patterns.append("hammer")

        # Shooting star
        if upper_wick1 > body1 * 2 and lower_wick1 < body1 * 0.5 and c1["close"] < c1["open"]:
            patterns.append("shooting_star")

        # Engulfing
        if c1["close"] > c1["open"] and c2["close"] < c2["open"]:
            if c1["open"] < c2["close"] and c1["close"] > c2["open"]:
                patterns.append("engulfing_bull")
        if c1["close"] < c1["open"] and c2["close"] > c2["open"]:
            if c1["open"] > c2["close"] and c1["close"] < c2["open"]:
                patterns.append("engulfing_bear")

        # Three white soldiers
        if len(candles) >= 3:
            if all(c["close"] > c["open"] for c in candles[-3:]):
                if candles[-1]["close"] > candles[-2]["close"] > candles[-3]["close"]:
                    patterns.append("three_white_soldiers")

        # Three black crows
        if len(candles) >= 3:
            if all(c["close"] < c["open"] for c in candles[-3:]):
                if candles[-1]["close"] < candles[-2]["close"] < candles[-3]["close"]:
                    patterns.append("three_black_crows")

        return patterns

    def _calculate_atr(self, candles: list[dict]) -> float:
        """Calculate Average True Range."""
        ranges = []
        for i in range(1, len(candles)):
            tr = max(
                candles[i]["high"] - candles[i]["low"],
                abs(candles[i]["high"] - candles[i-1]["close"]),
                abs(candles[i]["low"] - candles[i-1]["close"])
            )
            ranges.append(tr)
        return sum(ranges) / len(ranges) if ranges else 0.0


# ── Technical Analyst ─────────────────────────────────────────────────────
class TechnicalAnalyst:
    """Calculates technical indicators."""
    def __init__(self):
        self.name = "TechnicalAnalyst"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 14:
            return {"rsi": 50, "macd": 0, "bb_position": 0.5}

        closes = [c["close"] for c in candles]

        # RSI
        rsi = self._calculate_rsi(closes)

        # MACD
        macd, signal = self._calculate_macd(closes)

        # Bollinger Bands
        bb_position = self._calculate_bb_position(closes)

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "bb_position": bb_position,
            "oversold": rsi < 30,
            "overbought": rsi > 70,
            "macd_bull": macd > signal,
            "bb_low": bb_position < 0.1,
            "bb_high": bb_position > 0.9,
        }

    def _calculate_rsi(self, closes: list[float]) -> float:
        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
        if len(gains) < 14:
            return 50
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, closes: list[float]) -> tuple[float, float]:
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd = ema12 - ema26
        # Simplified signal line (9-period EMA of MACD)
        signal = macd * 0.9  # Approximation
        return macd, signal

    def _ema(self, data: list[float], period: int) -> float:
        if len(data) < period:
            return sum(data) / len(data)
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calculate_bb_position(self, closes: list[float]) -> float:
        if len(closes) < 20:
            return 0.5
        sma = sum(closes[-20:]) / 20
        std = math.sqrt(sum((c - sma) ** 2 for c in closes[-20:]) / 20)
        upper = sma + 2 * std
        lower = sma - 2 * std
        if upper == lower:
            return 0.5
        return (closes[-1] - lower) / (upper - lower)


# ── Trend Detector ───────────────────────────────────────────────────────
class TrendDetector:
    """Detects market trend using EMA crossovers."""
    def __init__(self):
        self.name = "TrendDetector"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 20:
            return {"trend": "unknown", "strength": 0.0}

        closes = [c["close"] for c in candles]

        ema5 = self._ema(closes, 5)
        ema10 = self._ema(closes, 10)
        ema20 = self._ema(closes, 20)

        # Trend strength
        trend_strength = abs(ema5 - ema20) / ema20 * 100

        # Direction
        if ema5 > ema10 > ema20:
            trend = "bull"
            strength = min(1.0, 0.5 + trend_strength * 0.02)
        elif ema5 < ema10 < ema20:
            trend = "bear"
            strength = min(1.0, 0.5 + trend_strength * 0.02)
        else:
            trend = "ranging"
            strength = 0.3

        # Higher highs / lower lows
        highs = [c["high"] for c in candles[-10:]]
        lows = [c["low"] for c in candles[-10:]]
        hh = max(highs[-5:]) > max(highs[:5])
        ll = min(lows[-5:]) < min(lows[:5])

        return {
            "trend": trend,
            "strength": strength,
            "ema5": ema5,
            "ema10": ema10,
            "ema20": ema20,
            "higher_highs": hh,
            "lower_lows": ll,
            "trend_strength": trend_strength,
        }

    def _ema(self, data: list[float], period: int) -> float:
        if len(data) < period:
            return sum(data) / len(data)
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema


# ── Volume Profiler ──────────────────────────────────────────────────────
class VolumeProfiler:
    """Analyzes volume and order flow."""
    def __init__(self):
        self.name = "VolumeProfiler"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 10:
            return {"sentiment": "neutral", "confidence": 0.5}

        recent = candles[-10:]

        # Volume trend
        avg_vol = sum(c["volume"] for c in recent) / len(recent)
        prev_avg = sum(c["volume"] for c in candles[-20:-10]) / 10 if len(candles) >= 20 else avg_vol
        vol_trend = avg_vol / prev_avg if prev_avg > 0 else 1.0

        # Taker buy ratio
        taker_ratios = [c.get("taker_buy_pct", 50.0) for c in recent]
        avg_taker = sum(taker_ratios) / len(taker_ratios)

        # Volume pressure
        buying_vol = sum(c["volume"] for c in recent if c["close"] > c["open"])
        selling_vol = sum(c["volume"] for c in recent if c["close"] < c["open"])
        total_vol = buying_vol + selling_vol

        if total_vol > 0:
            buy_pressure = buying_vol / total_vol
        else:
            buy_pressure = 0.5

        # Sentiment
        if avg_taker > 60 and buy_pressure > 0.6:
            sentiment = "strongly_bullish"
            confidence = 0.80
        elif avg_taker < 40 and buy_pressure < 0.4:
            sentiment = "strongly_bearish"
            confidence = 0.80
        elif avg_taker > 55:
            sentiment = "bullish"
            confidence = 0.60
        elif avg_taker < 45:
            sentiment = "bearish"
            confidence = 0.60
        else:
            sentiment = "neutral"
            confidence = 0.50

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "avg_taker": avg_taker,
            "buy_pressure": buy_pressure,
            "vol_trend": vol_trend,
            "avg_volume": avg_vol,
        }


# ── LLM Decision Maker ─────────────────────────────────────────────────────
class LLMDecisionMaker:
    """Calls LLM with all agent input for final decision."""
    def __init__(self):
        self.name = "LLM"

    def decide_entry(self, kronos: dict, technical: dict, trend: dict, volume: dict, candles: list[dict]) -> dict:
        """Ask LLM for entry decision."""
        current = candles[-1]

        prompt = f"""You are an expert BTC/USDT scalping trader.

AGENT ANALYSIS:
1. Kronos Predictor (Candlestick AI):
   - Direction: {kronos['direction']}
   - Confidence: {kronos['confidence']:.0%}
   - Patterns: {', '.join(kronos['patterns'])}
   - Momentum 5m: {kronos['momentum_5m']:.3f}%
   - Momentum 10m: {kronos['momentum_10m']:.3f}%
   - Volume trend: {kronos['vol_trend']:.2f}x

2. Technical Analyst:
   - RSI: {technical['rsi']:.1f} ({('oversold' if technical['oversold'] else 'overbought' if technical['overbought'] else 'neutral')})
   - MACD: {technical['macd']:.2f} (signal: {technical['macd_signal']:.2f})
   - Bollinger Position: {technical['bb_position']:.2f}
   - MACD Bullish: {technical['macd_bull']}
   - BB Low: {technical['bb_low']}
   - BB High: {technical['bb_high']}

3. Trend Detector:
   - Trend: {trend['trend']}
   - Strength: {trend['strength']:.0%}
   - EMA5: {trend['ema5']:.0f}
   - EMA10: {trend['ema10']:.0f}
   - EMA20: {trend['ema20']:.0f}
   - Higher Highs: {trend['higher_highs']}
   - Lower Lows: {trend['lower_lows']}

4. Volume Profiler:
   - Sentiment: {volume['sentiment']}
   - Confidence: {volume['confidence']:.0%}
   - Taker Buy: {volume['avg_taker']:.0f}%
   - Buy Pressure: {volume['buy_pressure']:.0%}
   - Volume Trend: {volume['vol_trend']:.2f}x

CURRENT PRICE:
- Open: {current['open']:.2f}
- High: {current['high']:.2f}
- Low: {current['low']:.2f}
- Close: {current['close']:.2f}
- Volume: {current['volume']:.2f}

DECISION: Should we BUY, SELL, or WAIT?

Rules:
- ONLY trade if Kronos and Trend agree (bullish → BUY, bearish → SELL)
- WAIT if technical says overbought (RSI > 70) and we want to BUY
- WAIT if technical says oversold (RSI < 30) and we want to SELL
- WAIT if volatility is extreme (Bollinger position > 0.95 or < 0.05)
- WAIT if volume is declining (vol_trend < 0.8)

Reply ONLY: BUY, SELL, or WAIT and a brief reason.
"""
        return call_llm(prompt)

    def decide_exit(self, position: dict, candles: list[dict], max_profit: float, elapsed: float) -> dict:
        """Ask LLM for exit decision."""
        current = candles[-1]
        profit = position["profit"]
        side = position["side"]

        prompt = f"""You are managing an open BTC/USDT trade.

POSITION:
- Side: {side}
- Current PnL: ${profit:.2f}
- Peak PnL: ${max_profit:.2f}
- Time Open: {elapsed:.0f}s

CURRENT PRICE:
- Open: {current['open']:.2f}
- High: {current['high']:.2f}
- Low: {current['low']:.2f}
- Close: {current['close']:.2f}

EXIT RULES:
1. CLOSE if profit >= ${MAX_TP:.0f} (profit target)
2. CLOSE if loss >= ${MAX_SL:.0f} (stop loss)
3. CLOSE if profit dropped ${TRAIL_DROP:.0f} from peak (trailing stop)
4. CLOSE if time open > {MAX_HOLD_SECONDS}s (max hold)
5. CLOSE if you see a strong reversal candle against our position

DECISION: CLOSE or HOLD?

Reply ONLY: CLOSE or HOLD and a brief reason.
"""
        return call_llm(prompt)


def call_llm(prompt: str) -> dict:
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
    """Fallback when LLM is unavailable."""
    if "POSITION" in prompt:
        # Exit decision
        if "Current PnL" in prompt:
            try:
                profit_line = [l for l in prompt.split('\n') if 'Current PnL' in l][0]
                profit = float(profit_line.split('$')[1].split()[0])
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
        # Entry decision
        kronos_bull = "Direction: bull" in prompt
        kronos_bear = "Direction: bear" in prompt
        trend_bull = "Trend: bull" in prompt
        trend_bear = "Trend: bear" in prompt
        overbought = "overbought" in prompt and "RSI" in prompt
        oversold = "oversold" in prompt and "RSI" in prompt
        bb_extreme = "BB High: True" in prompt or "BB Low: True" in prompt

        # Strong consensus - trade even if overbought/oversold (competition mode)
        if kronos_bull and trend_bull and not bb_extreme:
            return {"action": "BUY", "reason": "Kronos + Trend agree bullish"}
        elif kronos_bear and trend_bear and not bb_extreme:
            return {"action": "SELL", "reason": "Kronos + Trend agree bearish"}
        # Moderate consensus
        elif kronos_bull and trend_bull:
            return {"action": "BUY", "reason": "Strong bullish consensus"}
        elif kronos_bear and trend_bear:
            return {"action": "SELL", "reason": "Strong bearish consensus"}
        # Kronos-only (weak)
        elif kronos_bull and not trend_bear:
            return {"action": "BUY", "reason": "Kronos bullish signal"}
        elif kronos_bear and not trend_bull:
            return {"action": "SELL", "reason": "Kronos bearish signal"}
        return {"action": "WAIT", "reason": "No LLM - unclear signals"}


# ── Main Bot ───────────────────────────────────────────────────────────────
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
    print("KRONOS-ENHANCED COUNCIL TRADING SYSTEM")
    print("=" * 60)
    print("Agents: KronosPredictor | TechnicalAnalyst | TrendDetector | VolumeProfiler | LLM")
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

    # Initialize agents
    kronos = KronosPredictor()
    technical = TechnicalAnalyst()
    trend = TrendDetector()
    volume = VolumeProfiler()
    llm = LLMDecisionMaker()

    max_profit = 0.0
    start_time = None

    try:
        while True:
            positions = wrapper.get_positions(SYMBOL)

            if not positions:
                # ── ENTRY PHASE ──
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === ENTRY DECISION ===")
                candles = fetch_candles(SYMBOL, limit=50)

                # Run all agents
                kronos_pred = kronos.predict(candles)
                tech_pred = technical.analyze(candles)
                trend_pred = trend.analyze(candles)
                vol_pred = volume.analyze(candles)

                print(f"  Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f}) patterns={kronos_pred['patterns']}")
                print(f"  Tech: RSI={tech_pred['rsi']:.1f} MACD={tech_pred['macd']:.2f} BB={tech_pred['bb_position']:.2f}")
                print(f"  Trend: {trend_pred['trend']} (strength={trend_pred['strength']:.2f})")
                print(f"  Volume: {vol_pred['sentiment']} (taker={vol_pred['avg_taker']:.0f}%)")

                # LLM decides
                decision = llm.decide_entry(kronos_pred, tech_pred, trend_pred, vol_pred, candles)
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
                decision = llm.decide_exit(
                    {"side": side, "profit": profit, "open_price": open_price, "current_price": current_price},
                    candles,
                    max_profit,
                    elapsed,
                )
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
    print("KRONOS-COUNCIL BACKTEST")
    print("=" * 60)

    kronos = KronosPredictor()
    technical = TechnicalAnalyst()
    trend = TrendDetector()
    volume = VolumeProfiler()
    llm = LLMDecisionMaker()

    total_pnl = 0.0
    wins = 0
    losses = 0

    for i in range(trade_count):
        print(f"\n{'='*60}")
        print(f"BACKTEST #{i+1}")
        print(f"{'='*60}")

        candles = fetch_candles(SYMBOL, limit=50)

        kronos_pred = kronos.predict(candles)
        tech_pred = technical.analyze(candles)
        trend_pred = trend.analyze(candles)
        vol_pred = volume.analyze(candles)

        print(f"Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f})")
        print(f"Tech: RSI={tech_pred['rsi']:.1f} MACD={tech_pred['macd']:.2f}")
        print(f"Trend: {trend_pred['trend']} (strength={trend_pred['strength']:.2f})")
        print(f"Volume: {vol_pred['sentiment']}")

        decision = llm.decide_entry(kronos_pred, tech_pred, trend_pred, vol_pred, candles)
        signal = decision.get("action", "WAIT")
        reason = decision.get("reason", "")

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

        sim_candles = candles[-11:-1]
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
    if not candles:
        return 0.0
    current = candles[-1]["close"]
    if side == "buy":
        return (current - entry) * LOT_SIZE
    return (entry - current) * LOT_SIZE


if __name__ == "__main__":
    main()
