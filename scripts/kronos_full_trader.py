#!/usr/bin/env python3
"""Kronos Full Trading System - Real Model + Multi-Agent + LLM.

Uses actual Kronos model from HuggingFace for candlestick prediction.

Agents:
  1. KronosPredictor - Real Kronos model
  2. TechnicalAnalyst - RSI, MACD, Bollinger
  3. TrendDetector - EMA crossovers
  4. VolumeProfiler - Order flow
  5. LLMDecisionMaker - Final decision

Usage:
    .venv/Scripts/python.exe scripts/kronos_full_trader.py --backtest --trades 5
    .venv/Scripts/python.exe scripts/kronos_full_trader.py --mt5-account YOUR_ACCOUNT --mt5-password "..." --mt5-server "..."
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
import pandas as pd
import structlog
import torch
from dotenv import load_dotenv

load_dotenv()

# Add Kronos module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "kronos_module"))
from model import Kronos, KronosTokenizer, KronosPredictor

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


def fetch_5m_candles(symbol: str, limit: int = 400) -> pd.DataFrame:
    """Fetch 5m candles from Binance."""
    url = "https://data-api.binance.vision/api/v3/klines"
    r = httpx.get(url, params={"symbol": symbol.upper(), "interval": "5m", "limit": limit}, timeout=30.0)
    r.raise_for_status()

    data = []
    for row in r.json():
        data.append({
            "timestamps": pd.to_datetime(int(row[0]), unit="ms"),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "amount": float(row[7]),
        })

    return pd.DataFrame(data)


# ── Kronos Real Predictor ─────────────────────────────────────────────────
class KronosRealPredictor:
    """Real Kronos model for candlestick prediction."""
    def __init__(self):
        self.name = "KronosPredictor"
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self._initialized = False

    def _init_model(self):
        if self._initialized:
            return

        try:
            print("  Loading Kronos tokenizer...")
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            print("  Loading Kronos model...")
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512)
            self._initialized = True
            print(f"  Kronos loaded on {device}")
        except Exception as e:
            print(f"  [red]Kronos failed: {e}[/red]")
            raise

    def predict(self, candles: list[dict]) -> dict:
        """Predict next candlestick direction."""
        self._init_model()

        try:
            return self._model_predict(candles)
        except Exception as e:
            log.error("kronos_predict_error", error=str(e))
            return {"direction": "unknown", "confidence": 0.0, "error": str(e)}

    def _model_predict(self, candles: list[dict]) -> dict:
        """Use actual Kronos model."""
        df = pd.DataFrame(candles)
        df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')

        x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df['timestamps']

        last_time = df['timestamps'].iloc[-1]
        freq = pd.Timedelta(minutes=5)
        y_timestamp = pd.Series(pd.date_range(start=last_time + freq, periods=12, freq=freq))

        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=12,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        last_close = df['close'].iloc[-1]
        predicted_close = pred_df['close'].iloc[-1]
        predicted_high = pred_df['high'].max()
        predicted_low = pred_df['low'].min()

        change_pct = (predicted_close - last_close) / last_close * 100
        max_up = (predicted_high - last_close) / last_close * 100
        max_down = (predicted_low - last_close) / last_close * 100

        if change_pct > 0.1:
            direction = "bull"
            confidence = min(0.95, 0.5 + abs(change_pct) * 2)
        elif change_pct < -0.1:
            direction = "bear"
            confidence = min(0.95, 0.5 + abs(change_pct) * 2)
        else:
            direction = "neutral"
            confidence = 0.0

        return {
            "direction": direction,
            "confidence": confidence,
            "predicted_close": predicted_close,
            "last_close": last_close,
            "change_pct": change_pct,
            "max_up": max_up,
            "max_down": max_down,
            "method": "kronos_real",
        }


# ── Technical Analyst ─────────────────────────────────────────────────────
class TechnicalAnalyst:
    def __init__(self):
        self.name = "TechnicalAnalyst"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 14:
            return {"rsi": 50, "macd": 0, "bb_position": 0.5}

        closes = [c["close"] for c in candles]
        rsi = self._calculate_rsi(closes)
        macd, signal = self._calculate_macd(closes)
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
        signal = macd * 0.9
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
    def __init__(self):
        self.name = "TrendDetector"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 20:
            return {"trend": "unknown", "strength": 0.0}

        closes = [c["close"] for c in candles]
        ema5 = self._ema(closes, 5)
        ema10 = self._ema(closes, 10)
        ema20 = self._ema(closes, 20)

        trend_strength = abs(ema5 - ema20) / ema20 * 100

        if ema5 > ema10 > ema20:
            trend = "bull"
            strength = min(1.0, 0.5 + trend_strength * 0.02)
        elif ema5 < ema10 < ema20:
            trend = "bear"
            strength = min(1.0, 0.5 + trend_strength * 0.02)
        else:
            trend = "ranging"
            strength = 0.3

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
    def __init__(self):
        self.name = "VolumeProfiler"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 10:
            return {"sentiment": "neutral", "confidence": 0.5}

        recent = candles[-10:]
        avg_vol = sum(c["volume"] for c in recent) / len(recent)
        prev_avg = sum(c["volume"] for c in candles[-20:-10]) / 10 if len(candles) >= 20 else avg_vol
        vol_trend = avg_vol / prev_avg if prev_avg > 0 else 1.0

        taker_ratios = [c.get("taker_buy_pct", 50.0) for c in recent]
        avg_taker = sum(taker_ratios) / len(taker_ratios)

        buying_vol = sum(c["volume"] for c in recent if c["close"] > c["open"])
        selling_vol = sum(c["volume"] for c in recent if c["close"] < c["open"])
        total_vol = buying_vol + selling_vol
        buy_pressure = buying_vol / total_vol if total_vol > 0 else 0.5

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
        }


# ── LLM Decision Maker ───────────────────────────────────────────────────
class LLMDecisionMaker:
    def __init__(self):
        self.name = "LLM"

    def decide_entry(self, kronos: dict, technical: dict, trend: dict, volume: dict, candles: list[dict]) -> dict:
        current = candles[-1]

        prompt = f"""BTC/USDT trading decision. Reply with ONLY one word: BUY, SELL, or WAIT.

Kronos AI: {kronos['direction']} ({kronos['confidence']:.0%}) predicted_change={kronos.get('change_pct', 0):.2f}%
Technical: RSI={technical['rsi']:.0f} MACD={technical['macd']:.0f} BB={technical['bb_position']:.2f}
Trend: {trend['trend']} ({trend['strength']:.0%})
Volume: {volume['sentiment']}
Price: {current['close']:.0f}

DECISION: BUY, SELL, or WAIT?"""

        return call_llm(prompt)

    def decide_exit(self, position: dict, candles: list[dict], max_profit: float, elapsed: float) -> dict:
        current = candles[-1]
        profit = position["profit"]
        side = position["side"]

        prompt = f"""BTC/USDT trade management. Reply with ONLY one word: CLOSE or HOLD.

Position: {side}
PnL: ${profit:.0f}
Peak: ${max_profit:.0f}
Time: {elapsed:.0f}s
Price: {current['close']:.0f}

DECISION: CLOSE or HOLD?"""

        return call_llm(prompt)


def call_llm(prompt: str) -> dict:
    if not os.getenv("LLM_TOKEN"):
        return llm_fallback(prompt)

    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("LLM_TOKEN"), base_url="https://api.fireworks.ai/inference/v1")
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "accounts/fireworks/routers/kimi-k2p6-turbo"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip().upper()

        words = text.split()
        for word in words:
            clean = word.strip(".,:;!?()")
            if clean in ("BUY", "SELL", "CLOSE", "HOLD", "WAIT"):
                return {"action": clean, "reason": text[:200]}

        if "BUY" in text:
            return {"action": "BUY", "reason": text[:200]}
        elif "SELL" in text:
            return {"action": "SELL", "reason": text[:200]}
        elif "CLOSE" in text:
            return {"action": "CLOSE", "reason": text[:200]}
        elif "HOLD" in text:
            return {"action": "HOLD", "reason": text[:200]}
        else:
            return {"action": "WAIT", "reason": text[:200]}
    except Exception as exc:
        log.error("llm_error", error=str(exc))
        return llm_fallback(prompt)


def llm_fallback(prompt: str) -> dict:
    if "POSITION" in prompt:
        try:
            profit = float([l for l in prompt.split('\n') if 'PnL' in l][0].split('$')[1].split()[0])
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
        kronos_bull = "Kronos AI: bull" in prompt
        kronos_bear = "Kronos AI: bear" in prompt
        trend_bull = "Trend: bull" in prompt
        trend_bear = "Trend: bear" in prompt

        if kronos_bull and trend_bull:
            return {"action": "BUY", "reason": "Kronos + Trend agree bullish"}
        elif kronos_bear and trend_bear:
            return {"action": "SELL", "reason": "Kronos + Trend agree bearish"}
        elif kronos_bull and not trend_bear:
            return {"action": "BUY", "reason": "Kronos bullish"}
        elif kronos_bear and not trend_bull:
            return {"action": "SELL", "reason": "Kronos bearish"}
        return {"action": "WAIT", "reason": "No LLM - unclear"}


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
    print("KRONOS FULL TRADING SYSTEM")
    print("=" * 60)
    print("Agents: KronosRealPredictor | TechnicalAnalyst | TrendDetector | VolumeProfiler | LLM")
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

    kronos = KronosRealPredictor()
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
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === ENTRY DECISION ===")
                df = fetch_5m_candles(SYMBOL, limit=400)
                candles = df.to_dict('records')
                for i, c in enumerate(candles):
                    c['open_time'] = int(df['timestamps'].iloc[i].timestamp() * 1000)

                kronos_pred = kronos.predict(candles)
                tech_pred = technical.analyze(candles)
                trend_pred = trend.analyze(candles)
                vol_pred = volume.analyze(candles)

                print(f"  Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f}) change={kronos_pred.get('change_pct', 0):.2f}%")
                print(f"  Tech: RSI={tech_pred['rsi']:.1f} MACD={tech_pred['macd']:.2f}")
                print(f"  Trend: {trend_pred['trend']} (strength={trend_pred['strength']:.2f})")
                print(f"  Volume: {vol_pred['sentiment']}")

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
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                else:
                    print(f"  [yellow]WAIT — checking again in 60s[/yellow]")
                    time.sleep(60)
                    continue

            else:
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

                df = fetch_5m_candles(SYMBOL, limit=20)
                candles = df.to_dict('records')
                for i, c in enumerate(candles):
                    c['open_time'] = int(df['timestamps'].iloc[i].timestamp() * 1000)

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
    print("=" * 60)
    print("KRONOS FULL BACKTEST")
    print("=" * 60)

    kronos = KronosRealPredictor()
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

        df = fetch_5m_candles(SYMBOL, limit=400)
        candles = df.to_dict('records')
        for j, c in enumerate(candles):
            c['open_time'] = int(df['timestamps'].iloc[j].timestamp() * 1000)

        kronos_pred = kronos.predict(candles)
        tech_pred = technical.analyze(candles)
        trend_pred = trend.analyze(candles)
        vol_pred = volume.analyze(candles)

        print(f"Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f}) change={kronos_pred.get('change_pct', 0):.2f}%")
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

        # Simulate next 12 candles
        sim_candles = candles[-13:-1]
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
