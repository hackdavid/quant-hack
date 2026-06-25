#!/usr/bin/env python3
"""Kronos LLM Master Trader - Complete Multi-Agent System.

Architecture:
  1. KronosPredictor - Real Kronos model for candlestick prediction
  2. TechnicalAnalyst - RSI, MACD, Bollinger
  3. TrendDetector - EMA crossovers
  4. VolumeProfiler - Order flow analysis
  5. ConflictAnalyzer - Detects Kronos vs Trend conflicts
  6. LLMDecisionMaker - Final judge with ALL agent input

Strategy: Feed Kronos + Trend + conflict + all agents to LLM for decision.

Usage:
    .venv/Scripts/python.exe scripts/kronos_llm_master_trader.py --backtest --trades 5
    .venv/Scripts/python.exe scripts/kronos_llm_master_trader.py --mt5-account YOUR_ACCOUNT --mt5-password "..." --mt5-server "..."
"""
from __future__ import annotations

import json
import os
import sys
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import structlog
import torch
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "kronos_module"))
from model import Kronos, KronosTokenizer, KronosPredictor

from intraday.trader.mt5_wrapper import MT5TradingWrapper

# Import competition score calculator for live metrics
sys.path.insert(0, str(Path(__file__).parent))
from mt5_competition_score import fetch_deals, calculate_metrics as calc_competition_metrics

log = structlog.get_logger(__name__)

# ── Settings ───────────────────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
LOT_SIZE = 8.0
MAX_SL = 400.0
MAX_TP = 200.0
MAX_HOLD_SECONDS = 900
TRAIL_ACTIVATE = 150.0
TRAIL_DROP = 100.0

# ── Dynamic Position Sizing ────────────────────────────────────────────────
LOTS_WEAK = 4.0
LOTS_MEDIUM = 8.0
LOTS_STRONG = 12.0
LOTS_VERY_STRONG = 16.0
LOTS_MAX = 20.0

TP_WEAK = 150.0
TP_MEDIUM = 400.0
TP_STRONG = 800.0
TP_VERY_STRONG = 2000.0
TP_MAX = 3000.0

SL_WEAK = 200.0
SL_MEDIUM = 400.0
SL_STRONG = 400.0
SL_VERY_STRONG = 800.0
SL_MAX = 800.0

HOLD_WEAK = 300
HOLD_MEDIUM = 600
HOLD_STRONG = 900
HOLD_VERY_STRONG = 900
HOLD_MAX = 900

MONITOR_INTERVAL = 30


def calculate_signal_score(agents: dict) -> int:
    """Score signal strength 0-100 based on all agents."""
    score = 0
    kronos = agents["kronos"]
    technical = agents["technical"]
    trend = agents["trend"]
    volume = agents["volume"]
    conflict = agents["conflict"]

    # Kronos confidence (0-30) — disabled, always 0
    score += int(kronos.get("confidence", 0) * 30)

    # Trend strength (0-30) — increased weight since Kronos is disabled
    score += int(trend.get("strength", 0) * 30)

    # Trend direction alignment with technical (0-25)
    trend_dir = trend.get("trend", "unknown")
    if technical.get("supertrend_bull") and trend_dir == "bull":
        score += 25
    elif technical.get("supertrend_bear") and trend_dir == "bear":
        score += 25
    elif trend_dir in ("bull", "bear"):
        score += 10

    # Technical confirmation (0-15)
    if technical.get("strong_trend"):
        score += 10
    if technical.get("adx", 0) > 25:
        score += 5

    # Volume confirmation (0-10)
    if volume.get("sentiment") in ("bullish", "bearish"):
        score += 10
    elif volume.get("sentiment") in ("mildly_bullish", "mildly_bearish"):
        score += 5

    return min(100, max(0, score))


def get_position_settings(score: int) -> dict:
    """Return position size, TP, SL, hold time. NO SL — manual exit by user."""
    # Gambling mode: 300 lots, $15000 TP (~$50 move), NO SL, 1hr hold
    return {"lots": 300.0, "tp": 15000.0, "sl": 999999.0, "hold": 3600, "label": "GAMBLE_MODE"}


def fetch_1m_candles(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch 1m candles from Binance for faster indicator updates."""
    url = "https://data-api.binance.vision/api/v3/klines"
    r = httpx.get(url, params={"symbol": symbol.upper(), "interval": "1m", "limit": limit}, timeout=30.0)
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
            "open_time": int(row[0]),
        })

    return pd.DataFrame(data)


# ── Kronos Real Predictor ─────────────────────────────────────────────────
class KronosPredictorAgent:
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

        print("  Loading Kronos tokenizer...")
        self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        print("  Loading Kronos model...")
        self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512)
        self._initialized = True
        print(f"  Kronos loaded on {device}")

    def predict(self, candles: list[dict]) -> dict:
        """Predict next candlestick direction."""
        self._init_model()

        try:
            return self._model_predict(candles)
        except Exception as e:
            log.error("kronos_predict_error", error=str(e))
            return {"direction": "unknown", "confidence": 0.0, "error": str(e)}

    def _model_predict(self, candles: list[dict]) -> dict:
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
        # 1m candles: need 150 min history for 100-period BB + 70-period RSI
        if len(candles) < 150:
            return {"rsi": 50, "macd": 0, "bb_position": 0.5, "adx": 25, "stoch_rsi": 50, "vwap_dev": 0, "supertrend": "neutral", "atr": 0}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        rsi = self._calculate_rsi(closes, period=70)
        macd, signal = self._calculate_macd(closes, fast=60, slow=130)
        bb_position = self._calculate_bb_position(closes, period=100)
        adx = self._calculate_adx(highs, lows, closes, period=70)
        stoch_rsi = self._calculate_stoch_rsi(closes, period=70)
        vwap_dev = self._calculate_vwap_deviation(closes, volumes)
        supertrend = self._calculate_supertrend(highs, lows, closes, period=50)
        atr = self._calculate_atr(highs, lows, closes, period=70)

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "bb_position": bb_position,
            "adx": adx,
            "stoch_rsi": stoch_rsi,
            "vwap_dev": vwap_dev,
            "supertrend": supertrend,
            "atr": atr,
            "oversold": rsi < 30 and stoch_rsi < 20,
            "overbought": rsi > 70 and stoch_rsi > 80,
            "macd_bull": macd > signal,
            "bb_low": bb_position < 0.1,
            "bb_high": bb_position > 0.9,
            "strong_trend": adx > 25,
            "weak_trend": adx < 20,
            "vwap_below": vwap_dev < -0.5,
            "vwap_above": vwap_dev > 0.5,
            "supertrend_bull": supertrend == "bull",
            "supertrend_bear": supertrend == "bear",
        }

    def _calculate_rsi(self, closes: list[float], period: int = 70) -> float:
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
        if len(gains) < period:
            return 50
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, closes: list[float], fast: int = 60, slow: int = 130) -> tuple[float, float]:
        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)
        macd = ema_fast - ema_slow
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

    def _calculate_bb_position(self, closes: list[float], period: int = 100) -> float:
        if len(closes) < period:
            return 0.5
        sma = sum(closes[-period:]) / period
        std = math.sqrt(sum((c - sma) ** 2 for c in closes[-period:]) / period)
        upper = sma + 2 * std
        lower = sma - 2 * std
        if upper == lower:
            return 0.5
        return (closes[-1] - lower) / (upper - lower)

    def _calculate_adx(self, highs: list[float], lows: list[float], closes: list[float], period: int = 70) -> float:
        if len(closes) < period + 1:
            return 25.0
        trs = []
        plus_dms = []
        minus_dms = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0
            plus_dms.append(plus_dm)
            minus_dms.append(minus_dm)
        if len(trs) < period:
            return 25.0
        atr = sum(trs[-period:]) / period
        plus_di = 100 * sum(plus_dms[-period:]) / period / atr if atr > 0 else 0
        minus_di = 100 * sum(minus_dms[-period:]) / period / atr if atr > 0 else 0
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        return dx

    def _calculate_stoch_rsi(self, closes: list[float], period: int = 70) -> float:
        if len(closes) < period:
            return 50.0
        rsi_vals = []
        for i in range(period, len(closes) + 1):
            chunk = closes[i-period:i]
            gains = [max(0, chunk[j] - chunk[j-1]) for j in range(1, len(chunk))]
            losses = [max(0, chunk[j-1] - chunk[j]) for j in range(1, len(chunk))]
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi_vals.append(100 - (100 / (1 + rs)))
        if len(rsi_vals) < 3:
            return 50.0
        min_rsi = min(rsi_vals[-period:])
        max_rsi = max(rsi_vals[-period:])
        if max_rsi == min_rsi:
            return 50.0
        return 100 * (rsi_vals[-1] - min_rsi) / (max_rsi - min_rsi)

    def _calculate_vwap_deviation(self, closes: list[float], volumes: list[float]) -> float:
        if len(closes) < 2 or len(volumes) < 2:
            return 0.0
        tp = [(closes[i] + closes[i] + closes[i]) / 3 for i in range(len(closes))]
        cum_pv = sum(tp[i] * volumes[i] for i in range(len(tp)))
        cum_vol = sum(volumes)
        vwap = cum_pv / cum_vol if cum_vol > 0 else closes[-1]
        return ((closes[-1] - vwap) / vwap) * 100 if vwap > 0 else 0

    def _calculate_supertrend(self, highs: list[float], lows: list[float], closes: list[float], period: int = 50, multiplier: float = 3.0) -> str:
        if len(closes) < period + 1:
            return "neutral"
        atr = self._calculate_atr(highs, lows, closes, period)
        upper = (highs[-period] + lows[-period]) / 2 + multiplier * atr
        lower = (highs[-period] + lows[-period]) / 2 - multiplier * atr
        if closes[-1] > upper:
            return "bull"
        elif closes[-1] < lower:
            return "bear"
        return "neutral"

    def _calculate_atr(self, highs: list[float], lows: list[float], closes: list[float], period: int = 70) -> float:
        if len(closes) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return sum(trs[-period:]) / period if len(trs) >= period else 0.0


# ── Trend Detector ───────────────────────────────────────────────────────
class TrendDetector:
    def __init__(self):
        self.name = "TrendDetector"

    def analyze(self, candles: list[dict]) -> dict:
        if len(candles) < 100:
            return {"trend": "unknown", "strength": 0.0}

        closes = [c["close"] for c in candles]
        ema25 = self._ema(closes, 25)
        ema50 = self._ema(closes, 50)
        ema100 = self._ema(closes, 100)

        trend_strength = abs(ema25 - ema100) / ema100 * 100

        if ema25 > ema50 > ema100:
            trend = "bull"
            strength = min(1.0, 0.5 + trend_strength * 0.02)
        elif ema25 < ema50 < ema100:
            trend = "bear"
            strength = min(1.0, 0.5 + trend_strength * 0.02)
        else:
            trend = "ranging"
            strength = 0.3

        highs = [c["high"] for c in candles[-50:]]
        lows = [c["low"] for c in candles[-50:]]
        hh = max(highs[-25:]) > max(highs[:25])
        ll = min(lows[-25:]) < min(lows[:25])

        return {
            "trend": trend,
            "strength": strength,
            "ema5": ema25,
            "ema10": ema50,
            "ema20": ema100,
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
        if len(candles) < 50:
            return {"sentiment": "neutral", "confidence": 0.5}

        recent = candles[-50:]
        avg_vol = sum(c["volume"] for c in recent) / len(recent)
        prev_avg = sum(c["volume"] for c in candles[-100:-50]) / 50 if len(candles) >= 100 else avg_vol
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


# ── Conflict Analyzer ─────────────────────────────────────────────────────
class ConflictAnalyzer:
    """Analyzes Kronos vs Trend relationship."""
    def __init__(self):
        self.name = "ConflictAnalyzer"

    def analyze(self, kronos: dict, trend: dict) -> dict:
        kronos_dir = kronos.get("direction", "unknown")
        trend_dir = trend.get("trend", "unknown")
        kronos_conf = kronos.get("confidence", 0.0)
        trend_conf = trend.get("strength", 0.0)

        # Determine relationship
        if kronos_dir == trend_dir:
            if kronos_dir in ("bull", "bear"):
                relationship = "AGREE"
                trust = "high"
            else:
                relationship = "NEUTRAL"
                trust = "low"
        elif kronos_dir == "neutral" or trend_dir == "ranging":
            relationship = "NEUTRAL"
            trust = "low"
        else:
            relationship = "CONFLICT"
            trust = "medium"

        # Historical data: When conflict, Trend wins 35.7% vs Kronos 25%
        # When agree, both are 54.5% correct
        if relationship == "AGREE":
            recommended = kronos_dir
            confidence = min(kronos_conf, trend_conf)
            reasoning = "Kronos and Trend agree - historically 54.5% accurate"
        elif relationship == "CONFLICT":
            # Trend is slightly better in conflicts (35.7% vs 25%)
            if trend_conf > kronos_conf:
                recommended = trend_dir
                confidence = trend_conf * 0.7  # Reduce confidence
                reasoning = "Conflict detected - Trend has higher confidence, historically better in conflicts"
            else:
                recommended = kronos_dir
                confidence = kronos_conf * 0.7
                reasoning = "Conflict detected - Kronos has higher confidence but historically less reliable in conflicts"
        else:
            recommended = "wait"
            confidence = 0.0
            reasoning = "Neutral signals - no clear direction"

        return {
            "relationship": relationship,
            "trust": trust,
            "recommended": recommended,
            "confidence": confidence,
            "reasoning": reasoning,
            "kronos_dir": kronos_dir,
            "trend_dir": trend_dir,
            "kronos_conf": kronos_conf,
            "trend_conf": trend_conf,
        }


# ── LLM Decision Maker ───────────────────────────────────────────────────
def get_competition_stats(wrapper: Any | None = None) -> dict:
    """Fetch current competition metrics from MT5 account using the score script."""
    stats = {
        "final_score": 9.95,
        "win_rate": 40.7,
        "sharpe": -0.18,
        "pnl": -4480,
        "trades": 150,
    }
    try:
        if wrapper is not None and wrapper._connected:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                deals = fetch_deals(mt5, today, datetime.now())
                metrics = calc_competition_metrics(deals, info)
                if "error" not in metrics:
                    stats.update({
                        "pnl": metrics["net_pnl"],
                        "win_rate": metrics["win_rate"],
                        "sharpe": metrics["sharpe"],
                        "trades": metrics["wins"] + metrics["losses"],
                    })
    except Exception:
        pass
    return stats


class LLMDecisionMaker:
    """Calls LLM with ALL agent input for final decision."""
    def __init__(self):
        self.name = "LLM"

    def decide_entry(self, agents: dict, candles: list[dict], wrapper: Any | None = None) -> dict:
        current = candles[-1]
        kronos = agents["kronos"]
        technical = agents["technical"]
        trend = agents["trend"]
        volume = agents["volume"]
        conflict = agents["conflict"]

        # Fetch dynamic stats from MT5
        stats = get_competition_stats(wrapper)

        prompt = f"""BTC/USDT trading decision.

CURRENT COMPETITION METRICS:
- Final Score: {stats['final_score']:.2f} (Target: 75-80 for top 5)
- Win Rate: {stats['win_rate']:.1f}% (Need >55%)
- Sharpe: {stats['sharpe']:.4f} (Need >0.5)
- P&L: ${stats['pnl']:,.0f} (Need +$30K to +$50K)
- Trades: {stats['trades']} (Need >30)

PREDICTION AGENTS:
1. Kronos AI: DISABLED (not used)
2. Trend Detector: {trend['trend']} ({trend['strength']:.0%}) EMA5={trend['ema5']:.0f} EMA20={trend['ema20']:.0f}
3. Conflict Analyzer: DISABLED (Kronos not active)

CONFIRMATION AGENTS:
4. Technical: RSI={technical['rsi']:.0f} StochRSI={technical['stoch_rsi']:.0f} MACD={technical['macd']:.0f} BB={technical['bb_position']:.2f} ADX={technical['adx']:.0f} Supertrend={technical['supertrend']} VWAP_dev={technical['vwap_dev']:.2f}%
5. Volume: {volume['sentiment']} (taker={volume['avg_taker']:.0f}%)

STRATEGY RULES:
1. GOAL: Increase Final Score to 75-80 in the next 2 days. Win Rate >55%, Sharpe >0.5, Positive P&L.
2. TRADE FREQUENTLY: We NEED volume for competition scoring. Target 5-10 trades per hour. Do NOT sit out — idle time kills the score.
3. GO WITH THE DOMINANT DIRECTION: Follow the Trend Detector. If Trend is bullish → BUY. If Trend is bearish → SELL. If Trend is ranging → follow the recent 3-5 candle momentum (up → BUY, down → SELL).
4. AGREE: When Trend + Technical + Volume all align → trade immediately with max confidence.
5. CONFLICT: When Trend and Technical disagree, trust the Trend (it has higher historical accuracy). Do NOT wait — take the trade.
6. NEUTRAL: When no clear signal, still trade if Trend strength >50%. If completely neutral, follow the recent price action (last 3-5 candles). If up → BUY. If down → SELL.
7. REMOVE ADX FILTER: Trade regardless of ADX. Ranging markets still have small 0.1-0.3% moves we can capture. A small win is better than no trade.
8. ALWAYS use SL=${MAX_SL:.0f} and TP=${MAX_TP:.0f}. Cut losses fast, let winners run. Never remove the stop.

End with exactly: FINAL DECISION: [GO LONG / GO SHORT / STAY OUT]
"""

        return call_llm(prompt)

    def decide_exit(self, position: dict, candles: list[dict], current_tp: float, current_sl: float, elapsed: float, profit_history: list[dict]) -> dict:
        current = candles[-1]
        profit = position["profit"]
        side = position["side"]
        open_price = position["open_price"]
        current_price = position["current_price"]
        peak_profit = max((h["profit"] for h in profit_history), default=profit)
        drop_from_peak = peak_profit - profit if peak_profit > profit else 0

        # Build profit history table
        history_lines = "\n".join(
            f"  {h['ts']}: PnL=${h['profit']:.2f} @ ${h['price']:.2f}" for h in profit_history[-10:]
        )

        # Build last 10 candles table
        recent_candles = candles[-10:]
        candle_lines = "\n".join(
            f"  {i+1}. O:{c['open']:.2f} H:{c['high']:.2f} L:{c['low']:.2f} C:{c['close']:.2f} V:{c['volume']:.2f}"
            for i, c in enumerate(recent_candles)
        )

        # Price change since entry
        price_change = current_price - open_price
        price_change_pct = (price_change / open_price) * 100 if open_price else 0

        prompt = f"""BTC/USDT trade management - REAL-TIME TREND MONITORING.

GOAL: Maximize profit by following trend. NO STOP LOSS — user is monitoring manually. Focus on profit trailing and trend strength.

Position: {side}
Entry Price: ${open_price:.2f}
Current Price: ${current_price:.2f}
Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
PnL: ${profit:.0f}
Peak PnL: ${peak_profit:.0f}
Drop from Peak: ${drop_from_peak:.0f}
Target: ${current_tp:.0f}
Time: {elapsed:.0f}s / 3600s max

PROFIT HISTORY (last 10 readings):
{history_lines}

LAST 10 CANDLES (most recent = 10):
{candle_lines}

RULES:
- CLOSE if profit >= ${current_tp:.0f} (take profit reached)
- CLOSE if profit dropped >$2000 from peak (profit trailing — lock it in)
- HOLD if loss < $2000 and trend strong (give it room to recover)
- If loss > $2000 and trend weakening → advise user to consider manual close
- Max hold: 1 hour (3600s). After that, advise user to close manually.

REAL-TIME TREND MONITORING:
1. Is the price still moving in my direction? Compare entry vs current price.
2. Is the trend weakening? (lower highs / higher lows in recent candles)
3. Is volume drying up? (trend losing momentum)
4. Are candles showing reversal patterns? (engulfing, doji, wicks)
5. Is profit dropping from peak? (protect gains)
6. Is the price action against my position? (entry vs current)

DECISION LOGIC:
- **PROFIT TRAILING IS #1 PRIORITY**: If profit was >$2000 and now dropping >$2000 → CLOSE (code will handle this)
- If profit > $2000 and trend weakening → CLOSE (lock in profit)
- If profit > $4000 and trend still strong → HOLD (let winner run)
- If loss > $2000 and trend clearly against position → EXIT (advise manual close)
- If loss < $2000 and trend strong → HOLD (user wants to wait it out)
- DO NOT panic on small dips. The user has NO SL and wants to hold for 1 hour.
- **Give clear advice: KEEP, EXIT, or HOLD for manual user decision.**

End with exactly: FINAL DECISION: [KEEP POSITION / EXIT POSITION / HOLD FOR MANUAL]
"""

        return call_llm(prompt)


def call_llm(prompt: str) -> dict:
    """Call LLM via configured API."""
    api_key = os.getenv("LLM_TOKEN")
    base_url = os.getenv("LLM_BASE_URL", "https://api.fireworks.ai/inference/v1")
    model = os.getenv("LLM_MODEL", "accounts/fireworks/routers/kimi-k2p6-turbo")

    if not api_key:
        log.warning("llm_no_api_key", message="LLM_TOKEN not set in .env")
        return llm_fallback(prompt)

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an aggressive BTC/USDT competition trader with 2 days left to boost the Final Score. Your goal: maximize score (75-80 = top 5). Key metrics: Win Rate >55%, Sharpe >0.5, Positive P&L. You MUST trade frequently — 5-10 trades per hour. Do not wait for perfect alignment. A small win is better than no trade. Go with the dominant direction (Kronos > Trend). Reply with exactly one word: BUY, SELL, or WAIT."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip().upper()

        # Extract the decision word
        words = text.split()
        for word in words:
            clean = word.strip(".,:;!?()")
            if clean in ("BUY", "SELL", "WAIT", "CLOSE", "HOLD"):
                return {"action": clean, "reason": f"Claude: {clean}"}

        # Fallback to substring matching
        if "BUY" in text:
            return {"action": "BUY", "reason": text[:100]}
        elif "SELL" in text:
            return {"action": "SELL", "reason": text[:100]}
        elif "CLOSE" in text:
            return {"action": "CLOSE", "reason": text[:100]}
        elif "HOLD" in text:
            return {"action": "HOLD", "reason": text[:100]}
        else:
            return {"action": "WAIT", "reason": text[:100]}
    except Exception as exc:
        log.error("llm_error", error=str(exc))
        return llm_fallback(prompt)


def _extract_decision_from_thinking(text: str) -> str | None:
    """Extract BUY/SELL/WAIT/CLOSE/HOLD from LLM reasoning output."""
    upper_text = text.upper()
    lines = text.split('\n')

    # Strategy 1: Look for explicit FINAL DECISION marker
    for line in lines:
        upper_line = line.upper()
        if "FINAL DECISION" in upper_line:
            if "GO LONG" in upper_line or "BUY" in upper_line:
                return "BUY"
            elif "GO SHORT" in upper_line or "SELL" in upper_line:
                return "SELL"
            elif "STAY OUT" in upper_line or "WAIT" in upper_line:
                return "WAIT"
            elif "CLOSE" in upper_line:
                return "CLOSE"
            elif "HOLD" in upper_line:
                return "HOLD"

    # Strategy 2: Look for the LAST occurrence of BUY/SELL/WAIT
    # (LLM usually states the final decision at the end)
    last_buy = upper_text.rfind("BUY")
    last_sell = upper_text.rfind("SELL")
    last_wait = upper_text.rfind("WAIT")
    last_close = upper_text.rfind("CLOSE")
    last_hold = upper_text.rfind("HOLD")

    # Filter out occurrences from the prompt itself (they appear early)
    # Look for the LAST occurrence which is the actual decision
    positions = {
        "BUY": last_buy,
        "SELL": last_sell,
        "WAIT": last_wait,
        "CLOSE": last_close,
        "HOLD": last_hold,
    }

    valid_positions = {k: v for k, v in positions.items() if v != -1}
    if valid_positions:
        return max(valid_positions, key=valid_positions.get)

    return None


def llm_fallback(prompt: str) -> dict:
    """Rule-based fallback when LLM is unavailable or thinking."""
    if "POSITION" in prompt:
        # Exit decision
        try:
            profit_line = [l for l in prompt.split('\n') if 'PnL' in l]
            if profit_line:
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
        # Entry decision - parse the conflict info from prompt
        has_agree = "AGREE" in prompt
        has_conflict = "CONFLICT" in prompt
        has_bull = "bull" in prompt.lower()
        has_bear = "bear" in prompt.lower()
        kronos_conf = 0.0
        trend_conf = 0.0

        # Extract confidence values
        try:
            for line in prompt.split('\n'):
                if 'Kronos' in line and '(' in line:
                    kronos_conf = float(line.split('(')[1].split('%')[0]) / 100
                if 'Trend' in line and '(' in line:
                    trend_conf = float(line.split('(')[1].split('%')[0]) / 100
        except:
            pass

        if has_agree:
            if has_bull:
                return {"action": "BUY", "reason": f"AGREE bullish (K:{kronos_conf:.0%} T:{trend_conf:.0%})"}
            elif has_bear:
                return {"action": "SELL", "reason": f"AGREE bearish (K:{kronos_conf:.0%} T:{trend_conf:.0%})"}

        if has_conflict:
            # During conflict, follow the one with higher confidence
            if kronos_conf > trend_conf:
                direction = "BUY" if has_bull else "SELL"
                return {"action": direction, "reason": f"Conflict: Kronos higher confidence ({kronos_conf:.0%})"}
            else:
                direction = "BUY" if has_bull else "SELL"
                return {"action": direction, "reason": f"Conflict: Trend higher confidence ({trend_conf:.0%})"}

        # NEUTRAL but Kronos has high confidence - trade anyway
        if kronos_conf > 0.7:
            direction = "BUY" if has_bull else "SELL"
            return {"action": direction, "reason": f"Kronos high confidence ({kronos_conf:.0%}) in neutral trend"}

        return {"action": "WAIT", "reason": "No clear signal"}


# ── Main Bot ───────────────────────────────────────────────────────────────
def main():
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mt5-account", type=int, default=int(os.getenv("MT5_ACCOUNT", "0")) or None)
    parser.add_argument("--mt5-password", type=str, default=os.getenv("MT5_PASSWORD", "") or None)
    parser.add_argument("--mt5-server", type=str, default=os.getenv("MT5_SERVER", "") or None)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--trades", type=int, default=3)
    parser.add_argument("--sell", action="store_true", help="Force SELL entry immediately (skip signal/LLM)")
    parser.add_argument("--buy", action="store_true", help="Force BUY entry immediately (skip signal/LLM)")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.trades)
        return

    if not all([args.mt5_account, args.mt5_password, args.mt5_server]):
        print("Error: MT5 credentials required for live trading")
        print("  Pass via CLI: --mt5-account X --mt5-password Y --mt5-server Z")
        print("  Or set in .env: MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER")
        sys.exit(1)

    print("=" * 60)
    print("KRONOS LLM MASTER TRADER — RECOVERY MODE")
    print("=" * 60)
    print("Agents: Kronos | Technical | Trend | Volume | Conflict | LLM")
    print(f"[red]GAMBLE MODE: 300 lots | TP=$15000 | NO SL | 1hr hold | Manual close[/red]")
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

    # Import trade state module
    from trade_state import TradeState, read_state, write_state, clear_command

    kronos = KronosPredictorAgent()
    technical = TechnicalAnalyst()
    trend = TrendDetector()
    volume = VolumeProfiler()
    conflict = ConflictAnalyzer()
    llm = LLMDecisionMaker()

    start_time = None
    position_settings = None
    is_paused = False
    entry_interval = 300  # 5 minutes between entry checks
    profit_history: list[dict] = []

    try:
        while True:
            # Read monitor commands
            state = read_state()

            # Handle commands
            if state.command == "close_all":
                print("\n[red]Monitor command: CLOSE ALL POSITIONS[/red]")
                positions = wrapper.get_positions(SYMBOL)
                for p in positions:
                    wrapper.close_position(p.ticket)
                clear_command()
                start_time = None
                position_settings = None
                time.sleep(5)
                continue

            if state.command == "pause":
                is_paused = True
                clear_command()
                print("\n[yellow]Trading PAUSED by monitor[/yellow]")

            if state.command == "resume":
                is_paused = False
                clear_command()
                print("\n[green]Trading RESUMED by monitor[/green]")

            if state.command == "update_tp" and state.command_value > 0:
                if position_settings:
                    position_settings["tp"] = state.command_value
                    print(f"\n[cyan]Monitor updated TP to ${state.command_value:.0f}[/cyan]")
                clear_command()

            if state.command == "update_sl" and state.command_value > 0:
                if position_settings:
                    position_settings["sl"] = state.command_value
                    print(f"\n[cyan]Monitor updated SL to ${state.command_value:.0f}[/cyan]")
                clear_command()

            positions = wrapper.get_positions(SYMBOL)

            if not positions:
                # Update state
                state.has_position = False
                state.is_running = True
                state.is_paused = is_paused
                write_state(state)

                if is_paused:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PAUSED — waiting for resume")
                    time.sleep(30)
                    continue

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === ENTRY DECISION ===")
                print(f"  [bold]GAMBLE MODE: 300 lots | TP=$15000 | NO SL | 1hr hold | Manual close[/bold]")

                # ── FORCE ENTRY (skip signal/LLM) ───────────────────────────────
                if args.sell or args.buy:
                    force_side = "SELL" if args.sell else "BUY"
                    print(f"  [cyan]FORCE ENTRY: {force_side} — skipping signal/LLM[/cyan]")
                    position_settings = get_position_settings(0)
                    desired_side = "buy" if force_side == "BUY" else "sell"
                    df = fetch_1m_candles(SYMBOL, limit=60)
                    candles = df.to_dict('records')
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = candles[-1]["close"]

                    result = wrapper.market_order(SYMBOL, desired_side, position_settings["lots"], 0.0, 0.0)
                    if result.success:
                        print(f"  [green]Trade opened: {force_side} {position_settings['lots']:.0f} lots at {price:.2f}[/green]")
                        profit_history.clear()
                        time.sleep(30)
                        continue
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                        time.sleep(30)
                        continue

                df = fetch_1m_candles(SYMBOL, limit=1000)
                candles = df.to_dict('records')

                # Run all agents
                kronos_pred = {"direction": "neutral", "confidence": 0.0, "change_pct": 0.0}
                tech_pred = technical.analyze(candles)
                trend_pred = trend.analyze(candles)
                vol_pred = volume.analyze(candles)
                conflict_pred = {"relationship": "NEUTRAL", "recommended": "wait", "trust": "low", "reasoning": "Kronos disabled"}

                agents = {
                    "kronos": kronos_pred,
                    "technical": tech_pred,
                    "trend": trend_pred,
                    "volume": vol_pred,
                    "conflict": conflict_pred,
                }

                signal_score = calculate_signal_score(agents)
                position_settings = get_position_settings(signal_score)

                print(f"  [dim]Kronos: DISABLED[/dim]")
                print(f"  Trend: {trend_pred['trend']} (strength={trend_pred['strength']:.2f})")
                print(f"  Tech: RSI={tech_pred['rsi']:.0f} StochRSI={tech_pred['stoch_rsi']:.0f} ADX={tech_pred['adx']:.0f} Super={tech_pred['supertrend']} VWAP={tech_pred['vwap_dev']:.1f}%")
                print(f"  Volume: {vol_pred['sentiment']}")
                print(f"  Signal Score: {signal_score}/100 -> {position_settings['label']} (Lots={position_settings['lots']:.0f} TP=${position_settings['tp']:.0f} NO SL Hold={position_settings['hold']:.0f}s)")

                # LLM decides
                decision = llm.decide_entry(agents, candles, wrapper)
                signal = decision.get("action", "WAIT")
                reason = decision.get("reason", "")
                print(f"  LLM: {signal} - {reason}")

                # Aggressive override
                if signal == "WAIT" and trend_pred.get("strength", 0) >= 0.30 and trend_pred.get("trend") in ("bull", "bear"):
                    trend_dir = trend_pred.get("trend")
                    if tech_pred.get("adx", 0) > 10:
                        signal = "BUY" if trend_dir == "bull" else "SELL"
                        reason = f"AGGRESSIVE OVERRIDE: Trend {trend_dir} (strength={trend_pred.get('strength', 0):.2f}) + ADX={tech_pred.get('adx', 0):.0f}"
                        print(f"  [cyan]OVERRIDE: {signal} — {reason}[/cyan]")
                    else:
                        print(f"  [yellow]OVERRIDE BLOCKED: ADX={tech_pred.get('adx', 0):.0f} < 10[/yellow]")
                        signal = "WAIT"
                        reason = "Override blocked: ADX too low"

                if signal in ("BUY", "SELL"):
                    desired_side = "buy" if signal == "BUY" else "sell"
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = candles[-1]["close"]

                    result = wrapper.market_order(SYMBOL, desired_side, position_settings["lots"], 0.0, 0.0)
                    if result.success:
                        print(f"  [green]Trade opened: {signal} {position_settings['lots']:.0f} lots at {price:.2f}[/green]")
                        profit_history.clear()
                        time.sleep(30)
                        continue
                    else:
                        print(f"  [red]Trade failed: {result}[/red]")
                else:
                    print(f"  [yellow]WAIT — checking again in {entry_interval}s[/yellow]")
                    time.sleep(entry_interval)
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

                if position_settings is None:
                    position_settings = get_position_settings(0)

                current_tp = position_settings["tp"]
                current_sl = position_settings["sl"]
                current_hold = position_settings["hold"]

                # Fix resume bug: calculate agents if not already defined
                if 'kronos_pred' not in locals():
                    kronos_pred = {"direction": "neutral", "confidence": 0.0, "change_pct": 0.0}
                    tech_pred = technical.analyze(candles) if 'candles' in locals() else {"rsi": 50, "stoch_rsi": 50, "adx": 25, "supertrend": "neutral", "vwap_dev": 0, "strong_trend": False, "supertrend_bull": False, "supertrend_bear": False}
                    trend_pred = trend.analyze(candles) if 'candles' in locals() else {"trend": "unknown", "strength": 0.0}
                    vol_pred = volume.analyze(candles) if 'candles' in locals() else {"sentiment": "neutral", "confidence": 0.5}
                    conflict_pred = {"relationship": "NEUTRAL", "recommended": "wait", "trust": "low", "reasoning": "Kronos disabled"}

                # Update state for monitor
                state.has_position = True
                state.position_side = side
                state.position_lots = position_settings["lots"]
                state.position_profit = profit
                state.position_open_price = open_price
                state.current_tp = current_tp
                state.current_sl = current_sl
                state.current_hold = current_hold
                state.elapsed_seconds = elapsed
                state.signal_score = calculate_signal_score({"kronos": kronos_pred, "technical": tech_pred, "trend": trend_pred, "volume": vol_pred, "conflict": conflict_pred})
                state.signal_label = position_settings["label"]
                write_state(state)

                # Track profit history
                profit_history.append({"ts": datetime.now().strftime('%H:%M:%S'), "profit": round(profit, 2), "price": round(current_price, 2)})
                if len(profit_history) > 20:
                    profit_history.pop(0)

                # Calculate peak/drop
                peak_profit = max((h["profit"] for h in profit_history), default=profit)
                drop_from_peak = peak_profit - profit if peak_profit > profit else 0

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring #{ticket} | {side} | PnL=${profit:.2f} | TP=${current_tp:.0f} SL=NO SL Hold={elapsed:.0f}/{current_hold:.0f}s")

                # 1. TP hit
                if profit >= current_tp:
                    print(f"  [green]TP hit: ${profit:.2f}[/green]")
                    wrapper.close_position(ticket)
                    start_time = None
                    position_settings = None
                    profit_history.clear()
                    time.sleep(60)
                    continue

                # 2. Profit lock — close if dropped >$3000 from peak >$3000
                if peak_profit > 3000 and drop_from_peak > 3000:
                    print(f"  [yellow]PROFIT LOCK: Dropped ${drop_from_peak:.2f} from peak ${peak_profit:.2f} — closing at ${profit:.2f}[/yellow]")
                    wrapper.close_position(ticket)
                    start_time = None
                    position_settings = None
                    profit_history.clear()
                    time.sleep(60)
                    continue

                # 3. Max hold time — warn but do not close
                if elapsed >= current_hold:
                    print(f"  [yellow]Max hold: {elapsed:.0f}s — MANUAL CLOSE RECOMMENDED[/yellow]")

                # 4. Ask LLM for exit advice
                df = fetch_1m_candles(SYMBOL, limit=60)
                candles = df.to_dict('records')
                decision = llm.decide_exit(
                    {"side": side, "profit": profit, "open_price": open_price, "current_price": current_price},
                    candles,
                    current_tp,
                    current_sl,
                    elapsed,
                    profit_history,
                )
                action = decision.get("action", "HOLD")
                reason = decision.get("reason", "")
                print(f"  LLM: {action} - {reason}")

                if action == "CLOSE":
                    print(f"  [cyan]Closing at ${profit:.2f}[/cyan]")
                    wrapper.close_position(ticket)
                    start_time = None
                    position_settings = None
                    time.sleep(60)
                    continue
                else:
                    print(f"  [green]HOLD: PnL=${profit:.2f} target=${current_tp:.0f}[/green]")

                time.sleep(MONITOR_INTERVAL)

    except KeyboardInterrupt:
        print("\n[yellow]Stopping bot... positions LEFT OPEN for manual management.[/yellow]")
        wrapper.shutdown()
        print("[green]Done. Use close_all.py if you need to close positions.[/green]")


def run_backtest(trade_count: int):
    print("=" * 60)
    print("KRONOS LLM MASTER BACKTEST")
    print("=" * 60)

    kronos = KronosPredictorAgent()
    technical = TechnicalAnalyst()
    trend = TrendDetector()
    volume = VolumeProfiler()
    conflict = ConflictAnalyzer()
    llm = LLMDecisionMaker()

    total_pnl = 0.0
    wins = 0
    losses = 0

    for i in range(trade_count):
        print(f"\n{'='*60}")
        print(f"BACKTEST #{i+1}")
        print(f"{'='*60}")

        df = fetch_1m_candles(SYMBOL, limit=1000)
        candles = df.to_dict('records')

        kronos_pred = kronos.predict(candles)
        tech_pred = technical.analyze(candles)
        trend_pred = trend.analyze(candles)
        vol_pred = volume.analyze(candles)
        conflict_pred = conflict.analyze(kronos_pred, trend_pred)

        agents = {
            "kronos": kronos_pred,
            "technical": tech_pred,
            "trend": trend_pred,
            "volume": vol_pred,
            "conflict": conflict_pred,
        }

        print(f"Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f})")
        print(f"Trend: {trend_pred['trend']} (strength={trend_pred['strength']:.2f})")
        print(f"Conflict: {conflict_pred['relationship']} -> {conflict_pred['recommended']}")
        print(f"Tech: RSI={tech_pred['rsi']:.0f} StochRSI={tech_pred['stoch_rsi']:.0f} ADX={tech_pred['adx']:.0f} Super={tech_pred['supertrend']} VWAP={tech_pred['vwap_dev']:.1f}%")
        print(f"Volume: {vol_pred['sentiment']}")

        # Backtest mode: no wrapper, use default stats
        decision = llm.decide_entry(agents, candles, None)
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
            # No SL — user monitors manually. Only trail profit.
            if max_profit > 300 and profit <= max_profit - 300:
                closed = True
                exit_reason = "TRAIL"
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
