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

    # Kronos confidence (0-30)
    score += int(kronos.get("confidence", 0) * 30)

    # Trend strength (0-20)
    score += int(trend.get("strength", 0) * 20)

    # Agreement bonus (0-25)
    if conflict.get("relationship") == "AGREE":
        score += 25
    elif conflict.get("relationship") == "NEUTRAL":
        score += 10

    # Technical confirmation (0-15)
    if technical.get("strong_trend"):
        score += 10
    if technical.get("supertrend_bull") and kronos.get("direction") == "bull":
        score += 5
    if technical.get("supertrend_bear") and kronos.get("direction") == "bear":
        score += 5

    # Volume confirmation (0-10)
    if volume.get("sentiment") in ("bullish", "bearish"):
        score += 10
    elif volume.get("sentiment") in ("mildly_bullish", "mildly_bearish"):
        score += 5

    return min(100, max(0, score))


def get_position_settings(score: int) -> dict:
    """Return position size, TP, SL, hold time based on signal score."""
    if score >= 80:
        return {"lots": LOTS_MAX, "tp": TP_MAX, "sl": SL_MAX, "hold": HOLD_MAX, "label": "MAX"}
    elif score >= 65:
        return {"lots": LOTS_VERY_STRONG, "tp": TP_VERY_STRONG, "sl": SL_VERY_STRONG, "hold": HOLD_VERY_STRONG, "label": "VERY_STRONG"}
    elif score >= 50:
        return {"lots": LOTS_STRONG, "tp": TP_STRONG, "sl": SL_STRONG, "hold": HOLD_STRONG, "label": "STRONG"}
    elif score >= 35:
        return {"lots": LOTS_MEDIUM, "tp": TP_MEDIUM, "sl": SL_MEDIUM, "hold": HOLD_MEDIUM, "label": "MEDIUM"}
    else:
        return {"lots": LOTS_WEAK, "tp": TP_WEAK, "sl": SL_WEAK, "hold": HOLD_WEAK, "label": "WEAK"}


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
        if len(candles) < 30:
            return {"rsi": 50, "macd": 0, "bb_position": 0.5, "adx": 25, "stoch_rsi": 50, "vwap_dev": 0, "supertrend": "neutral", "atr": 0}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        rsi = self._calculate_rsi(closes)
        macd, signal = self._calculate_macd(closes)
        bb_position = self._calculate_bb_position(closes)
        adx = self._calculate_adx(highs, lows, closes)
        stoch_rsi = self._calculate_stoch_rsi(closes)
        vwap_dev = self._calculate_vwap_deviation(closes, volumes)
        supertrend = self._calculate_supertrend(highs, lows, closes)
        atr = self._calculate_atr(highs, lows, closes)

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

    def _calculate_adx(self, highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
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

    def _calculate_stoch_rsi(self, closes: list[float], period: int = 14) -> float:
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
        min_rsi = min(rsi_vals[-14:])
        max_rsi = max(rsi_vals[-14:])
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

    def _calculate_supertrend(self, highs: list[float], lows: list[float], closes: list[float], period: int = 10, multiplier: float = 3.0) -> str:
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

    def _calculate_atr(self, highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
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
1. Kronos AI: {kronos['direction']} ({kronos['confidence']:.0%}) predicted_change={kronos.get('change_pct', 0):.2f}%
2. Trend Detector: {trend['trend']} ({trend['strength']:.0%}) EMA5={trend['ema5']:.0f} EMA20={trend['ema20']:.0f}
3. Conflict Analyzer: {conflict['relationship']} | Trust: {conflict['trust']}
   - Recommended: {conflict['recommended']}
   - Reasoning: {conflict['reasoning']}

CONFIRMATION AGENTS:
4. Technical: RSI={technical['rsi']:.0f} StochRSI={technical['stoch_rsi']:.0f} MACD={technical['macd']:.0f} BB={technical['bb_position']:.2f} ADX={technical['adx']:.0f} Supertrend={technical['supertrend']} VWAP_dev={technical['vwap_dev']:.2f}%
5. Volume: {volume['sentiment']} (taker={volume['avg_taker']:.0f}%)

STRATEGY RULES:
1. GOAL: Increase Final Score. Win Rate >55%, Sharpe >0.5, Positive P&L.
2. DO NOT WAIT too long. If signals are 60%+ aligned, take the trade even if small.
3. AGREE: Trade immediately when Kronos + Trend + Supertrend align (54.5% accuracy).
4. CONFLICT: Follow Trend (35.7% win vs Kronos 25%). Only trade if ADX>25 (strong trend) and Supertrend confirms.
5. NEUTRAL: If StochRSI extreme (<20 or >80) OR VWAP dev >1% OR volume >60%, take the trade.
6. FILTER: Only trade if ADX>20 (avoid choppy markets). Strong trend = ADX>25.
7. Small winning trades are better than no trades. Target 3-5 trades per hour.
8. ALWAYS use SL=${MAX_SL:.0f} and TP=${MAX_TP:.0f}. Cut losses fast, let winners run.

End with exactly: FINAL DECISION: [GO LONG / GO SHORT / STAY OUT]
"""

        return call_llm(prompt)

    def decide_exit(self, position: dict, candles: list[dict], current_tp: float, elapsed: float) -> dict:
        current = candles[-1]
        profit = position["profit"]
        side = position["side"]

        prompt = f"""BTC/USDT trade management - REAL-TIME TREND MONITORING.

GOAL: Protect win rate and P&L. Maximize profit by following trend, not chasing peaks.

Position: {side}
PnL: ${profit:.0f}
Target: ${current_tp:.0f}
Time: {elapsed:.0f}s
Price: {current['close']:.0f}

RULES:
- CLOSE if profit >= ${current_tp:.0f} (take profit reached)
- CLOSE if loss > $200 (cut losses fast)
- CLOSE if time > 900s (max 15 min)

REAL-TIME TREND MONITORING:
1. Is the price still moving in my direction?
2. Is the trend weakening? (lower highs / higher lows)
3. Is volume drying up? (trend losing momentum)
4. Are candles showing reversal patterns?

DECISION LOGIC:
- If profit > $500 and trend weakening → CLOSE (lock in profit)
- If profit > $100 and time > 600s → CLOSE (time exit)
- If loss < $100 and trend strong → HOLD (give it room)
- If loss > $200 → CLOSE (stop loss)
- If profit > $1000 and trend still strong → HOLD (let winner run)
- DO NOT use trailing stops. Watch trend direction, not peak price.

End with exactly: FINAL DECISION: [KEEP POSITION / EXIT POSITION]
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
                {"role": "system", "content": "You are a BTC/USDT competition trader. Your goal is to maximize Final Score (75-80 = top 5). Key metrics: Win Rate >55%, Sharpe >0.5, Positive P&L. Do not wait too long. Take small winning trades. Reply with exactly one word: BUY, SELL, or WAIT."},
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
    print("KRONOS LLM MASTER TRADER")
    print("=" * 60)
    print("Agents: Kronos | Technical | Trend | Volume | Conflict | LLM")
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
                df = fetch_5m_candles(SYMBOL, limit=400)
                candles = df.to_dict('records')

                # Run all agents
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

                # Calculate signal strength
                signal_score = calculate_signal_score(agents)
                position_settings = get_position_settings(signal_score)

                print(f"  Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f})")
                print(f"  Trend: {trend_pred['trend']} (strength={trend_pred['strength']:.2f})")
                print(f"  Conflict: {conflict_pred['relationship']} -> {conflict_pred['recommended']}")
                print(f"  Tech: RSI={tech_pred['rsi']:.0f} StochRSI={tech_pred['stoch_rsi']:.0f} ADX={tech_pred['adx']:.0f} Super={tech_pred['supertrend']} VWAP={tech_pred['vwap_dev']:.1f}%")
                print(f"  Volume: {vol_pred['sentiment']}")
                print(f"  Signal Score: {signal_score}/100 -> {position_settings['label']} (Lots={position_settings['lots']:.0f} TP=${position_settings['tp']:.0f} SL=${position_settings['sl']:.0f} Hold={position_settings['hold']:.0f}s)")

                # LLM decides
                decision = llm.decide_entry(agents, candles, wrapper)
                signal = decision.get("action", "WAIT")
                reason = decision.get("reason", "")

                print(f"  LLM: {signal} - {reason}")

                if signal in ("BUY", "SELL"):
                    desired_side = "buy" if signal == "BUY" else "sell"
                    price = wrapper.get_current_price(SYMBOL, desired_side)
                    if price is None:
                        price = candles[-1]["close"]

                    result = wrapper.market_order(SYMBOL, desired_side, position_settings["lots"], 0.0, 0.0)
                    if result.success:
                        print(f"  [green]Trade opened: {signal} {position_settings['lots']:.0f} lots at {price:.2f}[/green]")
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

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring #{ticket} | {side} | PnL=${profit:.2f} | TP=${current_tp:.0f} SL=${current_sl:.0f} Hold={elapsed:.0f}/{current_hold:.0f}s")

                # Hard limits
                if profit >= current_tp:
                    print(f"  [green]TP hit: ${profit:.2f}[/green]")
                    wrapper.close_position(ticket)
                    start_time = None
                    position_settings = None
                    time.sleep(60)
                    continue

                if profit <= -current_sl:
                    print(f"  [red]SL hit: ${profit:.2f}[/red]")
                    wrapper.close_position(ticket)
                    start_time = None
                    position_settings = None
                    time.sleep(60)
                    continue

                if elapsed >= current_hold:
                    print(f"  [yellow]Max hold: {elapsed:.0f}s[/yellow]")
                    wrapper.close_position(ticket)
                    start_time = None
                    position_settings = None
                    time.sleep(60)
                    continue

                # Ask LLM for real-time trend monitoring
                df = fetch_5m_candles(SYMBOL, limit=20)
                candles = df.to_dict('records')
                decision = llm.decide_exit(
                    {"side": side, "profit": profit, "open_price": open_price, "current_price": current_price},
                    candles,
                    current_tp,
                    elapsed,
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
        print("\n[yellow]Stopping...[/yellow]")
        positions = wrapper.get_positions(SYMBOL)
        for p in positions:
            wrapper.close_position(p.ticket)
        wrapper.shutdown()
        print("[green]Done.[/green]")


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

        df = fetch_5m_candles(SYMBOL, limit=400)
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
