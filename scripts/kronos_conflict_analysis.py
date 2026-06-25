#!/usr/bin/env python3
"""Kronos Conflict Analysis - Walk-Forward Backtest.

For each candle i (from 61 to N):
  - Use candles [i-60:i] as context (60 candles)
  - Kronos predicts candle i+1 direction
  - Compare with actual trend of candle i+1
  - Record: agree, conflict, or disagree
  - Track what happens after conflicts

Usage:
    .venv/Scripts/python.exe scripts/kronos_conflict_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "kronos_module"))
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_5m_candles(symbol: str = "BTCUSDT", limit: int = 576) -> pd.DataFrame:
    """Fetch 5m candles from Binance (2 days = 576 candles)."""
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


class KronosAnalyzer:
    """Analyzes Kronos predictions vs actual market direction."""

    def __init__(self):
        self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512)

    def predict(self, df: pd.DataFrame, idx: int) -> dict:
        """Predict direction for candle at idx+1 using candles [idx-60:idx]."""
        lookback = 60
        start_idx = max(0, idx - lookback)
        end_idx = idx

        x_df = df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.iloc[start_idx:end_idx]['timestamps'].reset_index(drop=True)

        last_time = x_timestamp.iloc[-1]
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

        last_close = x_df['close'].iloc[-1]
        predicted_close = pred_df['close'].iloc[-1]
        change_pct = (predicted_close - last_close) / last_close * 100

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
            "change_pct": change_pct,
            "predicted_close": predicted_close,
        }


def analyze_trend(df: pd.DataFrame, idx: int) -> dict:
    """Analyze actual trend using EMA crossovers."""
    lookback = 60
    start_idx = max(0, idx - lookback)
    end_idx = idx + 1  # Include candle idx

    closes = df.iloc[start_idx:end_idx]['close'].tolist()
    if len(closes) < 20:
        return {"trend": "unknown", "strength": 0.0}

    ema5 = sum(closes[-5:]) / 5
    ema10 = sum(closes[-10:]) / 10
    ema20 = sum(closes[-20:]) / 20

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

    return {
        "trend": trend,
        "strength": strength,
        "ema5": ema5,
        "ema10": ema10,
        "ema20": ema20,
    }


def get_actual_direction(df: pd.DataFrame, idx: int) -> str:
    """Get actual direction of candle idx+1."""
    if idx + 1 >= len(df):
        return "unknown"

    current_close = df.iloc[idx]['close']
    next_close = df.iloc[idx + 1]['close']

    change_pct = (next_close - current_close) / current_close * 100

    if change_pct > 0.05:
        return "bull"
    elif change_pct < -0.05:
        return "bear"
    else:
        return "neutral"


def main():
    print("=" * 60)
    print("KRONOS CONFLICT ANALYSIS")
    print("=" * 60)
    print("Method: Walk-forward analysis using 60-candle context")
    print("=" * 60)

    # 1. Fetch data
    print("\nFetching 2 days of 5m data...")
    df = fetch_5m_candles("BTCUSDT", limit=576)
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")

    # 2. Load Kronos
    print("\nLoading Kronos model...")
    analyzer = KronosAnalyzer()
    print("Model loaded!")

    # 3. Run analysis
    print("\nRunning walk-forward analysis...")
    print("-" * 60)

    results = []
    start_idx = 60
    end_idx = min(160, len(df) - 1)  # At least 100 candles

    for i in range(start_idx, end_idx):
        print(f"\nCandle {i} / {end_idx} ({df['timestamps'].iloc[i]})")

        # Kronos prediction
        kronos_pred = analyzer.predict(df, i)
        print(f"  Kronos: {kronos_pred['direction']} (conf={kronos_pred['confidence']:.2f}) change={kronos_pred['change_pct']:.2f}%")

        # Trend analysis
        trend = analyze_trend(df, i)
        print(f"  Trend: {trend['trend']} (strength={trend['strength']:.2f})")

        # Actual direction
        actual = get_actual_direction(df, i)
        print(f"  Actual: {actual}")

        # Determine relationship
        kronos_dir = kronos_pred['direction']
        trend_dir = trend['trend']
        kronos_conf = kronos_pred['confidence']
        trend_conf = trend['strength']

        if kronos_dir == trend_dir:
            relationship = "AGREE"
        elif kronos_dir == "neutral" or trend_dir == "ranging":
            relationship = "NEUTRAL"
        else:
            relationship = "CONFLICT"

        # Check if Kronos was right
        if kronos_dir == actual:
            kronos_correct = True
        elif kronos_dir == "neutral" and actual == "neutral":
            kronos_correct = True
        else:
            kronos_correct = False

        # Check if Trend was right
        if trend_dir == actual:
            trend_correct = True
        elif trend_dir == "ranging" and actual == "neutral":
            trend_correct = True
        else:
            trend_correct = False

        result = {
            "idx": i,
            "timestamp": df['timestamps'].iloc[i],
            "kronos_dir": kronos_dir,
            "kronos_conf": kronos_conf,
            "trend_dir": trend_dir,
            "trend_conf": trend_conf,
            "actual": actual,
            "relationship": relationship,
            "kronos_correct": kronos_correct,
            "trend_correct": trend_correct,
            "kronos_change_pct": kronos_pred['change_pct'],
            "actual_change_pct": (df.iloc[i+1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'] * 100 if i + 1 < len(df) else 0,
        }
        results.append(result)

        print(f"  Result: {relationship} | Kronos right: {kronos_correct} | Trend right: {trend_correct}")

    # 4. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(results)
    agree = sum(1 for r in results if r["relationship"] == "AGREE")
    conflict = sum(1 for r in results if r["relationship"] == "CONFLICT")
    neutral = sum(1 for r in results if r["relationship"] == "NEUTRAL")

    kronos_correct = sum(1 for r in results if r["kronos_correct"])
    trend_correct = sum(1 for r in results if r["trend_correct"])

    # When they agree
    agree_results = [r for r in results if r["relationship"] == "AGREE"]
    agree_kronos_correct = sum(1 for r in agree_results if r["kronos_correct"])
    agree_trend_correct = sum(1 for r in agree_results if r["trend_correct"])

    # When they conflict
    conflict_results = [r for r in results if r["relationship"] == "CONFLICT"]
    conflict_kronos_correct = sum(1 for r in conflict_results if r["kronos_correct"])
    conflict_trend_correct = sum(1 for r in conflict_results if r["trend_correct"])

    print(f"\nTotal candles analyzed: {total}")
    print(f"  Agree: {agree} ({agree/total*100:.1f}%)")
    print(f"  Conflict: {conflict} ({conflict/total*100:.1f}%)")
    print(f"  Neutral: {neutral} ({neutral/total*100:.1f}%)")

    print(f"\nAccuracy:")
    print(f"  Kronos correct: {kronos_correct}/{total} ({kronos_correct/total*100:.1f}%)")
    print(f"  Trend correct: {trend_correct}/{total} ({trend_correct/total*100:.1f}%)")

    if agree_results:
        print(f"\nWhen AGREE ({len(agree_results)} cases):")
        print(f"  Kronos correct: {agree_kronos_correct}/{len(agree_results)} ({agree_kronos_correct/len(agree_results)*100:.1f}%)")
        print(f"  Trend correct: {agree_trend_correct}/{len(agree_results)} ({agree_trend_correct/len(agree_results)*100:.1f}%)")

    if conflict_results:
        print(f"\nWhen CONFLICT ({len(conflict_results)} cases):")
        print(f"  Kronos correct: {conflict_kronos_correct}/{len(conflict_results)} ({conflict_kronos_correct/len(conflict_results)*100:.1f}%)")
        print(f"  Trend correct: {conflict_trend_correct}/{len(conflict_results)} ({conflict_trend_correct/len(conflict_results)*100:.1f}%)")

        # Analyze high-confidence conflicts
        high_conf_conflicts = [r for r in conflict_results if r["kronos_conf"] > 0.7 or r["trend_conf"] > 0.7]
        if high_conf_conflicts:
            print(f"\nHigh-confidence conflicts (>70%): {len(high_conf_conflicts)}")
            hc_kronos_correct = sum(1 for r in high_conf_conflicts if r["kronos_correct"])
            hc_trend_correct = sum(1 for r in high_conf_conflicts if r["trend_correct"])
            print(f"  Kronos correct: {hc_kronos_correct}/{len(high_conf_conflicts)} ({hc_kronos_correct/len(high_conf_conflicts)*100:.1f}%)")
            print(f"  Trend correct: {hc_trend_correct}/{len(high_conf_conflicts)} ({hc_trend_correct/len(high_conf_conflicts)*100:.1f}%)")

    # 5. Detailed conflict log
    print("\n" + "=" * 60)
    print("CONFLICT LOG")
    print("=" * 60)
    for r in conflict_results:
        winner = "Kronos" if r["kronos_correct"] else ("Trend" if r["trend_correct"] else "Neither")
        print(f"  Candle {r['idx']} ({r['timestamp']}):")
        print(f"    Kronos: {r['kronos_dir']} ({r['kronos_conf']:.0%}) vs Trend: {r['trend_dir']} ({r['trend_conf']:.0%})")
        print(f"    Actual: {r['actual']} ({r['actual_change_pct']:+.2f}%)")
        print(f"    Winner: {winner}")

    # 6. Trading simulation
    print("\n" + "=" * 60)
    print("TRADING SIMULATION")
    print("=" * 60)

    lot_size = 8.0
    tp = 200.0
    sl = 400.0

    # Strategy 1: Trade when Kronos and Trend agree
    pnl_agree = 0.0
    wins_agree = 0
    losses_agree = 0
    for r in agree_results:
        if r["relationship"] != "AGREE":
            continue
        direction = r["kronos_dir"]  # They agree, so either works
        actual_change = r["actual_change_pct"]
        profit = actual_change / 100 * df.iloc[r["idx"]]['close'] * lot_size
        if direction == "bull":
            pnl_agree += profit
        else:
            pnl_agree -= profit
        if profit > 0:
            wins_agree += 1
        else:
            losses_agree += 1

    print(f"\nStrategy 1: Trade when AGREE")
    print(f"  Trades: {wins_agree + losses_agree}")
    print(f"  Wins: {wins_agree}")
    print(f"  Losses: {losses_agree}")
    print(f"  P&L: ${pnl_agree:.2f}")

    # Strategy 2: Trade only Kronos direction
    pnl_kronos = 0.0
    wins_kronos = 0
    losses_kronos = 0
    for r in results:
        if r["kronos_dir"] == "neutral":
            continue
        direction = r["kronos_dir"]
        actual_change = r["actual_change_pct"]
        profit = actual_change / 100 * df.iloc[r["idx"]]['close'] * lot_size
        if direction == "bull":
            pnl_kronos += profit
        else:
            pnl_kronos -= profit
        if (direction == "bull" and profit > 0) or (direction == "bear" and profit < 0):
            wins_kronos += 1
        else:
            losses_kronos += 1

    print(f"\nStrategy 2: Trade Kronos direction")
    print(f"  Trades: {wins_kronos + losses_kronos}")
    print(f"  Wins: {wins_kronos}")
    print(f"  Losses: {losses_kronos}")
    print(f"  P&L: ${pnl_kronos:.2f}")

    # Strategy 3: Trade only when Kronos high confidence
    pnl_kronos_high = 0.0
    wins_kronos_high = 0
    losses_kronos_high = 0
    for r in results:
        if r["kronos_dir"] == "neutral" or r["kronos_conf"] < 0.7:
            continue
        direction = r["kronos_dir"]
        actual_change = r["actual_change_pct"]
        profit = actual_change / 100 * df.iloc[r["idx"]]['close'] * lot_size
        if direction == "bull":
            pnl_kronos_high += profit
        else:
            pnl_kronos_high -= profit
        if (direction == "bull" and profit > 0) or (direction == "bear" and profit < 0):
            wins_kronos_high += 1
        else:
            losses_kronos_high += 1

    print(f"\nStrategy 3: Trade Kronos high confidence (>70%)")
    print(f"  Trades: {wins_kronos_high + losses_kronos_high}")
    print(f"  Wins: {wins_kronos_high}")
    print(f"  Losses: {losses_kronos_high}")
    print(f"  P&L: ${pnl_kronos_high:.2f}")


if __name__ == "__main__":
    main()
