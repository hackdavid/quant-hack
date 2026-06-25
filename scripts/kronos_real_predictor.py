#!/usr/bin/env python3
"""Kronos Real Model Integration Test.

Uses the actual Kronos model from GitHub + HuggingFace.

Usage:
    .venv/Scripts/python.exe scripts/kronos_real_predictor.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import torch

# Add Kronos module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "kronos_module"))
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_5m_candles(symbol: str = "BTCUSDT", limit: int = 400) -> pd.DataFrame:
    """Fetch 5m candles from Binance and return as DataFrame."""
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
            "amount": float(row[7]),  # Quote volume
        })

    return pd.DataFrame(data)


def load_kronos_model():
    """Load Kronos model and tokenizer."""
    print("Loading Kronos tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

    print("Loading Kronos model...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Kronos loaded on {device}")
    return model, tokenizer


def predict_next_candles(model, tokenizer, df: pd.DataFrame, pred_len: int = 12) -> pd.DataFrame:
    """Predict next candles using Kronos."""
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df['timestamps']

    # Create future timestamps
    last_time = df['timestamps'].iloc[-1]
    freq = pd.Timedelta(minutes=5)
    y_timestamp = pd.Series(pd.date_range(start=last_time + freq, periods=pred_len, freq=freq))

    print(f"Predicting {pred_len} candles...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    return pred_df


def main():
    print("=" * 60)
    print("KRONOS REAL MODEL INTEGRATION")
    print("=" * 60)

    # 1. Load model
    model, tokenizer = load_kronos_model()

    # 2. Fetch data
    print("\nFetching BTC/USDT 5m data...")
    df = fetch_5m_candles("BTCUSDT", limit=400)
    print(f"Data: {len(df)} rows")
    print(f"Last close: {df['close'].iloc[-1]:.2f}")

    # 3. Predict
    pred_df = predict_next_candles(model, tokenizer, df, pred_len=12)

    # 4. Show results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted next 12 candles:")
    print(pred_df.head(12))

    # Calculate direction
    last_close = df['close'].iloc[-1]
    predicted_close = pred_df['close'].iloc[-1]
    direction = "UP" if predicted_close > last_close else "DOWN"
    change_pct = (predicted_close - last_close) / last_close * 100

    print(f"\nLast close: {last_close:.2f}")
    print(f"Predicted close (12 candles): {predicted_close:.2f}")
    print(f"Direction: {direction} ({change_pct:+.2f}%)")


if __name__ == "__main__":
    main()
