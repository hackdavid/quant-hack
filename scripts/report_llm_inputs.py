#!/usr/bin/env python3
"""Report: What inputs does the LLM receive for each trading decision?

This script loads the pipeline, runs one bar, and prints the EXACT prompt
that would be sent to the LLM. Use this to inspect and verify context.

Usage:
    uv run python scripts/report_llm_inputs.py \
        --transformer-run models/transformer/20260623T132957Z \
        --mt5-account YOUR_ACCOUNT --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).parent))

import autonomous_trader as _at
AutonomousTrader = _at.AutonomousTrader
BinanceFeed = _at.BinanceFeed


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Print LLM input report for one bar")
    p.add_argument("--transformer-run", type=Path, required=True, help="Path to transformer run dir")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root")
    p.add_argument("--mt5-account", type=int, required=True, help="MT5 account")
    p.add_argument("--mt5-password", type=str, required=True, help="MT5 password")
    p.add_argument("--mt5-server", type=str, required=True, help="MT5 server")
    p.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    args = p.parse_args()

    print("=" * 70)
    print("  LLM INPUT REPORT — What the LLM sees before every trade")
    print("=" * 70)

    # Build trader with LLM debug enabled
    trader = AutonomousTrader(
        symbol=args.symbol,
        capital=1_000_000.0,
        transformer_run=args.transformer_run,
        data_dir=args.data_dir,
        mt5_account=args.mt5_account,
        mt5_password=args.mt5_password,
        mt5_server=args.mt5_server,
        use_llm=True,
        llm_debug=True,
    )

    # Load historical bars
    loaded = trader.feed.load_historical(bars_5m=128)
    print(f"\nLoaded {loaded} historical bars.")
    print(f"Feed buffer rows: {len(trader.feed.buffer)}")

    # Connect to MT5 for real account state
    if not trader.executor.connect():
        print("\n[WARNING] MT5 connection failed — using dummy account state")
    else:
        print("\nMT5 connected.")
        state = trader.executor.state()
        print(f"  Balance:  {state.get('balance', 0):,.2f}")
        print(f"  Equity:   {state.get('equity', 0):,.2f}")
        print(f"  Profit:   {state.get('profit', 0):+.2f}")
        positions = trader.executor.get_positions(args.symbol)
        print(f"  Positions: {len(positions)}")
        for pos in positions:
            print(f"    {pos['side']} {pos['volume']:.4f} @ {pos['open_price']:.2f}")

    # Run pipeline on latest bar
    df = trader.feed.to_df()
    if df is None:
        print("\n[ERROR] Not enough bars for pipeline.")
        return

    pipeline = trader._run_pipeline(df)
    last_row = df.row(len(df) - 1, named=True)
    candle_close = last_row.get("close", 0.0)

    # Account / risk state
    account = trader.executor.state()
    equity = account.get("equity", trader.capital)
    risk_state = trader.risk.update(equity)
    positions = trader.executor.get_positions(args.symbol)
    recent_logs = trader.logger.last_n(20)

    # Signal from pipeline
    signal = "HOLD"
    confidence = 0.5
    if pipeline.get("decision"):
        side = pipeline["decision"].get("side", "flat")
        if side == "long":
            signal = "BUY"
            confidence = pipeline["decision"].get("confidence", 0.5)
        elif side == "short":
            signal = "SELL"
            confidence = pipeline["decision"].get("confidence", 0.5)

    # Build full candle data
    raw = trader.feed._last_raw_bar or {}
    bar_data = {
        "open": raw.get("open", candle_close),
        "high": raw.get("high", candle_close),
        "low": raw.get("low", candle_close),
        "close": candle_close,
        "volume": raw.get("volume", last_row.get("vol_5m", 0.0)),
        "trade_count": raw.get("trade_count", 0),
        "taker_buy_ratio": last_row.get("taker_buy_ratio_5m", 0.5),
    }

    # Build the prompt using the same method as the trader
    system, user = trader.llm._build_prompt(
        signal=signal,
        confidence=confidence,
        bar=bar_data,
        positions=positions,
        account=account,
        risk_state=risk_state,
        recent_logs=recent_logs,
        pipeline=pipeline,
        competition_rules=None,  # uses default
    )

    print("\n" + "=" * 70)
    print("  SYSTEM PROMPT")
    print("=" * 70)
    print(system)

    print("\n" + "=" * 70)
    print("  USER PROMPT")
    print("=" * 70)
    print(user)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Signal:         {signal}")
    print(f"  Confidence:     {confidence:.4f}")
    print(f"  Candle:         {bar_data}")
    print(f"  Positions:      {len(positions)}")
    print(f"  Account:        {account}")
    print(f"  Risk state:     {risk_state}")
    print(f"  Recent logs:    {len(recent_logs)} entries")
    print(f"  Pipeline:       {pipeline}")
    print(f"  System chars:   {len(system)}")
    print(f"  User chars:     {len(user)}")
    print(f"  Total chars:    {len(system) + len(user)}")

    trader.executor.shutdown()
    print("\n[OK] Report complete.")


if __name__ == "__main__":
    main()
