#!/usr/bin/env python3
"""Live trading on Binance USDT-M perpetuals.

⚠️  REAL MONEY — test on paper first with run_paper_trade.py

Prerequisites:
  export BINANCE_API_KEY="your_key"
  export BINANCE_API_SECRET="your_secret"
  # Enable futures trading on your Binance account
  # Start with minimal capital until strategy is validated

Usage:
    python scripts/run_live_trade.py \\
        --transformer-dir models/transformer/20260623T132957Z \\
        --lgb-dir models/lgb \\
        --capital 1000 \\
        --threshold 0.57 \\
        --max-position-frac 0.10 \\
        --leverage 1

Ctrl+C to stop. All fills written to logs/trader/trade_log_*.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from intraday.signal.combiner import SignalCombiner
from intraday.risk.agent import RiskAgent
from intraday.trader.exchange import Exchange
from intraday.trader.loop import TradingLoop


def main() -> None:
    p = argparse.ArgumentParser(description="Live trade on Binance USDT-M futures")
    p.add_argument("--transformer-dir",  default=None)
    p.add_argument("--lgb-dir",          default=None)
    p.add_argument("--lgb-ensemble-dir", default=None)
    p.add_argument("--capital",          type=float, default=1_000.0,
                   help="Notional account capital in USD (used for position sizing)")
    p.add_argument("--threshold",        type=float, default=0.57,
                   help="Prob threshold to trade (higher = fewer but higher-confidence trades)")
    p.add_argument("--max-position-frac",type=float, default=0.10,
                   help="Max % of capital per trade (keep low until validated)")
    p.add_argument("--daily-loss-limit", type=float, default=0.01,
                   help="Stop trading today if daily loss > this % (default 1%)")
    p.add_argument("--max-drawdown",     type=float, default=0.03,
                   help="Stop trading permanently if drawdown > this % (default 3%)")
    p.add_argument("--leverage",         type=int,   default=1,
                   help="Futures leverage (1 = no leverage, recommended to start)")
    p.add_argument("--symbol",           default="BTCUSDT")
    p.add_argument("--log-dir",          default="logs/trader")
    args = p.parse_args()

    if not args.transformer_dir and not args.lgb_dir and not args.lgb_ensemble_dir:
        p.error("Provide at least one of --transformer-dir / --lgb-dir / --lgb-ensemble-dir")

    # Require API keys
    if not os.environ.get("BINANCE_API_KEY"):
        sys.exit("ERROR: Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables first.")

    print("=" * 60)
    print("  ⚠️  LIVE TRADING MODE — REAL MONEY")
    print("=" * 60)
    print(f"  Symbol:    {args.symbol}")
    print(f"  Capital:   ${args.capital:,.2f}")
    print(f"  Threshold: {args.threshold}  (higher = fewer trades)")
    print(f"  Max pos:   {args.max_position_frac*100:.0f}% of capital")
    print(f"  Leverage:  {args.leverage}x")
    print(f"  Daily lim: {args.daily_loss_limit*100:.1f}%")
    print(f"  Max DD:    {args.max_drawdown*100:.1f}%")
    print()

    confirm = input("Type 'yes' to start live trading: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    combiner = SignalCombiner(
        transformer_dir  = args.transformer_dir,
        lgb_dir          = args.lgb_dir,
        lgb_ensemble_dir = args.lgb_ensemble_dir,
    )

    risk = RiskAgent(
        max_position_frac = args.max_position_frac,
        daily_loss_limit  = args.daily_loss_limit,
        max_drawdown      = args.max_drawdown,
        leverage          = float(args.leverage),
    )

    exchange = Exchange(symbol=args.symbol, paper=False, leverage=args.leverage)

    loop = TradingLoop(
        combiner        = combiner,
        risk            = risk,
        exchange        = exchange,
        initial_capital = args.capital,
        threshold       = args.threshold,
        symbol          = args.symbol,
        log_dir         = args.log_dir,
    )

    print("\nStarting live loop (Ctrl+C to stop)...\n")
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("\nStopped. Cancelling open orders...")
        exchange.cancel_all()
        print("Done.")


if __name__ == "__main__":
    main()
