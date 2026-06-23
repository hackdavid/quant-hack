#!/usr/bin/env python3
"""Paper trading: live Binance WebSocket data, no real orders placed.

Prerequisites:
  - Trained model(s) in models/

Usage:
    python scripts/run_paper_trade.py \\
        --transformer-dir models/transformer/20260623T132957Z \\
        --lgb-dir models/lgb \\
        --capital 10000 \\
        --threshold 0.55

Ctrl+C to stop. All decisions written to logs/trader/trade_log_*.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from intraday.signal.combiner import SignalCombiner
from intraday.risk.agent import RiskAgent
from intraday.trader.exchange import Exchange
from intraday.trader.loop import TradingLoop


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-trade using live Binance 5-min bars")
    p.add_argument("--transformer-dir",  default=None,
                   help="Path to transformer run dir (contains best.pt)")
    p.add_argument("--lgb-dir",          default=None,
                   help="Path to LGB model dir (contains lgb_model.txt)")
    p.add_argument("--lgb-ensemble-dir", default=None,
                   help="Path to GBM ensemble dir (contains lgb_gbdt.txt)")
    p.add_argument("--capital",          type=float, default=10_000.0,
                   help="Starting capital in USD")
    p.add_argument("--threshold",        type=float, default=0.55,
                   help="Min prob to enter long (1-t for short)")
    p.add_argument("--max-position-frac",type=float, default=0.20,
                   help="Max fraction of capital per trade")
    p.add_argument("--daily-loss-limit", type=float, default=0.02,
                   help="Stop trading if daily loss exceeds this fraction")
    p.add_argument("--max-drawdown",     type=float, default=0.05,
                   help="Stop trading if drawdown from peak exceeds this fraction")
    p.add_argument("--symbol",           default="BTCUSDT")
    p.add_argument("--log-dir",          default="logs/trader")
    args = p.parse_args()

    if not args.transformer_dir and not args.lgb_dir and not args.lgb_ensemble_dir:
        p.error("Provide at least one of --transformer-dir / --lgb-dir / --lgb-ensemble-dir")

    print("=" * 60)
    print("  Paper Trading Mode (no real orders)")
    print("=" * 60)
    print(f"  Symbol:    {args.symbol}")
    print(f"  Capital:   ${args.capital:,.2f}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Max pos:   {args.max_position_frac*100:.0f}% of capital")
    print(f"  Daily lim: {args.daily_loss_limit*100:.0f}%")

    combiner = SignalCombiner(
        transformer_dir  = args.transformer_dir,
        lgb_dir          = args.lgb_dir,
        lgb_ensemble_dir = args.lgb_ensemble_dir,
    )

    risk = RiskAgent(
        max_position_frac = args.max_position_frac,
        daily_loss_limit  = args.daily_loss_limit,
        max_drawdown      = args.max_drawdown,
    )

    exchange = Exchange(symbol=args.symbol, paper=True)

    loop = TradingLoop(
        combiner        = combiner,
        risk            = risk,
        exchange        = exchange,
        initial_capital = args.capital,
        threshold       = args.threshold,
        symbol          = args.symbol,
        log_dir         = args.log_dir,
    )

    print("\nStarting loop (Ctrl+C to stop)...\n")
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
