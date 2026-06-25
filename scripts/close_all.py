#!/usr/bin/env python3
"""Emergency close-all script.

Closes every open BTCUSDT position on MT5 immediately.
Run this if the bot goes haywire or you need to stop all trading.

Usage:
    .venv\\Scripts\\python.exe scripts\\close_all.py \\
        --mt5-account 10408 --mt5-password "YOUR_PASSWORD" --mt5-server "YOUR_SERVER"

Or read from .env:
    .venv\\Scripts\\python.exe scripts\\close_all.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from intraday.trader.mt5_wrapper import MT5TradingWrapper


def main():
    parser = argparse.ArgumentParser(description="Emergency close all open positions")
    parser.add_argument("--mt5-account", type=int, default=int(os.getenv("MT5_ACCOUNT", "0")))
    parser.add_argument("--mt5-password", type=str, default=os.getenv("MT5_PASSWORD", ""))
    parser.add_argument("--mt5-server", type=str, default=os.getenv("MT5_SERVER", ""))
    args = parser.parse_args()

    if not args.mt5_account or not args.mt5_password or not args.mt5_server:
        print("[red]Error: MT5 credentials required.[/red]")
        print("  Pass via CLI: --mt5-account X --mt5-password Y --mt5-server Z")
        print("  Or set in .env: MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER")
        sys.exit(1)

    print("=" * 60)
    print(f"  EMERGENCY CLOSE ALL — {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    wrapper = MT5TradingWrapper(
        account_id=args.mt5_account,
        password=args.mt5_password,
        server=args.mt5_server,
        magic=999999,
    )

    if not wrapper.connect():
        print("[red]MT5 connection failed[/red]")
        sys.exit(1)

    positions = wrapper.get_positions("BTCUSDT")
    print(f"\nOpen positions found: {len(positions)}")

    if not positions:
        print("  Nothing to close.")
        wrapper.shutdown()
        sys.exit(0)

    total_pnl = 0.0
    for p in positions:
        total_pnl += p.profit
        print(f"\n  #{p.ticket}: {p.side} {p.volume} lot @ {p.open_price:.2f} | PnL=${p.profit:.2f}")
        print("  Closing...")
        result = wrapper.close_position(p.ticket)
        if result.success:
            print(f"  [green]Closed #{result.ticket} | {result.comment}[/green]")
        else:
            print(f"  [red]Failed: {result}[/red]")

    wrapper.shutdown()

    print("\n" + "=" * 60)
    print(f"  DONE — Total P&L: ${total_pnl:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
