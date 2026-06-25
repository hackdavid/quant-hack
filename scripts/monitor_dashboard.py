#!/usr/bin/env python3
"""Monitor Dashboard — Real-time trading monitor.

Shows:
  - Current positions
  - P&L
  - Agent signals
  - Market trend
  - Competition score

Usage:
    .venv\\Scripts\\python.exe scripts\\monitor_dashboard.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from intraday.trader.mt5_wrapper import MT5TradingWrapper

SYMBOL = "BTCUSDT"


def print_header():
    print("=" * 70)
    print("  KRONOS MONITOR DASHBOARD")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)


def print_positions(wrapper):
    positions = wrapper.get_positions(SYMBOL)
    if not positions:
        print("\n  [NO OPEN POSITIONS]")
        return

    print(f"\n  OPEN POSITIONS: {len(positions)}")
    print("-" * 70)
    total_pnl = 0.0
    for p in positions:
        total_pnl += p.profit
        print(f"  #{p.ticket} | {p.side.upper():6s} | {p.volume:.1f} lots | Entry: ${p.open_price:.2f}")
        print(f"         Current: ${p.current_price:.2f} | PnL: ${p.profit:+.2f}")
    print("-" * 70)
    print(f"  TOTAL P&L: ${total_pnl:+.2f}")


def print_account(wrapper):
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info:
            print(f"\n  ACCOUNT")
            print("-" * 70)
            print(f"  Balance:      ${info.balance:,.2f}")
            print(f"  Equity:       ${info.equity:,.2f}")
            print(f"  Margin:       ${info.margin:,.2f}")
            print(f"  Free Margin:  ${info.margin_free:,.2f}")
            print(f"  Margin Level: {info.margin_level:.2f}%")
    except Exception:
        pass


def print_market(wrapper):
    try:
        import MetaTrader5 as mt5
        mt5_sym = "BTCUSD"
        tick = mt5.symbol_info_tick(mt5_sym)
        if tick:
            print(f"\n  MARKET PRICE")
            print("-" * 70)
            print(f"  Bid: ${tick.bid:.2f}")
            print(f"  Ask: ${tick.ask:.2f}")
            print(f"  Spread: ${tick.ask - tick.bid:.2f}")
    except Exception:
        pass


def print_trade_state():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from trade_state import read_state
        state = read_state()
        print(f"\n  BOT STATE")
        print("-" * 70)
        print(f"  Running:      {state.is_running}")
        print(f"  Has Position: {state.has_position}")
        print(f"  Active Tickets: {state.active_ticket_count}")
        if state.has_position:
            print(f"  Side:         {state.position_side}")
            print(f"  Lots:         {state.position_lots}")
            print(f"  Open Price:   ${state.position_open_price:.2f}")
            print(f"  Current P&L:  ${state.position_profit:.2f}")
            print(f"  TP:           ${state.current_tp:.2f}")
            print(f"  SL:           ${state.current_sl:.2f}")
            print(f"  Hold:         {state.elapsed_seconds:.0f}s / {state.current_hold:.0f}s")
            print(f"  Signal:       {state.signal_label} ({state.signal_score:.0f}/100)")
        if state.command:
            print(f"  Command:      {state.command}={state.command_value}")
    except Exception as e:
        print(f"\n  BOT STATE: Unavailable ({e})")


def print_competition_stats():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from mt5_competition_score import get_competition_stats
        stats = get_competition_stats()
        print(f"\n  COMPETITION STATS")
        print("-" * 70)
        print(f"  Final Score:  {stats.get('final_score', 0):.2f}")
        print(f"  Win Rate:     {stats.get('win_rate', 0):.1f}%")
        print(f"  Sharpe:       {stats.get('sharpe', 0):.4f}")
        print(f"  P&L:          ${stats.get('pnl', 0):,.2f}")
        print(f"  Trades:       {stats.get('trades', 0)}")
    except Exception:
        pass


def main():
    account = int(os.getenv("MT5_ACCOUNT", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")

    if not account or not password or not server:
        print("Error: Set MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER in .env")
        sys.exit(1)

    wrapper = MT5TradingWrapper(
        account_id=account,
        password=password,
        server=server,
        magic=999999,
    )

    if not wrapper.connect():
        print("[red]Failed to connect to MT5[/red]")
        sys.exit(1)

    print("[green]MT5 Connected[/green]")
    print("[cyan]Press Ctrl+C to stop monitoring[/cyan]\n")

    try:
        while True:
            print_header()
            print_positions(wrapper)
            print_account(wrapper)
            print_market(wrapper)
            print_trade_state()
            print_competition_stats()
            print("\n" + "=" * 70)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[yellow]Monitoring stopped[/yellow]")
        wrapper.shutdown()


if __name__ == "__main__":
    main()
