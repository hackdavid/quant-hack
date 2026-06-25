#!/usr/bin/env python3
"""Fetch real trade history from MT5 and calculate competition score.

Usage:
    .venv/Scripts/python.exe scripts/fetch_mt5_score.py \
        --account YOUR_ACCOUNT \
        --password "YOUR_PASSWORD" \
        --server "YOUR_SERVER" \
        --from-date "2026-06-20"
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def connect_mt5(account: int, password: str, server: str) -> Any:
    """Connect to MT5 and return the module."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 not installed. Run: pip install MetaTrader5")
        sys.exit(1)

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    info = mt5.account_info()
    if info is not None and info.login == account:
        print(f"Already logged in: {info.login}, balance: ${info.balance:,.2f}")
        return mt5

    if not mt5.login(login=account, password=password, server=server):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    info = mt5.account_info()
    print(f"Connected: {info.login}, balance: ${info.balance:,.2f}")
    return mt5


def fetch_deals(mt5: Any, from_date: datetime, to_date: datetime) -> list[dict]:
    """Fetch closed deals (trades) from MT5 history."""
    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None:
        print(f"No deals found: {mt5.last_error()}")
        return []

    results = []
    for d in deals:
        results.append({
            "ticket": d.ticket,
            "order": d.order,
            "symbol": d.symbol,
            "type": "buy" if d.type == 0 else "sell",
            "entry": d.entry,
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "swap": d.swap,
            "commission": d.commission,
            "time": datetime.fromtimestamp(d.time),
            "comment": d.comment,
        })
    return results


def calculate_metrics(deals: list[dict]) -> dict:
    """Calculate competition metrics from MT5 deals."""
    if not deals:
        return {"error": "No deals found"}

    df = pd.DataFrame(deals)

    # Filter actual trades (entry == 1 = market exit, we need the position close)
    # MT5 deals: entry=0 (in), entry=1 (out)
    trades_df = df[df["entry"] == 1].copy()

    if trades_df.empty:
        # Fallback: use all deals
        trades_df = df.copy()

    total_pnl = trades_df["profit"].sum()
    wins = int((trades_df["profit"] > 0).sum())
    losses = int((trades_df["profit"] < 0).sum())
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0

    # Build equity curve from profit
    initial = 1_000_000.0
    equity = [initial]
    current = initial
    for p in trades_df["profit"]:
        current += p
        equity.append(current)

    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()

    if len(returns) >= 8 and returns.std() > 0:
        sharpe = returns.mean() / returns.std()
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (peak - equity_series) / peak
    max_dd = drawdown.max()

    final_equity = initial + total_pnl
    return_pct = (final_equity - initial) / initial * 100

    return {
        "total_pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "final_equity": final_equity,
        "return_pct": return_pct,
        "total_trades": total,
    }


def print_scorecard(metrics: dict):
    """Print competition scorecard."""
    print("\n" + "=" * 60)
    print("MT5 COMPETITION SCORECARD")
    print("=" * 60)

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return

    print(f"\nP&L:              ${metrics['total_pnl']:,.2f}")
    print(f"Win Rate:         {metrics['win_rate']:.1f}% ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"Sharpe Ratio:     {metrics['sharpe']:.4f}")
    print(f"Max Drawdown:     {metrics['max_dd']*100:.2f}%")
    print(f"Return:           {metrics['return_pct']:+.2f}%")
    print(f"Final Equity:     ${metrics['final_equity']:,.2f}")
    print(f"Total Trades:     {metrics['total_trades']}")

    print("\n" + "=" * 60)
    print("YOUR REPORTED SCORE")
    print("=" * 60)
    print(f"Final Score:    9.95")
    print(f"P&L:            $4,480")
    print(f"Win Rate:       41%")
    print(f"Sharpe:         0")

    print("\n" + "=" * 60)
    print("DIFFERENCE")
    print("=" * 60)
    print(f"P&L:            ${metrics['total_pnl'] - 4480:,.2f}")
    print(f"Win Rate:       {metrics['win_rate'] - 41:.1f}%")
    print(f"Sharpe:         {metrics['sharpe'] - 0:.4f}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=int, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--from-date", type=str, default="2026-06-20")
    parser.add_argument("--to-date", type=str, default=None)
    args = parser.parse_args()

    from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date else datetime.now()

    print(f"Fetching MT5 history from {from_date.date()} to {to_date.date()}")
    print(f"Account: {args.account}, Server: {args.server}")

    mt5 = connect_mt5(args.account, args.password, args.server)
    deals = fetch_deals(mt5, from_date, to_date)
    print(f"Found {len(deals)} deals")

    metrics = calculate_metrics(deals)
    print_scorecard(metrics)

    mt5.shutdown()


if __name__ == "__main__":
    main()
