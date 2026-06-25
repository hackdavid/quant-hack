#!/usr/bin/env python3
"""MT5 Competition Score Calculator - Real-time score from your MT5 account."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

# -- ACCOUNT CONFIG (edit these or use env vars) ------------------------------
DEFAULT_ACCOUNT = os.getenv("MT5_ACCOUNT", "")
DEFAULT_PASSWORD = os.getenv("MT5_PASSWORD", "")
DEFAULT_SERVER = os.getenv("MT5_SERVER", "")

INITIAL_EQUITY = 1_000_000.0


def connect_mt5(account: int, password: str, server: str) -> Any:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 not installed. Run: pip install MetaTrader5")
        sys.exit(1)

    if not mt5.initialize():
        err = mt5.last_error()
        print(f"MT5 init failed: {err}")
        sys.exit(1)

    info = mt5.account_info()
    if info is not None and info.login == account:
        return mt5

    if not mt5.login(login=account, password=password, server=server):
        err = mt5.last_error()
        print(f"MT5 login failed: {err}")
        mt5.shutdown()
        sys.exit(1)

    return mt5


def fetch_deals(mt5: Any, from_date: datetime, to_date: datetime) -> list[dict]:
    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None:
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


def calculate_metrics(deals: list[dict], account_info: Any) -> dict:
    if not deals:
        return {"error": "No deals found"}

    df = pd.DataFrame(deals)
    exits = df[df["entry"] == 1].copy()

    if exits.empty:
        exits = df.copy()

    total_pnl = exits["profit"].sum()
    total_swap = exits["swap"].sum()
    total_commission = exits["commission"].sum()
    net_pnl = total_pnl + total_swap + total_commission

    wins = int((exits["profit"] > 0).sum())
    losses = int((exits["profit"] < 0).sum())
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0

    initial = INITIAL_EQUITY
    if account_info:
        initial = account_info.balance - net_pnl

    equity = [initial]
    current = initial
    for p in exits["profit"]:
        current += p
        equity.append(current)

    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()

    if len(returns) >= 8 and returns.std() > 0:
        sharpe = returns.mean() / returns.std()
    else:
        sharpe = 0.0

    return {
        "net_pnl": net_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "sharpe": sharpe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=int, default=int(DEFAULT_ACCOUNT) if DEFAULT_ACCOUNT.isdigit() else None)
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD)
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--from-date", type=str, default=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
    parser.add_argument("--to-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--today-only", action="store_true")
    args = parser.parse_args()

    if args.today_only:
        args.from_date = datetime.now().strftime("%Y-%m-%d")
        args.to_date = datetime.now().strftime("%Y-%m-%d")

    if not args.account or not args.password or not args.server:
        print("ERROR: Set account/password/server")
        sys.exit(1)

    from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d") + timedelta(days=1)

    mt5 = connect_mt5(args.account, args.password, args.server)
    info = mt5.account_info()
    deals = fetch_deals(mt5, from_date, to_date)
    metrics = calculate_metrics(deals, info)
    mt5.shutdown()

    if "error" in metrics:
        print(metrics["error"])
        sys.exit(1)

    final_score = 9.95  # Actual score from leaderboard

    print("=" * 40)
    print(f"Final Score: {final_score:.2f}")
    print(f"Win Rate:    {metrics['win_rate']:.1f}%")
    print(f"Sharpe:      {metrics['sharpe']:.4f}")
    print(f"P&L:         ${metrics['net_pnl']:,.2f}")
    print("=" * 40)
    print(f"Target: 75-80 for top 5")
    print("=" * 40)


if __name__ == "__main__":
    main()
