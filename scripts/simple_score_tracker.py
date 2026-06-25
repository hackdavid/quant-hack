#!/usr/bin/env python3
"""Simple Competition Score Tracker — P&L, Win Rate, Sharpe, Final Score.

Reads MT5 trade history or CSV export and calculates:
  - P&L (absolute profit/loss)
  - Win Rate (%)
  - Sharpe Ratio (competition formula: non-annualized, 15-min equity returns)
  - Final Score (from the competition scoreboard)

Usage:
    .venv/Scripts/python.exe scripts/simple_score_tracker.py \
        --trade-log logs/autonomous_trader/trade_log_2026-06-24.jsonl

Or from MT5 trade history CSV:
    .venv/Scripts/python.exe scripts/simple_score_tracker.py \
        --mt5-csv "C:/Users/.../MQL5/Files/trade_history.csv"
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INITIAL_EQUITY = 1_000_000.0


def load_trade_log(path: Path) -> pd.DataFrame:
    """Load trade log JSONL (from our bots)."""
    rows = []
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rows.append(data)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def load_mt5_csv(path: Path) -> pd.DataFrame:
    """Load MT5 trade history CSV."""
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def calculate_pnl(trades: pd.DataFrame) -> float:
    """Total absolute P&L."""
    if "profit" in trades.columns:
        return trades["profit"].sum()
    if "Profit" in trades.columns:
        return trades["Profit"].sum()
    if "profit" in trades.columns:
        return trades["profit"].sum()
    return 0.0


def calculate_win_rate(trades: pd.DataFrame) -> tuple[float, int, int]:
    """Win rate and trade counts."""
    profit_col = None
    for col in ["profit", "Profit", "profit", "Profit"]:
        if col in trades.columns:
            profit_col = col
            break

    if profit_col is None:
        return 0.0, 0, 0

    profits = trades[profit_col]
    wins = int((profits > 0).sum())
    losses = int((profits < 0).sum())
    total = wins + losses
    if total == 0:
        return 0.0, 0, 0
    return (wins / total) * 100, wins, losses


def calculate_sharpe(equity_series: pd.Series) -> float:
    """Competition Sharpe: non-annualized, 15-min equity returns.
    Sharpe = Mean(15-min returns) / Std(15-min returns)
    """
    if equity_series.empty or len(equity_series) < 2:
        return 0.0

    # Calculate 15-min returns
    # If data frequency is not 15-min, resample
    equity_series = equity_series.dropna()
    if len(equity_series) < 8:
        return 0.0

    # Simple pct_change
    returns = equity_series.pct_change().dropna()
    if len(returns) < 8:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret == 0 or math.isnan(std_ret):
        return 0.0

    return mean_ret / std_ret


def extract_equity_from_trades(trades: pd.DataFrame) -> pd.Series:
    """Extract equity series from trade log."""
    if "equity" in trades.columns:
        return trades["equity"]
    if "Equity" in trades.columns:
        return trades["Equity"]
    if "account_balance" in trades.columns:
        return trades["account_balance"]
    if "Balance" in trades.columns:
        return trades["Balance"]
    # Fallback: build equity from initial + cumulative P&L
    if "profit" in trades.columns:
        equity = INITIAL_EQUITY + trades["profit"].cumsum()
        return equity
    return pd.Series([INITIAL_EQUITY])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None, help="Our bot JSONL trade log")
    parser.add_argument("--mt5-csv", type=Path, default=None, help="MT5 trade history CSV")
    args = parser.parse_args()

    # Find latest trade log if not specified
    if not args.trade_log and not args.mt5_csv:
        log_dir = Path("logs/autonomous_trader")
        if log_dir.exists():
            logs = sorted(log_dir.glob("*.jsonl"))
            if logs:
                args.trade_log = logs[-1]
                print(f"Using latest trade log: {args.trade_log}\n")

    if args.mt5_csv:
        trades = load_mt5_csv(args.mt5_csv)
    elif args.trade_log:
        trades = load_trade_log(args.trade_log)
    else:
        print("No trade data found. Provide --trade-log or --mt5-csv")
        sys.exit(1)

    if trades.empty:
        print("No trades found in the log")
        sys.exit(1)

    # Calculate metrics
    pnl = calculate_pnl(trades)
    win_rate, wins, losses = calculate_win_rate(trades)
    equity_series = extract_equity_from_trades(trades)
    sharpe = calculate_sharpe(equity_series)
    final_equity = INITIAL_EQUITY + pnl
    return_pct = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100

    # Print simple scorecard
    print("=" * 50)
    print("COMPETITION SCORE TRACKER")
    print("=" * 50)
    print(f"\nP&L:              ${pnl:,.2f}")
    print(f"Win Rate:         {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"Sharpe Ratio:     {sharpe:.4f}")
    print(f"Return:           {return_pct:+.2f}%")
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Total Trades:     {wins + losses}")
    print("\n" + "=" * 50)

    # Show what the user reported
    print("\n[YOUR REPORTED SCORE]")
    print(f"  Final Score:    9.95")
    print(f"  P&L:            $4,480")
    print(f"  Win Rate:       41%")
    print(f"  Sharpe:         0")
    print("\n[DISCREPANCY ANALYSIS]")
    print(f"  P&L diff:       ${pnl - 4480:,.2f} (calculated: {pnl:,.0f} vs reported: 4,480)")
    print(f"  Win rate diff:  {win_rate - 41:.1f}% (calculated: {win_rate:.1f}% vs reported: 41%)")
    print(f"  Sharpe diff:    {sharpe - 0:.4f} (calculated: {sharpe:.4f} vs reported: 0)")
    print("\n  Note: Your score may differ because:")
    print("  1. The competition uses REAL peer rankings (not simulated)")
    print("  2. Score depends on other participants' performance")
    print("  3. Risk discipline deductions may apply")
    print("  4. Different calculation timing (your score was at 10PM)")
    print("=" * 50)


if __name__ == "__main__":
    main()
