#!/usr/bin/env python3
"""Calculate competition score from live trading logs.

Reads the trade log JSONL and computes:
  - Return, Drawdown, Sharpe, Win Rate, P&L
  - Final Score (per competition rules)

Usage:
    .venv/Scripts/python.exe scripts/calculate_live_score.py
        --trade-log logs/autonomous_trader/trade_log_2026-06-24.jsonl
        --n-participants 500
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────
INITIAL_EQUITY = 1_000_000.0
WEIGHT_RETURN = 0.70
WEIGHT_DRAWDOWN = 0.15
WEIGHT_SHARPE = 0.10
WEIGHT_RISK = 0.05


def load_trade_log(path: Path) -> list[dict]:
    """Load trade log JSONL."""
    trades = []
    if not path.exists():
        print(f"Error: Trade log not found: {path}")
        return trades
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return trades


def extract_equity_series(trades: list[dict]) -> pd.Series:
    """Extract equity over time from trade log."""
    timestamps = []
    equities = []
    balances = []
    open_pls = []
    max_dds = []

    for t in trades:
        ts = t.get("ts", "")
        equity = t.get("equity", INITIAL_EQUITY)
        balance = t.get("account_balance", INITIAL_EQUITY)
        open_pl = t.get("open_pl", 0.0)
        max_dd = t.get("max_drawdown_pct", 0.0)

        timestamps.append(ts)
        equities.append(equity)
        balances.append(balance)
        open_pls.append(open_pl)
        max_dds.append(max_dd)

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "equity": equities,
        "balance": balances,
        "open_pl": open_pls,
        "max_dd": max_dds,
    })
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_actual_trades(trades: list[dict]) -> list[dict]:
    """Extract actual executed trades (not HOLD/WAIT)."""
    executed = []
    for t in trades:
        action = t.get("action", "")
        if action in ("BUY", "SELL"):
            # Check if executed successfully
            llm_out = t.get("llm_output", {})
            if isinstance(llm_out, str):
                try:
                    llm_out = json.loads(llm_out)
                except:
                    llm_out = {}

            # Check reason for execution result
            reason = t.get("reason", "")
            if "executed" in reason:
                # Try to parse profit
                profit = 0.0
                if "success" in reason:
                    if "False" in reason or "failed" in reason.lower():
                        continue  # Skip failed trades

                executed.append({
                    "action": action,
                    "profit": profit,
                    "volume": t.get("position_size", 0.0),
                    "entry_price": t.get("bar_close", 0.0),
                    "symbol": t.get("symbol", "BTCUSD"),
                    "side": "buy" if action == "BUY" else "sell",
                    "timestamp": t.get("ts", ""),
                })

    return executed


def calculate_return(final_equity: float) -> float:
    return (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY


def calculate_max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    peak = equity_series.expanding().max()
    drawdown = (peak - equity_series) / peak
    return drawdown.max()


def calculate_sharpe(equity_series: pd.Series) -> float:
    if equity_series.empty or len(equity_series) < 2:
        return 0.0
    returns = equity_series.pct_change().dropna()
    if len(returns) < 8:
        return 0.0
    mean = returns.mean()
    std = returns.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return mean / std


def simulate_ranks(return_pct: float, max_dd: float, sharpe: float, n: int = 500) -> tuple[float, float, float]:
    np.random.seed(42)
    peers_ret = np.random.normal(0.0, 0.05, n - 1)
    peers_ret = np.append(peers_ret, return_pct)
    sorted_ret = sorted(peers_ret, reverse=True)
    rank_ret = sorted_ret.index(return_pct) + 1
    ret_rank = 100.0 * (n - rank_ret) / (n - 1)

    peers_dd = np.random.exponential(0.10, n - 1)
    peers_dd = np.append(peers_dd, max_dd)
    sorted_dd = sorted(peers_dd)
    rank_dd = sorted_dd.index(max_dd) + 1
    dd_rank = 100.0 * (n - rank_dd) / (n - 1)

    peers_sharpe = np.random.normal(0.0, 0.5, n - 1)
    peers_sharpe = np.append(peers_sharpe, sharpe)
    sorted_sharpe = sorted(peers_sharpe, reverse=True)
    rank_sharpe = sorted_sharpe.index(sharpe) + 1
    sharpe_rank = 100.0 * (n - rank_sharpe) / (n - 1)

    return ret_rank, dd_rank, sharpe_rank


def calculate_final_score(ret_rank: float, dd_rank: float, sharpe_rank: float, risk: float) -> float:
    return WEIGHT_RETURN * ret_rank + WEIGHT_DRAWDOWN * dd_rank + WEIGHT_SHARPE * sharpe_rank + WEIGHT_RISK * risk


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--n-participants", type=int, default=500)
    args = parser.parse_args()

    if not args.trade_log:
        # Find latest trade log
        log_dir = Path("logs/autonomous_trader")
        if log_dir.exists():
            logs = sorted(log_dir.glob("*.jsonl"))
            if logs:
                args.trade_log = logs[-1]
                print(f"Using latest trade log: {args.trade_log}")
            else:
                print("No trade logs found")
                sys.exit(1)
        else:
            print("No trade logs found")
            sys.exit(1)

    print("=" * 70)
    print("COMPETITION SCORE CALCULATOR")
    print("=" * 70)
    print(f"\nReading: {args.trade_log}")

    trades = load_trade_log(args.trade_log)
    print(f"Total log entries: {len(trades)}")

    if not trades:
        print("No trades found")
        sys.exit(1)

    # Extract equity series
    equity_df = extract_equity_series(trades)
    final_equity = equity_df["equity"].iloc[-1]
    return_pct = calculate_return(final_equity)
    max_dd = calculate_max_drawdown(equity_df["equity"])
    sharpe = calculate_sharpe(equity_df["equity"])

    # Extract actual trades
    executed = extract_actual_trades(trades)
    n_trades = len(executed)
    wins = sum(1 for t in executed if t.get("profit", 0) > 0)
    losses = n_trades - wins
    win_rate = (wins / n_trades * 100) if n_trades > 0 else 0
    total_pnl = final_equity - INITIAL_EQUITY

    # Calculate ranks
    ret_rank, dd_rank, sharpe_rank = simulate_ranks(return_pct, max_dd, sharpe, args.n_participants)
    risk_discipline = 100.0  # Simplified
    final_score = calculate_final_score(ret_rank, dd_rank, sharpe_rank, risk_discipline)

    # Print scorecard
    print("\n" + "=" * 70)
    print("COMPETITION SCORECARD")
    print("=" * 70)
    print(f"\n[PERFORMANCE] Trading Performance")
    print(f"  Initial Equity:      ${INITIAL_EQUITY:,.2f}")
    print(f"  Final Equity:        ${final_equity:,.2f}")
    print(f"  Total P&L:           ${total_pnl:,.2f}")
    print(f"  Return:              {return_pct*100:+.2f}%")
    print(f"  Win Rate:            {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Trades Executed:     {n_trades}")
    print(f"  Log Entries:         {len(trades)}")

    print(f"\n[RISK] Risk Metrics")
    print(f"  Max Drawdown:        {max_dd*100:.2f}%")
    print(f"  Sharpe Ratio:        {sharpe:.4f}")

    print(f"\n[RANKING] Ranking (simulated vs {args.n_participants} peers)")
    print(f"  Return Rank:         {ret_rank:.1f}/100")
    print(f"  Drawdown Rank:       {dd_rank:.1f}/100")
    print(f"  Sharpe Rank:         {sharpe_rank:.1f}/100")
    print(f"  Risk Discipline:     {risk_discipline:.1f}/100")

    print(f"\n[FINAL SCORE]")
    print(f"  = {WEIGHT_RETURN*100:.0f}% x Return Rank")
    print(f"  + {WEIGHT_DRAWDOWN*100:.0f}% x Drawdown Rank")
    print(f"  + {WEIGHT_SHARPE*100:.0f}% x Sharpe Rank")
    print(f"  + {WEIGHT_RISK*100:.0f}% x Risk Discipline")
    print(f"  = {final_score:.2f} / 100")

    print(f"\n[SHARPE PRIZE]")
    if n_trades >= 30:
        print(f"  [OK] Trade count: {n_trades} (required: 30+)")
    else:
        print(f"  [NEED MORE] Trade count: {n_trades} (need 30+ for Sharpe prize)")

    print(f"\n[ANALYSIS]")
    print(f"  1. Return (70% weight): {ret_rank:.1f}/100")
    if ret_rank < 50:
        print(f"     -> NEEDS IMPROVEMENT: Increase P&L with more winning trades")
    else:
        print(f"     -> GOOD")

    print(f"  2. Drawdown (15% weight): {dd_rank:.1f}/100")
    if dd_rank < 50:
        print(f"     -> NEEDS IMPROVEMENT: Reduce max drawdown")
    else:
        print(f"     -> GOOD")

    print(f"  3. Sharpe (10% weight): {sharpe_rank:.1f}/100")
    if sharpe_rank < 50:
        print(f"     -> NEEDS IMPROVEMENT: More consistent returns")
    else:
        print(f"     -> GOOD")

    print(f"\n[TO IMPROVE SCORE]")
    print(f"  1. Trade more when signals are clear (target: 30+ trades)")
    print(f"  2. Tighten SL to reduce drawdown")
    print(f"  3. Use trailing stop to lock in profits")
    print(f"  4. Keep margin < 90%, leverage < 28x")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
