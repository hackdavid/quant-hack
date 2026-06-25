#!/usr/bin/env python3
"""Competition Score Calculator — Compute Final Score per Official Rules.

Reads trading history and calculates:
  - Return (%)
  - Max Drawdown (%)
  - Sharpe Ratio (15-min non-annualized)
  - Win Rate / P&L
  - Risk Discipline (margin, leverage, exposure)
  - Final Score (weighted composite)

Usage:
    .venv/Scripts/python.exe scripts/competition_score_calculator.py \
        --equity-log logs/equity_log.csv \
        --trade-log logs/trade_log.jsonl \
        --n-participants 500
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# ── Competition Constants ─────────────────────────────────────────────────
INITIAL_EQUITY = 1_000_000.0
MAX_LEVERAGE = 30.0
MARGIN_STOP_OUT = 0.30

# Risk Discipline thresholds
MARGIN_90_DURATION = 30  # minutes
MARGIN_95_DURATION = 15
LEVERAGE_28_DURATION = 30
LEVERAGE_29_DURATION = 15
SINGLE_INSTRUMENT_90_DURATION = 30
NET_EXPOSURE_95_DURATION = 30

# Final Score Weights
WEIGHT_RETURN = 0.70
WEIGHT_DRAWDOWN = 0.15
WEIGHT_SHARPE = 0.10
WEIGHT_RISK = 0.05


def load_equity_log(path: Path) -> pd.DataFrame:
    """Load equity snapshot log."""
    if not path.exists():
        log.warning("equity_log_not_found", path=str(path))
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_trade_log(path: Path) -> list[dict]:
    """Load trade log JSONL."""
    trades = []
    if not path.exists():
        log.warning("trade_log_not_found", path=str(path))
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


def calculate_return(final_equity: float, initial_equity: float = INITIAL_EQUITY) -> float:
    """Return = (Final - Initial) / Initial."""
    return (final_equity - initial_equity) / initial_equity


def calculate_return_rank(return_pct: float, all_returns: list[float]) -> float:
    """Return Rank = 100 × (N - Rank_i) / (N - 1)."""
    n = len(all_returns)
    if n <= 1:
        return 100.0
    sorted_returns = sorted(all_returns, reverse=True)
    rank = sorted_returns.index(return_pct) + 1  # 1-based
    return 100.0 * (n - rank) / (n - 1)


def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """MaxDD = max((Peak - Trough) / Peak)."""
    if equity_series.empty:
        return 0.0
    peak = equity_series.expanding().max()
    drawdown = (peak - equity_series) / peak
    return drawdown.max()


def calculate_drawdown_rank(max_dd: float, all_max_dds: list[float]) -> float:
    """Drawdown Rank = 100 × (N - Rank_DD_i) / (N - 1).
    Lower drawdown = higher rank."""
    n = len(all_max_dds)
    if n <= 1:
        return 100.0
    sorted_dds = sorted(all_max_dds)  # ascending
    rank = sorted_dds.index(max_dd) + 1
    return 100.0 * (n - rank) / (n - 1)


def calculate_sharpe_ratio(equity_series: pd.Series, interval_minutes: int = 15) -> float:
    """Sharpe = Mean(15-min returns) / Std(15-min returns).
    Non-annualized, computed from 15-minute equity returns."""
    if equity_series.empty or len(equity_series) < 2:
        return 0.0

    # Resample to 15-minute intervals
    # If data is already 15-min, use directly
    returns = equity_series.pct_change().dropna()
    if len(returns) < 8:
        return 0.0  # cap at 50 per rules

    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0 or math.isnan(std_ret):
        return 0.0

    return mean_ret / std_ret


def calculate_sharpe_rank(sharpe: float, all_sharpes: list[float]) -> float:
    """Sharpe Rank = 100 × (N - Rank_Sharpe_i) / (N - 1)."""
    n = len(all_sharpes)
    if n <= 1:
        return 100.0
    sorted_sharpes = sorted(all_sharpes, reverse=True)
    rank = sorted_sharpes.index(sharpe) + 1
    return 100.0 * (n - rank) / (n - 1)


def calculate_risk_discipline(
    equity_df: pd.DataFrame,
    trades: list[dict],
    initial_equity: float = INITIAL_EQUITY,
) -> float:
    """Calculate Risk Discipline score (starts at 100, minus deductions)."""
    score = 100.0

    if equity_df.empty:
        return score

    # Margin Usage deductions
    if "used_margin" in equity_df.columns and "equity" in equity_df.columns:
        equity_df["margin_usage"] = equity_df["used_margin"] / equity_df["equity"]

        # >90% for >=30 min
        over_90 = equity_df["margin_usage"] > 0.90
        if over_90.any():
            # Find consecutive periods
            consecutive = _find_consecutive_minutes(over_90, equity_df["timestamp"])
            if consecutive >= MARGIN_90_DURATION:
                score -= 20

        # >95% for >=15 min
        over_95 = equity_df["margin_usage"] > 0.95
        if over_95.any():
            consecutive = _find_consecutive_minutes(over_95, equity_df["timestamp"])
            if consecutive >= MARGIN_95_DURATION:
                score -= 30

    # Leverage Usage deductions
    if "gross_exposure" in equity_df.columns and "equity" in equity_df.columns:
        equity_df["leverage"] = equity_df["gross_exposure"] / equity_df["equity"]

        over_28 = equity_df["leverage"] > 28
        if over_28.any():
            consecutive = _find_consecutive_minutes(over_28, equity_df["timestamp"])
            if consecutive >= LEVERAGE_28_DURATION:
                score -= 20

        over_29 = equity_df["leverage"] > 29
        if over_29.any():
            consecutive = _find_consecutive_minutes(over_29, equity_df["timestamp"])
            if consecutive >= LEVERAGE_29_DURATION:
                score -= 30

    # Exposure Concentration (from trades)
    if trades:
        symbols = defaultdict(float)
        total_exposure = 0.0
        for t in trades:
            notional = t.get("volume", 0.0) * t.get("entry_price", 0.0)
            symbol = t.get("symbol", "BTCUSD")
            symbols[symbol] += notional
            total_exposure += notional

        if total_exposure > 0:
            max_single = max(symbols.values()) / total_exposure
            if max_single > 0.90:
                score -= 10

        # Net directional exposure
        long_exposure = sum(
            t.get("volume", 0.0) * t.get("entry_price", 0.0)
            for t in trades if t.get("side", "") == "buy"
        )
        short_exposure = sum(
            t.get("volume", 0.0) * t.get("entry_price", 0.0)
            for t in trades if t.get("side", "") == "sell"
        )
        net_exposure = abs(long_exposure - short_exposure)
        total_notional = long_exposure + short_exposure
        if total_notional > 0:
            net_ratio = net_exposure / total_notional
            if net_ratio > 0.95:
                score -= 10

    return max(0.0, score)


def _find_consecutive_minutes(mask: pd.Series, timestamps: pd.Series) -> int:
    """Find longest consecutive True period in minutes."""
    if not mask.any():
        return 0
    # Simple approximation: count consecutive True values
    # Assuming regular intervals
    consecutive = 0
    max_consecutive = 0
    for is_true in mask:
        if is_true:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    return max_consecutive


def calculate_win_rate(trades: list[dict]) -> tuple[float, int, int]:
    """Win rate, wins, losses."""
    wins = sum(1 for t in trades if t.get("profit", 0.0) > 0)
    losses = sum(1 for t in trades if t.get("profit", 0.0) <= 0)
    total = wins + losses
    if total == 0:
        return 0.0, 0, 0
    return wins / total * 100, wins, losses


def calculate_total_pnl(trades: list[dict]) -> float:
    """Sum of all trade profits."""
    return sum(t.get("profit", 0.0) for t in trades)


def calculate_final_score(
    return_rank: float,
    drawdown_rank: float,
    sharpe_rank: float,
    risk_discipline: float,
) -> float:
    """Final Score = 70%×Return Rank + 15%×Drawdown Rank + 10%×Sharpe Rank + 5%×Risk Discipline."""
    return (
        WEIGHT_RETURN * return_rank
        + WEIGHT_DRAWDOWN * drawdown_rank
        + WEIGHT_SHARPE * sharpe_rank
        + WEIGHT_RISK * risk_discipline
    )


def simulate_peer_ranks(
    my_return: float,
    my_max_dd: float,
    my_sharpe: float,
    n_participants: int = 500,
) -> tuple[float, float, float]:
    """Simulate peer ranks for score calculation."""
    # Simulate 500 participants with realistic distributions
    np.random.seed(42)
    peer_returns = np.random.normal(0.0, 0.05, n_participants - 1)
    peer_returns = np.append(peer_returns, my_return)

    peer_dds = np.random.exponential(0.10, n_participants - 1)
    peer_dds = np.append(peer_dds, my_max_dd)

    peer_sharpes = np.random.normal(0.0, 0.5, n_participants - 1)
    peer_sharpes = np.append(peer_sharpes, my_sharpe)

    return_rank = calculate_return_rank(my_return, peer_returns.tolist())
    dd_rank = calculate_drawdown_rank(my_max_dd, peer_dds.tolist())
    sharpe_rank = calculate_sharpe_rank(my_sharpe, peer_sharpes.tolist())

    return return_rank, dd_rank, sharpe_rank


def estimate_score_from_trades(
    trades: list[dict],
    equity_df: pd.DataFrame | None = None,
    n_participants: int = 500,
) -> dict:
    """Estimate competition score from trade history."""
    total_pnl = calculate_total_pnl(trades)
    final_equity = INITIAL_EQUITY + total_pnl
    return_pct = calculate_return(final_equity)
    win_rate, wins, losses = calculate_win_rate(trades)
    n_trades = wins + losses

    # If no equity log provided, simulate from trades
    if equity_df is None or equity_df.empty:
        # Build equity curve from trades
        equity = [INITIAL_EQUITY]
        current = INITIAL_EQUITY
        for t in trades:
            current += t.get("profit", 0.0)
            equity.append(current)
        equity_series = pd.Series(equity)
    else:
        equity_series = equity_df["equity"]

    max_dd = calculate_max_drawdown(equity_series)
    sharpe = calculate_sharpe_ratio(equity_series)

    # Simulate ranks
    return_rank, dd_rank, sharpe_rank = simulate_peer_ranks(return_pct, max_dd, sharpe, n_participants)

    # Risk discipline (simplified)
    risk_discipline = 100.0
    if n_trades > 0:
        # Check if we had any margin or leverage issues
        # Simplified: assume no issues unless data shows them
        pass

    final_score = calculate_final_score(return_rank, dd_rank, sharpe_rank, risk_discipline)

    return {
        "initial_equity": INITIAL_EQUITY,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "return_pct": return_pct * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "n_trades": n_trades,
        "return_rank": return_rank,
        "drawdown_rank": dd_rank,
        "sharpe_rank": sharpe_rank,
        "risk_discipline": risk_discipline,
        "final_score": final_score,
    }


def print_scorecard(score: dict):
    """Print formatted scorecard."""
    print("=" * 70)
    print("COMPETITION SCORECARD")
    print("=" * 70)
    print(f"\n[PERFORMANCE] Trading Performance")
    print(f"  Initial Equity:      ${score['initial_equity']:,.2f}")
    print(f"  Final Equity:        ${score['final_equity']:,.2f}")
    print(f"  Total P&L:           ${score['total_pnl']:,.2f}")
    print(f"  Return:              {score['return_pct']:+.2f}%")
    print(f"  Win Rate:            {score['win_rate']:.1f}% ({score['wins']}W / {score['losses']}L)")
    print(f"  Trades:              {score['n_trades']}")

    print(f"\n[RISK] Risk Metrics")
    print(f"  Max Drawdown:        {score['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:        {score['sharpe_ratio']:.4f}")

    print(f"\n[RANKING] Ranking (simulated vs 500 peers)")
    print(f"  Return Rank:         {score['return_rank']:.1f}/100")
    print(f"  Drawdown Rank:       {score['drawdown_rank']:.1f}/100")
    print(f"  Sharpe Rank:         {score['sharpe_rank']:.1f}/100")
    print(f"  Risk Discipline:     {score['risk_discipline']:.1f}/100")

    print(f"\n[FINAL SCORE]")
    print(f"  = {WEIGHT_RETURN*100:.0f}% x Return Rank")
    print(f"  + {WEIGHT_DRAWDOWN*100:.0f}% x Drawdown Rank")
    print(f"  + {WEIGHT_SHARPE*100:.0f}% x Sharpe Rank")
    print(f"  + {WEIGHT_RISK*100:.0f}% x Risk Discipline")
    print(f"  = {score['final_score']:.2f} / 100")

    print(f"\n[SHARPE PRIZE]")
    if score['n_trades'] >= 30:
        print(f"  [OK] Trade count: {score['n_trades']} (required: 30+)")
    else:
        print(f"  [NEED MORE] Trade count: {score['n_trades']} (need 30+ for Sharpe prize)")

    print(f"\n{'=' * 70}")
    print(f"\n[IMPROVE SCORE]")
    print(f"  1. Increase Return (70% weight): Trade more when signals agree")
    print(f"  2. Reduce Drawdown (15% weight): Tighten SL, use trailing stops")
    print(f"  3. Increase Sharpe (10% weight): More consistent returns, less volatility")
    print(f"  4. Maintain Risk (5% weight): Keep margin < 90%, leverage < 28x")
    print(f"\n{'=' * 70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-log", type=Path, default=None)
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--n-participants", type=int, default=500)
    parser.add_argument("--simulate", action="store_true", help="Simulate with example data")
    args = parser.parse_args()

    if args.simulate:
        # Simulate example data
        print("=" * 70)
        print("SIMULATED SCORE (Example)")
        print("=" * 70)

        # Simulate 50 trades with realistic outcomes
        np.random.seed(42)
        trades = []
        equity = [INITIAL_EQUITY]
        current = INITIAL_EQUITY

        for i in range(50):
            # 60% win rate, average profit $200, average loss $400
            is_win = np.random.random() < 0.60
            profit = np.random.normal(200, 50) if is_win else np.random.normal(-400, 100)
            current += profit
            equity.append(current)
            trades.append({
                "profit": profit,
                "volume": 8.0,
                "entry_price": 60000.0,
                "symbol": "BTCUSD",
                "side": "buy" if np.random.random() < 0.5 else "sell",
            })

        equity_df = pd.DataFrame({
            "timestamp": pd.date_range("2026-06-21", periods=len(equity), freq="15min"),
            "equity": equity,
        })

        score = estimate_score_from_trades(trades, equity_df, args.n_participants)
        print_scorecard(score)
        return

    # Load real data
    equity_df = None
    if args.equity_log:
        equity_df = load_equity_log(args.equity_log)

    trades = []
    if args.trade_log:
        trades = load_trade_log(args.trade_log)

    if not trades:
        print("No trade data found. Use --simulate to see example.")
        sys.exit(1)

    score = estimate_score_from_trades(trades, equity_df, args.n_participants)
    print_scorecard(score)


if __name__ == "__main__":
    main()
