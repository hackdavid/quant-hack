"""Vectorized backtest engine for intraday BTC direction signals.

Simulates trading on the validation split using saved model predictions.
Accounts for Binance USDT-M perpetual costs:
  - Taker fee:    0.04% per side  (0.08% round-trip)
  - Slippage:     0.01% per side  (market impact estimate)
  - Funding rate: 0.01% per 8h   → 0.000125% per 5-min bar

Usage:
    from intraday.backtest.engine import BacktestEngine
    result = BacktestEngine().run(probs, fwd_returns)
    BacktestEngine.print_report(result)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


# ── Binance USDT-M perpetual cost constants ────────────────────────────────────
TAKER_FEE      = 0.0004          # 0.04% per side
SLIPPAGE       = 0.0001          # 0.01% per side (market impact)
FUNDING_8H     = 0.0001          # 0.01% per 8h (typical BTC perp rate)
BARS_PER_8H    = 96              # 5-min bars in 8 hours
FUNDING_PER_BAR = FUNDING_8H / BARS_PER_8H   # per 5-min bar on open position
BARS_PER_YEAR  = 365 * 24 * 12  # 5-min bars per year


@dataclass
class BacktestResult:
    signals:      np.ndarray   # {-1, 0, +1} per bar
    probs:        np.ndarray   # raw model probability (up)
    gross_log_ret: np.ndarray  # signal * fwd_return (before costs)
    net_log_ret:  np.ndarray   # after fees, slippage, funding
    equity:       np.ndarray   # cumulative equity curve (starts at 1.0)
    timestamps:   np.ndarray   # bar_time_ms for each bar (optional)


class BacktestEngine:
    def __init__(
        self,
        threshold:        float = 0.55,    # min prob to go long (1-t to go short)
        fee_per_side:     float = TAKER_FEE + SLIPPAGE,
        funding_per_bar:  float = FUNDING_PER_BAR,
        initial_capital:  float = 10_000.0,
    ) -> None:
        self.threshold       = threshold
        self.fee_per_side    = fee_per_side
        self.funding_per_bar = funding_per_bar
        self.initial_capital = initial_capital

    def run(
        self,
        probs:       np.ndarray,
        fwd_returns: np.ndarray,
        timestamps:  np.ndarray | None = None,
    ) -> BacktestResult:
        """Vectorized backtest.

        Args:
            probs:       (N,) P(up) from model, in [0, 1]
            fwd_returns: (N,) actual log-returns over the holding period
            timestamps:  (N,) optional bar_time_ms for reporting
        """
        t = self.threshold
        signals = np.where(probs > t, 1, np.where(probs < (1 - t), -1, 0)).astype(np.int8)

        # Gross P&L: signal × actual return
        gross = signals.astype(np.float64) * fwd_returns.astype(np.float64)

        # Transaction cost: paid when position CHANGES (entry + exit = 2 sides)
        prev_signals = np.concatenate([[0], signals[:-1]])
        position_changed = (signals != prev_signals).astype(np.float64)
        trade_cost = position_changed * self.fee_per_side * 2

        # Funding: paid every bar we hold a position
        funding_cost = np.abs(signals).astype(np.float64) * self.funding_per_bar

        net = gross - trade_cost - funding_cost

        # Equity curve (compounded)
        equity = self.initial_capital * np.exp(np.cumsum(net))

        return BacktestResult(
            signals=signals,
            probs=probs,
            gross_log_ret=gross,
            net_log_ret=net,
            equity=equity,
            timestamps=timestamps if timestamps is not None else np.arange(len(probs)),
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def metrics(result: BacktestResult) -> dict:
        net  = result.net_log_ret
        eq   = result.equity
        sig  = result.signals

        in_market   = sig != 0
        n_bars      = len(net)
        n_in_market = int(in_market.sum())

        # Return
        total_log_ret   = net.sum()
        total_return    = np.expm1(total_log_ret)
        ann_return      = np.expm1(total_log_ret * BARS_PER_YEAR / n_bars)

        # Sharpe (annualised, on all bars including flat)
        sharpe = (net.mean() / (net.std() + 1e-12)) * np.sqrt(BARS_PER_YEAR)

        # Calmar
        mdd = BacktestEngine._max_drawdown(eq)
        calmar = ann_return / (mdd + 1e-12)

        # Trade-level stats (each contiguous block of same signal = 1 trade)
        trade_returns = BacktestEngine._trade_returns(result)
        n_trades   = len(trade_returns)
        win_rate   = float((trade_returns > 0).mean()) if n_trades else 0.0
        avg_win    = float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
        avg_loss   = float(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        return {
            "total_return_pct":   round(total_return * 100, 2),
            "ann_return_pct":     round(ann_return * 100, 2),
            "sharpe":             round(sharpe, 3),
            "calmar":             round(calmar, 3),
            "max_drawdown_pct":   round(mdd * 100, 2),
            "n_trades":           n_trades,
            "win_rate_pct":       round(win_rate * 100, 1),
            "profit_factor":      round(profit_factor, 3),
            "avg_win_pct":        round(avg_win * 100, 3),
            "avg_loss_pct":       round(avg_loss * 100, 3),
            "pct_time_in_market": round(n_in_market / n_bars * 100, 1),
            "n_bars":             n_bars,
        }

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd   = (peak - equity) / peak
        return float(dd.max())

    @staticmethod
    def _trade_returns(result: BacktestResult) -> np.ndarray:
        """Group consecutive same-direction bars into trades, return per-trade net log-return."""
        trades = []
        sig = result.signals
        net = result.net_log_ret
        i   = 0
        while i < len(sig):
            if sig[i] == 0:
                i += 1
                continue
            j = i + 1
            while j < len(sig) and sig[j] == sig[i]:
                j += 1
            trades.append(net[i:j].sum())
            i = j
        return np.array(trades) if trades else np.array([0.0])

    @staticmethod
    def print_report(result: BacktestResult, title: str = "Backtest Report") -> None:
        m = BacktestEngine.metrics(result)
        print(f"\n{'='*52}")
        print(f"  {title}")
        print(f"{'='*52}")
        print(f"  Total return     : {m['total_return_pct']:>8.2f}%")
        print(f"  Ann. return      : {m['ann_return_pct']:>8.2f}%")
        print(f"  Sharpe (ann.)    : {m['sharpe']:>8.3f}")
        print(f"  Calmar           : {m['calmar']:>8.3f}")
        print(f"  Max drawdown     : {m['max_drawdown_pct']:>8.2f}%")
        print(f"  Trades           : {m['n_trades']:>8d}")
        print(f"  Win rate         : {m['win_rate_pct']:>8.1f}%")
        print(f"  Profit factor    : {m['profit_factor']:>8.3f}")
        print(f"  Avg win          : {m['avg_win_pct']:>8.3f}%")
        print(f"  Avg loss         : {m['avg_loss_pct']:>8.3f}%")
        print(f"  Time in market   : {m['pct_time_in_market']:>8.1f}%")
        print(f"{'='*52}\n")
        return m
