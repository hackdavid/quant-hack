"""Risk agent: position sizing and trading guards.

Position sizing uses a vol-adjusted fractional Kelly approach:
    raw_kelly  = edge / variance          (edge = |p - 0.5| * 2)
    vol_scaled = target_vol / realised_vol
    final_size = min(raw_kelly, vol_scaled, max_position_frac) * capital

Guards (checked before every trade):
  - Daily loss limit (default -2%)
  - Total drawdown limit (default -5%)
  - Cooldown after hitting a limit (resumes next UTC day)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


class RiskAgent:
    def __init__(
        self,
        max_position_frac: float = 0.20,   # max fraction of capital per trade
        target_vol_annual: float = 0.40,   # target annualised vol (40%)
        daily_loss_limit:  float = 0.02,   # stop trading if daily loss > 2%
        max_drawdown:      float = 0.05,   # stop if drawdown from peak > 5%
        min_edge:          float = 0.04,   # min |prob - 0.5| to trade (2% edge)
        leverage:          float = 1.0,    # leverage multiplier
    ) -> None:
        self.max_pos   = max_position_frac
        self.target_vol = target_vol_annual / (365 * 24 * 12) ** 0.5  # per 5-min bar
        self.daily_limit = daily_loss_limit
        self.max_dd    = max_drawdown
        self.min_edge  = min_edge
        self.leverage  = leverage

        self._peak_equity     = None
        self._day_start_equity = None
        self._day_key         = None        # YYYY-MM-DD UTC
        self._halted          = False
        self._halt_reason     = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        """Call after every bar with current mark-to-market equity."""
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        now_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._day_key != now_day:
            self._day_key          = now_day
            self._day_start_equity = equity
            if self._halted:               # reset halt at start of new UTC day
                self._halted      = False
                self._halt_reason = ""
                print("  Risk: new UTC day — trading resumed")

    def check_can_trade(self, equity: float) -> tuple[bool, str]:
        """Returns (can_trade, reason). Call before generating any order."""
        if self._halted:
            return False, f"halted: {self._halt_reason}"

        if self._peak_equity and equity < self._peak_equity * (1 - self.max_dd):
            self._halt("max_drawdown_exceeded")
            return False, self._halt_reason

        if self._day_start_equity:
            daily_pnl = (equity - self._day_start_equity) / self._day_start_equity
            if daily_pnl < -self.daily_limit:
                self._halt("daily_loss_limit")
                return False, self._halt_reason

        return True, "ok"

    def size_position(
        self,
        prob_up:       float,
        capital:       float,
        realized_vol:  float | None = None,
    ) -> float:
        """Return position size in USD (positive = long, negative = short).

        Args:
            prob_up:      model probability that price goes up [0, 1]
            capital:      current account equity in USD
            realized_vol: per-bar realised vol; if None, uses target_vol directly
        """
        edge = abs(prob_up - 0.5) * 2          # 0 → no edge, 1 → perfect predictor
        if edge < self.min_edge:
            return 0.0

        # Kelly fraction: edge / (per-bar variance estimate)
        bar_vol = realized_vol if realized_vol and realized_vol > 1e-6 else self.target_vol
        kelly   = edge * self.target_vol / (bar_vol ** 2 + 1e-10)
        kelly   = min(kelly, self.max_pos)

        direction = 1 if prob_up > 0.5 else -1
        size_usd  = capital * kelly * self.leverage * direction
        return size_usd

    def position_to_contracts(
        self, size_usd: float, price: float, contract_size_usd: float = 1.0
    ) -> float:
        """Convert USD position size to number of contracts."""
        return size_usd / price / contract_size_usd

    # ── Internal ───────────────────────────────────────────────────────────────

    def _halt(self, reason: str) -> None:
        self._halted      = True
        self._halt_reason = reason
        print(f"  Risk HALT: {reason} — trading suspended until next UTC day")
