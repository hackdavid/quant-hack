"""Account state with avg-cost FIFO position tracking.

Cash accounting: buying deducts notional from cash; selling adds notional.
PnL is embedded in cash after closing. The equity formula is simply:
  equity = cash_quote + position_base * mark_price

position_base is signed: positive=long, negative=short.
avg_entry_price tracks cost basis for PnL attribution only.
realized_pnl_quote accumulates realized gains/losses (already in cash).
funding_paid_quote and fee_paid_quote are cost trackers (already in cash).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intraday.sim.loop import Fill


class Account:
    def __init__(self, cash_quote: float) -> None:
        self.position_base: float = 0.0
        self.avg_entry_price: float = 0.0
        self.cash_quote: float = cash_quote
        self.realized_pnl_quote: float = 0.0
        self.funding_paid_quote: float = 0.0
        self.fee_paid_quote: float = 0.0

    def update_on_fill(self, fill: "Fill", fee: float) -> None:
        """Update account using avg-cost FIFO.

        Cash is debited/credited by notional on every fill.
        On position reduction, realized PnL is computed and added to the tracker
        (it's already reflected in cash via the notional credit/debit).
        """
        self.fee_paid_quote += fee
        self.cash_quote -= fee

        fill_sign = 1.0 if fill.side == "buy" else -1.0
        notional = fill.qty_base * fill.price

        if self.position_base == 0.0:
            # Opening from flat
            self.position_base = fill_sign * fill.qty_base
            self.avg_entry_price = fill.price
            # Debit cash for long, credit for short (short receives margin)
            self.cash_quote -= fill_sign * notional

        elif (fill_sign > 0) == (self.position_base > 0):
            # Adding to existing position (same direction)
            old_notional = abs(self.position_base) * self.avg_entry_price
            new_abs_pos = abs(self.position_base) + fill.qty_base
            self.avg_entry_price = (old_notional + notional) / new_abs_pos
            self.position_base += fill_sign * fill.qty_base
            self.cash_quote -= fill_sign * notional

        else:
            # Reducing or reversing position
            reduce_qty = min(fill.qty_base, abs(self.position_base))
            remain_qty = fill.qty_base - reduce_qty

            # Realize PnL on the reduced portion
            if self.position_base > 0:
                pnl = reduce_qty * (fill.price - self.avg_entry_price)
            else:
                pnl = reduce_qty * (self.avg_entry_price - fill.price)
            self.realized_pnl_quote += pnl

            # Update position
            old_sign = 1.0 if self.position_base > 0 else -1.0
            self.position_base += fill_sign * fill.qty_base

            if abs(self.position_base) < 1e-12:
                self.position_base = 0.0
                self.avg_entry_price = 0.0
            elif (self.position_base > 0) != (old_sign > 0):
                # Reversed direction
                self.avg_entry_price = fill.price

            # Cash: receive back the original notional for reduced portion at fill price
            # For a long reduce: we're selling reduce_qty — credit notional
            # For a short reduce: we're buying  reduce_qty — debit notional
            self.cash_quote += old_sign * reduce_qty * fill.price

            # Any remainder opens a new position in the opposite direction
            if remain_qty > 1e-12:
                self.cash_quote -= fill_sign * remain_qty * fill.price

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.position_base == 0.0 or mark_price == 0.0 or self.avg_entry_price == 0.0:
            return 0.0
        return self.position_base * (mark_price - self.avg_entry_price)

    def equity(self, mark_price: float) -> float:
        """Total account value: cash + open position mark-to-market."""
        if mark_price <= 0.0:
            mark_price = self.avg_entry_price
        return self.cash_quote + self.position_base * mark_price

    def drawdown_pct(self, mark_price: float, peak_equity: float) -> float:
        if peak_equity <= 0.0:
            return 0.0
        current = self.equity(mark_price)
        return max(0.0, (peak_equity - current) / peak_equity * 100.0)


__all__ = ["Account"]
