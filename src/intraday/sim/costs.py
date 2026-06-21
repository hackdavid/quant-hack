"""Fee and funding cost calculations.

MAKER_BPS: rebate-adjusted maker fee (positive = cost, negative = rebate).
TAKER_BPS: taker fee.
Funding is applied to position notional: positive rate means longs pay shorts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intraday.sim.account import Account
    from intraday.sim.events import FundingEvent
    from intraday.sim.loop import Fill

MAKER_BPS: float = 2.0
TAKER_BPS: float = 5.0


def fee_for_fill(fill: "Fill") -> float:
    """Returns the USDT fee amount for a fill (always positive)."""
    rate_bps = MAKER_BPS if fill.is_maker else TAKER_BPS
    notional = fill.qty_base * fill.price
    return notional * rate_bps / 10_000.0


def apply_funding(account: "Account", funding_event: "FundingEvent") -> float:
    """Apply funding payment to the account.

    Returns the payment amount (negative = account paid out, positive = received).
    Long positions pay when rate > 0, receive when rate < 0.
    """
    if account.position_base == 0.0:
        return 0.0
    notional = account.position_base * funding_event.mark_price
    # Positive rate: longs pay shorts → payment is negative for long positions
    payment = -notional * funding_event.funding_rate
    account.funding_paid_quote -= payment  # funding_paid accumulates as cost
    account.cash_quote += payment
    return payment


__all__ = ["MAKER_BPS", "TAKER_BPS", "fee_for_fill", "apply_funding"]
