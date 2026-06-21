"""Square-root market impact model.

IMPACT_C and DAILY_VOLUME_BTC are calibrated for BTCUSDT perpetual futures.
The square-root law gives impact proportional to sqrt(order_size / adv).
"""

from __future__ import annotations

import math

IMPACT_C: float = 0.4
DAILY_VOLUME_BTC: float = 300_000.0


def sqrt_impact_bps(size_btc: float) -> float:
    """Market impact in basis points using the square-root law."""
    return IMPACT_C * math.sqrt(size_btc / DAILY_VOLUME_BTC) * 10_000.0


def adjusted_fill_price(base_price: float, size_btc: float, side: str) -> float:
    """Fill price worsened by market impact — buys pay more, sells receive less."""
    impact = sqrt_impact_bps(size_btc)
    multiplier = impact / 10_000.0
    if side == "buy":
        return base_price * (1.0 + multiplier)
    return base_price * (1.0 - multiplier)


__all__ = ["IMPACT_C", "DAILY_VOLUME_BTC", "sqrt_impact_bps", "adjusted_fill_price"]
