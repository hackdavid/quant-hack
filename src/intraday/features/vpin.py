"""VPIN — Volume-synchronized Probability of Informed Trading.

Reference: Easley, López de Prado, O'Hara (2012).

Algorithm:
  1. Divide the trade stream into volume buckets of size V_S BTC each.
  2. In each bucket classify volume as buy/sell via the `is_buyer_maker` flag:
       is_buyer_maker=False → taker was buyer → BUY volume
       is_buyer_maker=True  → taker was seller → SELL volume
  3. VPIN = mean(|V_buy_i - V_sell_i| / V_S)  over the last `window` buckets.

State is maintained across 5m bar boundaries — the rolling bucket window
carries over so VPIN is continuous, not reset each bar.

Calibration for BTC/USD futures (~112K BTC/day):
  V_S = 100 BTC  (~1.3 minutes to fill at average volume)
  window = 50    (~65 minutes of rolling history)
"""

from collections import deque
from typing import Optional


class VPINCalculator:
    """Stateful VPIN calculator. Call update() for every trade."""

    def __init__(self, bucket_btc: float = 100.0, window: int = 50) -> None:
        self.bucket_btc = bucket_btc
        self.window = window

        # Current (open) bucket accumulators
        self._bucket_buy: float = 0.0
        self._bucket_sell: float = 0.0
        self._bucket_total: float = 0.0

        # Completed bucket |imbalance| values: |V_buy - V_sell| / V_S
        self._completed: deque[float] = deque(maxlen=window)

    def update(self, quantity: float, is_buyer_maker: bool) -> None:
        """Feed one aggTrade. Closes buckets automatically when they fill."""
        remaining = quantity

        while remaining > 0:
            space = self.bucket_btc - self._bucket_total
            fill = min(remaining, space)

            if is_buyer_maker:
                self._bucket_sell += fill
            else:
                self._bucket_buy += fill
            self._bucket_total += fill
            remaining -= fill

            if self._bucket_total >= self.bucket_btc:
                # Close this bucket
                imbalance = abs(self._bucket_buy - self._bucket_sell) / self.bucket_btc
                self._completed.append(imbalance)
                # Reset (any overflow already handled by loop)
                self._bucket_buy = 0.0
                self._bucket_sell = 0.0
                self._bucket_total = 0.0

    def vpin(self) -> Optional[float]:
        """Current VPIN estimate. None until at least window/2 buckets are complete."""
        n = len(self._completed)
        if n < self.window // 2:
            return None
        return sum(self._completed) / n

    def current_bucket_imbalance(self) -> Optional[float]:
        """Buy fraction of the current (open) bucket. None if no trades yet."""
        total = self._bucket_buy + self._bucket_sell
        if total < 1e-9:
            return None
        return self._bucket_buy / total

    def buckets_completed(self) -> int:
        return len(self._completed)
