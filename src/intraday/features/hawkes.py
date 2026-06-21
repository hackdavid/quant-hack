"""Hawkes Process — self-exciting point process for trade arrival intensity.

Models buy and sell trade arrivals as independent Hawkes processes:
  λ_buy(t)  = μ + Σ_i α·exp(−β·(t − t_i))   for t_i in buy arrivals
  λ_sell(t) = μ + Σ_i α·exp(−β·(t − t_i))   for t_i in sell arrivals

Updated via the efficient recurrence (no need to sum over all past events):
  λ(t_new) = μ + (λ(t_prev) − μ)·exp(−β·Δt) + α

Features emitted at each 5m bar close:
  hawkes_buy_intensity   — λ_buy(T)
  hawkes_sell_intensity  — λ_sell(T)
  hawkes_net             — (λ_buy − λ_sell) / (λ_buy + λ_sell), ∈ [−1, 1]

Calibration for BTC/USD futures (~12 trades/second):
  α   = 1.0    jump per trade
  β   = 10.0   per second  (half-life ≈ 69ms, captures micro-burst clusters)
  μ   = 6.0    background rate per side (assumes 50/50 split at baseline)
"""

import math
from typing import Optional


class HawkesCalculator:
    """Stateful Hawkes intensity calculator for buy and sell trade streams."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        mu: float = 6.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

        self._lambda_buy: float = mu
        self._lambda_sell: float = mu
        self._last_ms: Optional[int] = None

    def update(self, time_ms: int, is_buyer_maker: bool) -> None:
        """Feed one aggTrade and update intensities."""
        if self._last_ms is not None:
            dt_s = (time_ms - self._last_ms) / 1000.0
            if dt_s > 0:
                decay = math.exp(-self.beta * dt_s)
                # Decay both toward background μ
                self._lambda_buy  = self.mu + (self._lambda_buy  - self.mu) * decay
                self._lambda_sell = self.mu + (self._lambda_sell - self.mu) * decay

        # Add jump for this trade
        if is_buyer_maker:     # taker sold → sell aggressor
            self._lambda_sell += self.alpha
        else:                  # taker bought → buy aggressor
            self._lambda_buy  += self.alpha

        self._last_ms = time_ms

    def decay_to(self, time_ms: int) -> None:
        """Advance intensities to `time_ms` without a new trade (bar boundary decay)."""
        if self._last_ms is not None and time_ms > self._last_ms:
            dt_s = (time_ms - self._last_ms) / 1000.0
            decay = math.exp(-self.beta * dt_s)
            self._lambda_buy  = self.mu + (self._lambda_buy  - self.mu) * decay
            self._lambda_sell = self.mu + (self._lambda_sell - self.mu) * decay
            self._last_ms = time_ms

    @property
    def buy_intensity(self) -> float:
        return self._lambda_buy

    @property
    def sell_intensity(self) -> float:
        return self._lambda_sell

    @property
    def net(self) -> float:
        """Normalised net intensity: (λ_buy − λ_sell) / (λ_buy + λ_sell). ∈ [−1, 1]."""
        denom = self._lambda_buy + self._lambda_sell
        if denom < 1e-9:
            return 0.0
        return (self._lambda_buy - self._lambda_sell) / denom
