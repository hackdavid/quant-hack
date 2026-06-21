"""Latency model for order round-trip simulation.

Uses lognormal distribution to model realistic network + exchange latency.
mu_ms is the median latency; sigma is the log-scale spread.
Seeding ensures reproducible simulations.
"""

from __future__ import annotations

import math
import random


class LatencyModel:
    def __init__(self, mu_ms: float = 80.0, sigma: float = 0.4, seed: int | None = None) -> None:
        self._mu_ms = mu_ms
        self._sigma = sigma
        self._rng = random.Random(seed)
        # Convert median to lognormal mu parameter: E[X] = exp(mu + sigma^2/2)
        # We want the median to be mu_ms, so log-space mean = log(mu_ms)
        self._log_mu = math.log(mu_ms)

    def sample_ms(self) -> float:
        """Sample a lognormal latency in milliseconds."""
        return math.exp(self._rng.gauss(self._log_mu, self._sigma))


__all__ = ["LatencyModel"]
