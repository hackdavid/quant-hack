"""Pure functional orderflow agent.

Combines OFI, Hawkes, microprice drift, and VPIN into a single flow_bias signal.
No learned model — all logic is rule-based z-score composition.
"""

import time
from collections import deque
from typing import Any

import structlog

from intraday.agents.base import Agent, AgentOpinion

log = structlog.get_logger(__name__)

# Signal names and their equal weights (normalised inside predict)
_SIGNAL_KEYS = [
    "ofi_5m",
    "hawkes_net",
    "log_ret_5m",
    "depth_imbalance_02pct",
    "taker_buy_ratio_5m",
]

# Minimum observations before z-scores are meaningful
_MIN_OBS = 5


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Return float(v) or default when v is None / non-numeric."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class _RollingStats:
    """Welford online algorithm for mean and variance."""

    __slots__ = ("_n", "_mean", "_M2", "_window", "_buf")

    def __init__(self, window: int) -> None:
        self._window: int = window
        self._buf: deque[float] = deque()
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def push(self, x: float) -> None:
        self._buf.append(x)
        # Update Welford
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

        # Evict oldest when over window
        if len(self._buf) > self._window:
            old = self._buf.popleft()
            self._n -= 1
            if self._n == 0:
                self._mean = 0.0
                self._M2 = 0.0
            else:
                delta_old = old - self._mean
                self._mean -= delta_old / self._n
                delta_old2 = old - self._mean
                self._M2 -= delta_old * delta_old2
                if self._M2 < 0.0:
                    self._M2 = 0.0

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return max((self._M2 / self._n) ** 0.5, 1e-9)

    @property
    def n(self) -> int:
        return self._n

    def zscore(self, x: float) -> float:
        if self._n < _MIN_OBS:
            return 0.0
        return (x - self._mean) / self.std


class OrderflowAgent(Agent):
    """Orderflow specialist agent.

    Maintains rolling z-score statistics for each input signal and combines them
    into a single flow_bias scalar in [-1, +1].
    """

    name = "orderflow"

    def __init__(
        self,
        vpin_threshold: float = 0.7,
        spread_median_window: int = 288,
    ) -> None:
        self.vpin_threshold = vpin_threshold
        self._stats: dict[str, _RollingStats] = {
            key: _RollingStats(spread_median_window) for key in _SIGNAL_KEYS
        }
        self._vol_stats = _RollingStats(spread_median_window)
        log.debug(
            "orderflow_agent_init",
            vpin_threshold=vpin_threshold,
            window=spread_median_window,
        )

    def predict(self, features: dict[str, Any]) -> AgentOpinion:
        t0 = time.perf_counter()

        # ── Extract raw signals ────────────────────────────────────────────
        ofi = _safe_float(features.get("ofi_5m"), 0.0)
        hawkes_net = _safe_float(features.get("hawkes_net"), 0.0)
        log_ret = _safe_float(features.get("log_ret_5m"), 0.0)
        depth_imb = _safe_float(features.get("depth_imbalance_02pct"), 0.0)
        tbr = _safe_float(features.get("taker_buy_ratio_5m"), 0.5)
        vpin = _safe_float(features.get("vpin_50"), 0.0)
        realized_vol = _safe_float(features.get("realized_vol_30m"), 0.0)

        # taker_buy_ratio is [0,1]; centre it so z-scores are symmetric
        tbr_centred = tbr - 0.5

        raw = {
            "ofi_5m": ofi,
            "hawkes_net": hawkes_net,
            "log_ret_5m": log_ret,
            "depth_imbalance_02pct": depth_imb,
            "taker_buy_ratio_5m": tbr_centred,
        }

        # ── Update rolling stats and compute z-scores ──────────────────────
        zscores: dict[str, float] = {}
        for key, val in raw.items():
            self._stats[key].push(val)
            zscores[key] = self._stats[key].zscore(val)

        self._vol_stats.push(realized_vol)

        # ── Weighted combination (equal weights after L1 normalisation) ────
        n_signals = len(zscores)
        raw_bias = sum(zscores.values()) / n_signals

        # Clip flow_bias to [-3, +3] then map to [-1, +1]
        flow_bias = max(-1.0, min(1.0, raw_bias / 3.0))

        # flow_strength: magnitude of the bias
        flow_strength = abs(flow_bias)

        # ── VPIN guard ────────────────────────────────────────────────────
        step_away = vpin > self.vpin_threshold

        ts_ms = int(features.get("bar_time_ms") or (time.time() * 1000))

        payload: dict[str, Any] = {
            "flow_bias": round(flow_bias, 6),
            "flow_strength": round(flow_strength, 6),
            "vpin": round(vpin, 6),
            "step_away": step_away,
            "ofi_l5_z": round(zscores["ofi_5m"], 6),
            "hawkes_imb": round(zscores["hawkes_net"], 6),
        }

        inference_ms = (time.perf_counter() - t0) * 1000.0

        log.debug(
            "orderflow_predict",
            flow_bias=flow_bias,
            vpin=vpin,
            step_away=step_away,
            inference_ms=round(inference_ms, 3),
        )

        return AgentOpinion(
            agent=self.name,
            ts_ms=ts_ms,
            payload=payload,
            confidence=flow_strength,
            inference_ms=inference_ms,
        )
