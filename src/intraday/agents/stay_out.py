"""News-shock / extreme-event stay-out detector.

Rule-based, no ML. Uses rolling z-scores of market stress indicators to flag
'stay_out' (score > 3.0) or 'defensive' (score > 2.0) conditions.
"""

import time
from collections import deque
from typing import Any

import structlog

from intraday.agents.base import Agent, AgentOpinion

log = structlog.get_logger(__name__)

_MIN_OBS = 10  # Minimum observations before z-scores are reliable


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class _OnlineZScore:
    """Online Welford mean/variance for a fixed-length rolling window."""

    __slots__ = ("_window", "_buf", "_n", "_mean", "_M2")

    def __init__(self, window: int) -> None:
        self._window = window
        self._buf: deque[float] = deque()
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x: float) -> float:
        """Push x into the window, return z-score of x (or 0 if insufficient data)."""
        self._buf.append(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

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

        if self._n < _MIN_OBS:
            return 0.0

        std = (self._M2 / self._n) ** 0.5
        if std < 1e-9:
            return 0.0
        return (x - self._mean) / std


class StayOutDetector(Agent):
    """Detects news-shock and extreme-event conditions."""

    name = "stay_out"
    STAY_OUT_THRESHOLD: float = 3.0
    DEFENSIVE_THRESHOLD: float = 2.0
    WINDOW: int = 60  # bars for rolling z-score

    def __init__(self) -> None:
        self._z_vol = _OnlineZScore(self.WINDOW)
        self._z_spread = _OnlineZScore(self.WINDOW)
        self._z_oi = _OnlineZScore(self.WINDOW)
        self._z_abs_ret = _OnlineZScore(self.WINDOW)
        log.debug("stay_out_detector_init", window=self.WINDOW)

    def predict(self, features: dict[str, Any]) -> AgentOpinion:
        """Evaluate market stress and return stay-out mode.

        Expected feature keys:
            realized_vol_30m  — rolling realised volatility
            taker_buy_ratio_5m — used as a proxy for spread pressure when extreme
            oi_change_1h      — fractional OI change
            rsi_14            — RSI(14), not directly z-scored but used to inform
            log_ret_5m        — single-bar log return (abs → z-scored)
        """
        t0 = time.perf_counter()
        ts_ms = int(features.get("bar_time_ms") or (time.time() * 1000))

        realized_vol = _safe_float(features.get("realized_vol_30m"), 0.0)
        tbr = _safe_float(features.get("taker_buy_ratio_5m"), 0.5)
        oi_chg = _safe_float(features.get("oi_change_1h"), 0.0)
        log_ret = _safe_float(features.get("log_ret_5m"), 0.0)

        # taker_buy_ratio used as spread proxy: extremes (near 0 or 1) signal wide spreads
        spread_proxy = abs(tbr - 0.5) * 2.0  # maps [0,1] → [0,1], max at extremes

        abs_ret = abs(log_ret)

        # Update rolling z-scorers
        z_vol = self._z_vol.update(realized_vol)
        z_spread = self._z_spread.update(spread_proxy)
        z_oi = self._z_oi.update(oi_chg)
        z_abs_ret = self._z_abs_ret.update(abs_ret)

        # Aggregate stress score: worst of the four z-scores
        score = max(z_vol, z_spread, abs(z_oi), z_abs_ret)

        # Determine mode
        if score > self.STAY_OUT_THRESHOLD:
            mode = "stay_out"
        elif score > self.DEFENSIVE_THRESHOLD:
            mode = "defensive"
        else:
            mode = "normal"

        drivers: dict[str, float] = {
            "z_realized_vol": round(z_vol, 6),
            "z_spread_proxy": round(z_spread, 6),
            "z_oi_change": round(z_oi, 6),
            "z_abs_ret": round(z_abs_ret, 6),
        }

        # Confidence inversely correlated with severity of stay-out
        if mode == "stay_out":
            confidence = 1.0
        elif mode == "defensive":
            confidence = 0.7
        else:
            confidence = 0.3

        inference_ms = (time.perf_counter() - t0) * 1000.0

        log.debug(
            "stay_out_predict",
            mode=mode,
            score=round(score, 4),
            drivers=drivers,
            inference_ms=round(inference_ms, 3),
        )

        return AgentOpinion(
            agent=self.name,
            ts_ms=ts_ms,
            payload={
                "mode": mode,
                "score": round(score, 6),
                "drivers": drivers,
            },
            confidence=confidence,
            inference_ms=inference_ms,
        )
