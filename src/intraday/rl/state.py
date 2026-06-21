"""Build the RL state vector (15-dim) from execution context."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from intraday.sim.loop import Fill

STATE_DIM = 15
STATE_FEATURES = [
    "ts_normalized",           # how far into 5m window (0..1)
    "target_usd_normalized",   # remaining target / equity (signed)
    "already_filled_usd",      # / target
    "microprice_drift_5m_z",   # z-scored microprice drift
    "spread_bps",
    "ofi_5m_l5_z",             # z-scored OFI
    "queue_imbalance_l5",
    "vpin",
    "vol_regime_0",            # one-hot 3-dim for vol regime
    "vol_regime_1",
    "vol_regime_2",
    "forecast_confidence",
    "recent_fill_slippage_bps",
    "time_remaining_s",
    "recent_cancel_rate",
]

_SPREAD_BPS_MAX = 20.0
_SLIPPAGE_BPS_MAX = 50.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_state_vector(
    *,
    ts_ms: int,
    window_start_ms: int,
    window_end_ms: int,
    target_usd: float,
    filled_usd: float,
    book_features: dict,    # spread_bps, ofi, queue_imbalance, vpin
    vol_regime_id: int,     # 0=low, 1=normal, 2=high
    forecast_confidence: float,
    recent_fills: list,     # list of Fill
    equity_usd: float,
) -> np.ndarray:
    """Returns float32 array shape (STATE_DIM,). All values bounded."""
    window_duration_ms = max(window_end_ms - window_start_ms, 1)
    elapsed_ms = max(ts_ms - window_start_ms, 0)
    ts_normalized = _clip(elapsed_ms / window_duration_ms, 0.0, 1.0)

    remaining_usd = target_usd - filled_usd
    target_usd_normalized = (
        _clip(remaining_usd / max(abs(equity_usd), 1.0), -1.0, 1.0)
        if equity_usd != 0.0 else 0.0
    )

    already_filled_frac = (
        _clip(filled_usd / max(abs(target_usd), 1.0), 0.0, 1.0)
        if target_usd != 0.0 else 0.0
    )

    microprice_drift_z = _clip(float(book_features.get("microprice_drift_5m_z", 0.0)), -5.0, 5.0)
    spread_bps_raw = float(book_features.get("spread_bps", 0.0))
    spread_bps = _clip(spread_bps_raw / _SPREAD_BPS_MAX, 0.0, 1.0)
    ofi_z = _clip(float(book_features.get("ofi", 0.0)), -5.0, 5.0)
    queue_imbalance = _clip(float(book_features.get("queue_imbalance", 0.0)), -1.0, 1.0)
    vpin = _clip(float(book_features.get("vpin", 0.5)), 0.0, 1.0)

    vol_regime_id = int(vol_regime_id) % 3
    vol_regime_0 = 1.0 if vol_regime_id == 0 else 0.0
    vol_regime_1 = 1.0 if vol_regime_id == 1 else 0.0
    vol_regime_2 = 1.0 if vol_regime_id == 2 else 0.0

    conf = _clip(float(forecast_confidence), 0.0, 1.0)

    if recent_fills:
        slippage_vals: list[float] = []
        for fill in recent_fills:
            slip = getattr(fill, "slippage_bps", None)
            if slip is not None:
                slippage_vals.append(float(slip))
        if slippage_vals:
            avg_slip = sum(slippage_vals) / len(slippage_vals)
        else:
            avg_slip = 0.0
    else:
        avg_slip = 0.0
    recent_fill_slippage = _clip(avg_slip / _SLIPPAGE_BPS_MAX, 0.0, 1.0)

    time_remaining_ms = max(window_end_ms - ts_ms, 0)
    time_remaining_s_raw = time_remaining_ms / 1000.0
    time_remaining_s = _clip(time_remaining_s_raw / 300.0, 0.0, 1.0)

    recent_cancel_rate = _clip(float(book_features.get("recent_cancel_rate", 0.0)), 0.0, 1.0)

    vec = np.array(
        [
            ts_normalized,
            target_usd_normalized,
            already_filled_frac,
            microprice_drift_z / 5.0,
            spread_bps,
            ofi_z / 5.0,
            queue_imbalance,
            vpin,
            vol_regime_0,
            vol_regime_1,
            vol_regime_2,
            conf,
            recent_fill_slippage,
            time_remaining_s,
            recent_cancel_rate,
        ],
        dtype=np.float32,
    )

    assert vec.shape == (STATE_DIM,), f"State vector wrong shape: {vec.shape}"
    return vec


__all__ = ["STATE_DIM", "STATE_FEATURES", "build_state_vector"]
