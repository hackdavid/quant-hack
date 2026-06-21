"""Triple-barrier labelling — Lopez de Prado (2018) ch. 3.

Produces directional labels from price series using profit-take / stop-loss
barriers and a fixed time horizon.
"""

from __future__ import annotations

import structlog
import numpy as np
import polars as pl

log = structlog.get_logger(__name__)


def triple_barrier_labels(
    bars: pl.DataFrame,
    *,
    pt_sl: tuple[float, float] = (1.5, 1.0),
    horizon_minutes: int = 15,
    vol_window_minutes: int = 60,
) -> pl.DataFrame:
    """Compute triple-barrier labels (Lopez de Prado 2018 ch. 3).

    Args:
        bars: DataFrame with columns ``bar_time_ms``, ``close``,
              and optionally ``realized_vol_30m``.  Sorted ascending by
              ``bar_time_ms`` (1-minute bars assumed).
        pt_sl: Profit-take and stop-loss multipliers in σ units.
        horizon_minutes: Maximum look-ahead (bars at 1-minute resolution).
        vol_window_minutes: Rolling window for realised-vol calculation.

    Returns:
        Input DataFrame augmented with:
        - ``label_sign``            int8  ∈ {-1, 0, +1}
        - ``label_first_touch_ms``  int64 timestamp of first barrier touch
        - ``label_realized_return`` float64 log-return at first touch / horizon
    """
    pt_mult, sl_mult = pt_sl

    # ── Ensure sorted ─────────────────────────────────────────────────────
    bars = bars.sort("bar_time_ms")

    n = len(bars)
    ts_arr: np.ndarray = bars["bar_time_ms"].to_numpy()
    close_arr: np.ndarray = bars["close"].to_numpy(allow_copy=True).astype(np.float64)

    # ── Compute log-returns for rolling vol ───────────────────────────────
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = 0.0
    log_ret[1:] = np.log(close_arr[1:] / close_arr[:-1])

    # Rolling realised vol (std-dev of log-returns over vol_window_minutes bars)
    if "realized_vol_30m" in bars.columns:
        vol_arr = bars["realized_vol_30m"].to_numpy(allow_copy=True).astype(np.float64)
        # Fill nulls with rolling fallback
        null_mask = np.isnan(vol_arr) | (vol_arr == 0.0)
    else:
        vol_arr = np.full(n, np.nan)
        null_mask = np.ones(n, dtype=bool)

    # Fill missing vol with rolling std
    if null_mask.any():
        for i in np.where(null_mask)[0]:
            start = max(0, i - vol_window_minutes + 1)
            window = log_ret[start : i + 1]
            if len(window) > 1:
                vol_arr[i] = float(np.std(window, ddof=1))
            else:
                vol_arr[i] = 1e-8  # guard against zero

    # Ensure no zeros slip through
    vol_arr = np.where(vol_arr <= 0.0, 1e-8, vol_arr)

    # ── Triple-barrier scan ────────────────────────────────────────────────
    label_sign = np.zeros(n, dtype=np.int8)
    label_first_touch_ms = ts_arr.copy()
    label_realized_return = np.zeros(n, dtype=np.float64)

    for i in range(n):
        c0 = close_arr[i]
        vol_i = vol_arr[i]
        pt_barrier = pt_mult * vol_i  # log-return threshold (upper)
        sl_barrier = sl_mult * vol_i  # log-return threshold (lower)

        horizon_end = min(i + horizon_minutes, n - 1)
        sign_out: int = 0
        touch_idx: int = horizon_end
        first_touch_ret: float = 0.0

        for j in range(i + 1, horizon_end + 1):
            ret = float(np.log(close_arr[j] / c0))
            if ret >= pt_barrier:
                sign_out = 1
                touch_idx = j
                first_touch_ret = ret
                break
            if ret <= -sl_barrier:
                sign_out = -1
                touch_idx = j
                first_touch_ret = ret
                break
        else:
            # Horizon expired — label 0
            first_touch_ret = float(np.log(close_arr[horizon_end] / c0))

        label_sign[i] = sign_out
        label_first_touch_ms[i] = int(ts_arr[touch_idx])
        label_realized_return[i] = first_touch_ret

    log.info(
        "triple_barrier_labels.done",
        n_bars=n,
        horizon_minutes=horizon_minutes,
        pt=pt_mult,
        sl=sl_mult,
        n_up=int((label_sign == 1).sum()),
        n_down=int((label_sign == -1).sum()),
        n_flat=int((label_sign == 0).sum()),
    )

    return bars.with_columns(
        pl.Series("label_sign", label_sign, dtype=pl.Int8),
        pl.Series("label_first_touch_ms", label_first_touch_ms, dtype=pl.Int64),
        pl.Series("label_realized_return", label_realized_return, dtype=pl.Float64),
    )
