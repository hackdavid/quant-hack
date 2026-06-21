"""PyTorch Dataset for forecast model training.

Each sample combines:
  - klines_norm:  (seq_klines, 5) z-normalised OHLCV — input to TCN (unused; kept for compat)
  - klines_raw:   (seq_klines, 6) raw OHLCV+amount   — input to Kronos tokenizer
  - klines_stamp: (seq_klines, 5) temporal features  — minute/hour/weekday/day/month
  - state_window: (seq_state, n_features) z-normed   — input to SmallTCN
  - label:        int64 bin index 0..10
  - meta_y:       int64 {0, 1} — was primary direction correct?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import structlog
import torch
from torch.utils.data import Dataset

from intraday.features.schema import ALL_FEATURES

log = structlog.get_logger(__name__)

_KLINE_COLS     = ["open", "high", "low", "close", "volume"]
_KLINE_RAW_COLS = ["open", "high", "low", "close", "volume", "quote_volume"]
_STAMP_COLS     = ["minute", "hour", "weekday", "day", "month"]
N_BINS = 2   # binary: 0=down, 1=up  (flat samples filtered at mask time)


# ── Normalisation helper ───────────────────────────────────────────────────────

class RunningNorm:
    """Incremental mean/std (Welford's algorithm)."""

    def __init__(self, n_features: int) -> None:
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2   = np.zeros(n_features, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        for row in batch:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (row - self.mean)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.M2 / (self.n - 1)) if self.n > 1 else np.ones_like(self.mean)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.where(self.std > 1e-8, self.std, 1.0)


# ── Dataset ────────────────────────────────────────────────────────────────────

class ForecastDataset(Dataset):
    """Paired samples for training the Kronos + TCN + ForecastHead pipeline.

    Returns 6-tuple per item:
        (klines_norm, klines_raw, klines_stamp, state_window, label, meta_y)
    """

    def __init__(
        self,
        klines_dir: Path,
        features_dir: Path,
        labels_df: pl.DataFrame,
        seq_klines: int = 256,
        seq_state: int = 128,
        klines_norm: RunningNorm | None = None,
        state_norm: RunningNorm | None = None,
    ) -> None:
        self._klines_dir   = Path(klines_dir)
        self._features_dir = Path(features_dir)
        self._seq_klines   = seq_klines
        self._seq_state    = seq_state
        self._labels       = labels_df.sort("bar_time_ms")

        log.info(
            "forecast_dataset.loading",
            klines_dir=str(klines_dir),
            features_dir=str(features_dir),
        )

        self._klines_df  = self._load_klines(self._klines_dir)
        self._state_df   = self._load_features(self._features_dir)

        # Fast timestamp→row lookup
        self._klines_ts = self._klines_df["bar_time_ms"].to_numpy()
        self._state_ts  = self._state_df["bar_time_ms"].to_numpy()

        # ── Normalised arrays for TCN ──────────────────────────────────────
        klines_np = self._klines_df.select(_KLINE_COLS).fill_null(0.0).to_numpy().astype(np.float64)
        state_np  = self._state_df.select(ALL_FEATURES).fill_null(0.0).to_numpy().astype(np.float64)

        if klines_norm is None:
            self._klines_norm = RunningNorm(len(_KLINE_COLS))
            self._klines_norm.update(klines_np)
        else:
            self._klines_norm = klines_norm

        if state_norm is None:
            self._state_norm = RunningNorm(len(ALL_FEATURES))
            self._state_norm.update(state_np)
        else:
            self._state_norm = state_norm

        self._klines_norm_arr = self._klines_norm.normalise(klines_np).astype(np.float32)
        self._state_norm_arr  = self._state_norm.normalise(state_np).astype(np.float32)

        # ── Raw OHLCV+amount for Kronos ────────────────────────────────────
        self._klines_raw_arr = (
            self._klines_df
            .select(_KLINE_RAW_COLS)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )

        # ── Temporal stamp features ────────────────────────────────────────
        self._klines_stamps = _compute_stamps(self._klines_ts)  # (N, 5) float32

        # ── Filter labels with sufficient history ──────────────────────────
        label_ts = self._labels["bar_time_ms"].to_numpy()
        self._valid_indices = np.where(self._compute_valid_mask(label_ts))[0]

        log.info(
            "forecast_dataset.ready",
            total_labels=len(self._labels),
            valid_samples=int(len(self._valid_indices)),
            seq_klines=seq_klines,
            seq_state=seq_state,
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return int(len(self._valid_indices))

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Return (klines_norm, klines_raw, klines_stamp, state_window, label, meta_y).

        Shapes:
            klines_norm:  float32 (seq_klines, 5)
            klines_raw:   float32 (seq_klines, 6)  — raw OHLCV+amount for Kronos
            klines_stamp: float32 (seq_klines, 5)  — temporal features
            state_window: float32 (seq_state, n_features)
            label:        int64   scalar ∈ 0..10
            meta_y:       int64   scalar ∈ {0, 1}
        """
        label_row_idx = int(self._valid_indices[idx])
        row = self._labels.row(label_row_idx, named=True)
        ts_ms         = int(row["bar_time_ms"])
        label_sign    = int(row["label_sign"])
        realized_ret  = float(row["label_realized_return"])

        # ── klines window ──────────────────────────────────────────────────
        k_end   = int(np.searchsorted(self._klines_ts, ts_ms, side="right"))
        k_start = max(k_end - self._seq_klines, 0)

        klines_norm_sl = self._klines_norm_arr[k_start:k_end]
        klines_raw_sl  = self._klines_raw_arr[k_start:k_end]
        klines_stamp_sl = self._klines_stamps[k_start:k_end]

        pad_k = self._seq_klines - len(klines_norm_sl)
        if pad_k > 0:
            klines_norm_sl  = np.concatenate([np.zeros((pad_k, klines_norm_sl.shape[1]),  dtype=np.float32), klines_norm_sl])
            klines_raw_sl   = np.concatenate([np.zeros((pad_k, klines_raw_sl.shape[1]),   dtype=np.float32), klines_raw_sl])
            klines_stamp_sl = np.concatenate([np.zeros((pad_k, klines_stamp_sl.shape[1]), dtype=np.float32), klines_stamp_sl])

        # ── state window ───────────────────────────────────────────────────
        s_end   = int(np.searchsorted(self._state_ts, ts_ms, side="right"))
        s_start = max(s_end - self._seq_state, 0)
        state_sl = self._state_norm_arr[s_start:s_end]

        pad_s = self._seq_state - len(state_sl)
        if pad_s > 0:
            state_sl = np.concatenate([np.zeros((pad_s, state_sl.shape[1]), dtype=np.float32), state_sl])

        # ── labels ─────────────────────────────────────────────────────────
        bin_label  = self._sign_to_bin(label_sign)
        meta_y     = int(label_sign != 0 and int(np.sign(realized_ret)) == label_sign)

        return (
            torch.from_numpy(klines_norm_sl),
            torch.from_numpy(klines_raw_sl),
            torch.from_numpy(klines_stamp_sl),
            torch.from_numpy(state_sl),
            torch.tensor(bin_label, dtype=torch.int64),
            torch.tensor(meta_y,    dtype=torch.int64),
        )

    # ── Normalisation accessors ────────────────────────────────────────────────

    @property
    def klines_norm(self) -> RunningNorm:
        return self._klines_norm

    @property
    def state_norm(self) -> RunningNorm:
        return self._state_norm

    # ── Loaders ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_klines(directory: Path) -> pl.DataFrame:
        """Load 1m klines, normalise timestamp column to bar_time_ms."""
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No klines Parquet files in {directory}")

        frames: list[pl.DataFrame] = []
        for f in files:
            df = pl.read_parquet(f)
            # klines_1m stores timestamp as open_time_ms — rename
            if "open_time_ms" in df.columns and "bar_time_ms" not in df.columns:
                df = df.rename({"open_time_ms": "bar_time_ms"})
            # Ensure quote_volume exists (used as "amount" for Kronos)
            if "quote_volume" not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias("quote_volume"))
            wanted = [c for c in _KLINE_RAW_COLS + ["bar_time_ms"] if c in df.columns]
            frames.append(df.select(wanted))

        return pl.concat(frames).sort("bar_time_ms")

    @staticmethod
    def _load_features(directory: Path) -> pl.DataFrame:
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No feature Parquet files in {directory}")

        frames: list[pl.DataFrame] = []
        for f in files:
            df = pl.read_parquet(f)
            wanted = [c for c in ALL_FEATURES + ["bar_time_ms"] if c in df.columns]
            frames.append(df.select(wanted))

        return pl.concat(frames).sort("bar_time_ms")

    def _compute_valid_mask(self, label_ts: np.ndarray) -> np.ndarray:
        valid = np.ones(len(label_ts), dtype=bool)
        label_signs = self._labels["label_sign"].to_numpy()
        for i, ts in enumerate(label_ts):
            # Drop flat samples — binary classification only (down=0, up=1)
            if label_signs[i] == 0:
                valid[i] = False
                continue
            k_pos = int(np.searchsorted(self._klines_ts, ts, side="right"))
            s_pos = int(np.searchsorted(self._state_ts,  ts, side="right"))
            if k_pos < self._seq_klines // 2:
                valid[i] = False
            if s_pos < self._seq_state // 2:
                valid[i] = False
        return valid

    @staticmethod
    def _sign_to_bin(label_sign: int) -> int:
        """Map label_sign {-1, +1} → bin {0=down, 1=up}."""
        return 0 if label_sign < 0 else 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_stamps(ts_ms: np.ndarray) -> np.ndarray:
    """Convert millisecond UTC timestamps → (N, 5) temporal feature array."""
    dts = pd.to_datetime(ts_ms, unit="ms", utc=True)
    return np.stack([
        dts.minute.to_numpy(dtype=np.float32),
        dts.hour.to_numpy(dtype=np.float32),
        dts.day_of_week.to_numpy(dtype=np.float32),
        dts.day.to_numpy(dtype=np.float32),
        dts.month.to_numpy(dtype=np.float32),
    ], axis=-1)
