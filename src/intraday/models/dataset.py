"""FeatureDataset — sliding-window sequences over the BTCUSDT feature store.

Each sample:
  X : float32 tensor (seq_len, n_features)  — normalized feature history
  y_ret : float32 scalar                    — fwd_ret_5m (regression target)
  y_dir : int64 scalar                      — fwd_direction_5m mapped to {0=down, 1=flat, 2=up}

Missing values (null depth before 2023, metrics gaps) are forward-filled then
zero-filled so the model sees 0 as "no data" for that bar.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from intraday.features.schema import ALL_FEATURES
from intraday.models.normalizer import FeatureNormalizer

# label map: {-1 → 0, 0 → 1, 1 → 2}
_DIR_MAP = {-1: 0, 0: 1, 1: 2}


def load_bars(
    data_dir: Path,
    symbol: str,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Load feature parquets for [start, end] into a single sorted DataFrame."""
    features_dir = data_dir / "features" / symbol
    files = sorted(features_dir.glob("*.parquet"))

    if start:
        files = [f for f in files if f.stem >= start.isoformat()]
    if end:
        files = [f for f in files if f.stem <= end.isoformat()]

    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir} for range {start}→{end}")

    df = pl.read_parquet(files).sort("bar_time_ms")
    return df


def _impute(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract feature matrix with imputation: forward-fill → zero-fill."""
    sub = df.select(feature_cols)
    # forward-fill metrics / depth gaps
    sub = sub.fill_null(strategy="forward")
    # remaining nulls (start of series, depth before 2023) → 0
    sub = sub.fill_null(0.0)
    return sub.to_numpy().astype(np.float32)


def build_arrays(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_raw, y_ret, y_dir) arrays aligned by row."""
    X = _impute(df, feature_cols)

    y_ret = df["fwd_ret_5m"].fill_null(0.0).to_numpy().astype(np.float32)

    # Map direction labels and drop nulls (use 1=flat for null bars at tail)
    y_dir_raw = df["fwd_direction_5m"].fill_null(0).to_numpy()
    y_dir = np.vectorize(_DIR_MAP.get)(y_dir_raw).astype(np.int64)

    return X, y_ret, y_dir


class FeatureDataset(Dataset):
    """Sliding window over normalized feature bars.

    Valid indices are seq_len..N-1 so every sequence has full history and
    a non-null target. Sequences can cross calendar-day boundaries (BTC
    perpetuals trade 24/7, no overnight gap).
    """

    def __init__(
        self,
        X_norm: np.ndarray,
        y_ret: np.ndarray,
        y_dir: np.ndarray,
        seq_len: int,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        self._X     = torch.from_numpy(X_norm)
        self._y_ret = torch.from_numpy(y_ret)
        self._y_dir = torch.from_numpy(y_dir)
        self.seq_len = seq_len

        # valid_mask marks which global indices belong to this split
        if valid_mask is not None:
            # indices where target falls in this split AND we have seq_len history
            raw = np.where(valid_mask)[0]
            self._idx = raw[raw >= seq_len]
        else:
            self._idx = np.arange(seq_len, len(X_norm))

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self._idx[i]
        x = self._X[t - self.seq_len : t]          # (seq_len, n_features)
        return x, self._y_ret[t], self._y_dir[t]


# ── Public factory ──────────────────────────────────────────────────────────────

def create_datasets(
    data_dir: Path,
    symbol: str = "BTCUSDT",
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2024-06-30",
    seq_len: int = 60,
    feature_cols: list[str] | None = None,
) -> tuple[FeatureDataset, FeatureDataset, FeatureDataset, FeatureNormalizer]:
    """Load all data, fit normalizer on train, return (train, val, test, normalizer)."""
    feature_cols = feature_cols or ALL_FEATURES

    # Load full history once
    df = load_bars(data_dir, symbol)
    dates = df["bar_time_ms"].cast(pl.Utf8)  # not used for masking

    bar_dates = pl.from_numpy(
        (df["bar_time_ms"] // 86_400_000).cast(pl.Int64).to_numpy()
    )

    # Date thresholds as epoch days
    def _eday(s: str) -> int:
        return int(date.fromisoformat(s).toordinal() - date(1970, 1, 1).toordinal())

    train_end_d  = _eday(train_end)
    val_start_d  = _eday(val_start)
    val_end_d    = _eday(val_end)

    epoch_days = (df["bar_time_ms"] // 86_400_000).to_numpy()

    train_mask = epoch_days <= train_end_d
    val_mask   = (epoch_days >= val_start_d) & (epoch_days <= val_end_d)
    test_mask  = epoch_days > val_end_d

    X_raw, y_ret, y_dir = build_arrays(df, feature_cols)

    # Fit normalizer on training rows only
    normalizer = FeatureNormalizer(feature_cols)
    normalizer.fit(X_raw[train_mask])
    X_norm = normalizer.transform(X_raw)

    train_ds = FeatureDataset(X_norm, y_ret, y_dir, seq_len, valid_mask=train_mask)
    val_ds   = FeatureDataset(X_norm, y_ret, y_dir, seq_len, valid_mask=val_mask)
    test_ds  = FeatureDataset(X_norm, y_ret, y_dir, seq_len, valid_mask=test_mask)

    return train_ds, val_ds, test_ds, normalizer
