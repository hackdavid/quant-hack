"""Feature normalizer: log1p + z-score, fitted on training data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Features with heavy right-skew — apply log1p before z-scoring
LOG1P_COLS: set[str] = {
    "vol_5m",
    "trade_count_5m",
    "avg_trade_size_5m",
    "oi_btc",
    "hawkes_buy_intensity",
    "hawkes_sell_intensity",
}


class FeatureNormalizer:
    """Fit on training set, transform any split.

    Pipeline per column:
      1. log1p  (only for LOG1P_COLS)
      2. z-score: (x - mean) / (std + eps)
      3. clip to [-5, 5]
    """

    _EPS = 1e-8

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        self._log1p_mask: np.ndarray | None = None  # bool array len(feature_cols)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """Fit on a (N, F) array of raw feature values. NaN-safe."""
        assert X.shape[1] == len(self.feature_cols)
        self._log1p_mask = np.array([c in LOG1P_COLS for c in self.feature_cols])

        X = X.copy().astype(np.float32)
        X[:, self._log1p_mask] = np.log1p(np.clip(X[:, self._log1p_mask], 0, None))

        self._mean = np.nanmean(X, axis=0)
        self._std  = np.nanstd(X,  axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return normalized (N, F) float32 array with NaN → 0."""
        assert self._mean is not None, "call fit() first"
        X = X.copy().astype(np.float32)
        X[:, self._log1p_mask] = np.log1p(np.clip(X[:, self._log1p_mask], 0, None))
        X = (X - self._mean) / (self._std + self._EPS)
        X = np.clip(X, -5.0, 5.0)
        np.nan_to_num(X, copy=False, nan=0.0)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_cols": self.feature_cols,
            "log1p_mask": self._log1p_mask.tolist(),
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "FeatureNormalizer":
        data = json.loads(path.read_text())
        obj = cls(data["feature_cols"])
        obj._log1p_mask = np.array(data["log1p_mask"])
        obj._mean       = np.array(data["mean"], dtype=np.float32)
        obj._std        = np.array(data["std"],  dtype=np.float32)
        return obj
