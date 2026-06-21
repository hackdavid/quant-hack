"""Isotonic regression calibration for probability outputs.

Maps raw model probabilities to calibrated probabilities using monotone
regression, which is more flexible than Platt scaling but requires more data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression

log = structlog.get_logger(__name__)


class IsotonicCalibrator:
    """Isotonic regression wrapper for probability calibration.

    Fits a monotone non-decreasing function from raw predicted probabilities
    to empirical positive-class frequencies.  Clips outputs to [0, 1].

    Example::

        cal = IsotonicCalibrator()
        cal.fit(val_probs, val_labels)
        test_cal = cal.transform(test_probs)
        cal.save(Path("models/calibrator.pkl"))
    """

    def __init__(self) -> None:
        self._ir: IsotonicRegression | None = None

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> None:
        """Fit the isotonic regression on validation predictions.

        Args:
            p_raw: Raw predicted probabilities, shape (N,).
            y:     Binary labels (0 or 1), shape (N,).
        """
        p_raw = np.asarray(p_raw, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(p_raw) != len(y):
            raise ValueError(
                f"p_raw and y must have the same length, got {len(p_raw)} vs {len(y)}"
            )
        self._ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._ir.fit(p_raw, y)
        log.info(
            "calibrator.fit_done",
            n_samples=int(len(y)),
            pos_rate=float(y.mean()),
            raw_mean=float(p_raw.mean()),
        )

    # ── Inference ─────────────────────────────────────────────────────────

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to raw probabilities.

        Args:
            p_raw: Raw probabilities, shape (N,) or scalar-compatible.

        Returns:
            Calibrated probabilities in [0, 1], same shape as input.
        """
        if self._ir is None:
            raise RuntimeError(
                "IsotonicCalibrator is not fitted. Call fit() or load() first."
            )
        p_raw = np.asarray(p_raw, dtype=np.float64)
        original_shape = p_raw.shape
        calibrated = self._ir.predict(p_raw.ravel())
        return np.clip(calibrated, 0.0, 1.0).reshape(original_shape)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the calibrator to a pickle file."""
        if self._ir is None:
            raise RuntimeError("Nothing to save — calibrator has not been fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._ir, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("calibrator.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        """Load a previously saved calibrator from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator not found: {path}")
        instance = cls()
        with open(path, "rb") as fh:
            instance._ir = pickle.load(fh)
        log.info("calibrator.loaded", path=str(path))
        return instance
