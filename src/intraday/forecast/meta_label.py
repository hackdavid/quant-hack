"""LightGBM meta-label classifier.

Secondary classifier that estimates P(primary forecast direction is correct)
given a set of context features.  Trained with purged k-fold cross-validation
to avoid label overlap leakage.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

log = structlog.get_logger(__name__)

# Features consumed by the meta-label model (all must be present in X)
META_FEATURE_COLS: list[str] = [
    "fc_confidence",
    "fc_expected_move_sigma",
    "vol_regime_id",
    "funding_rate",
    "rsi_14",
    "log_ret_60m",
    "realized_vol_30m",
    "hour_utc",
]


class MetaLabelClassifier:
    """LightGBM binary classifier: P(primary forecast direction is correct).

    Attributes:
        model_dir: Directory from which/to which the model is serialised.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir
        self._model: Any | None = None  # lgb.Booster after fit/load

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        *,
        timestamps: pl.Series,
        label_first_touch: pl.Series | None = None,
        n_folds: int = 5,
    ) -> dict[str, float]:
        """Train using purged k-fold cross-validation.

        Args:
            X:                   Feature DataFrame with columns in META_FEATURE_COLS.
            y:                   Binary labels (1 = direction correct, 0 = wrong).
            timestamps:          Bar open timestamps in ms UTC (same length as X).
            label_first_touch:   First-touch timestamps for purging; defaults to
                                 timestamps + 15 min if not provided.
            n_folds:             Number of CV folds.

        Returns:
            ``{"auc": float, "brier": float}`` mean scores over all folds.
        """
        try:
            import lightgbm as lgb
            from sklearn.metrics import brier_score_loss, roc_auc_score
        except ImportError as exc:
            raise ImportError(
                "lightgbm and scikit-learn are required for MetaLabelClassifier. "
                "Install with: pip install lightgbm scikit-learn"
            ) from exc

        from intraday.forecast.splits import purged_kfold

        X_arr = self._to_numpy(X)
        y_arr = y.to_numpy().astype(np.float32)
        ts_arr = timestamps.to_numpy().astype(np.int64)

        if label_first_touch is not None:
            ft_arr = label_first_touch.to_numpy().astype(np.int64)
        else:
            # Default: assume 15-minute horizon
            ft_arr = ts_arr + 15 * 60 * 1000

        folds = purged_kfold(ts_arr, ft_arr, n_splits=n_folds, embargo_pct=0.01)

        auc_scores: list[float] = []
        brier_scores: list[float] = []
        oof_preds = np.full(len(y_arr), np.nan)

        lgb_params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        }

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

            model = lgb.train(
                lgb_params,
                train_set,
                num_boost_round=500,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

            val_preds = model.predict(X_val)
            auc = float(roc_auc_score(y_val, val_preds))
            brier = float(brier_score_loss(y_val, val_preds))
            auc_scores.append(auc)
            brier_scores.append(brier)
            oof_preds[val_idx] = val_preds

            log.info(
                "meta_label.fold_done",
                fold=fold_i + 1,
                n_folds=n_folds,
                auc=round(auc, 4),
                brier=round(brier, 4),
            )

        # Retrain on full dataset
        full_set = lgb.Dataset(X_arr, label=y_arr)
        self._model = lgb.train(
            lgb_params,
            full_set,
            num_boost_round=500,
            callbacks=[lgb.log_evaluation(-1)],
        )

        metrics = {
            "auc": float(np.mean(auc_scores)),
            "brier": float(np.mean(brier_scores)),
        }
        log.info("meta_label.fit_done", **{k: round(v, 4) for k, v in metrics.items()})
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Return P(direction correct) ∈ [0, 1] for each row in X.

        Args:
            X: DataFrame with META_FEATURE_COLS columns.

        Returns:
            float64 array of shape (N,).
        """
        if self._model is None:
            raise RuntimeError("MetaLabelClassifier is not fitted. Call fit() or load() first.")
        X_arr = self._to_numpy(X)
        return self._model.predict(X_arr).astype(np.float64)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise model to disk using pickle."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("meta_label.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "MetaLabelClassifier":
        """Load a previously saved MetaLabelClassifier from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MetaLabel model not found: {path}")
        instance = cls()
        with open(path, "rb") as fh:
            instance._model = pickle.load(fh)
        log.info("meta_label.loaded", path=str(path))
        return instance

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(X: pl.DataFrame) -> np.ndarray:
        """Select META_FEATURE_COLS (filling nulls with 0) and return float32 array."""
        available = [c for c in META_FEATURE_COLS if c in X.columns]
        missing = [c for c in META_FEATURE_COLS if c not in X.columns]
        if missing:
            log.warning("meta_label.missing_features", missing=missing)

        arr = (
            X.select(available)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )
        return arr
