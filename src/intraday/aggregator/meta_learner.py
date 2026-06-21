"""LightGBM stacked meta-learner: P(trade will be profitable given all agent inputs)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

LGBM_PARAMS: dict[str, Any] = {
    "num_leaves": 31,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Categorical columns that must be label-encoded before fitting
_CAT_COLS = ["rg_regime", "rg_vol_regime", "so_mode"]


def _encode_categoricals(X: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    """Replace string columns with integer codes.  Returns encoded frame + mapping."""
    encodings: dict[str, dict[str, int]] = {}
    for col in _CAT_COLS:
        if col not in X.columns:
            continue
        uniq = X[col].unique().sort().to_list()
        mapping = {v: i for i, v in enumerate(uniq)}
        encodings[col] = mapping
        X = X.with_columns(
            pl.col(col).replace_strict(list(mapping.keys()), list(mapping.values()), default=0).alias(col)
        )
    return X, encodings


def _apply_encodings(X: pl.DataFrame, encodings: dict[str, dict[str, int]]) -> pl.DataFrame:
    """Apply a pre-built encoding mapping; unknown values map to 0."""
    for col, mapping in encodings.items():
        if col not in X.columns:
            continue
        keys = list(mapping.keys())
        vals = list(mapping.values())
        X = X.with_columns(
            pl.col(col).replace_strict(keys, vals, default=0).alias(col)
        )
    return X


def _calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


class MetaLearner:
    """LightGBM stacked meta-learner trained with purged k-fold CV.

    Attributes
    ----------
    model_dir : Path | None
        Directory where the model artefact is saved / loaded.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir is not None else None
        self._models: list[Any] = []           # one per fold
        self._encodings: dict[str, dict[str, int]] = {}
        self._feature_cols: list[str] = []
        self._oof_preds: np.ndarray | None = None
        self._threshold: float = 0.5
        log.debug("meta_learner_init", model_dir=str(model_dir))

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        *,
        ts: pl.Series,
        n_folds: int = 5,
        embargo_pct: float = 0.01,
    ) -> dict:
        """Train LightGBM with purged k-fold CV.

        Parameters
        ----------
        X : pl.DataFrame
            Feature matrix (aggregator rows).
        y : pl.Series
            Binary labels (1 = profitable trade, 0 = not).
        ts : pl.Series
            Timestamps (int64 ms) — used for purging / embargo.
        n_folds : int
            Number of time-series folds.
        embargo_pct : float
            Fraction of the total time range to embargo between train/test.

        Returns
        -------
        dict
            OOF metrics: ``auc``, ``brier``, ``ece``.
        """
        try:
            import lightgbm as lgb
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:
            raise ImportError(
                "lightgbm and scikit-learn are required for MetaLearner.fit(). "
                "Install with: pip install lightgbm scikit-learn"
            ) from exc

        self._feature_cols = X.columns
        X_enc, self._encodings = _encode_categoricals(X)

        X_np = X_enc.to_numpy().astype(np.float32)
        y_np = y.to_numpy().astype(np.float32)
        ts_np = ts.to_numpy().astype(np.int64)

        n = len(X_np)
        fold_size = n // n_folds
        embargo_gap = max(1, int(n * embargo_pct))

        oof_preds = np.full(n, np.nan, dtype=np.float64)
        self._models = []

        log.info(
            "meta_learner_fit_start",
            n_rows=n,
            n_folds=n_folds,
            embargo_gap=embargo_gap,
        )

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n
            train_end = max(0, test_start - embargo_gap)

            X_train = X_np[:train_end]
            y_train = y_np[:train_end]
            X_test = X_np[test_start:test_end]
            y_test = y_np[test_start:test_end]

            if len(X_train) < 50:
                log.warning("meta_learner_skip_fold", fold=fold, reason="insufficient train samples")
                continue

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

            params = {
                **LGBM_PARAMS,
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
                "random_state": 42,
            }
            # n_estimators is passed separately to train()
            n_estimators = params.pop("n_estimators")

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=25, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            self._models.append(model)

            fold_preds = model.predict(X_test)
            oof_preds[test_start:test_end] = fold_preds

            log.info(
                "meta_learner_fold_done",
                fold=fold,
                best_iteration=model.best_iteration,
                n_train=len(X_train),
                n_test=len(X_test),
            )

        self._oof_preds = oof_preds

        # Compute metrics on filled OOF predictions
        valid_mask = ~np.isnan(oof_preds)
        if valid_mask.sum() < 10:
            log.warning("meta_learner_insufficient_oof")
            return {"auc": 0.0, "brier": 1.0, "ece": 1.0}

        y_valid = y_np[valid_mask]
        p_valid = oof_preds[valid_mask]

        auc = float(roc_auc_score(y_valid, p_valid))
        brier = _brier(y_valid, p_valid)
        ece = _calibration_ece(y_valid, p_valid)

        # Calibrate threshold: maximise F1 on OOF
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.linspace(0.3, 0.8, 51):
            preds_bin = (p_valid >= thresh).astype(np.float32)
            tp = float(((preds_bin == 1) & (y_valid == 1)).sum())
            fp = float(((preds_bin == 1) & (y_valid == 0)).sum())
            fn = float(((preds_bin == 0) & (y_valid == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
        self._threshold = best_thresh

        metrics = {"auc": auc, "brier": brier, "ece": ece}
        log.info("meta_learner_fit_done", **metrics, threshold=self._threshold)
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Return P(profitable) for each row in X.

        Averages predictions from all fold models.  Returns zeros when no
        models are loaded (graceful degradation).
        """
        if not self._models:
            log.warning("meta_learner_predict_no_model")
            return np.zeros(len(X), dtype=np.float64)

        X_enc = _apply_encodings(X, self._encodings)
        # Align columns to training order
        missing = [c for c in self._feature_cols if c not in X_enc.columns]
        for c in missing:
            X_enc = X_enc.with_columns(pl.lit(0.0).alias(c))
        X_enc = X_enc.select(self._feature_cols)
        X_np = X_enc.to_numpy().astype(np.float32)

        preds = np.stack([m.predict(X_np) for m in self._models], axis=0)
        return preds.mean(axis=0)

    # ── Feature importance ───────────────────────────────────────────────────

    def feature_importance(self) -> pl.DataFrame:
        """Return averaged feature importance across all fold models."""
        if not self._models:
            return pl.DataFrame({"feature": [], "importance": []})

        imps = np.stack(
            [m.feature_importance(importance_type="gain") for m in self._models],
            axis=0,
        ).mean(axis=0)

        return (
            pl.DataFrame({"feature": self._feature_cols, "importance": imps.tolist()})
            .sort("importance", descending=True)
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the full meta-learner state to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models": self._models,
            "encodings": self._encodings,
            "feature_cols": self._feature_cols,
            "threshold": self._threshold,
            "oof_preds": self._oof_preds,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)
        log.info("meta_learner_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "MetaLearner":
        """Load a previously saved meta-learner."""
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)
        ml = cls(model_dir=path.parent)
        ml._models = state["models"]
        ml._encodings = state["encodings"]
        ml._feature_cols = state["feature_cols"]
        ml._threshold = state.get("threshold", 0.5)
        ml._oof_preds = state.get("oof_preds")
        log.info("meta_learner_loaded", path=str(path), n_models=len(ml._models))
        return ml


__all__ = ["MetaLearner", "LGBM_PARAMS"]
