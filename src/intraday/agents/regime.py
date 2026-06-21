"""Regime detection agent: HMM (6 hidden states) + LightGBM classifier.

hmmlearn and lightgbm are optional — import errors produce a clear message and
predict() returns a neutral opinion when the model is not fitted.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from intraday.agents.base import Agent, AgentOpinion

log = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

REGIME_LABELS: list[str] = [
    "trend_up",
    "trend_down",
    "mean_revert",
    "breakout",
    "high_volatility",
    "low_liquidity",
    "liquidation_cascade",
]

VOL_REGIMES: list[str] = ["low", "normal", "high"]

N_HMM_STATES: int = 6

HMM_FEATURES: list[str] = [
    "log_ret_5m",
    "realized_vol_30m",
    "taker_buy_ratio_5m",
    "funding_rate",
    "oi_change_1h",
]

_NEUTRAL_REGIME_PROBS: dict[str, float] = {r: 1.0 / len(REGIME_LABELS) for r in REGIME_LABELS}
_NEUTRAL_NEXT_PROBS: dict[str, float] = {r: 1.0 / len(REGIME_LABELS) for r in REGIME_LABELS}


# ── Helpers ────────────────────────────────────────────────────────────────

def _col_zscore(series: pl.Series) -> pl.Series:
    """Z-score a Polars series, handling zero-std gracefully."""
    mu = series.mean()
    sd = series.std()
    if sd is None or sd < 1e-9:
        return series * 0.0
    return (series - mu) / sd


def _rolling_sum(series: pl.Series, window: int) -> pl.Series:
    return series.rolling_sum(window_size=window, min_periods=1)


def _auto_label(df: pl.DataFrame) -> pl.Series:
    """Rule-based regime label assignment.

    Priority order (first matching rule wins):
      1. liquidation_cascade
      2. breakout
      3. trend_up
      4. trend_down
      5. high_volatility
      6. mean_revert
      7. low_liquidity (default)
    """
    n = len(df)

    vol = df["realized_vol_30m"].fill_null(0.0)
    oi_chg = df["oi_change_1h"].fill_null(0.0)
    log_ret = df["log_ret_5m"].fill_null(0.0)
    tbr = df["taker_buy_ratio_5m"].fill_null(0.5)

    vol_arr = vol.to_numpy()
    oi_arr = oi_chg.to_numpy()
    ret_arr = log_ret.to_numpy()
    tbr_arr = tbr.to_numpy()

    # Thresholds
    vol_p90 = float(np.nanpercentile(vol_arr, 90)) if len(vol_arr) > 0 else 0.0
    vol_p95 = float(np.nanpercentile(vol_arr, 95)) if len(vol_arr) > 0 else 0.0
    vol_p60 = float(np.nanpercentile(vol_arr, 60)) if len(vol_arr) > 0 else 0.0

    ret_std = float(np.nanstd(ret_arr)) if len(ret_arr) > 0 else 1e-9
    if ret_std < 1e-9:
        ret_std = 1e-9

    # Rolling sums (12 bars = 60 min, 6 bars = 30 min)
    roll12 = np.convolve(ret_arr, np.ones(12), mode="full")[:n]
    roll6 = np.convolve(ret_arr, np.ones(6), mode="full")[:n]

    labels: list[str] = []
    for i in range(n):
        rv = vol_arr[i]
        oi = oi_arr[i]
        r = ret_arr[i]
        r12 = roll12[i]
        r6 = roll6[i]
        tb = tbr_arr[i]

        # 1. liquidation_cascade
        if oi < -0.05 and rv > vol_p95:
            labels.append("liquidation_cascade")
        # 2. breakout — single bar abs return > 3σ
        elif abs(r) > 3.0 * ret_std:
            labels.append("breakout")
        # 3. trend_up
        elif r12 > 0.5 * ret_std and tb > 0.55:
            labels.append("trend_up")
        # 4. trend_down
        elif r12 < -0.5 * ret_std and tb < 0.45:
            labels.append("trend_down")
        # 5. high_volatility
        elif rv > vol_p90:
            labels.append("high_volatility")
        # 6. mean_revert
        elif abs(r6) < 0.3 * ret_std and rv < vol_p60:
            labels.append("mean_revert")
        # 7. default
        else:
            labels.append("low_liquidity")

    return pl.Series("regime_label", labels)


def _vol_regime(realized_vol: float, low_q: float, high_q: float) -> str:
    if realized_vol > high_q:
        return "high"
    if realized_vol < low_q:
        return "low"
    return "normal"


# ── Agent ──────────────────────────────────────────────────────────────────

class RegimeAgent(Agent):
    """Regime detection: HMM posteriors → LightGBM multi-class classifier."""

    name = "regime"

    def __init__(self, model_dir: Path | None = None) -> None:
        self.hmm = None
        self.gbm = None
        self._model_dir = model_dir
        self._label_encoder: list[str] = REGIME_LABELS[:]
        self._vol_low_q: float = 0.0
        self._vol_high_q: float = 1.0
        # Keep last HMM state for next-regime transition matrix
        self._last_hmm_state: int | None = None
        log.debug("regime_agent_init", model_dir=str(model_dir))

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, features_df: pl.DataFrame) -> None:
        """Train HMM then LightGBM on top of HMM posteriors + raw features."""
        try:
            from hmmlearn import hmm as hmmlearn_hmm
        except ImportError as exc:
            raise ImportError(
                "hmmlearn is required for RegimeAgent.fit(). "
                "Install it with: pip install hmmlearn"
            ) from exc

        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for RegimeAgent.fit(). "
                "Install it with: pip install lightgbm"
            ) from exc

        log.info("regime_fit_start", n_rows=len(features_df))

        # ── Prepare HMM input ──────────────────────────────────────────────
        hmm_cols = [c for c in HMM_FEATURES if c in features_df.columns]
        hmm_df = features_df.select(hmm_cols).fill_null(0.0)

        # Z-score each column
        normed_cols = [_col_zscore(hmm_df[c]).rename(c) for c in hmm_cols]
        X_hmm = np.column_stack([s.to_numpy() for s in normed_cols]).astype(np.float64)

        # ── Fit HMM ───────────────────────────────────────────────────────
        self.hmm = hmmlearn_hmm.GaussianHMM(
            n_components=N_HMM_STATES,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )
        self.hmm.fit(X_hmm)
        log.info("hmm_fitted", n_states=N_HMM_STATES, n_rows=X_hmm.shape[0])

        # ── HMM posteriors for each bar ───────────────────────────────────
        posteriors = self.hmm.predict_proba(X_hmm)  # shape (n, 6)

        # ── Auto-label regimes ────────────────────────────────────────────
        labels_series = _auto_label(features_df)
        y = np.array([REGIME_LABELS.index(l) for l in labels_series.to_list()])

        # Store vol quantiles for predict()
        vol_arr = features_df["realized_vol_30m"].fill_null(0.0).to_numpy()
        self._vol_low_q = float(np.nanpercentile(vol_arr, 33))
        self._vol_high_q = float(np.nanpercentile(vol_arr, 67))

        # ── Build GBM feature matrix ──────────────────────────────────────
        raw_arr = X_hmm  # already z-scored raw features
        X_gbm = np.hstack([posteriors, raw_arr])

        # ── Fit LightGBM ──────────────────────────────────────────────────
        self.gbm = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.gbm.fit(X_gbm, y)
        log.info(
            "gbm_fitted",
            n_classes=len(REGIME_LABELS),
            n_features=X_gbm.shape[1],
        )

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, features: dict[str, Any]) -> AgentOpinion:
        t0 = time.perf_counter()
        ts_ms = int(features.get("bar_time_ms") or (time.time() * 1000))

        if self.hmm is None or self.gbm is None:
            log.debug("regime_predict_unfitted")
            return AgentOpinion(
                agent=self.name,
                ts_ms=ts_ms,
                payload={
                    "regime": "unknown",
                    "regime_probs": _NEUTRAL_REGIME_PROBS.copy(),
                    "next_regime_probs": _NEUTRAL_NEXT_PROBS.copy(),
                    "is_transition": False,
                    "vol_regime": "normal",
                },
                confidence=0.0,
                inference_ms=0.0,
            )

        # ── Build HMM input vector ────────────────────────────────────────
        hmm_input = np.array(
            [float(features.get(k) or 0.0) for k in HMM_FEATURES],
            dtype=np.float64,
        ).reshape(1, -1)

        posteriors = self.hmm.predict_proba(hmm_input)  # (1, 6)
        current_state = int(np.argmax(posteriors[0]))

        # ── Build GBM input ───────────────────────────────────────────────
        X_gbm = np.hstack([posteriors, hmm_input])  # (1, 6+n_features)
        gbm_probs = self.gbm.predict_proba(X_gbm)[0]  # (n_classes,)
        best_class = int(np.argmax(gbm_probs))
        regime = REGIME_LABELS[best_class]

        regime_probs = {
            REGIME_LABELS[i]: float(gbm_probs[i])
            for i in range(len(gbm_probs))
        }

        # ── Next-regime probabilities via HMM transition matrix ───────────
        trans_row = self.hmm.transmat_[current_state]  # (6,)
        # Map HMM states to regimes via GBM: next_state_posterior → GBM preds
        # Simplified: weight GBM output by transition probability per state
        # We use the transition matrix as a mixture over GBM predictions for each state
        next_gbm_probs = np.zeros(len(REGIME_LABELS))
        for s in range(N_HMM_STATES):
            state_post = np.zeros(N_HMM_STATES)
            state_post[s] = 1.0
            state_gbm_input = np.hstack([state_post.reshape(1, -1), hmm_input])
            state_preds = self.gbm.predict_proba(state_gbm_input)[0]
            next_gbm_probs += trans_row[s] * state_preds

        next_regime_probs = {
            REGIME_LABELS[i]: float(next_gbm_probs[i])
            for i in range(len(next_gbm_probs))
        }

        # ── Transition detection ──────────────────────────────────────────
        is_transition = (
            self._last_hmm_state is not None
            and self._last_hmm_state != current_state
        )
        self._last_hmm_state = current_state

        # ── Vol regime ────────────────────────────────────────────────────
        realized_vol = float(features.get("realized_vol_30m") or 0.0)
        vol_reg = _vol_regime(realized_vol, self._vol_low_q, self._vol_high_q)

        confidence = float(gbm_probs[best_class])

        inference_ms = (time.perf_counter() - t0) * 1000.0

        log.debug(
            "regime_predict",
            regime=regime,
            confidence=round(confidence, 4),
            vol_regime=vol_reg,
            is_transition=is_transition,
            inference_ms=round(inference_ms, 3),
        )

        return AgentOpinion(
            agent=self.name,
            ts_ms=ts_ms,
            payload={
                "regime": regime,
                "regime_probs": regime_probs,
                "next_regime_probs": next_regime_probs,
                "is_transition": is_transition,
                "vol_regime": vol_reg,
            },
            confidence=confidence,
            inference_ms=inference_ms,
        )

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist HMM + GBM to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "hmm": self.hmm,
            "gbm": self.gbm,
            "vol_low_q": self._vol_low_q,
            "vol_high_q": self._vol_high_q,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)
        log.info("regime_model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "RegimeAgent":
        """Load HMM + GBM from a pickle file."""
        path = Path(path)
        with path.open("rb") as f:
            state = pickle.load(f)
        agent = cls(model_dir=path.parent)
        agent.hmm = state["hmm"]
        agent.gbm = state["gbm"]
        agent._vol_low_q = state["vol_low_q"]
        agent._vol_high_q = state["vol_high_q"]
        log.info("regime_model_loaded", path=str(path))
        return agent
