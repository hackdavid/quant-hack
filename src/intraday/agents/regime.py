"""RegimeAgent — HMM regime detection (bull/sideways/bear) + LightGBM classifier."""
from __future__ import annotations
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from intraday.agents.base import Agent, AgentOpinion

# Exported constants
N_HMM_STATES = 3
REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}
VOL_REGIMES = {0: "low", 1: "normal", 2: "high"}
HMM_FEATURES = ["log_ret_5m", "realized_vol_30m"]


class RegimeAgent(Agent):
    name = "regime"
    LABELS = REGIME_LABELS

    def __init__(self, n_states: int = N_HMM_STATES) -> None:
        self.n_states = n_states
        self._hmm = None
        self._lgb = None
        self._state_map: dict[int, int] = {}   # raw HMM state → semantic label

    def _safe(self, df, col, default=0.0):
        if isinstance(df, dict):
            v = df.get(col, default)
            return float(v) if v is not None else default
        if col not in df.columns:
            return default
        v = df[col].tail(1)[0]
        return float(v) if v is not None else default

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, df) -> "RegimeAgent":
        from hmmlearn.hmm import GaussianHMM
        import lightgbm as lgb

        log_ret = df["log_ret_5m"].fill_null(0).to_numpy().astype(np.float64)
        if "realized_vol_30m" in df.columns:
            vol = df["realized_vol_30m"].fill_null(0).to_numpy().astype(np.float64)
        else:
            vol = np.array([np.std(log_ret[max(0, i - 6):i + 1]) for i in range(len(log_ret))])

        X_hmm = np.column_stack([log_ret, vol])
        self._hmm = GaussianHMM(
            n_components=self.n_states, covariance_type="diag",
            n_iter=100, random_state=42,
        )
        self._hmm.fit(X_hmm)
        raw_states = self._hmm.predict(X_hmm)

        # Map raw states to bull/sideways/bear by mean return
        state_means = {s: log_ret[raw_states == s].mean() for s in range(self.n_states)}
        sorted_states = sorted(state_means, key=state_means.get)   # bear→sideways→bull
        self._state_map = {sorted_states[i]: i for i in range(self.n_states)}
        semantic = np.array([self._state_map[s] for s in raw_states])

        # LightGBM: predict regime from features
        feat_cols = [c for c in [
            "log_ret_1m", "log_ret_5m", "log_ret_15m", "log_ret_60m",
            "realized_vol_30m", "rsi_14", "taker_buy_ratio_5m",
            "vpin_50", "hawkes_net", "oi_change_1h", "ls_count_ratio",
        ] if c in df.columns]
        X_lgb = df.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)

        ds = lgb.Dataset(X_lgb, label=semantic)
        params = {
            "objective": "multiclass", "num_class": self.n_states,
            "num_leaves": 31, "learning_rate": 0.05, "verbosity": -1, "n_jobs": -1,
        }
        self._lgb = lgb.train(params, ds, num_boost_round=200)
        self._feat_cols = feat_cols
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, history_df) -> np.ndarray:
        if self._lgb is None:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        if isinstance(history_df, dict):
            X = np.array([[history_df.get(c, 0.0) for c in self._feat_cols]], dtype=np.float32)
        else:
            X = history_df.tail(1).select(self._feat_cols).fill_null(0).to_numpy().astype(np.float32)
        probs = self._lgb.predict(X)[0]          # shape (n_states,)
        return probs

    def predict(self, history_df) -> AgentOpinion:
        t0 = time.perf_counter()
        probs = self.predict_proba(history_df)
        regime_id = int(np.argmax(probs))
        payload = self.signal_dict(history_df)
        ts = self._safe(history_df, "bar_time_ms", 0)
        return AgentOpinion(
            agent=self.name,
            ts_ms=int(ts) if ts else 0,
            payload=payload,
            confidence=float(max(probs)),
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def signal_dict(self, history_df) -> dict:
        probs = self.predict_proba(history_df)
        regime_id = int(np.argmax(probs))
        return {
            "regime": REGIME_LABELS.get(regime_id, "unknown"),
            "regime_id": regime_id,
            "regime_probs": {
                REGIME_LABELS.get(i, f"state_{i}"): float(p)
                for i, p in enumerate(probs)
            },
            "regime_bear_prob": float(probs[0]),
            "regime_sideways_prob": float(probs[1]),
            "regime_bull_prob": float(probs[2]),
            "is_transition": False,
            "vol_regime": "normal",
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "RegimeAgent":
        return pickle.loads(Path(path).read_bytes())