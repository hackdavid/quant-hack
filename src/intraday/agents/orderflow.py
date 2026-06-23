"""OrderflowAgent — rule-based signal from taker flow, depth, VPIN, Hawkes."""
from __future__ import annotations
import time
import numpy as np

from intraday.agents.base import Agent, AgentOpinion


class OrderflowAgent(Agent):
    name = "orderflow"

    def __init__(self, taker_long=0.60, taker_short=0.40,
                 depth_thresh=0.25, vpin_thresh=0.70, hawkes_scale=0.02):
        self.tl, self.ts, self.dt = taker_long, taker_short, depth_thresh
        self.vt, self.hs = vpin_thresh, hawkes_scale

    def _safe(self, df, col, default=0.0):
        if isinstance(df, dict):
            v = df.get(col, default)
            return float(v) if v is not None else default
        if col not in df.columns:
            return default
        v = df[col].tail(1)[0]
        return float(v) if v is not None else default

    def _predict_raw(self, history_df) -> float:
        t  = self._safe(history_df, "taker_buy_ratio_5m", 0.5)
        d  = self._safe(history_df, "depth_imbalance_1pct", 0.0)
        vp = self._safe(history_df, "vpin_50", 0.0)
        hn = self._safe(history_df, "hawkes_net", 0.0)
        ls = self._safe(history_df, "ls_count_ratio", 1.0)
        scores = []
        scores.append((t - self.tl) / (1 - self.tl) if t > self.tl else
                      (-(self.ts - t) / self.ts if t < self.ts else 0.0))
        scores.append(float(np.clip(d / max(self.dt, 1e-6), -1.0, 1.0)))
        scores.append(float(np.sign(t - 0.5) * min((vp - self.vt) / (1 - self.vt), 1.0))
                      if vp > self.vt else 0.0)
        scores.append(float(np.clip(hn / max(self.hs, 1e-6), -1.0, 1.0)))
        scores.append(float(np.clip((ls - 1.0) / 0.5, -1.0, 1.0)))
        return float(np.mean(scores))

    def predict(self, history_df) -> AgentOpinion:
        t0 = time.perf_counter()
        raw = self._predict_raw(history_df)
        payload = self.signal_dict(history_df)
        ts = self._safe(history_df, "bar_time_ms", 0)
        return AgentOpinion(
            agent=self.name,
            ts_ms=int(ts) if ts else 0,
            payload=payload,
            confidence=min(abs(raw) / 0.5, 1.0),
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def signal_dict(self, history_df) -> dict:
        s = self._predict_raw(history_df)
        return {
            "flow_bias": float(np.sign(s)),
            "flow_strength": abs(s),
            "step_away": abs(s) < 0.1,
            "vpin": self._safe(history_df, "vpin_50", 0.0),
        }
