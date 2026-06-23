"""StayOutDetector — rule-based filter to skip low-quality trading conditions."""
from __future__ import annotations
import time

import numpy as np

from intraday.agents.base import Agent, AgentOpinion


class StayOutDetector(Agent):
    name = "stay_out"

    def __init__(self,
                 dead_vol: float = 0.0003,
                 spike_mult: float = 3.0,
                 quiet_hours_utc: tuple = (0, 1, 2, 3, 4),
                 max_oi_change: float = 0.05) -> None:
        self.dead_vol = dead_vol
        self.spike_mult = spike_mult
        self.quiet_hours = set(quiet_hours_utc)
        self.max_oi_chg = max_oi_change

    def _safe(self, df, col, default=0.0):
        if isinstance(df, dict):
            v = df.get(col, default)
            return float(v) if v is not None else default
        if col not in df.columns:
            return default
        v = df[col].tail(1)[0]
        return float(v) if v is not None else default

    def _predict_raw(self, history_df) -> tuple[bool, str]:
        # Dead market: last 3 bars all below vol floor
        if "realized_vol_30m" in history_df.columns:
            last3 = [v for v in history_df["realized_vol_30m"].tail(3).to_list() if v]
            if len(last3) == 3 and all(v < self.dead_vol for v in last3):
                return True, "dead_market"
        elif isinstance(history_df, dict):
            vol = self._safe(history_df, "realized_vol_30m", 0.0)
            if vol < self.dead_vol:
                return True, "dead_market"

        # Quiet UTC hours (low BTC liquidity)
        if "bar_time_ms" in history_df.columns:
            ts = history_df["bar_time_ms"].tail(1)[0]
            if ts is not None:
                hour = (int(ts) // 3_600_000) % 24
                if hour in self.quiet_hours:
                    return True, "quiet_hours"
        elif isinstance(history_df, dict):
            ts = history_df.get("bar_time_ms")
            if ts is not None:
                hour = (int(ts) // 3_600_000) % 24
                if hour in self.quiet_hours:
                    return True, "quiet_hours"

        # Vol spike: last bar > spike_mult × rolling mean
        if "realized_vol_30m" in history_df.columns:
            vols = [v for v in history_df["realized_vol_30m"].tail(31).to_list() if v]
            if len(vols) > 5:
                last_vol = vols[-1]
                mean_vol = np.mean(vols[:-1])
                if mean_vol > 0 and last_vol > self.spike_mult * mean_vol:
                    return True, "vol_spike"
        elif isinstance(history_df, dict):
            vol = self._safe(history_df, "realized_vol_30m", 0.0)
            if vol > self.dead_vol * self.spike_mult:
                return True, "vol_spike"

        # OI extreme change
        oi = abs(self._safe(history_df, "oi_change_1h", 0.0))
        if oi > self.max_oi_chg:
            return True, "oi_extreme"

        return False, ""

    def predict(self, history_df) -> AgentOpinion:
        t0 = time.perf_counter()
        stay_out, reason = self._predict_raw(history_df)
        payload = self.signal_dict(history_df)
        ts = self._safe(history_df, "bar_time_ms", 0)
        return AgentOpinion(
            agent=self.name,
            ts_ms=int(ts) if ts else 0,
            payload=payload,
            confidence=0.0 if stay_out else 1.0,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def signal_dict(self, history_df) -> dict:
        stay_out, reason = self._predict_raw(history_df)
        return {
            "stay_out": stay_out,
            "stay_out_reason": reason,
            "mode": "stay_out" if stay_out else "normal",
            "score": 1.0 if stay_out else 0.0,
        }
