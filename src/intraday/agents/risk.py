"""RiskAgent — rule-based risk scoring from volatility and market conditions."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from intraday.agents.base import Agent, AgentOpinion


@dataclass
class RiskConfig:
    vol_cap: float = 0.008
    vol_floor: float = 0.0002
    max_oi_change: float = 0.05


class RiskAgent(Agent):
    name = "risk"

    def __init__(self, vol_cap: float = 0.008, vol_floor: float = 0.0002) -> None:
        self.vol_cap = vol_cap
        self.vol_floor = vol_floor

    def _safe(self, df, col, default=0.0):
        if isinstance(df, dict):
            v = df.get(col, default)
            return float(v) if v is not None else default
        if col not in df.columns:
            return default
        v = df[col].tail(1)[0]
        return float(v) if v is not None else default

    def _predict_raw(self, history_df) -> float:
        vol = self._safe(history_df, "realized_vol_30m", 0.001)

        # Too volatile → max risk
        if vol > self.vol_cap:
            return 1.0

        # Dead market → liquidity risk
        if vol < self.vol_floor:
            return 0.8

        # Momentum streak: 4 of last 5 same sign → mean reversion risk
        if "log_ret_5m" in history_df.columns:
            rets = history_df["log_ret_5m"].tail(5).to_list()
            rets = [r for r in rets if r is not None]
            if len(rets) >= 4:
                signs = [1 if r > 0 else -1 for r in rets]
                if abs(sum(signs)) >= 3:
                    return 0.65

        # OI spike: sudden large position unwinding
        oi_chg = abs(self._safe(history_df, "oi_change_1h", 0.0))
        if oi_chg > 0.05:
            return 0.75

        return float(np.clip(vol / self.vol_cap, 0.0, 1.0))

    def predict(self, history_df) -> AgentOpinion:
        t0 = time.perf_counter()
        score = self._predict_raw(history_df)
        payload = self.signal_dict(history_df)
        ts = self._safe(history_df, "bar_time_ms", 0)
        return AgentOpinion(
            agent=self.name,
            ts_ms=int(ts) if ts else 0,
            payload=payload,
            confidence=1.0 - score,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def signal_dict(self, history_df) -> dict:
        score = self._predict_raw(history_df)
        return {
            "risk_score": score,
            "risk_ok": score < 0.70,
            "risk_multiplier": 1.0 - score,
            "allow_trade": score < 0.70,
            "stop_trading": score > 0.90,
        }
