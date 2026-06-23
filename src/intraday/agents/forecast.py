"""ForecastAgent — wraps CryptoTransformer checkpoint for live inference.

Drop-in replacement for the original Kronos+LoRA ForecastAgent.
Loads best.pt and exposes predict(feature_window_df) → prob_up.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from intraday.agents.base import Agent, AgentOpinion
from intraday.forecast.output import ForecastOutput


class ForecastAgent(Agent):
    """Wraps a trained CryptoTransformer for per-bar inference.

    Args:
        run_dir:  Path to transformer run directory (contains best.pt + config.json)
        device:   'cuda' or 'cpu'
    """

    name = "forecast"

    def __init__(self, run_dir: str | Path, device: str = "cuda") -> None:
        import json
        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        cfg = ckpt["config"]

        self.feat_cols = cfg["feat_cols"]
        self.seq_len = cfg["seq_len"]
        self.n_time = cfg["n_time_feat"]
        self.norm_mean = ckpt["norm_mean"].astype(np.float32)
        self.norm_std = ckpt["norm_std"].astype(np.float32)
        self.val_auc = ckpt.get("best_val_auc", 0.5)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        from intraday.forecast.transformer_model import CryptoTransformer
        self._model = CryptoTransformer(
            n_features=cfg["n_features"], n_time_feat=cfg["n_time_feat"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dim_ff=cfg["dim_ff"],
            seq_len=cfg["seq_len"], dropout=0.0,
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

    @torch.no_grad()
    def _predict_raw(self, history_df) -> float:
        """Return P(up) ∈ [0,1] from the last seq_len bars of history_df."""
        import polars as pl
        df = history_df.tail(self.seq_len) if hasattr(history_df, "tail") else history_df
        if len(df) < self.seq_len:
            return 0.5   # not enough history

        feat = df.select(self.feat_cols).fill_null(0).to_numpy().astype(np.float32)
        feat = (feat - self.norm_mean) / np.where(self.norm_std > 1e-8, self.norm_std, 1.0)
        feat = np.clip(feat, -8.0, 8.0)

        ts = df["bar_time_ms"].to_numpy().astype(np.float64)
        hour = ((ts // 3_600_000) % 24).astype(np.float32)
        dow = ((ts // 86_400_000) % 7).astype(np.float32)
        tf = np.stack([
            hour / 23.0, np.sin(2*math.pi*hour/24), np.cos(2*math.pi*hour/24),
            dow / 6.0,   np.sin(2*math.pi*dow/7),   np.cos(2*math.pi*dow/7),
        ], axis=1).astype(np.float32)

        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        tf_ = torch.from_numpy(tf).unsqueeze(0).to(self.device)
        logits = self._model(x, tf_)
        return float(torch.softmax(logits, -1)[0, 1].cpu())

    def predict(self, history_df) -> AgentOpinion:
        t0 = time.perf_counter()
        prob = self._predict_raw(history_df)
        payload = self.signal_dict(history_df)
        ts = 0
        if hasattr(history_df, "columns") and "bar_time_ms" in history_df.columns:
            ts = history_df["bar_time_ms"].tail(1)[0]
        elif isinstance(history_df, dict):
            ts = history_df.get("bar_time_ms", 0)
        return AgentOpinion(
            agent=self.name,
            ts_ms=int(ts) if ts else 0,
            payload=payload,
            confidence=abs(prob - 0.5) * 2.0,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def signal_dict(self, history_df) -> dict:
        prob = self._predict_raw(history_df)
        return {
            "forecast_prob_up": prob,
            "forecast_signal": 1 if prob > 0.55 else (-1 if prob < 0.45 else 0),
        }
