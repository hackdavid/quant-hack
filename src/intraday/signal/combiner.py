"""Signal combiner: loads trained models and produces a blended probability.

Supports:
  - CryptoTransformer (best.pt checkpoint)
  - LightGBM GBDT / DART (lgb_model.txt or lgb_gbdt.txt)
  - Weighted average blend (weighted by val AUC)

Usage:
    combiner = SignalCombiner(
        transformer_dir="models/transformer/20260623T132957Z",
        lgb_dir="models/lgb",           # optional
        lgb_ensemble_dir="models/gbm_ensemble",  # optional
    )
    prob = combiner.predict_from_df(feature_window_df)  # returns float in [0,1]
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from intraday.features.schema import ALL_FEATURES


# ── Lag / rolling feature names (must match train_lgb.py) ─────────────────────
_LAG_FEATURES = [
    "ls_count_ratio", "oi_change_1h", "depth_imbalance_1pct",
    "vpin_50", "taker_buy_ratio_5m", "realized_vol_30m",
    "hawkes_net", "log_ret_5m", "log_ret_15m", "log_ret_60m",
    "taker_ls_vol_ratio", "vpin_bucket_imbalance",
]
_LAG_PERIODS = [1, 3, 6, 12, 24]

_ROLL_FEATURES = [
    "taker_buy_ratio_5m", "vpin_50", "ls_count_ratio",
    "hawkes_net", "depth_imbalance_1pct", "oi_change_1h",
]
_ROLL_WINDOWS = [6, 12, 24, 48]


def _build_lgb_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lag + rolling features to a DataFrame (same as train_lgb.py)."""
    exprs = []
    for feat in _LAG_FEATURES:
        if feat not in df.columns:
            continue
        for lag in _LAG_PERIODS:
            exprs.append(pl.col(feat).shift(lag).alias(f"{feat}_lag{lag}"))
    for feat in _ROLL_FEATURES:
        if feat not in df.columns:
            continue
        for w in _ROLL_WINDOWS:
            exprs.append(pl.col(feat).rolling_mean(w).alias(f"{feat}_rmean{w}"))
            exprs.append(pl.col(feat).rolling_std(w).alias(f"{feat}_rstd{w}"))
    ms   = pl.col("bar_time_ms")
    hour = (ms // 3_600_000) % 24
    dow  = (ms // 86_400_000) % 7
    exprs += [
        hour.cast(pl.Float32).alias("hour_utc"),
        (hour.cast(pl.Float64) * (2 * math.pi / 24)).sin().cast(pl.Float32).alias("sin_hour"),
        (hour.cast(pl.Float64) * (2 * math.pi / 24)).cos().cast(pl.Float32).alias("cos_hour"),
        dow.cast(pl.Float32).alias("day_of_week"),
        (dow.cast(pl.Float64) * (2 * math.pi / 7)).sin().cast(pl.Float32).alias("sin_dow"),
        (dow.cast(pl.Float64) * (2 * math.pi / 7)).cos().cast(pl.Float32).alias("cos_dow"),
    ]
    return df.with_columns(exprs)


class TransformerPredictor:
    def __init__(self, run_dir: Path, device: str = "cuda") -> None:
        from intraday.forecast.transformer_model import CryptoTransformer

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]

        self.feat_cols  = cfg["feat_cols"]
        self.seq_len    = cfg["seq_len"]
        self.norm_mean  = ckpt["norm_mean"].astype(np.float32)
        self.norm_std   = ckpt["norm_std"].astype(np.float32)
        self.val_auc    = ckpt.get("best_val_auc", 0.5)

        self.model = CryptoTransformer(
            n_features  = cfg["n_features"],
            n_time_feat = cfg["n_time_feat"],
            d_model     = cfg["d_model"],
            n_heads     = cfg["n_heads"],
            n_layers    = cfg["n_layers"],
            dim_ff      = cfg["dim_ff"],
            seq_len     = cfg["seq_len"],
            dropout     = 0.0,   # no dropout at inference
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_window(self, feat_window: np.ndarray, ts_window: np.ndarray) -> float:
        """
        Args:
            feat_window: (seq_len, n_features) float32
            ts_window:   (seq_len,) int64 bar_time_ms
        Returns:
            prob_up: float in [0, 1]
        """
        feat = (feat_window - self.norm_mean) / np.where(self.norm_std > 1e-8, self.norm_std, 1.0)
        feat = np.clip(feat, -8.0, 8.0).astype(np.float32)

        hour = ((ts_window // 3_600_000) % 24).astype(np.float32)
        dow  = ((ts_window // 86_400_000) % 7).astype(np.float32)
        time_feat = np.stack([
            hour / 23.0,
            np.sin(2 * math.pi * hour / 24),
            np.cos(2 * math.pi * hour / 24),
            dow / 6.0,
            np.sin(2 * math.pi * dow / 7),
            np.cos(2 * math.pi * dow / 7),
        ], axis=1).astype(np.float32)

        x  = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        tf = torch.from_numpy(time_feat).unsqueeze(0).to(self.device)
        logits = self.model(x, tf)
        return float(torch.softmax(logits, dim=-1)[0, 1].cpu())


class LGBPredictor:
    def __init__(self, model_path: Path, feat_cols: list[str], val_auc: float = 0.5) -> None:
        import lightgbm as lgb
        self.model     = lgb.Booster(model_file=str(model_path))
        self.feat_cols = feat_cols
        self.val_auc   = val_auc

    def predict_row(self, df_row: pl.DataFrame) -> float:
        """df_row: single-row DataFrame with all required feature columns."""
        X = df_row.select(self.feat_cols).fill_null(0).to_numpy().astype(np.float32)
        return float(self.model.predict(X)[0])


class SignalCombiner:
    """Blends transformer + LGB predictions weighted by their val AUC.

    Pass at least one of transformer_dir or lgb_dir.
    """

    def __init__(
        self,
        transformer_dir:  str | Path | None = None,
        lgb_dir:          str | Path | None = None,
        lgb_ensemble_dir: str | Path | None = None,
        device:           str = "cuda",
    ) -> None:
        self._transformer: TransformerPredictor | None = None
        self._lgb: LGBPredictor | None = None

        if transformer_dir:
            print(f"Loading transformer from {transformer_dir}...")
            self._transformer = TransformerPredictor(Path(transformer_dir), device)
            print(f"  val AUC: {self._transformer.val_auc:.4f}")

        if lgb_ensemble_dir:
            p = Path(lgb_ensemble_dir)
            model_path = p / "lgb_gbdt.txt"
            if not model_path.exists():
                model_path = p / "lgb_model.txt"
            meta = json.loads((p / "results.json").read_text()) if (p / "results.json").exists() else {}
            feat_cols = meta.get("feat_cols", [])
            val_auc   = meta.get("models", {}).get("lgb_gbdt", {}).get("holdout_auc", 0.5)
            print(f"Loading LGB ensemble from {lgb_ensemble_dir}...")
            self._lgb = LGBPredictor(model_path, feat_cols, val_auc)
            print(f"  val AUC: {val_auc:.4f}")
        elif lgb_dir:
            p = Path(lgb_dir)
            model_path = p / "lgb_model.txt"
            meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
            feat_cols = meta.get("feat_cols", [])
            val_auc   = meta.get("val_auc", 0.5)
            print(f"Loading LGB from {lgb_dir}...")
            self._lgb = LGBPredictor(model_path, feat_cols, val_auc)
            print(f"  val AUC: {val_auc:.4f}")

        if not self._transformer and not self._lgb:
            raise ValueError("Provide at least one of transformer_dir or lgb_dir")

    @property
    def _weights(self) -> tuple[float, float]:
        """AUC-based weights for (transformer, lgb)."""
        t_auc = self._transformer.val_auc if self._transformer else 0.0
        l_auc = self._lgb.val_auc         if self._lgb         else 0.0
        total = t_auc + l_auc
        if total == 0:
            return 0.5, 0.5
        return t_auc / total, l_auc / total

    def predict_from_df(self, history_df: pl.DataFrame) -> tuple[float, dict]:
        """Generate blended prediction from a rolling feature DataFrame.

        Args:
            history_df: DataFrame with at least seq_len rows of feature data,
                        sorted ascending by bar_time_ms. Must have all ALL_FEATURES columns.
        Returns:
            (prob_up, components_dict)
        """
        components = {}
        preds, weights = [], []

        if self._transformer:
            seq_len   = self._transformer.seq_len
            feat_cols = self._transformer.feat_cols
            df_tail   = history_df.tail(seq_len)
            if len(df_tail) < seq_len:
                raise ValueError(f"Need {seq_len} bars, got {len(df_tail)}")
            feat_window = df_tail.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)
            ts_window   = df_tail["bar_time_ms"].to_numpy().astype(np.int64)
            p = self._transformer.predict_window(feat_window, ts_window)
            components["transformer"] = p
            t_w, _ = self._weights
            preds.append(p * t_w)
            weights.append(t_w)

        if self._lgb:
            df_with_lags = _build_lgb_features(history_df)
            last_row     = df_with_lags.tail(1)
            p = self._lgb.predict_row(last_row)
            components["lgb"] = p
            _, l_w = self._weights
            preds.append(p * l_w)
            weights.append(l_w)

        blended = sum(preds) / sum(weights) if weights else 0.5
        components["blended"] = blended
        return blended, components
