"""Inference interface for the trained forecast model.

Loads all artefacts from a model directory and provides a single
``predict()`` method that returns a fully populated ``ForecastOutput``.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import torch

from intraday.features.schema import ALL_FEATURES
from intraday.forecast.calibration import IsotonicCalibrator
from intraday.forecast.head import ForecastHead
from intraday.forecast.meta_label import META_FEATURE_COLS, MetaLabelClassifier
from intraday.forecast.output import ForecastOutput
from intraday.forecast.tcn import SmallTCN

log = structlog.get_logger(__name__)

_KLINE_COLS = ["open", "high", "low", "close", "volume"]
N_BINS = 11


class ForecastModel:
    """All-in-one inference wrapper.

    Loads Kronos + LoRA, TCN, head, meta-label classifier and isotonic
    calibrator from ``model_dir`` on construction.  Thread-safe for read
    access (no mutable state during ``predict``).

    Args:
        model_dir: Directory produced by ``train_forecast()``.
    """

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # ── Metadata ───────────────────────────────────────────────────────
        meta_path = model_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {model_dir}")
        with open(meta_path) as fh:
            self._meta: dict = json.load(fh)

        self._model_version: str = self._meta.get("model_version", "unknown")
        self._seq_klines: int = int(self._meta.get("seq_klines", 256))
        self._seq_state: int = int(self._meta.get("seq_state", 128))
        n_features: int = int(self._meta.get("n_features", len(ALL_FEATURES)))
        kronos_hidden: int = int(self._meta.get("kronos_hidden", 512))

        # ── Device ─────────────────────────────────────────────────────────
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        log.info("forecast_model.loading", model_dir=str(model_dir), device=str(self._device))

        # ── TCN + Head ─────────────────────────────────────────────────────
        self._tcn = SmallTCN(n_features=n_features, channels=64, dropout=0.0)
        self._head = ForecastHead(kronos_dim=kronos_hidden, tcn_dim=64, hidden=256, n_bins=N_BINS, dropout=0.0)

        self._tcn = self._load_weights(self._tcn, model_dir, "tcn")
        self._head = self._load_weights(self._head, model_dir, "head")
        self._tcn = self._tcn.to(self._device).eval()
        self._head = self._head.to(self._device).eval()

        # ── Kronos + LoRA ──────────────────────────────────────────────────
        self._kronos: torch.nn.Module | None = None
        kronos_lora_dir = model_dir / "kronos_lora"
        kronos_pt = model_dir / "kronos_lora.pt"
        if kronos_lora_dir.exists():
            try:
                from peft import PeftModel
                from transformers import AutoModel
                base_cfg_path = self._meta.get("kronos_base_checkpoint")
                if base_cfg_path and Path(base_cfg_path).exists():
                    base = AutoModel.from_pretrained(base_cfg_path, torch_dtype=torch.float32)
                    self._kronos = PeftModel.from_pretrained(base, str(kronos_lora_dir))
                else:
                    # Try loading purely as a HuggingFace pretrained model
                    from transformers import AutoModel
                    self._kronos = AutoModel.from_pretrained(str(kronos_lora_dir), torch_dtype=torch.float32)
                self._kronos = self._kronos.to(self._device).eval()
                log.info("forecast_model.kronos_loaded", source=str(kronos_lora_dir))
            except Exception as exc:
                log.warning("forecast_model.kronos_load_failed", error=str(exc))
                self._kronos = None
        elif kronos_pt.exists():
            log.warning("forecast_model.kronos_pt_fallback", path=str(kronos_pt))
        else:
            log.warning("forecast_model.kronos_not_found", detail="will use zero embedding")

        # ── Normalisers ────────────────────────────────────────────────────
        self._klines_norm = self._load_pickle(model_dir / "klines_norm.pkl")
        self._state_norm = self._load_pickle(model_dir / "state_norm.pkl")

        # ── Meta-label classifier ──────────────────────────────────────────
        meta_path2 = model_dir / "meta_label.lgbm"
        self._meta_clf: MetaLabelClassifier | None = None
        if meta_path2.exists():
            self._meta_clf = MetaLabelClassifier.load(meta_path2)
        else:
            log.warning("forecast_model.meta_label_not_found")

        # ── Calibrator ─────────────────────────────────────────────────────
        cal_path = model_dir / "calibrator.pkl"
        self._calibrator: IsotonicCalibrator | None = None
        if cal_path.exists():
            self._calibrator = IsotonicCalibrator.load(cal_path)
        else:
            log.warning("forecast_model.calibrator_not_found")

        log.info("forecast_model.ready", model_version=self._model_version)

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(
        self,
        *,
        klines_window: pl.DataFrame,
        state_window: pl.DataFrame,
        ts_ms: int,
        horizon_minutes: int = 15,
    ) -> ForecastOutput:
        """Run a full forward pass and return a ForecastOutput.

        Args:
            klines_window:   Last N 1m klines (columns: bar_time_ms, open, high,
                             low, close, volume).  At least 1 row; will be
                             zero-padded if shorter than seq_klines.
            state_window:    Last M 5m state rows (ALL_FEATURES columns).
            ts_ms:           Current timestamp in ms UTC.
            horizon_minutes: Forecast horizon (5 | 15 | 60).

        Returns:
            Fully populated ForecastOutput.
        """
        t0 = time.perf_counter()

        # ── Prepare klines tensor ──────────────────────────────────────────
        klines_np = self._prep_klines(klines_window)   # (seq_klines, 5)
        state_np = self._prep_state(state_window)       # (seq_state, n_features)

        klines_t = torch.from_numpy(klines_np).unsqueeze(0).to(self._device)  # (1, T, 5)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(self._device)    # (1, T, n_f)

        with torch.no_grad():
            # Kronos embedding
            if self._kronos is not None:
                from intraday.forecast.kronos_loader import kronos_embed
                kronos_emb = kronos_embed(self._kronos, klines_t)  # (1, hidden)
            else:
                # Fallback: zero vector of expected dim
                hidden = self._meta.get("kronos_hidden", 512)
                kronos_emb = torch.zeros(1, hidden, device=self._device)

            tcn_emb = self._tcn(state_t)              # (1, 64)
            logits = self._head(kronos_emb, tcn_emb)  # (1, 11)

        logits_list: list[float] = logits.squeeze(0).cpu().tolist()

        # ── Meta-label ─────────────────────────────────────────────────────
        meta_features = self._build_meta_features(state_window, logits_list)
        meta_p_correct = 0.5
        meta_act = False

        if self._meta_clf is not None:
            try:
                meta_proba = self._meta_clf.predict_proba(meta_features)
                raw_meta_p = float(meta_proba[0])
                if self._calibrator is not None:
                    import math
                    probs_np = np.array(
                        [np.exp(v) for v in logits_list], dtype=np.float64
                    )
                    probs_np /= probs_np.sum()
                    raw_max_p = float(probs_np.max())
                    meta_p_correct = float(self._calibrator.transform(np.array([raw_meta_p]))[0])
                else:
                    meta_p_correct = raw_meta_p
                meta_act = meta_p_correct >= 0.55
            except Exception as exc:
                log.warning("forecast_model.meta_label_inference_error", error=str(exc))

        inference_ms = (time.perf_counter() - t0) * 1000.0

        return ForecastOutput.from_logits(
            logits_list,
            ts_ms=ts_ms,
            horizon_minutes=horizon_minutes,
            meta_act=meta_act,
            meta_p_correct=meta_p_correct,
            model_version=self._model_version,
            inference_ms=round(inference_ms, 2),
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prep_klines(self, df: pl.DataFrame) -> np.ndarray:
        """Normalise and pad/truncate klines to (seq_klines, 5)."""
        available = [c for c in _KLINE_COLS if c in df.columns]
        arr = df.select(available).fill_null(0.0).to_numpy().astype(np.float64)

        # Trim to last seq_klines rows
        if len(arr) > self._seq_klines:
            arr = arr[-self._seq_klines:]

        if self._klines_norm is not None:
            arr = self._klines_norm.normalise(arr)

        # Pad left with zeros if needed
        if len(arr) < self._seq_klines:
            pad = np.zeros((self._seq_klines - len(arr), arr.shape[1]), dtype=np.float64)
            arr = np.concatenate([pad, arr], axis=0)

        return arr.astype(np.float32)

    def _prep_state(self, df: pl.DataFrame) -> np.ndarray:
        """Normalise and pad/truncate state to (seq_state, n_features)."""
        available = [c for c in ALL_FEATURES if c in df.columns]
        arr = df.select(available).fill_null(0.0).to_numpy().astype(np.float64)

        if len(arr) > self._seq_state:
            arr = arr[-self._seq_state:]

        if self._state_norm is not None:
            arr = self._state_norm.normalise(arr)

        if len(arr) < self._seq_state:
            pad = np.zeros((self._seq_state - len(arr), arr.shape[1]), dtype=np.float64)
            arr = np.concatenate([pad, arr], axis=0)

        return arr.astype(np.float32)

    def _build_meta_features(
        self, state_window: pl.DataFrame, logits: list[float]
    ) -> pl.DataFrame:
        """Build a 1-row meta-label feature DataFrame."""
        import math

        probs = np.exp(np.array(logits, dtype=np.float64))
        probs /= probs.sum()
        BIN_CENTRES = [-4.0, -2.5, -1.5, -0.75, -0.35, 0.0, 0.35, 0.75, 1.5, 2.5, 4.0]

        entropy = -sum(float(p) * math.log(float(p)) for p in probs if float(p) > 1e-12)
        confidence = 1.0 - entropy / math.log(N_BINS)
        expected_move = float(sum(float(p) * c for p, c in zip(probs, BIN_CENTRES)))

        last_row: dict = {}
        if len(state_window) > 0:
            last_row = state_window.tail(1).row(0, named=True)

        data: dict[str, list] = {
            "fc_confidence": [confidence],
            "fc_expected_move_sigma": [expected_move],
            "vol_regime_id": [0.0],
            "funding_rate": [float(last_row.get("funding_rate") or 0.0)],
            "rsi_14": [float(last_row.get("rsi_14") or 50.0)],
            "log_ret_60m": [float(last_row.get("log_ret_60m") or 0.0)],
            "realized_vol_30m": [float(last_row.get("realized_vol_30m") or 0.0)],
            "hour_utc": [float((int(time.time()) // 3600) % 24)],
        }
        return pl.DataFrame(data)

    @staticmethod
    def _load_weights(model: torch.nn.Module, model_dir: Path, name: str) -> torch.nn.Module:
        """Try safetensors first, then .pt fallback."""
        st_path = model_dir / f"{name}.safetensors"
        pt_path = model_dir / f"{name}.pt"
        if st_path.exists():
            try:
                from safetensors.torch import load_file
                state = load_file(str(st_path))
                model.load_state_dict(state, strict=True)
                log.info(f"forecast_model.{name}_loaded", source="safetensors")
                return model
            except Exception as exc:
                log.warning(f"forecast_model.{name}_safetensors_failed", error=str(exc))
        if pt_path.exists():
            state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            log.info(f"forecast_model.{name}_loaded", source="pt")
            return model
        log.warning(f"forecast_model.{name}_not_found", tried=[str(st_path), str(pt_path)])
        return model

    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ── Module-level loader ────────────────────────────────────────────────────────

def load_forecast(
    version: str = "latest",
    models_dir: Path = Path("models/forecast"),
) -> ForecastModel:
    """Locate and load a ForecastModel by version string.

    Args:
        version:    "latest" picks the most recently modified sub-directory;
                    otherwise treated as a sub-directory name or absolute path.
        models_dir: Root directory for forecast model versions.

    Returns:
        Loaded ForecastModel instance.
    """
    models_dir = Path(models_dir)

    if version == "latest":
        candidates = sorted(
            [d for d in models_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()],
            key=lambda d: d.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No forecast model versions found in {models_dir}. "
                "Run: intraday forecast train"
            )
        model_dir = candidates[-1]
    else:
        model_dir = models_dir / version
        if not model_dir.exists():
            # Maybe it's an absolute path
            model_dir = Path(version)
        if not model_dir.exists():
            raise FileNotFoundError(f"Forecast model version not found: {version}")

    log.info("load_forecast", version=version, model_dir=str(model_dir))
    return ForecastModel(model_dir)
