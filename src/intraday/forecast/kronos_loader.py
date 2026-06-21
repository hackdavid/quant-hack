"""Load Kronos time-series foundation model for embedding extraction.

Uses the Kronos repo cloned at <project_root>/Kronos/.
Weights are loaded from local HuggingFace snapshots.
Both model and tokenizer are frozen; only TCN + ForecastHead are trained.

Embedding pipeline:
  raw OHLCV+amount  →  per-window z-norm + clip  →  KronosTokenizer.encode()
  →  Kronos.decode_s1()  →  mean-pool over time  →  (B, 832) embedding
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

log = structlog.get_logger(__name__)

# Path to the user-cloned Kronos repo (project_root/Kronos)
# __file__ = src/intraday/forecast/kronos_loader.py → parents[3] = project root
_KRONOS_REPO = Path(__file__).parents[3] / "Kronos"
_CLIP = 5.0


def _ensure_kronos_on_path() -> None:
    repo = str(_KRONOS_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if not _KRONOS_REPO.exists():
        raise FileNotFoundError(
            f"Kronos repo not found at {_KRONOS_REPO}. "
            "Clone it with: git clone https://github.com/shiyu-coder/Kronos.git"
        )


def load_kronos(
    *,
    model_checkpoint: Path,
    tokenizer_checkpoint: Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Load frozen Kronos model + tokenizer from local checkpoints.

    Args:
        model_checkpoint:     Path to Kronos-base model directory.
        tokenizer_checkpoint: Path to Kronos-Tokenizer-base directory.
        device:               Target device. None keeps default (cpu).

    Returns:
        (kronos_model, tokenizer, config_dict)
        config_dict has at least: {"d_model": 832, "s1_bits": 10, "s2_bits": 10}

    Raises:
        FileNotFoundError: If checkpoint dirs or Kronos repo are missing.
    """
    _ensure_kronos_on_path()
    from model import Kronos, KronosTokenizer  # type: ignore[import]

    model_checkpoint = Path(model_checkpoint)
    tokenizer_checkpoint = Path(tokenizer_checkpoint)

    for p in (model_checkpoint, tokenizer_checkpoint):
        if not p.exists():
            raise FileNotFoundError(
                f"Kronos checkpoint not found: {p}\n"
                "Download with:\n"
                "  python -c \"from huggingface_hub import snapshot_download; "
                "snapshot_download('NeoQuasar/Kronos-base', local_dir='models/kronos-base')\""
            )

    log.info(
        "kronos.loading",
        model=str(model_checkpoint),
        tokenizer=str(tokenizer_checkpoint),
    )

    tokenizer: nn.Module = KronosTokenizer.from_pretrained(str(tokenizer_checkpoint))
    model: nn.Module = Kronos.from_pretrained(str(model_checkpoint))

    # Freeze everything — only TCN + ForecastHead are trained
    for p in model.parameters():
        p.requires_grad_(False)
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    model.eval()
    tokenizer.eval()

    if device is not None:
        model = model.to(device)
        tokenizer = tokenizer.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "kronos.loaded",
        params_M=round(n_params / 1e6, 1),
        d_model=model.d_model,
        frozen=True,
    )

    cfg: dict[str, Any] = {
        "d_model": model.d_model,
        "s1_bits": model.s1_bits,
        "s2_bits": model.s2_bits,
    }
    return model, tokenizer, cfg


def kronos_embed(
    model: nn.Module,
    tokenizer: nn.Module,
    klines_raw: torch.Tensor,
    stamps: torch.Tensor,
    clip: float = _CLIP,
) -> torch.Tensor:
    """Get mean-pooled Kronos embedding for a batch of kline windows.

    Args:
        model:       Frozen Kronos model.
        tokenizer:   Frozen KronosTokenizer.
        klines_raw:  (B, T, 6) float32 — open/high/low/close/volume/amount,
                     **unnormalized**. Normalisation is applied per-window here.
        stamps:      (B, T, 5) float32 — minute/hour/weekday/day/month.
        clip:        z-score clip threshold (default 5, matching Kronos training).

    Returns:
        (B, d_model=832) float32 — mean-pooled transformer context.
    """
    # Per-window z-normalisation (matches Kronos training convention)
    mean = klines_raw.mean(dim=1, keepdim=True)          # (B, 1, 6)
    std  = klines_raw.std(dim=1, keepdim=True) + 1e-5    # (B, 1, 6)
    x_norm = ((klines_raw - mean) / std).clamp(-clip, clip)  # (B, T, 6)

    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(x_norm, half=True)          # both (B, T)
        _s1_logits, context = model.decode_s1(s1_ids, s2_ids, stamps) # (B, T, d_model)

    return context.mean(dim=1)  # (B, d_model=832)
