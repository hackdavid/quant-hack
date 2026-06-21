"""Load Kronos time-series foundation model for embedding extraction.

Uses the Kronos repo cloned at <project_root>/Kronos/.
Weights are loaded from local HuggingFace snapshots.

Two modes:
  frozen (default):   Kronos 100% frozen — only TCN + ForecastHead train.
                      Good for quick runs and smoke tests.
  lora (recommended): Last `unfreeze_top_k` transformer layers get lightweight
                      LoRA adapters (r=16, 0.21% of Kronos params per 4 layers).
                      Lets the model adapt to crypto-specific patterns while
                      keeping the pre-trained temporal knowledge intact.

Embedding pipeline:
  raw OHLCV+amount  →  per-window z-norm + clip  →  KronosTokenizer.encode()
  →  Kronos.decode_s1()  →  mean-pool over time  →  (B, 832) embedding
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

log = structlog.get_logger(__name__)

# __file__ = src/intraday/forecast/kronos_loader.py → parents[3] = project root
_KRONOS_REPO = Path(__file__).parents[3] / "Kronos"
_CLIP = 5.0


# ── Lightweight LoRA layer ─────────────────────────────────────────────────────

class _LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank delta.

    The base weight is kept frozen. Only lora_A and lora_B are trained.
    Output = base(x) + scale * (x @ A.T @ B.T)
    Initialised so the LoRA delta is exactly zero at the start (B = 0).
    """

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: int = 32) -> None:
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        fan_in  = linear.in_features
        fan_out = linear.out_features
        dev     = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, fan_in, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(fan_out, r, device=dev))
        self.scale   = alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora


def _apply_lora(model: nn.Module, top_k: int, r: int, alpha: int) -> int:
    """Replace q_proj + v_proj in the last `top_k` TransformerBlocks with LoRA."""
    n_layers = model.n_layers
    start    = max(0, n_layers - top_k)
    replaced = 0

    for i in range(start, n_layers):
        block = model.transformer[i]
        block.self_attn.q_proj = _LoRALinear(block.self_attn.q_proj, r=r, alpha=alpha)
        block.self_attn.v_proj = _LoRALinear(block.self_attn.v_proj, r=r, alpha=alpha)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


# ── Public API ─────────────────────────────────────────────────────────────────

def _ensure_kronos_on_path() -> None:
    repo = str(_KRONOS_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    if not _KRONOS_REPO.exists():
        raise FileNotFoundError(
            f"Kronos repo not found at {_KRONOS_REPO}. "
            "Clone with: git clone https://github.com/shiyu-coder/Kronos.git"
        )


def load_kronos(
    *,
    model_checkpoint: Path,
    tokenizer_checkpoint: Path,
    device: torch.device | None = None,
    unfreeze_top_k: int = 0,
    lora_rank: int = 16,
    lora_alpha: int = 32,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Load Kronos model + tokenizer from local checkpoints.

    Args:
        model_checkpoint:     Path to Kronos-base model directory.
        tokenizer_checkpoint: Path to Kronos-Tokenizer-base directory.
        device:               Target device (None = cpu).
        unfreeze_top_k:       Number of top transformer layers to fine-tune
                              via LoRA. 0 = fully frozen (fast, smoke test).
                              4 = recommended for GPU training runs.
        lora_rank:            LoRA rank r (default 16).
        lora_alpha:           LoRA alpha scaling (default 32 = 2×rank).

    Returns:
        (kronos_model, tokenizer, config_dict)
        config_dict: {"d_model": 832, "s1_bits": 10, "s2_bits": 10,
                      "lora_trainable": <int>}
    """
    _ensure_kronos_on_path()
    from model import Kronos, KronosTokenizer  # type: ignore[import]

    model_checkpoint     = Path(model_checkpoint)
    tokenizer_checkpoint = Path(tokenizer_checkpoint)

    for p in (model_checkpoint, tokenizer_checkpoint):
        if not p.exists():
            raise FileNotFoundError(
                f"Kronos checkpoint not found: {p}\n"
                "Download:\n"
                "  from huggingface_hub import snapshot_download\n"
                "  snapshot_download('NeoQuasar/Kronos-base',           local_dir='models/kronos-base')\n"
                "  snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='models/kronos-tokenizer')"
            )

    log.info("kronos.loading", model=str(model_checkpoint), tokenizer=str(tokenizer_checkpoint))

    tokenizer: nn.Module = KronosTokenizer.from_pretrained(str(tokenizer_checkpoint))
    model: nn.Module     = Kronos.from_pretrained(str(model_checkpoint))

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    tokenizer.eval()

    # Apply LoRA to top-k layers if requested
    lora_trainable = 0
    if unfreeze_top_k > 0:
        lora_trainable = _apply_lora(model, top_k=unfreeze_top_k, r=lora_rank, alpha=lora_alpha)
        model.train()  # needed so LoRA dropout and BN work correctly
        log.info(
            "kronos.lora_applied",
            top_k=unfreeze_top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_trainable_params=lora_trainable,
        )
    else:
        model.eval()

    if device is not None:
        model     = model.to(device)
        tokenizer = tokenizer.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        "kronos.loaded",
        total_params_M=round(total_params / 1e6, 1),
        lora_params=lora_trainable,
        d_model=model.d_model,
        mode="lora" if unfreeze_top_k > 0 else "frozen",
    )

    cfg: dict[str, Any] = {
        "d_model":        model.d_model,
        "s1_bits":        model.s1_bits,
        "s2_bits":        model.s2_bits,
        "lora_trainable": lora_trainable,
    }
    return model, tokenizer, cfg


def kronos_embed(
    model: nn.Module,
    tokenizer: nn.Module,
    klines_raw: torch.Tensor,
    stamps: torch.Tensor,
    clip: float = _CLIP,
    no_grad: bool = True,
) -> torch.Tensor:
    """Get mean-pooled Kronos embedding for a batch of kline windows.

    Args:
        model:       Kronos model (frozen or LoRA-adapted).
        tokenizer:   Frozen KronosTokenizer.
        klines_raw:  (B, T, 6) float32 — OHLCV+amount, **unnormalized**.
        stamps:      (B, T, 5) float32 — minute/hour/weekday/day/month.
        clip:        z-score clip (default 5, matches Kronos training).
        no_grad:     Use torch.no_grad() for the Kronos forward pass.
                     Set False when LoRA gradients need to flow.

    Returns:
        (B, d_model=832) float32 — mean-pooled transformer context.
    """
    # Per-window z-normalisation (matches Kronos training convention)
    mean  = klines_raw.mean(dim=1, keepdim=True)
    std   = klines_raw.std(dim=1, keepdim=True) + 1e-5
    x_norm = ((klines_raw - mean) / std).clamp(-clip, clip)

    # Tokenizer is always frozen
    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(x_norm, half=True)

    # Kronos forward — grad only flows if LoRA is active
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        _s1_logits, context = model.decode_s1(s1_ids, s2_ids, stamps)  # (B, T, d_model)

    return context.mean(dim=1)  # (B, 832)
