"""CryptoTransformer — intraday direction prediction from 5-min feature sequences.

Architecture:
    Input (B, T, F+6)   ← 5-min bar features + cyclical time embeddings
        ↓  Linear projection → d_model
        ↓  LocalConvBlock (causal k=3, dil=1,2) — captures short-term momentum
        ↓  CLS token prepended → (B, T+1, d_model)
        ↓  Sinusoidal positional encoding
        ↓  N × Pre-LN TransformerEncoderLayer (d, h heads, dim_ff FFN)
        ↓  CLS output → LayerNorm → MLP head → 2 logits

Design notes:
    - Pre-LayerNorm (more stable than post-LN, especially at small batch sizes)
    - LocalConvBlock is depthwise-separable so adds minimal parameter overhead
    - No causal mask on the transformer — all T bars are past observations;
      CLS attends freely to aggregate them
    - Sinusoidal PE is fixed (no extra params, generalises to shifted sequences)
    - Weight init: truncated normal 0.02 (GPT-style)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional encoding ────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ── Local convolutional block ──────────────────────────────────────────────────

class LocalConvBlock(nn.Module):
    """Depthwise causal convs at dilation 1 and 2 for local pattern capture.

    Keeps the same sequence length. The global transformer layers then handle
    long-range dependencies across the full window.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Depthwise (groups=d_model) — cheap, one filter per channel
        self.dw1  = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2,  dilation=1, groups=d_model)
        self.dw2  = nn.Conv1d(d_model, d_model, kernel_size=3, padding=4,  dilation=2, groups=d_model)
        # Pointwise projection fuses channels
        self.pw   = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = x.transpose(1, 2)                    # (B, d, T)
        T = h.size(2)
        h1 = self.act(self.dw1(h)[:, :, :T])     # causal: drop right padding
        h2 = self.act(self.dw2(h)[:, :, :T])
        h  = self.act(self.pw(h1 + h2))
        h  = h.transpose(1, 2)                    # (B, T, d)
        return self.norm(x + self.drop(h))


# ── Pre-LN Transformer layer ───────────────────────────────────────────────────

class PreLNLayer(nn.Module):
    """Transformer encoder block with LayerNorm before attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class CryptoTransformer(nn.Module):
    """Transformer for 5-min crypto bar sequences.

    Args:
        n_features:  Raw feature channels per bar (e.g. 21 from ALL_FEATURES)
        n_time_feat: Cyclical time channels appended (e.g. 6: hour, sin_h, cos_h, dow, sin_dow, cos_dow)
        d_model:     Internal hidden dim (e.g. 256)
        n_heads:     Attention heads — must divide d_model (e.g. 8)
        n_layers:    Transformer depth (e.g. 8)
        dim_ff:      FFN expansion dim (e.g. 1024)
        seq_len:     Context window in bars (e.g. 128 = 10.7h at 5-min)
        n_classes:   2 for binary up/down
        dropout:     Applied after attention, FFN, local conv, and head
    """

    def __init__(
        self,
        n_features:  int   = 21,
        n_time_feat: int   = 6,
        d_model:     int   = 256,
        n_heads:     int   = 8,
        n_layers:    int   = 8,
        dim_ff:      int   = 1024,
        seq_len:     int   = 128,
        n_classes:   int   = 2,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection: features + time → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features + n_time_feat, d_model),
            nn.LayerNorm(d_model),
        )

        # Local short-term pattern extractor
        self.local_conv = LocalConvBlock(d_model, dropout)

        # Positional encoding (seq_len + 1 for CLS token)
        self.pos_enc = SinusoidalPE(d_model, max_len=seq_len + 2)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer stack
        self.layers = nn.ModuleList([
            PreLNLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        time_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, n_features)   — normalised feature windows
            time_feat: (B, T, n_time_feat)  — cyclical time features

        Returns:
            logits: (B, n_classes)
        """
        B = x.size(0)

        if time_feat is not None:
            x = torch.cat([x, time_feat], dim=-1)   # (B, T, F+n_time_feat)

        h = self.input_proj(x)                       # (B, T, d_model)
        h = self.local_conv(h)                       # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, -1)
        h   = torch.cat([cls, h], dim=1)             # (B, T+1, d_model)
        h   = self.pos_enc(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        return self.head(h[:, 0])                    # CLS → (B, n_classes)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
