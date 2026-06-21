"""Temporal Convolutional Network (TCN) encoder for intraday feature sequences.

Architecture: 4 dilated causal 1-D conv layers (dilation 1, 2, 4, 8),
64 channels, kernel size 3, GroupNorm, GELU activation, residual connections,
dropout 0.1. Output: mean-pooled (batch, 64) representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualDilatedBlock(nn.Module):
    """One causal dilated conv block with residual connection.

    Padding is left-padded so that output length == input length
    (causal: each position only attends to itself and past positions).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Causal padding: (kernel - 1) * dilation on the left only
        self._causal_pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # we pad manually for causal masking
        )
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # 1x1 projection for residual when channel dims differ
        self.downsample: nn.Module
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, T)
        residual = self.downsample(x)

        # Causal left-padding
        out = F.pad(x, (self._causal_pad, 0))
        out = self.conv(out)       # (batch, out_channels, T)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)

        return out + residual


class SmallTCN(nn.Module):
    """4-layer dilated TCN.

    Input:  (batch, T, n_features)  — time-major as produced by the dataset
    Output: (batch, 64) — mean-pooled over the time axis
    """

    DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = n_features
        for dilation in self.DILATIONS:
            layers.append(
                _ResidualDilatedBlock(
                    in_channels=in_ch,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = channels

        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, n_features) → transpose to (batch, n_features, T)
        h = x.transpose(1, 2)

        for layer in self.layers:
            h = layer(h)

        # Mean-pool over time → (batch, channels)
        return h.mean(dim=2)
