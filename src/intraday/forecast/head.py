"""Forecast head: fuses Kronos embedding + TCN embedding → 11-bin logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """MLP that fuses Kronos and TCN representations into bin logits.

    Architecture:
        [kronos_emb | tcn_emb]  →  Linear(kronos_dim+tcn_dim, hidden)
                                →  GELU  →  Dropout
                                →  Linear(hidden, hidden // 2)
                                →  GELU  →  Dropout
                                →  Linear(hidden // 2, n_bins)

    Returns raw logits (pre-softmax); pass through softmax for probabilities.
    """

    def __init__(
        self,
        kronos_dim: int = 512,
        tcn_dim: int = 64,
        hidden: int = 256,
        n_bins: int = 11,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        in_dim = kronos_dim + tcn_dim
        mid_dim = hidden // 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, n_bins),
        )

        self.n_bins = n_bins
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        kronos_emb: torch.Tensor,
        tcn_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bin logits.

        Args:
            kronos_emb: (batch, kronos_dim) from Kronos backbone.
            tcn_emb:    (batch, tcn_dim)    from SmallTCN.

        Returns:
            (batch, n_bins) raw logits (not softmaxed).
        """
        combined = torch.cat([kronos_emb, tcn_emb], dim=-1)
        return self.net(combined)
