"""Dual-head GRU/LSTM model for 5-minute BTC return prediction.

Two output heads share one RNN encoder:
  - Regression head  → predicted fwd_ret_5m  (float)
  - Classification head → fwd_direction_5m  (3 classes: down/flat/up)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_features: int = 20
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.3
    model_type: str = "gru"  # "gru" | "lstm"


def _head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )


class DualHeadRNN(nn.Module):
    """GRU or LSTM encoder with regression + classification heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        rnn_cls = nn.GRU if cfg.model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )

        self.reg_head = _head(cfg.hidden_dim, 1,  cfg.dropout)
        self.clf_head = _head(cfg.hidden_dim, 3,  cfg.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            reg_pred:  (batch,)    — predicted fwd_ret_5m
            clf_logits:(batch, 3)  — logits for {down, flat, up}
        """
        out, _ = self.rnn(x)
        h = out[:, -1, :]                          # last timestep hidden state
        reg_pred   = self.reg_head(h).squeeze(-1)
        clf_logits = self.clf_head(h)
        return reg_pred, clf_logits

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
