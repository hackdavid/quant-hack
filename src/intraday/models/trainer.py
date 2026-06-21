"""Training loop, metrics, and checkpointing for DualHeadRNN."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from intraday.models.gru import DualHeadRNN, ModelConfig


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "data"
    symbol: str = "BTCUSDT"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    seq_len: int = 60

    # ── Model ──────────────────────────────────────────────────────────────────
    model_type: str = "gru"
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.3

    # ── Optimization ──────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clf_loss_weight: float = 0.5
    grad_clip: float = 1.0
    label_smoothing: float = 0.05

    # ── Early stopping ─────────────────────────────────────────────────────────
    patience: int = 10

    # ── Output ────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    run_name: str = ""          # auto-generated if empty


# ── Metrics ────────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    loss: float = 0.0
    reg_loss: float = 0.0
    clf_loss: float = 0.0
    # regression
    mae: float = 0.0
    rmse: float = 0.0
    ic: float = 0.0          # Pearson corr between predicted and actual returns
    # classification
    accuracy: float = 0.0
    f1_macro: float = 0.0

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f}  reg={self.reg_loss:.4f}  clf={self.clf_loss:.4f}"
            f"  mae={self.mae:.5f}  ic={self.ic:.4f}"
            f"  acc={self.accuracy:.4f}  f1={self.f1_macro:.4f}"
        )


def _compute_metrics(
    all_ret_true: np.ndarray,
    all_ret_pred: np.ndarray,
    all_dir_true: np.ndarray,
    all_dir_pred: np.ndarray,
    total_loss: float,
    total_reg_loss: float,
    total_clf_loss: float,
    n_batches: int,
) -> EpochMetrics:
    m = EpochMetrics(
        loss     = total_loss / n_batches,
        reg_loss = total_reg_loss / n_batches,
        clf_loss = total_clf_loss / n_batches,
    )

    # Regression metrics
    err = all_ret_pred - all_ret_true
    m.mae  = float(np.abs(err).mean())
    m.rmse = float(np.sqrt((err ** 2).mean()))

    if all_ret_true.std() > 0 and all_ret_pred.std() > 0:
        m.ic = float(np.corrcoef(all_ret_true, all_ret_pred)[0, 1])

    # Classification metrics
    m.accuracy = float((all_dir_pred == all_dir_true).mean())

    # Macro F1 (3 classes)
    f1s = []
    for c in range(3):
        tp = ((all_dir_pred == c) & (all_dir_true == c)).sum()
        fp = ((all_dir_pred == c) & (all_dir_true != c)).sum()
        fn = ((all_dir_pred != c) & (all_dir_true == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    m.f1_macro = float(np.mean(f1s))

    return m


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: DualHeadRNN,
        cfg: TrainConfig,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.cfg   = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Compile for ~15% throughput gain on A100 (PyTorch 2.x)
        if self.device.type == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")  # type: ignore[assignment]

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None

    def _loss(
        self,
        reg_pred: torch.Tensor,
        clf_logits: torch.Tensor,
        y_ret: torch.Tensor,
        y_dir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reg_l = F.huber_loss(reg_pred, y_ret, delta=0.002)
        clf_l = F.cross_entropy(
            clf_logits, y_dir,
            weight=self.class_weights,
            label_smoothing=self.cfg.label_smoothing,
        )
        total = reg_l + self.cfg.clf_loss_weight * clf_l
        return total, reg_l, clf_l

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> EpochMetrics:
        self.model.train(train)
        ctx = torch.enable_grad() if train else torch.no_grad()

        total_loss = total_reg = total_clf = 0.0
        all_ret_true, all_ret_pred = [], []
        all_dir_true, all_dir_pred = [], []

        with ctx:
            for x, y_ret, y_dir in loader:
                x     = x.to(self.device, non_blocking=True)
                y_ret = y_ret.to(self.device, non_blocking=True)
                y_dir = y_dir.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    reg_pred, clf_logits = self.model(x)
                    loss, reg_l, clf_l = self._loss(reg_pred, clf_logits, y_ret, y_dir)

                if train:
                    self.opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    if scheduler is not None:
                        scheduler.step()

                total_loss += loss.item()
                total_reg  += reg_l.item()
                total_clf  += clf_l.item()

                all_ret_true.append(y_ret.cpu().float().numpy())
                all_ret_pred.append(reg_pred.detach().cpu().float().numpy())
                all_dir_true.append(y_dir.cpu().numpy())
                all_dir_pred.append(clf_logits.detach().cpu().argmax(-1).numpy())

        return _compute_metrics(
            np.concatenate(all_ret_true),
            np.concatenate(all_ret_pred),
            np.concatenate(all_dir_true),
            np.concatenate(all_dir_pred),
            total_loss, total_reg, total_clf,
            len(loader),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
    ) -> dict:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=self.cfg.lr,
            steps_per_epoch=len(train_loader),
            epochs=self.cfg.epochs,
            pct_start=0.1,
        )

        best_val_loss = float("inf")
        patience_left = self.cfg.patience
        history: list[dict] = []

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(train_loader, train=True, scheduler=scheduler)
            va = self._run_epoch(val_loader,   train=False)
            elapsed = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": round(tr.loss, 5),
                "val_loss":   round(va.loss, 5),
                "val_ic":     round(va.ic, 4),
                "val_acc":    round(va.accuracy, 4),
                "val_f1":     round(va.f1_macro, 4),
                "val_mae":    round(va.mae, 6),
                "lr":         round(self.opt.param_groups[0]["lr"], 7),
                "secs":       round(elapsed, 1),
            }
            history.append(row)

            tqdm.write(
                f"Ep {epoch:03d}/{self.cfg.epochs}"
                f"  train={tr.loss:.4f}  val={va.loss:.4f}"
                f"  ic={va.ic:.4f}  acc={va.accuracy:.4f}  f1={va.f1_macro:.4f}"
                f"  [{elapsed:.0f}s]"
            )

            # Save best checkpoint
            if va.loss < best_val_loss:
                best_val_loss = va.loss
                patience_left = self.cfg.patience
                self._save(checkpoint_dir / "best.pt", epoch, va.loss)
                tqdm.write(f"  ↑ new best val_loss={va.loss:.5f}  saved → {checkpoint_dir}/best.pt")
            else:
                patience_left -= 1
                if patience_left == 0:
                    tqdm.write(f"  Early stopping at epoch {epoch}")
                    break

        # Save final + history
        self._save(checkpoint_dir / "last.pt", epoch, va.loss)
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch}

    def _save(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save({
            "epoch":     epoch,
            "val_loss":  val_loss,
            "model_state": self.model.state_dict(),  # type: ignore[union-attr]
            "opt_state": self.opt.state_dict(),
            "cfg":       asdict(self.cfg),
        }, path)

    def evaluate(self, loader: DataLoader) -> EpochMetrics:
        """Run one full evaluation pass and return metrics."""
        return self._run_epoch(loader, train=False)
