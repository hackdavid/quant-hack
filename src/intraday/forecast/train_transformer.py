"""Training loop for CryptoTransformer on 5-min BTCUSDT feature sequences.

Usage:
    python -m intraday.forecast.train_transformer [options]
    python -m intraday.forecast.train_transformer --resume-from models/transformer/<run>/latest.pt

Features:
    - Saves best.pt (highest val AUC) and latest.pt (most recent epoch) — no disk bloat
    - Resume-safe: checkpoint stores model + optimizer + scheduler + scaler + norm stats
    - Mixed-precision (bf16 on Ampere+, fp16 otherwise)
    - Cosine LR with linear warmup
    - Label smoothing (0.1)
    - Reports train loss / val loss / val AUC every epoch
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from intraday.features.schema import ALL_FEATURES
from intraday.forecast.transformer_model import CryptoTransformer


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding-window dataset that returns (feat_window, time_feat, label).

    Each sample covers `seq_len` consecutive 5-min bars ending at the label bar.
    Labels: 1 = up (fwd_ret_15m > threshold), 0 = down (< -threshold).
    Flat bars are excluded.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feat_cols: list[str],
        seq_len: int,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        ret_threshold: float = 0.0005,
    ) -> None:
        self._seq_len   = seq_len
        self._feat_mean = feat_mean.astype(np.float32)
        self._feat_std  = feat_std.astype(np.float32)

        # Build feature array
        self._feat = df.select(feat_cols).fill_null(0).to_numpy().astype(np.float32)

        # Cyclical time features: 6 channels
        ts    = df["bar_time_ms"].to_numpy().astype(np.float64)
        hour  = ((ts // 3_600_000) % 24).astype(np.float32)
        dow   = ((ts // 86_400_000) % 7).astype(np.float32)
        self._time_feat = np.stack([
            hour / 23.0,
            np.sin(2 * math.pi * hour / 24),
            np.cos(2 * math.pi * hour / 24),
            dow / 6.0,
            np.sin(2 * math.pi * dow / 7),
            np.cos(2 * math.pi * dow / 7),
        ], axis=1).astype(np.float32)  # (N, 6)

        # Labels from fwd_ret_15m
        if "fwd_ret_15m" not in df.columns:
            raise ValueError("DataFrame must contain 'fwd_ret_15m'")
        fwd = df["fwd_ret_15m"].fill_null(0).to_numpy().astype(np.float32)
        label = np.where(fwd > ret_threshold, 1,
                np.where(fwd < -ret_threshold, 0, -1)).astype(np.int8)

        # Valid: labeled AND enough history for a full window
        self._valid_idx = np.where(
            (label >= 0) & (np.arange(len(label)) >= seq_len)
        )[0]
        self._label = label

    def __len__(self) -> int:
        return len(self._valid_idx)

    def __getitem__(self, i: int):
        end   = int(self._valid_idx[i]) + 1
        start = end - self._seq_len

        feat = self._feat[start:end].copy()                        # (T, F)
        feat = (feat - self._feat_mean) / np.where(
            self._feat_std > 1e-8, self._feat_std, 1.0
        )
        feat      = np.clip(feat, -8.0, 8.0)                      # guard outliers
        time_feat = self._time_feat[start:end]                     # (T, 6)
        label     = int(self._label[self._valid_idx[i]])

        return (
            torch.from_numpy(feat),
            torch.from_numpy(time_feat),
            torch.tensor(label, dtype=torch.long),
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def compute_norm_stats(
    df: pl.DataFrame, feat_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    arr = df.select(feat_cols).fill_null(0).to_numpy().astype(np.float64)
    return arr.mean(axis=0).astype(np.float32), arr.std(axis=0).astype(np.float32)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    best_val_auc: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    cfg: dict,
) -> None:
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "best_val_auc":    best_val_auc,
        "norm_mean":       norm_mean,
        "norm_std":        norm_std,
        "config":          cfg,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return (
        ckpt["epoch"],
        ckpt["best_val_auc"],
        ckpt["norm_mean"],
        ckpt["norm_std"],
        ckpt["config"],
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp_dtype,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for feat, time_feat, label in loader:
        feat      = feat.to(device, non_blocking=True)
        time_feat = time_feat.to(device, non_blocking=True)
        label     = label.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(feat, time_feat)
            loss   = criterion(logits, label)
        total_loss += loss.item() * label.size(0)
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_probs))
    return avg_loss, auc


# ── Main training function ────────────────────────────────────────────────────

def train_transformer(
    features_dir:    str   = "data/features/BTCUSDT",
    output_dir:      str   = "models/transformer",
    seq_len:         int   = 128,
    d_model:         int   = 256,
    n_heads:         int   = 8,
    n_layers:        int   = 8,
    dim_ff:          int   = 1024,
    dropout:         float = 0.2,
    batch_size:      int   = 256,
    epochs:          int   = 50,
    lr:              float = 5e-5,
    weight_decay:    float = 0.05,
    warmup_steps:    int   = 300,
    label_smoothing: float = 0.1,
    ret_threshold:   float = 0.0005,
    val_frac:        float = 0.15,
    patience:        int   = 10,
    resume_from:     str | None = None,
    num_workers:     int   = 4,
) -> Path:

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp  = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\n{'='*60}")
    print(f"CryptoTransformer Training  [{run_id}]")
    print(f"{'='*60}")
    print(f"Device: {device}   AMP: {use_amp} ({amp_dtype})")
    print(f"Seq len: {seq_len} bars ({seq_len * 5} min)   "
          f"Model: d={d_model} h={n_heads} L={n_layers} ff={dim_ff}")

    # ── Load features ──────────────────────────────────────────────────────
    print("\n[1/5] Loading feature files...")
    files = sorted(glob.glob(f"{features_dir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {features_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("bar_time_ms")
    print(f"  {len(df):,} bars  ({len(files)} daily files)")

    feat_cols   = [c for c in ALL_FEATURES if c in df.columns]
    n_features  = len(feat_cols)
    n_time_feat = 6
    print(f"  {n_features} feature columns")

    # ── Time split ─────────────────────────────────────────────────────────
    print("\n[2/5] Splitting train / val...")
    n     = len(df)
    split = int(n * (1 - val_frac))
    train_df, val_df = df[:split], df[split:]

    # Norm stats from train only (prevent leakage)
    norm_mean, norm_std = compute_norm_stats(train_df, feat_cols)

    train_ds = SequenceDataset(train_df, feat_cols, seq_len, norm_mean, norm_std, ret_threshold)
    val_ds   = SequenceDataset(val_df,   feat_cols, seq_len, norm_mean, norm_std, ret_threshold)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  "
          f"label balance={sum(int(train_ds._label[i]) for i in train_ds._valid_idx)/len(train_ds):.3f}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_amp, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=use_amp,
        persistent_workers=num_workers > 0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n[3/5] Building model...")
    model = CryptoTransformer(
        n_features=n_features,
        n_time_feat=n_time_feat,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        seq_len=seq_len,
        n_classes=2,
        dropout=dropout,
    ).to(device)
    n_params = model.count_params()
    print(f"  Trainable params: {n_params:,}  ({n_params/1e6:.2f}M)")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = epochs * len(train_loader)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler(enabled=use_amp)
    train_crit   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_crit     = nn.CrossEntropyLoss()   # no smoothing for evaluation

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_auc = 0.0

    if resume_from:
        print(f"\n  Resuming from {resume_from}")
        start_epoch, best_val_auc, norm_mean, norm_std, _ = load_checkpoint(
            Path(resume_from), model, optimizer, scheduler, scaler
        )
        print(f"  → epoch {start_epoch}, best AUC so far {best_val_auc:.4f}")

    # ── Save config ────────────────────────────────────────────────────────
    cfg = dict(
        seq_len=seq_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        dim_ff=dim_ff, dropout=dropout, batch_size=batch_size, epochs=epochs,
        lr=lr, weight_decay=weight_decay, warmup_steps=warmup_steps,
        label_smoothing=label_smoothing, ret_threshold=ret_threshold, val_frac=val_frac,
        patience=patience, feat_cols=feat_cols, n_features=n_features,
        n_time_feat=n_time_feat, n_params=n_params,
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    ckpt_kwargs = dict(
        model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, norm_mean=norm_mean, norm_std=norm_std, cfg=cfg,
    )

    history: list[dict] = []
    no_improve  = 0   # epochs since last val AUC improvement (early stopping)
    print(f"\n[4/5] Training for up to {epochs} epochs (patience={patience})...")
    print(f"  Steps/epoch: {len(train_loader)}   Total steps: {total_steps}")
    print(f"  Checkpoints → {out}/\n", flush=True)

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        t0         = time.time()
        total_loss = 0.0
        n_batches  = 0

        for feat, time_feat, label in train_loader:
            feat      = feat.to(device, non_blocking=True)
            time_feat = time_feat.to(device, non_blocking=True)
            label     = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(feat, time_feat)
                loss   = train_crit(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / n_batches
        val_loss, val_auc = validate(model, val_loader, device, val_crit, amp_dtype, use_amp)

        elapsed  = time.time() - t0
        is_best  = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            no_improve   = 0
        else:
            no_improve  += 1

        lr_now = scheduler.get_last_lr()[0]
        marker = "  ← best" if is_best else f"  (no improve {no_improve}/{patience})"
        print(
            f"  Ep {epoch+1:02d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"auc={val_auc:.4f}  lr={lr_now:.2e}  [{elapsed:.0f}s]{marker}",
            flush=True,
        )

        row = dict(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss,
                   val_auc=val_auc, lr=lr_now, elapsed_s=round(elapsed, 1))
        history.append(row)

        # ── Checkpoints ───────────────────────────────────────────────────
        save_checkpoint(out / "latest.pt", epoch=epoch + 1, best_val_auc=best_val_auc, **ckpt_kwargs)
        if is_best:
            save_checkpoint(out / "best.pt", epoch=epoch + 1, best_val_auc=best_val_auc, **ckpt_kwargs)
            print(f"    → best.pt  (AUC {best_val_auc:.4f})", flush=True)

        (out / "history.json").write_text(json.dumps(history, indent=2))

        # ── Early stopping ────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n  Early stop: no improvement for {patience} epochs.", flush=True)
            break

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Done.")
    print(f"  Best val AUC : {best_val_auc:.4f}")
    print(f"  Outputs      : {out}")
    print(f"    best.pt    — best model by val AUC")
    print(f"    latest.pt  — last epoch (use for --resume-from)")
    print(f"    config.json / history.json")

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CryptoTransformer on 5-min BTC features")
    p.add_argument("--features-dir",    default="data/features/BTCUSDT")
    p.add_argument("--output-dir",      default="models/transformer")
    p.add_argument("--seq-len",         type=int,   default=128,   help="Context window in bars (128=10.7h)")
    p.add_argument("--d-model",         type=int,   default=256,   help="Transformer hidden dim")
    p.add_argument("--n-heads",         type=int,   default=8,     help="Attention heads")
    p.add_argument("--n-layers",        type=int,   default=8,     help="Transformer depth")
    p.add_argument("--dim-ff",          type=int,   default=1024,  help="FFN expansion dim")
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--epochs",          type=int,   default=60)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-2)
    p.add_argument("--warmup-steps",    type=int,   default=500)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--ret-threshold",   type=float, default=0.0005,
                   help="fwd_ret_15m threshold for up/down label")
    p.add_argument("--val-frac",        type=float, default=0.15)
    p.add_argument("--patience",         type=int,   default=10,
                   help="Early stopping: epochs with no val AUC improvement before stopping")
    p.add_argument("--resume-from",     default=None,
                   help="Path to latest.pt checkpoint to resume from")
    p.add_argument("--num-workers",     type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_transformer(
        features_dir    = args.features_dir,
        output_dir      = args.output_dir,
        seq_len         = args.seq_len,
        d_model         = args.d_model,
        n_heads         = args.n_heads,
        n_layers        = args.n_layers,
        dim_ff          = args.dim_ff,
        dropout         = args.dropout,
        batch_size      = args.batch_size,
        epochs          = args.epochs,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_steps    = args.warmup_steps,
        label_smoothing = args.label_smoothing,
        ret_threshold   = args.ret_threshold,
        val_frac        = args.val_frac,
        patience        = args.patience,
        resume_from     = args.resume_from,
        num_workers     = args.num_workers,
    )
