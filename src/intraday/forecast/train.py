"""Training loop for the full forecast model (Kronos + TCN + ForecastHead).

Pipeline:
  1. Load features + labels, compute triple-barrier labels
  2. Build ForecastDataset (purged train/val split)
  3. Load Kronos (frozen or with LoRA on top-K layers) + KronosTokenizer
  4. Train SmallTCN + ForecastHead (+ optional LoRA params) with:
       - CrossEntropyLoss (label_smoothing=0.0)
       - AdamW with separate LR groups for LoRA vs TCN/head
       - Linear warmup → cosine decay LR schedule
       - Gradient accumulation for large effective batch sizes
  5. Train MetaLabelClassifier (purged k-fold)
  6. Fit IsotonicCalibrator on validation predictions
  7. Save all artefacts + checkpoints to output_dir

Checkpoint / resume:
  A checkpoint is saved after every epoch at output_dir/checkpoint_epoch{N}.pt.
  Pass resume_from=<path> to restart from a saved checkpoint.

Smoke-test:
  Set max_batches=1 to run exactly one forward+backward batch and exit.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

log = structlog.get_logger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _fmt_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to min_lr_ratio × peak_lr."""
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    tcn: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_loss: float,
    klines_norm: Any,
    state_norm: Any,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "tcn": tcn.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "klines_norm": pickle.dumps(klines_norm),
            "state_norm": pickle.dumps(state_norm),
        },
        tmp,
    )
    tmp.replace(path)


def _load_checkpoint(path: Path, *, tcn: nn.Module, head: nn.Module,
                     optimizer: torch.optim.Optimizer, scheduler: Any) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tcn.load_state_dict(ckpt["tcn"])
    head.load_state_dict(ckpt["head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def train_forecast(
    *,
    klines_dir: Path,
    features_dir: Path,
    output_dir: Path,
    train_end: date,
    val_start: date,
    val_end: date,
    kronos_checkpoint: Path,
    tokenizer_checkpoint: Path,
    unfreeze_top_k: int = 0,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 1,
    lr_lora: float = 5e-5,
    lr_tcn_head: float = 2e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 500,
    device: str = "auto",
    seed: int = 42,
    max_batches: int | None = None,
    resume_from: Path | None = None,
    log_every: int = 10,
) -> Path:
    """Train the forecast model and save all artefacts.

    Args:
        klines_dir:           Directory of daily 1m klines Parquet files.
        features_dir:         Directory of daily 5m feature Parquet files.
        output_dir:           Where to write model artefacts + checkpoints.
        train_end:            Last date (inclusive) of training window.
        val_start:            First date of validation window.
        val_end:              Last date of validation window.
        kronos_checkpoint:    Path to Kronos-base model directory.
        tokenizer_checkpoint: Path to Kronos-Tokenizer-base directory.
        unfreeze_top_k:       Number of top Kronos transformer layers to
                              fine-tune via LoRA. 0 = fully frozen (CPU/smoke).
                              4 = recommended for GPU runs (213K extra params).
        lora_rank:            LoRA rank r (default 16).
        lora_alpha:           LoRA alpha scaling (default 32 = 2×rank).
        epochs:               Training epochs.
        batch_size:           Mini-batch size per step (4-8 on CPU, 32-64 GPU).
        grad_accum:           Gradient accumulation steps. Effective batch =
                              batch_size × grad_accum. Use 4 on GPU for 128.
        lr_lora:              Learning rate for LoRA parameters (lower, 5e-5).
        lr_tcn_head:          Learning rate for TCN + head (2e-4).
        weight_decay:         AdamW weight decay.
        warmup_steps:         Linear LR warmup steps before cosine decay.
        device:               "auto" | "cpu" | "cuda" | "mps".
        seed:                 Random seed.
        max_batches:          Stop after this many batches (1 = smoke test).
        resume_from:          Path to checkpoint_epoch*.pt to resume from.
        log_every:            Print progress every N optimizer steps.

    Returns:
        Path to the output directory.
    """
    from intraday.forecast.calibration import IsotonicCalibrator
    from intraday.forecast.dataset import ForecastDataset
    from intraday.forecast.head import ForecastHead
    from intraday.forecast.kronos_loader import kronos_embed, load_kronos
    from intraday.forecast.labels import triple_barrier_labels
    from intraday.forecast.meta_label import META_FEATURE_COLS, MetaLabelClassifier
    from intraday.features.schema import ALL_FEATURES

    _set_seed(seed)
    dev = _resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    is_smoke = max_batches is not None

    print(f"\n{'='*60}")
    print(f"  Forecast Training {'[SMOKE TEST]' if is_smoke else ''}")
    print(f"  device={dev}  epochs={epochs}  batch={batch_size}")
    print(f"  output={output_dir}")
    print(f"{'='*60}\n")

    log.info("train_forecast.start", output_dir=str(output_dir), device=str(dev),
             epochs=epochs, model_version=model_version, smoke_test=is_smoke)

    # ── 1. Load features and compute labels ───────────────────────────────
    print("[1/7] Loading features and computing triple-barrier labels...")
    features_dir = Path(features_dir)
    klines_dir   = Path(klines_dir)

    feature_files = sorted(features_dir.glob("*.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"No feature Parquet files in {features_dir}")

    all_frames = [pl.read_parquet(f) for f in feature_files
                  if _is_date_file(f)]
    bars_df = pl.concat(all_frames).sort("bar_time_ms")

    log.info("train_forecast.computing_labels", n_bars=len(bars_df))
    labeled_df = triple_barrier_labels(bars_df, pt_sl=(1.0, 1.0), horizon_minutes=15, vol_window_minutes=60)

    train_end_ms = _date_to_ms(train_end, end_of_day=True)
    val_start_ms = _date_to_ms(val_start, end_of_day=False)
    val_end_ms   = _date_to_ms(val_end,   end_of_day=True)

    train_labels = labeled_df.filter(pl.col("bar_time_ms") <= train_end_ms)
    val_labels   = labeled_df.filter(
        (pl.col("bar_time_ms") >= val_start_ms) & (pl.col("bar_time_ms") <= val_end_ms)
    )

    print(f"    train_labels={len(train_labels)}  val_labels={len(val_labels)}")
    log.info("train_forecast.split_done", n_train=len(train_labels), n_val=len(val_labels))

    # ── 2. Datasets ───────────────────────────────────────────────────────
    print("[2/7] Building ForecastDatasets...")
    train_ds = ForecastDataset(
        klines_dir=klines_dir,
        features_dir=features_dir,
        labels_df=train_labels,
        seq_klines=512,
        seq_state=128,
    )
    val_ds = ForecastDataset(
        klines_dir=klines_dir,
        features_dir=features_dir,
        labels_df=val_labels,
        seq_klines=512,
        seq_state=128,
        klines_norm=train_ds.klines_norm,
        state_norm=train_ds.state_norm,
    )
    print(f"    train={len(train_ds)} samples  val={len(val_ds)} samples")

    n_features   = len(ALL_FEATURES)
    pin          = dev.type == "cuda"
    import os
    n_workers    = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=n_workers, pin_memory=pin, persistent_workers=n_workers>0, prefetch_factor=2 if n_workers>0 else None)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin, persistent_workers=n_workers>0, prefetch_factor=2 if n_workers>0 else None)

    # ── 3. Load Kronos (frozen or LoRA on top-k layers) ───────────────────
    mode_label = f"LoRA top-{unfreeze_top_k} layers" if unfreeze_top_k > 0 else "fully frozen"
    print(f"[3/7] Loading Kronos ({mode_label})...")
    kronos_model, tokenizer, kronos_cfg = load_kronos(
        model_checkpoint=Path(kronos_checkpoint),
        tokenizer_checkpoint=Path(tokenizer_checkpoint),
        device=dev,
        unfreeze_top_k=unfreeze_top_k,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    kronos_hidden  = int(kronos_cfg["d_model"])           # 832
    lora_trainable = int(kronos_cfg.get("lora_trainable", 0))
    if lora_trainable:
        print(f"    LoRA trainable params: {lora_trainable:,} ({100*lora_trainable/102_310_592:.2f}% of Kronos)")

    # ── 4. Build trainable components ────────────────────────────────────
    print("[4/7] Building TCN + ForecastHead...")
    from intraday.forecast.tcn import SmallTCN
    tcn  = SmallTCN(n_features=n_features, channels=128, dropout=0.1).to(dev)
    head = ForecastHead(kronos_dim=kronos_hidden, tcn_dim=128, hidden=256, n_bins=2, dropout=0.1).to(dev)

    # Separate LR for LoRA (lower) vs TCN/head (higher)
    param_groups: list[dict] = []
    lora_params = [p for p in kronos_model.parameters() if p.requires_grad]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": 0.01})
    param_groups.append({
        "params": list(tcn.parameters()) + list(head.parameters()),
        "lr": lr_tcn_head,
        "weight_decay": weight_decay,
    })

    optimizer   = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    steps_per_epoch = (min(len(train_loader), max_batches) if max_batches else len(train_loader))
    total_steps = epochs * math.ceil(steps_per_epoch / grad_accum)
    scheduler   = _warmup_cosine_schedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    criterion   = nn.CrossEntropyLoss()
    eff_batch   = batch_size * grad_accum
    amp_enabled = dev.type == "cuda"
    scaler      = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"    effective batch={eff_batch}  warmup={warmup_steps}  total_opt_steps={total_steps}")
    print(f"    AMP mixed-precision: {'ON (fp16 activations)' if amp_enabled else 'OFF'}")

    # ── 5. Resume from checkpoint ─────────────────────────────────────────
    start_epoch    = 1
    global_step    = 0
    best_val_loss  = float("inf")

    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"[5/7] Resuming from checkpoint: {resume_path}")
            ckpt = _load_checkpoint(resume_path, tcn=tcn, head=head,
                                    optimizer=optimizer, scheduler=scheduler)
            start_epoch   = int(ckpt["epoch"]) + 1
            global_step   = int(ckpt.get("global_step", 0))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            log.info("train_forecast.resumed", epoch=start_epoch, best_val_loss=best_val_loss)
            print(f"    Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"    Warning: checkpoint not found at {resume_path}, starting fresh")
    else:
        print("[5/7] No checkpoint to resume — starting fresh")

    # ── 6. Training loop ──────────────────────────────────────────────────
    print(f"\n[6/7] Training: {epochs - start_epoch + 1} epoch(s) to go...\n")

    all_val_logits: list[np.ndarray] = []
    all_val_labels: list[np.ndarray] = []

    kronos_grads = unfreeze_top_k > 0
    all_trainable = (
        list(kronos_model.parameters() if kronos_grads else [])
        + list(tcn.parameters())
        + list(head.parameters())
    )
    total_opt_per_epoch = math.ceil((min(len(train_loader), max_batches)
                                     if max_batches else len(train_loader)) / grad_accum)

    epoch_bar = tqdm(
        range(start_epoch, epochs + 1),
        desc="Training", unit="ep",
        position=0, leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────
        tcn.train(); head.train()
        train_loss_sum = 0.0
        train_steps    = 0
        epoch_t0       = time.time()
        accum_loss     = 0.0

        step_bar = tqdm(
            total=total_opt_per_epoch,
            desc=f"  Ep {epoch:02d}/{epochs} train",
            unit="step", position=1, leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        for batch_idx, (klines_norm, klines_raw, klines_stamp, state_win, labels, _) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            klines_raw   = klines_raw.to(dev, non_blocking=True)
            klines_stamp = klines_stamp.to(dev, non_blocking=True)
            state_win    = state_win.to(dev, non_blocking=True)
            labels       = labels.to(dev, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                kronos_emb = kronos_embed(
                    kronos_model, tokenizer, klines_raw, klines_stamp,
                    no_grad=not kronos_grads,
                )
                tcn_emb = tcn(state_win)
                logits  = head(kronos_emb, tcn_emb)
                loss    = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            accum_loss += float(loss.item())

            is_update_step = (batch_idx + 1) % grad_accum == 0
            is_last_batch  = (batch_idx + 1) == (max_batches or len(train_loader))

            if is_update_step or is_last_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step        = (batch_idx + 1) // grad_accum
                train_loss_sum += accum_loss
                train_steps    += 1
                global_step    += 1
                accum_loss      = 0.0

                lr_now      = optimizer.param_groups[-1]["lr"]
                lr_lora_now = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else lr_now
                running_avg = train_loss_sum / train_steps

                step_bar.update(1)
                step_bar.set_postfix(
                    loss=f"{running_avg:.4f}",
                    lr_lora=f"{lr_lora_now:.1e}" if kronos_grads else None,
                    lr=f"{lr_now:.1e}",
                    gpu=f"{torch.cuda.memory_allocated()/1024**2:.0f}MB",
                )

                if opt_step % log_every == 0 or opt_step == 1:
                    log.info(
                        "train_forecast.step",
                        epoch=epoch, opt_step=opt_step,
                        total_opt_steps=total_opt_per_epoch,
                        loss=round(running_avg, 4),
                        lr=round(lr_now, 8),
                    )

        step_bar.close()
        avg_train = train_loss_sum / max(train_steps, 1)

        if is_smoke:
            epoch_bar.close()
            elapsed = _fmt_seconds(time.time() - epoch_t0)
            tqdm.write(f"\n[SMOKE TEST DONE] train_loss={avg_train:.4f}  ({elapsed})")
            tqdm.write("Pipeline verified — all components functional.\n")
            log.info("train_forecast.smoke_test_done", train_loss=round(avg_train, 4))
            _save_checkpoint(
                output_dir / "checkpoint_smoke.pt",
                epoch=epoch, global_step=global_step,
                tcn=tcn, head=head, optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                klines_norm=train_ds.klines_norm, state_norm=train_ds.state_norm,
            )
            tqdm.write(f"Checkpoint saved: {output_dir / 'checkpoint_smoke.pt'}")
            return _save_artefacts(
                output_dir=output_dir, tcn=tcn, head=head,
                kronos_model=kronos_model if unfreeze_top_k > 0 else None,
                train_ds=train_ds, model_version=model_version,
                train_end=train_end, val_start=val_start, val_end=val_end,
                epochs=epochs, batch_size=batch_size, lora_rank=lora_rank,
                unfreeze_top_k=unfreeze_top_k, grad_accum=grad_accum,
                kronos_hidden=kronos_hidden, n_features=n_features, dev=dev, seed=seed,
                best_val_loss=avg_train, meta_metrics={"auc": 0.0, "brier": 0.5},
                calibrator=None,
            )

        # ── Validate ───────────────────────────────────────────────────────
        tcn.eval(); head.eval()
        val_loss_sum = 0.0
        val_steps    = 0
        epoch_val_logits: list[np.ndarray] = []
        epoch_val_labels: list[np.ndarray] = []

        val_bar = tqdm(
            val_loader,
            desc=f"  Ep {epoch:02d}/{epochs} val  ",
            unit="batch", position=1, leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        )
        with torch.no_grad():
            for klines_norm, klines_raw, klines_stamp, state_win, labels, _ in val_bar:
                klines_raw   = klines_raw.to(dev, non_blocking=True)
                klines_stamp = klines_stamp.to(dev, non_blocking=True)
                state_win    = state_win.to(dev, non_blocking=True)
                labels       = labels.to(dev, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    kronos_emb = kronos_embed(kronos_model, tokenizer, klines_raw, klines_stamp)
                    tcn_emb    = tcn(state_win)
                    logits     = head(kronos_emb, tcn_emb)
                    loss       = criterion(logits, labels)

                val_loss_sum += float(loss.item())
                val_steps    += 1
                epoch_val_logits.append(logits.cpu().numpy())
                epoch_val_labels.append(labels.cpu().numpy())
                val_bar.set_postfix(val_loss=f"{val_loss_sum/val_steps:.4f}")

        val_bar.close()
        avg_val   = val_loss_sum / max(val_steps, 1)
        epoch_sec = time.time() - epoch_t0
        improved  = avg_val < best_val_loss

        tqdm.write(
            f"  Ep {epoch:02d}/{epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}"
            f"{'  ← best' if improved else ''}  "
            f"[{_fmt_seconds(epoch_sec)}]"
        )
        log.info(
            "train_forecast.epoch_done",
            epoch=epoch, epochs=epochs,
            train_loss=round(avg_train, 4),
            val_loss=round(avg_val, 4),
            epoch_secs=round(epoch_sec, 1),
        )
        epoch_bar.set_postfix(
            train=f"{avg_train:.4f}",
            val=f"{avg_val:.4f}",
            best=f"{min(best_val_loss, avg_val):.4f}",
        )

        # Save epoch checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch:02d}.pt"
        _save_checkpoint(
            ckpt_path,
            epoch=epoch, global_step=global_step,
            tcn=tcn, head=head, optimizer=optimizer, scheduler=scheduler,
            best_val_loss=best_val_loss,
            klines_norm=train_ds.klines_norm, state_norm=train_ds.state_norm,
        )

        if improved:
            best_val_loss = avg_val
            best_ckpt = output_dir / "checkpoint_best.pt"
            _save_checkpoint(
                best_ckpt,
                epoch=epoch, global_step=global_step,
                tcn=tcn, head=head, optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                klines_norm=train_ds.klines_norm, state_norm=train_ds.state_norm,
            )
            tqdm.write(f"    ✓ New best val_loss={best_val_loss:.4f}  → {best_ckpt.name}")

        # Accumulate val predictions for calibration
        all_val_logits.extend(epoch_val_logits)
        all_val_labels.extend(epoch_val_labels)

    epoch_bar.close()

    # ── 7. Meta-label + calibration + save ───────────────────────────────
    print("[7/7] Training meta-label classifier + calibration...")
    meta_metrics, calibrator = _train_meta_and_calibrate(
        all_val_logits=all_val_logits,
        all_val_labels=all_val_labels,
        val_labels_df=val_labels,
        output_dir=output_dir,
    )

    return _save_artefacts(
        output_dir=output_dir, tcn=tcn, head=head,
        kronos_model=kronos_model if unfreeze_top_k > 0 else None,
        train_ds=train_ds, model_version=model_version,
        train_end=train_end, val_start=val_start, val_end=val_end,
        epochs=epochs, batch_size=batch_size, lora_rank=lora_rank,
        unfreeze_top_k=unfreeze_top_k, grad_accum=grad_accum,
        kronos_hidden=kronos_hidden, n_features=n_features, dev=dev, seed=seed,
        best_val_loss=best_val_loss, meta_metrics=meta_metrics,
        calibrator=calibrator,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_date_file(f: Path) -> bool:
    try:
        date.fromisoformat(f.stem)
        return True
    except ValueError:
        return False


def _date_to_ms(d: date, *, end_of_day: bool) -> int:
    if end_of_day:
        dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _train_meta_and_calibrate(
    all_val_logits: list[np.ndarray],
    all_val_labels: list[np.ndarray],
    val_labels_df: pl.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, float], Any]:
    from intraday.forecast.calibration import IsotonicCalibrator
    from intraday.forecast.meta_label import META_FEATURE_COLS, MetaLabelClassifier

    metrics = {"auc": 0.0, "brier": 0.5}
    calibrator = None

    if not all_val_logits:
        return metrics, calibrator

    logits_np = np.concatenate(all_val_logits, axis=0)
    labels_np = np.concatenate(all_val_labels, axis=0)

    # ── Meta-label classifier ──────────────────────────────────────────────
    if len(logits_np) > 10:
        probs      = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
        BIN_CTR    = [-1.0, 1.0]  # binary: down, up
        import math
        fc_conf_all = [1.0 - (-sum(p * math.log(p + 1e-12) for p in row) / math.log(11)) for row in probs]
        fc_move_all = [sum(p * c for p, c in zip(row, BIN_CTR)) for row in probs]
        # all_val_logits accumulates across epochs; cap to val_labels_df rows using last epoch
        n_df     = len(val_labels_df)
        n        = min(len(fc_conf_all), n_df)
        fc_conf  = fc_conf_all[-n:]
        fc_move  = fc_move_all[-n:]
        labels_meta = labels_np[-n:]

        def _col(name: str, default: float = 0.0) -> list[float]:
            if name in val_labels_df.columns:
                return [float(v) for v in val_labels_df[name].head(n).fill_null(default).to_list()]
            return [default] * n

        hour_utc = []
        if "bar_time_ms" in val_labels_df.columns:
            for ts_ms in val_labels_df["bar_time_ms"].head(n).to_list():
                hour_utc.append(float((int(ts_ms) // (3600 * 1000)) % 24))
        else:
            hour_utc = [0.0] * n

        meta_X = pl.DataFrame({
            "fc_confidence": fc_conf,
            "fc_expected_move_sigma": fc_move,
            "vol_regime_id": [0.0] * n,
            "rsi_14": _col("rsi_14"),
            "log_ret_60m": _col("log_ret_60m"),
            "realized_vol_30m": _col("realized_vol_30m"),
            "hour_utc": hour_utc,
        })
        meta_y  = pl.Series("y", labels_meta.astype(np.int32))
        ts_ser  = pl.Series("ts", val_labels_df["bar_time_ms"].head(n))
        try:
            meta_clf = MetaLabelClassifier()
            metrics  = meta_clf.fit(meta_X, meta_y, timestamps=ts_ser, n_folds=3)
            meta_clf.save(output_dir / "meta_label.lgbm")
            print(f"  meta_label AUC={metrics.get('auc', 0):.3f}")
        except Exception as exc:
            log.warning("train_forecast.meta_label_failed", error=str(exc))

    # ── Isotonic calibration ───────────────────────────────────────────────
    import torch.nn.functional as F
    preds_bin   = np.argmax(logits_np, axis=1)
    raw_max_p   = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy().max(axis=1)
    is_correct  = (preds_bin == labels_np).astype(np.float32)
    calibrator  = IsotonicCalibrator()
    calibrator.fit(raw_max_p, is_correct)
    calibrator.save(output_dir / "calibrator.pkl")
    print("  Isotonic calibrator saved")

    return metrics, calibrator


def _save_artefacts(
    *,
    output_dir: Path,
    tcn: nn.Module,
    head: nn.Module,
    kronos_model: nn.Module | None,
    train_ds: Any,
    model_version: str,
    train_end: date,
    val_start: date,
    val_end: date,
    epochs: int,
    batch_size: int,
    lora_rank: int,
    unfreeze_top_k: int,
    grad_accum: int,
    kronos_hidden: int,
    n_features: int,
    dev: torch.device,
    seed: int,
    best_val_loss: float,
    meta_metrics: dict[str, float],
    calibrator: Any,
) -> Path:
    log.info("train_forecast.saving_models")
    try:
        from safetensors.torch import save_file as sf_save
        sf_save(tcn.state_dict(),  str(output_dir / "tcn.safetensors"))
        sf_save(head.state_dict(), str(output_dir / "head.safetensors"))
    except ImportError:
        torch.save(tcn.state_dict(),  output_dir / "tcn.pt")
        torch.save(head.state_dict(), output_dir / "head.pt")

    # Save LoRA weights separately if Kronos was fine-tuned
    if kronos_model is not None:
        lora_state = {
            k: v for k, v in kronos_model.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        torch.save(lora_state, output_dir / "kronos_lora.pt")
        print(f"  LoRA weights saved: {len(lora_state)} tensors → kronos_lora.pt")

    with open(output_dir / "klines_norm.pkl", "wb") as fh:
        pickle.dump(train_ds.klines_norm, fh)
    with open(output_dir / "state_norm.pkl", "wb") as fh:
        pickle.dump(train_ds.state_norm, fh)

    metadata: dict[str, Any] = {
        "model_version": model_version,
        "train_end": str(train_end),
        "val_start": str(val_start),
        "val_end": str(val_end),
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch": batch_size * grad_accum,
        "grad_accum": grad_accum,
        "lora_rank": lora_rank,
        "unfreeze_top_k": unfreeze_top_k,
        "kronos_hidden": kronos_hidden,
        "n_features": n_features,
        "seq_klines": 512,
        "seq_state": 128,
        "best_val_loss": round(best_val_loss, 4),
        "meta_label_metrics": meta_metrics,
        "device": str(dev),
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nAll artefacts saved to: {output_dir}")
    log.info("train_forecast.done", output_dir=str(output_dir))
    return output_dir
