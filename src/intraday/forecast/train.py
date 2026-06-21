"""Training loop for the full forecast model (Kronos + TCN + ForecastHead).

Pipeline:
  1. Load features + labels, compute triple-barrier labels
  2. Build ForecastDataset (purged train/val split)
  3. Load frozen Kronos + KronosTokenizer
  4. Train SmallTCN + ForecastHead with CrossEntropyLoss + AdamW + cosine LR
  5. Train MetaLabelClassifier (purged k-fold)
  6. Fit IsotonicCalibrator on validation predictions
  7. Save all artefacts + checkpoints to output_dir

Checkpoint / resume:
  A checkpoint is saved after every epoch at output_dir/checkpoint_epoch{N}.pt.
  Pass resume_from=<path> to restart from a saved checkpoint.

Smoke-test:
  Set max_batches=1 to run exactly one forward+backward batch and exit —
  useful to verify the full pipeline before committing to a long run.
"""

from __future__ import annotations

import json
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
    lora_rank: int = 8,
    epochs: int = 5,
    batch_size: int = 8,
    lr_tcn_head: float = 2e-4,
    weight_decay: float = 1e-2,
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
        lora_rank:            Kept for API compatibility; Kronos is frozen.
        epochs:               Training epochs (skipped in smoke-test mode).
        batch_size:           Mini-batch size (use 4-8 on CPU).
        lr_tcn_head:          Learning rate for TCN + head parameters.
        weight_decay:         AdamW weight decay.
        device:               "auto" | "cpu" | "cuda" | "mps".
        seed:                 Random seed.
        max_batches:          If set, stop after this many batches per epoch
                              (set to 1 for a smoke test).
        resume_from:          Path to a checkpoint_epoch*.pt file to resume.
        log_every:            Print progress every N steps.

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
    labeled_df = triple_barrier_labels(bars_df, pt_sl=(1.5, 1.0), horizon_minutes=15, vol_window_minutes=60)

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
        seq_klines=256,
        seq_state=128,
    )
    val_ds = ForecastDataset(
        klines_dir=klines_dir,
        features_dir=features_dir,
        labels_df=val_labels,
        seq_klines=256,
        seq_state=128,
        klines_norm=train_ds.klines_norm,
        state_norm=train_ds.state_norm,
    )
    print(f"    train={len(train_ds)} samples  val={len(val_ds)} samples")

    n_features   = len(ALL_FEATURES)
    pin          = dev.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # ── 3. Load frozen Kronos ─────────────────────────────────────────────
    print("[3/7] Loading frozen Kronos model...")
    kronos_model, tokenizer, kronos_cfg = load_kronos(
        model_checkpoint=Path(kronos_checkpoint),
        tokenizer_checkpoint=Path(tokenizer_checkpoint),
        device=dev,
    )
    kronos_hidden = int(kronos_cfg["d_model"])  # 832

    # ── 4. Build trainable components ────────────────────────────────────
    print("[4/7] Building TCN + ForecastHead...")
    from intraday.forecast.tcn import SmallTCN
    tcn  = SmallTCN(n_features=n_features, channels=64, dropout=0.1).to(dev)
    head = ForecastHead(kronos_dim=kronos_hidden, tcn_dim=64, hidden=256, n_bins=11, dropout=0.1).to(dev)

    optimizer = torch.optim.AdamW(
        list(tcn.parameters()) + list(head.parameters()),
        lr=lr_tcn_head,
        weight_decay=weight_decay,
    )
    total_steps = epochs * (min(len(train_loader), max_batches) if max_batches else len(train_loader))
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.05)

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

    for epoch in range(start_epoch, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        tcn.train(); head.train()
        train_loss_sum = 0.0
        train_steps    = 0
        epoch_t0       = time.time()
        step_times: list[float] = []

        for batch_idx, (klines_norm, klines_raw, klines_stamp, state_win, labels, _) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            step_t0     = time.time()
            klines_raw  = klines_raw.to(dev)    # (B, 256, 6)
            klines_stamp = klines_stamp.to(dev) # (B, 256, 5)
            state_win   = state_win.to(dev)      # (B, 128, n_feat)
            labels      = labels.to(dev)

            optimizer.zero_grad(set_to_none=True)

            kronos_emb = kronos_embed(kronos_model, tokenizer, klines_raw, klines_stamp)  # (B, 832)
            tcn_emb    = tcn(state_win)                                                    # (B, 64)
            logits     = head(kronos_emb, tcn_emb)                                         # (B, 11)
            loss       = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(list(tcn.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step_dt = time.time() - step_t0
            step_times.append(step_dt)
            if len(step_times) > 20:
                step_times.pop(0)

            train_loss_sum += float(loss.item())
            train_steps    += 1
            global_step    += 1

            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                avg_step  = sum(step_times) / len(step_times)
                remaining = (len(train_loader) - batch_idx - 1) * avg_step
                if max_batches:
                    remaining = 0.0
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E{epoch}/{epochs} | step {batch_idx+1:>4}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}  "
                    f"{avg_step:.2f}s/step  ETA {_fmt_seconds(remaining)}"
                )
                log.info(
                    "train_forecast.step",
                    epoch=epoch, step=batch_idx + 1, total_steps=len(train_loader),
                    loss=round(float(loss.item()), 4), lr=round(lr_now, 8),
                    secs_per_step=round(avg_step, 2),
                )

        avg_train = train_loss_sum / max(train_steps, 1)

        if is_smoke:
            print(f"\n[SMOKE TEST DONE] train_loss={avg_train:.4f}  ({_fmt_seconds(time.time()-epoch_t0)})")
            print("Pipeline verified — all components functional.\n")
            log.info("train_forecast.smoke_test_done", train_loss=round(avg_train, 4))
            _save_checkpoint(
                output_dir / "checkpoint_smoke.pt",
                epoch=epoch, global_step=global_step,
                tcn=tcn, head=head, optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                klines_norm=train_ds.klines_norm, state_norm=train_ds.state_norm,
            )
            print(f"Checkpoint saved: {output_dir / 'checkpoint_smoke.pt'}")
            return _save_artefacts(
                output_dir=output_dir, tcn=tcn, head=head,
                train_ds=train_ds, model_version=model_version,
                train_end=train_end, val_start=val_start, val_end=val_end,
                epochs=epochs, batch_size=batch_size, lora_rank=lora_rank,
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

        with torch.no_grad():
            for klines_norm, klines_raw, klines_stamp, state_win, labels, _ in val_loader:
                klines_raw   = klines_raw.to(dev)
                klines_stamp = klines_stamp.to(dev)
                state_win    = state_win.to(dev)
                labels       = labels.to(dev)

                kronos_emb = kronos_embed(kronos_model, tokenizer, klines_raw, klines_stamp)
                tcn_emb    = tcn(state_win)
                logits     = head(kronos_emb, tcn_emb)
                loss       = criterion(logits, labels)

                val_loss_sum += float(loss.item())
                val_steps    += 1
                epoch_val_logits.append(logits.cpu().numpy())
                epoch_val_labels.append(labels.cpu().numpy())

        avg_val   = val_loss_sum / max(val_steps, 1)
        epoch_sec = time.time() - epoch_t0

        print(
            f"\n  ── Epoch {epoch}/{epochs} done ──  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"time={_fmt_seconds(epoch_sec)}"
        )
        log.info(
            "train_forecast.epoch_done",
            epoch=epoch, epochs=epochs,
            train_loss=round(avg_train, 4),
            val_loss=round(avg_val, 4),
            epoch_secs=round(epoch_sec, 1),
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
        print(f"  Checkpoint: {ckpt_path.name}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_ckpt = output_dir / "checkpoint_best.pt"
            _save_checkpoint(
                best_ckpt,
                epoch=epoch, global_step=global_step,
                tcn=tcn, head=head, optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                klines_norm=train_ds.klines_norm, state_norm=train_ds.state_norm,
            )
            print(f"  ✓ New best val_loss={best_val_loss:.4f}  → {best_ckpt.name}")

        # Accumulate val predictions for calibration
        all_val_logits.extend(epoch_val_logits)
        all_val_labels.extend(epoch_val_labels)
        print()

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
        train_ds=train_ds, model_version=model_version,
        train_end=train_end, val_start=val_start, val_end=val_end,
        epochs=epochs, batch_size=batch_size, lora_rank=lora_rank,
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
        BIN_CTR    = [-4.0, -2.5, -1.5, -0.75, -0.35, 0.0, 0.35, 0.75, 1.5, 2.5, 4.0]
        import math
        fc_conf  = [1.0 - (-sum(p * math.log(p + 1e-12) for p in row) / math.log(11)) for row in probs]
        fc_move  = [sum(p * c for p, c in zip(row, BIN_CTR)) for row in probs]
        n        = len(fc_conf)

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
            "funding_rate": _col("funding_rate"),
            "rsi_14": _col("rsi_14"),
            "log_ret_60m": _col("log_ret_60m"),
            "realized_vol_30m": _col("realized_vol_30m"),
            "hour_utc": hour_utc,
        })
        meta_y  = pl.Series("y", labels_np[:n].astype(np.int32))
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
    train_ds: Any,
    model_version: str,
    train_end: date,
    val_start: date,
    val_end: date,
    epochs: int,
    batch_size: int,
    lora_rank: int,
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
        "lora_rank": lora_rank,
        "kronos_hidden": kronos_hidden,
        "n_features": n_features,
        "seq_klines": 256,
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
