"""CLI commands for LSTM/GRU model training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

ml_app  = typer.Typer(help="ML model training (LSTM/GRU)")
console = Console()


@ml_app.command("train")
def train_cmd(
    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir:   Annotated[Path, typer.Option(help="Data root")] = Path("data"),
    symbol:     Annotated[str,  typer.Option()] = "BTCUSDT",
    train_end:  Annotated[str,  typer.Option(help="Last date of training set")] = "2023-12-31",
    val_start:  Annotated[str,  typer.Option(help="First date of val set")] = "2024-01-01",
    val_end:    Annotated[str,  typer.Option(help="Last date of val set")] = "2024-06-30",
    seq_len:    Annotated[int,  typer.Option(help="Lookback window in 5m bars")] = 60,
    # ── Model ─────────────────────────────────────────────────────────────────
    model_type: Annotated[str,  typer.Option(help="gru | lstm")] = "gru",
    hidden_dim: Annotated[int,  typer.Option()] = 256,
    n_layers:   Annotated[int,  typer.Option()] = 2,
    dropout:    Annotated[float, typer.Option()] = 0.3,
    # ── Training ──────────────────────────────────────────────────────────────
    epochs:     Annotated[int,   typer.Option()] = 100,
    batch_size: Annotated[int,   typer.Option()] = 1024,
    lr:         Annotated[float, typer.Option()] = 1e-3,
    weight_decay: Annotated[float, typer.Option()] = 1e-4,
    clf_weight: Annotated[float, typer.Option(help="Weight of clf loss vs reg loss")] = 0.5,
    patience:   Annotated[int,   typer.Option(help="Early stopping patience")] = 10,
    # ── Output ────────────────────────────────────────────────────────────────
    checkpoint_dir: Annotated[Path, typer.Option()] = Path("checkpoints"),
    run_name:   Annotated[str, typer.Option(help="Subfolder name (auto if empty)")] = "",
    workers:    Annotated[int, typer.Option(help="DataLoader worker processes")] = 4,
) -> None:
    """Train a dual-head GRU/LSTM to predict fwd_ret_5m and fwd_direction_5m.

    Split:
      train  → [dataset start] .. --train-end
      val    → --val-start      .. --val-end
      test   → --val-end + 1day .. [dataset end]  (evaluated after training)

    Examples:
        # Quick default run (GRU, hidden=256, seq=60)
        intraday ml train

        # LSTM with larger hidden, longer lookback
        intraday ml train --model-type lstm --hidden-dim 512 --seq-len 96
    """
    import torch
    from torch.utils.data import DataLoader

    from intraday.features.schema import ALL_FEATURES
    from intraday.models.dataset import create_datasets
    from intraday.models.gru import DualHeadRNN, ModelConfig
    from intraday.models.trainer import Trainer, TrainConfig

    # ── Build run name ────────────────────────────────────────────────────────
    if not run_name:
        run_name = f"{model_type}_h{hidden_dim}_l{n_layers}_seq{seq_len}"
    ckpt_dir = checkpoint_dir / run_name

    console.print(f"\n[bold yellow]BTCUSDT Dual-Head {model_type.upper()} Training[/bold yellow]")
    console.print(f"Run      : [cyan]{run_name}[/cyan]")
    console.print(f"Split    : train→{train_end}  val {val_start}→{val_end}")
    console.print(f"Model    : {model_type.upper()}  hidden={hidden_dim}  layers={n_layers}  seq={seq_len}")
    console.print(f"Training : epochs={epochs}  batch={batch_size}  lr={lr}  patience={patience}")
    console.print(f"Output   : {ckpt_dir}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    console.print("[dim]Loading feature data...[/dim]")
    train_ds, val_ds, test_ds, normalizer = create_datasets(
        data_dir=data_dir,
        symbol=symbol,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        seq_len=seq_len,
    )

    console.print(
        f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} sequences"
    )

    # Save normalizer alongside checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(ckpt_dir / "normalizer.json")
    console.print(f"  normalizer saved → {ckpt_dir}/normalizer.json\n")

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    # ── Class weights (inverse frequency) ─────────────────────────────────────
    import numpy as np
    dirs = train_ds._y_dir[train_ds._idx].numpy()
    counts = np.bincount(dirs, minlength=3).astype(float)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32)
    console.print(f"Class weights (down/flat/up): {weights.numpy().round(3)}\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        n_features=len(ALL_FEATURES),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        model_type=model_type,
    )
    model = DualHeadRNN(model_cfg)
    console.print(f"Parameters: [bold]{model.n_params:,}[/bold]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device    : [bold]{device}[/bold]")
    if device == "cuda":
        import torch
        console.print(f"GPU       : {torch.cuda.get_device_name(0)}\n")

    # Save model config
    (ckpt_dir / "model_cfg.json").write_text(
        json.dumps({"model_type": model_type, "hidden_dim": hidden_dim,
                    "n_layers": n_layers, "dropout": dropout,
                    "n_features": len(ALL_FEATURES), "seq_len": seq_len}, indent=2)
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        data_dir=str(data_dir), symbol=symbol,
        train_end=train_end, val_start=val_start, val_end=val_end,
        seq_len=seq_len, model_type=model_type,
        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, clf_loss_weight=clf_weight,
        patience=patience, checkpoint_dir=str(ckpt_dir),
    )
    trainer = Trainer(model, train_cfg, class_weights=weights)

    console.print("[bold]─── Training ───[/bold]")
    result = trainer.fit(train_loader, val_loader, ckpt_dir)
    console.print(f"\n[green]Training complete[/green]  best_val_loss={result['best_val_loss']:.5f}  epochs={result['epochs_trained']}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    console.print("\n[bold]─── Test set evaluation ───[/bold]")
    # Load best checkpoint
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    # Re-create a non-compiled model for clean state_dict loading
    best_model = DualHeadRNN(model_cfg)
    # Strip _orig_mod. prefix if present (from torch.compile)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    best_model.load_state_dict(sd)
    best_trainer = Trainer(best_model, train_cfg, class_weights=weights)
    test_m = best_trainer.evaluate(test_loader)

    table = Table(title=f"Test results ({val_end} → end)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="bold")
    rows = [
        ("Loss",     f"{test_m.loss:.5f}"),
        ("Reg loss", f"{test_m.reg_loss:.5f}"),
        ("Clf loss", f"{test_m.clf_loss:.5f}"),
        ("MAE",      f"{test_m.mae:.6f}"),
        ("RMSE",     f"{test_m.rmse:.6f}"),
        ("IC",       f"{test_m.ic:.4f}"),
        ("Accuracy", f"{test_m.accuracy:.4f}"),
        ("F1 macro", f"{test_m.f1_macro:.4f}"),
    ]
    for name, val in rows:
        table.add_row(name, val)
    console.print(table)

    # Save test results
    (ckpt_dir / "test_results.json").write_text(
        json.dumps({k: v for k, v in zip([r[0] for r in rows], [r[1] for r in rows])}, indent=2)
    )
    console.print(f"\n[dim]All artefacts saved to {ckpt_dir}/[/dim]")


@ml_app.command("eval")
def eval_cmd(
    checkpoint: Annotated[Path, typer.Argument(help="Path to checkpoint directory")],
    data_dir:   Annotated[Path, typer.Option()] = Path("data"),
    split:      Annotated[str, typer.Option(help="train | val | test")] = "test",
    batch_size: Annotated[int, typer.Option()] = 1024,
    workers:    Annotated[int, typer.Option()] = 4,
) -> None:
    """Evaluate a saved checkpoint on train/val/test split."""
    import torch
    from torch.utils.data import DataLoader

    from intraday.models.dataset import create_datasets
    from intraday.models.gru import DualHeadRNN, ModelConfig
    from intraday.models.normalizer import FeatureNormalizer
    from intraday.models.trainer import Trainer, TrainConfig

    cfg_path  = checkpoint / "model_cfg.json"
    ckpt_path = checkpoint / "best.pt"
    norm_path = checkpoint / "normalizer.json"

    if not ckpt_path.exists():
        console.print(f"[red]No best.pt found in {checkpoint}[/red]")
        raise typer.Exit(1)

    model_cfg_d = json.loads(cfg_path.read_text())
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg_d = ckpt["cfg"]

    model_cfg = ModelConfig(**{k: model_cfg_d[k] for k in ModelConfig.__dataclass_fields__})
    model = DualHeadRNN(model_cfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(sd)

    train_ds, val_ds, test_ds, _ = create_datasets(
        data_dir=data_dir,
        symbol=train_cfg_d.get("symbol", "BTCUSDT"),
        train_end=train_cfg_d["train_end"],
        val_start=train_cfg_d["val_start"],
        val_end=train_cfg_d["val_end"],
        seq_len=train_cfg_d["seq_len"],
    )

    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = ds_map[split]

    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        pin_memory=torch.cuda.is_available())
    train_cfg = TrainConfig(**train_cfg_d)
    trainer = Trainer(model, train_cfg)
    m = trainer.evaluate(loader)

    console.print(f"\n[bold]{split.upper()} evaluation — {checkpoint.name}[/bold]")
    console.print(f"  Loss={m.loss:.5f}  MAE={m.mae:.6f}  IC={m.ic:.4f}"
                  f"  Acc={m.accuracy:.4f}  F1={m.f1_macro:.4f}")
