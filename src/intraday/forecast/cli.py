"""CLI commands for the Forecast Agent (training and inference)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

log = structlog.get_logger(__name__)
console = Console()

forecast_app = typer.Typer(help="Forecast model training and inference")


@forecast_app.command("train")
def forecast_train(
    mode: Annotated[str, typer.Option(help="Training mode: pretrain")] = "pretrain",
    train_start: Annotated[Optional[str], typer.Option(help="Train start date YYYY-MM-DD")] = None,
    train_end: Annotated[Optional[str], typer.Option(help="Train end date YYYY-MM-DD")] = None,
    val_start: Annotated[Optional[str], typer.Option(help="Validation start date YYYY-MM-DD")] = None,
    val_end: Annotated[Optional[str], typer.Option(help="Validation end date YYYY-MM-DD")] = None,
    kronos_checkpoint: Annotated[Path, typer.Option(help="Path to Kronos-base model directory")] = Path("models/kronos-base"),
    tokenizer_checkpoint: Annotated[Path, typer.Option(help="Path to Kronos-Tokenizer-base directory")] = Path("models/kronos-tokenizer"),
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 10,
    batch_size: Annotated[int, typer.Option(help="Mini-batch size per step (4-8 CPU, 32-64 GPU)")] = 4,
    grad_accum: Annotated[int, typer.Option(help="Gradient accumulation steps (effective batch = batch×accum)")] = 1,
    unfreeze_top_k: Annotated[int, typer.Option(help="Fine-tune last K Kronos layers via LoRA (0=frozen, 4=GPU recommended)")] = 0,
    lora_rank: Annotated[int, typer.Option(help="LoRA rank r (default 16)")] = 16,
    lora_alpha: Annotated[int, typer.Option(help="LoRA alpha scaling (default 32)")] = 32,
    lr_lora: Annotated[float, typer.Option(help="Learning rate for LoRA params (lower)")] = 5e-5,
    lr_head: Annotated[float, typer.Option(help="Learning rate for TCN+head")] = 2e-4,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay")] = 1e-2,
    warmup_steps: Annotated[int, typer.Option(help="LR warmup steps before cosine decay")] = 500,
    device: Annotated[str, typer.Option(help="Device: auto | cpu | cuda | mps")] = "auto",
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    data_dir: Annotated[Path, typer.Option(help="Root data directory")] = Path("data"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Output directory for model artefacts")] = None,
    symbol: Annotated[str, typer.Option(help="Symbol")] = "BTCUSDT",
    smoke_test: Annotated[bool, typer.Option("--smoke-test", help="Run 1 batch to verify the pipeline, then exit")] = False,
    resume_from: Annotated[Optional[Path], typer.Option(help="Resume from a checkpoint_epoch*.pt file")] = None,
    log_every: Annotated[int, typer.Option(help="Print progress every N optimizer steps")] = 10,
) -> None:
    """Train the Kronos + TCN + meta-label forecast model.

    Kronos is used as a FROZEN feature extractor (102M params); only the
    SmallTCN + ForecastHead (~500K params) are trained.

    Examples:

      # Smoke test — verify the pipeline with 1 batch:
      uv run intraday forecast train --smoke-test

      # Full run on available data:
      uv run intraday forecast train \\
        --train-end 2026-05-19 --val-start 2026-05-20 --val-end 2026-06-19 \\
        --epochs 5 --batch-size 4

      # Resume after interruption:
      uv run intraday forecast train \\
        --resume-from models/forecast/latest/checkpoint_epoch02.pt \\
        --train-end 2026-05-19 --val-start 2026-05-20 --val-end 2026-06-19
    """
    from intraday.forecast.train import train_forecast
    from intraday.utils.logging import setup_logging

    setup_logging(log_level="info", console=True)

    # Date defaults: use full available data range
    today = date.today()
    if train_end:
        _train_end = date.fromisoformat(train_end)
    else:
        # Default: everything up to 5 days ago
        from datetime import timedelta
        _train_end = today - timedelta(days=5)

    if val_start:
        _val_start = date.fromisoformat(val_start)
    else:
        _val_start = _train_end

    if val_end:
        _val_end = date.fromisoformat(val_end)
    else:
        _val_end = today

    klines_dir   = data_dir / "raw" / "binance" / "klines_1m" / symbol
    features_dir = data_dir / "features" / symbol

    if output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("models") / "forecast" / ts

    rprint(f"\n[bold yellow]Forecast Training ({mode})[/bold yellow]")
    rprint(f"  Train end    : {_train_end}")
    rprint(f"  Val window   : {_val_start} → {_val_end}")
    rprint(f"  klines_dir   : {klines_dir}")
    rprint(f"  features_dir : {features_dir}")
    rprint(f"  Kronos model : {kronos_checkpoint}")
    rprint(f"  Kronos tok   : {tokenizer_checkpoint}")
    rprint(f"  output_dir   : {output_dir}")
    rprint(f"  epochs={epochs}  batch={batch_size}  grad_accum={grad_accum}  eff_batch={batch_size*grad_accum}")
    rprint(f"  unfreeze_top_k={unfreeze_top_k}  lora_rank={lora_rank}  warmup={warmup_steps}")
    rprint(f"  device={device}")
    if smoke_test:
        rprint("  [bold red]SMOKE TEST MODE — 1 batch only[/bold red]")
    if resume_from:
        rprint(f"  resume_from  : {resume_from}")

    for required_dir in [klines_dir, features_dir]:
        if not required_dir.exists():
            rprint(f"[red]Directory not found: {required_dir}[/red]")
            rprint("Run: uv run intraday data download-bulk  &&  uv run intraday features compute")
            raise typer.Exit(code=1)

    for ckpt_path in [kronos_checkpoint, tokenizer_checkpoint]:
        if not ckpt_path.exists():
            rprint(f"[red]Checkpoint not found: {ckpt_path}[/red]")
            rprint("Download with:")
            rprint("  python -c \"from huggingface_hub import snapshot_download")
            rprint("  snapshot_download('NeoQuasar/Kronos-base',          local_dir='models/kronos-base')\"")
            rprint("  snapshot_download('NeoQuasar/Kronos-Tokenizer-base', local_dir='models/kronos-tokenizer')\"")
            raise typer.Exit(code=1)

    try:
        saved = train_forecast(
            klines_dir=klines_dir,
            features_dir=features_dir,
            output_dir=output_dir,
            train_end=_train_end,
            val_start=_val_start,
            val_end=_val_end,
            kronos_checkpoint=kronos_checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            unfreeze_top_k=unfreeze_top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr_lora=lr_lora,
            lr_tcn_head=lr_head,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            device=device,
            seed=seed,
            max_batches=1 if smoke_test else None,
            resume_from=resume_from,
            log_every=log_every,
        )
        rprint(f"\n[green]Model saved to: {saved}[/green]")
    except FileNotFoundError as exc:
        rprint(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1)


@forecast_app.command("predict")
def forecast_predict(
    at_ts: Annotated[str, typer.Argument(help="ISO timestamp or unix ms for prediction point")],
    horizon: Annotated[int, typer.Option(help="Forecast horizon in minutes (5|15|60)")] = 15,
    version: Annotated[str, typer.Option(help="Model version or 'latest'")] = "latest",
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="Symbol")] = "BTCUSDT",
    models_dir: Annotated[Path, typer.Option(help="Forecast models directory")] = Path("models/forecast"),
) -> None:
    """Run a single forecast at a given timestamp.

    AT_TS can be:
      - ISO datetime: "2026-06-01T12:00:00"
      - Unix milliseconds: "1748779200000"

    Examples:

      intraday forecast predict "2026-06-01T12:00:00" --horizon 15

      intraday forecast predict 1748779200000 --horizon 5 --version latest
    """
    import polars as pl
    from intraday.forecast.predict import load_forecast
    from intraday.utils.logging import setup_logging

    setup_logging(log_level="warning", console=True)

    # Parse timestamp
    try:
        if at_ts.isdigit():
            ts_ms = int(at_ts)
        else:
            dt = datetime.fromisoformat(at_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts_ms = int(dt.timestamp() * 1000)
    except ValueError as exc:
        rprint(f"[red]Invalid timestamp: {at_ts!r} — {exc}[/red]")
        raise typer.Exit(code=1)

    # Load model
    try:
        model = load_forecast(version=version, models_dir=models_dir)
    except FileNotFoundError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # Load recent klines and features windows
    klines_dir = data_dir / "raw" / "binance" / "klines_1m" / symbol
    features_dir = data_dir / "features" / symbol

    klines_window = _load_window(klines_dir, ts_ms, n_bars=256, bar_ms=60_000)
    state_window = _load_window(features_dir, ts_ms, n_bars=128, bar_ms=300_000)

    if klines_window is None or len(klines_window) == 0:
        rprint(f"[red]No klines data found near {at_ts}[/red]")
        raise typer.Exit(code=1)

    if state_window is None or len(state_window) == 0:
        rprint(f"[red]No feature state data found near {at_ts}[/red]")
        raise typer.Exit(code=1)

    out = model.predict(
        klines_window=klines_window,
        state_window=state_window,
        ts_ms=ts_ms,
        horizon_minutes=horizon,
    )

    # Display results
    table = Table(title=f"Forecast @ {at_ts}  (horizon={horizon}m)")
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("model_version", out.model_version)
    table.add_row("horizon_minutes", str(out.horizon_minutes))
    table.add_row("expected_move_sigma", f"{out.expected_move_sigma:+.3f}")
    table.add_row("p_up_05sigma", f"{out.p_up_05sigma:.3f}")
    table.add_row("p_down_05sigma", f"{out.p_down_05sigma:.3f}")
    table.add_row("confidence", f"{out.confidence:.3f}")
    table.add_row("meta_act", str(out.meta_act))
    table.add_row("meta_p_correct", f"{out.meta_p_correct:.3f}")
    table.add_row("inference_ms", f"{out.inference_ms:.1f}")

    console.print(table)

    bins_table = Table(title="Bin Probabilities")
    bins_table.add_column("Bin", justify="right")
    bins_table.add_column("Range", style="dim")
    bins_table.add_column("P", justify="right")

    bin_labels = [
        "< -3σ", "-3..-2σ", "-2..-1σ", "-1..-0.5σ", "-0.5..-0.2σ",
        "-0.2..+0.2σ", "+0.2..+0.5σ", "+0.5..+1σ", "+1..+2σ", "+2..+3σ", "> +3σ",
    ]
    for i, (label, p) in enumerate(zip(bin_labels, out.p_bins)):
        bar = "█" * int(p * 40)
        bins_table.add_row(str(i), label, f"{p:.4f} {bar}")

    console.print(bins_table)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_window(
    directory: Path,
    ts_ms: int,
    n_bars: int,
    bar_ms: int,
) -> Optional["pl.DataFrame"]:
    """Load the last n_bars from Parquet files up to ts_ms."""
    import polars as pl

    if not directory.exists():
        return None

    # Determine which daily files we need
    window_start_ms = ts_ms - n_bars * bar_ms * 2  # load 2x for safety
    files = sorted(directory.glob("*.parquet"))
    relevant: list[Path] = []
    for f in files:
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        # Day end in ms
        day_end_ms = int(
            datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000
        )
        day_start_ms = int(
            datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
        )
        if day_end_ms >= window_start_ms and day_start_ms <= ts_ms:
            relevant.append(f)

    if not relevant:
        return None

    frames: list[pl.DataFrame] = []
    for f in relevant:
        df = pl.read_parquet(f)
        if "bar_time_ms" not in df.columns:
            continue
        frames.append(df.filter(pl.col("bar_time_ms") <= ts_ms))

    if not frames:
        return None

    combined = pl.concat(frames).sort("bar_time_ms").tail(n_bars)
    return combined


# Allow Optional import above
from typing import Optional  # noqa: E402
