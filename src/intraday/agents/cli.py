"""CLI commands for specialist agents.

Sub-commands:
    train   — fit a learnable agent (e.g. regime) on historical feature data
    predict — run a single-bar prediction from the feature store
    inspect — show opinions across a date range in a Rich table
"""

import inspect
import json
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Optional

import polars as pl
import structlog
import typer
from rich.console import Console
from rich.table import Table

from intraday.agents.registry import get_agent, _REGISTRY, _auto_register
from intraday.utils.logging import setup_logging

log = structlog.get_logger(__name__)
console = Console()

agents_app = typer.Typer(help="Specialist agents")


def _ensure_registered() -> None:
    _auto_register()


def _load_features(
    data_dir: Path,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
) -> pl.DataFrame:
    """Load feature parquet files for the given date range."""
    features_dir = data_dir / "features" / symbol
    if not features_dir.exists():
        console.print(f"[red]Feature directory not found: {features_dir}[/red]")
        console.print("Run: intraday features compute --start YYYY-MM-DD")
        raise typer.Exit(1)

    files = sorted(features_dir.glob("*.parquet"))
    if not files:
        console.print("[red]No feature files found.[/red]")
        raise typer.Exit(1)

    filtered: list[Path] = []
    for f in files:
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if start and d < date.fromisoformat(start):
            continue
        if end and d > date.fromisoformat(end):
            continue
        filtered.append(f)

    if not filtered:
        console.print(f"[red]No feature files in range {start} → {end}[/red]")
        raise typer.Exit(1)

    dfs = [pl.read_parquet(f) for f in filtered]
    return pl.concat(dfs).sort("bar_time_ms")


# ── train ──────────────────────────────────────────────────────────────────

@agents_app.command("train")
def agent_train(
    name: Annotated[str, typer.Argument(help="Agent name (e.g. 'regime')")],
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD")] = None,
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="Futures symbol")] = "BTCUSDT",
    model_out: Annotated[Optional[Path], typer.Option(help="Path to save fitted model")] = None,
) -> None:
    """Fit a learnable agent on historical feature data.

    Only agents with a fit() method support training (currently: regime).

    Examples:
        intraday agents train regime --start 2026-01-01 --end 2026-06-01
    """
    setup_logging(log_level="info")
    _ensure_registered()

    try:
        # Only pass model_dir if the agent accepts it (RegimeAgent doesn't)
        agent_cls = _REGISTRY.get(name)
        if agent_cls is None:
            raise KeyError(f"Unknown agent '{name}'")
        sig = inspect.signature(agent_cls.__init__)
        kwargs: dict[str, Any] = {}
        if model_out and "model_dir" in sig.parameters:
            kwargs["model_dir"] = model_out.parent
        agent = agent_cls(**kwargs)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not hasattr(agent, "fit"):
        console.print(f"[red]Agent '{name}' does not support training (no fit() method).[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Loading features for {symbol}...[/yellow]")
    features_df = _load_features(data_dir, symbol, start, end)
    console.print(f"Loaded {len(features_df):,} rows, {features_df.select(pl.first()).shape[1]} columns")

    console.print(f"[yellow]Training agent '{name}'...[/yellow]")
    try:
        agent.fit(features_df)
    except ImportError as exc:
        console.print(f"[red]Missing dependency: {exc}[/red]")
        raise typer.Exit(1) from exc

    if model_out is None:
        model_out = data_dir / "models" / f"{name}.pkl"

    if hasattr(agent, "save"):
        agent.save(model_out)
        console.print(f"[green]Model saved to {model_out}[/green]")
    else:
        console.print(f"[yellow]Agent '{name}' has no save() method; model not persisted.[/yellow]")


# ── predict ────────────────────────────────────────────────────────────────

@agents_app.command("predict")
def agent_predict(
    name: Annotated[str, typer.Argument(help="Agent name")],
    at_ts: Annotated[str, typer.Argument(help="Bar timestamp (YYYY-MM-DD HH:MM or ms int)")],
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="Futures symbol")] = "BTCUSDT",
    model_path: Annotated[Optional[Path], typer.Option(help="Path to pre-fitted model")] = None,
) -> None:
    """Run a single-bar prediction from the feature store.

    Prints the AgentOpinion as JSON to stdout.

    Examples:
        intraday agents predict orderflow "2026-06-01 12:00"
        intraday agents predict regime "2026-06-01 12:00" --model-path data/models/regime.pkl
    """
    setup_logging(log_level="warning")
    _ensure_registered()

    # ── Resolve timestamp ──────────────────────────────────────────────────
    ts_ms: Optional[int] = None
    if at_ts.isdigit():
        ts_ms = int(at_ts)
    else:
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(at_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts_ms = int(dt.timestamp() * 1000)
        except ValueError:
            console.print(f"[red]Cannot parse timestamp: {at_ts!r}[/red]")
            raise typer.Exit(1)

    # ── Load agent ────────────────────────────────────────────────────────
    try:
        if model_path and model_path.exists():
            from intraday.agents import RegimeAgent
            if name == "regime":
                agent = RegimeAgent.load(model_path)
            else:
                agent = get_agent(name)
        else:
            agent = get_agent(name)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    # ── Find nearest bar ──────────────────────────────────────────────────
    features_dir = data_dir / "features" / symbol
    if not features_dir.exists():
        console.print(f"[red]Feature directory not found: {features_dir}[/red]")
        raise typer.Exit(1)

    # Find the file containing ts_ms
    from datetime import datetime, timezone
    dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    day_stem = dt_utc.date().isoformat()
    f = features_dir / f"{day_stem}.parquet"

    if not f.exists():
        # Fall back to nearest available file
        files = sorted(features_dir.glob("*.parquet"))
        if not files:
            console.print("[red]No feature files found.[/red]")
            raise typer.Exit(1)
        f = files[-1]
        console.print(f"[yellow]Exact date not found; using {f.stem}[/yellow]")

    df = pl.read_parquet(f).sort("bar_time_ms")

    # Find nearest row by ts_ms
    diffs = (df["bar_time_ms"] - ts_ms).abs()
    nearest_idx = int(diffs.arg_min())
    row = df.row(nearest_idx, named=True)

    opinion = agent.predict(row)
    sys.stdout.write(opinion.model_dump_json(indent=2) + "\n")


# ── inspect ────────────────────────────────────────────────────────────────

@agents_app.command("inspect")
def agent_inspect(
    name: Annotated[str, typer.Argument(help="Agent name")],
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD")] = None,
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    symbol: Annotated[str, typer.Option(help="Futures symbol")] = "BTCUSDT",
    model_path: Annotated[Optional[Path], typer.Option(help="Path to pre-fitted model")] = None,
    max_rows: Annotated[int, typer.Option(help="Maximum rows to display")] = 50,
) -> None:
    """Show agent opinions across a date range in a Rich table.

    Examples:
        intraday agents inspect orderflow --start 2026-06-01 --end 2026-06-10
        intraday agents inspect regime --start 2026-06-01 --model-path data/models/regime.pkl
    """
    setup_logging(log_level="warning")
    _ensure_registered()

    # ── Load agent ────────────────────────────────────────────────────────
    try:
        if model_path and model_path.exists() and name == "regime":
            from intraday.agents import RegimeAgent
            agent = RegimeAgent.load(model_path)
        else:
            agent = get_agent(name)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    # ── Load features ─────────────────────────────────────────────────────
    features_df = _load_features(data_dir, symbol, start, end)

    # ── Run predictions ───────────────────────────────────────────────────
    console.print(
        f"[yellow]Running {name} agent on {min(len(features_df), max_rows)} bars...[/yellow]"
    )

    rows_to_process = features_df.head(max_rows)
    opinions = []
    for row in rows_to_process.iter_rows(named=True):
        opinion = agent.predict(row)
        opinions.append(opinion)

    # ── Build Rich table ──────────────────────────────────────────────────
    table = Table(title=f"Agent: {name}", show_header=True, header_style="bold cyan")
    table.add_column("ts_ms", style="dim", no_wrap=True)
    table.add_column("confidence", justify="right")
    table.add_column("inference_ms", justify="right")

    # Determine payload keys from first opinion
    payload_keys: list[str] = []
    if opinions:
        payload_keys = [
            k for k in opinions[0].payload
            if not isinstance(opinions[0].payload[k], dict)
        ]
        for k in payload_keys:
            table.add_column(k, justify="right")

    for op in opinions:
        from datetime import datetime, timezone
        dt_str = datetime.fromtimestamp(op.ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        row_vals = [
            dt_str,
            f"{op.confidence:.3f}",
            f"{op.inference_ms:.2f}",
        ]
        for k in payload_keys:
            v = op.payload.get(k)
            if isinstance(v, float):
                row_vals.append(f"{v:.4f}")
            elif isinstance(v, bool):
                row_vals.append("[red]YES[/red]" if v else "no")
            else:
                row_vals.append(str(v))
        table.add_row(*row_vals)

    console.print(table)
    console.print(f"\nTotal opinions: {len(opinions)}")
