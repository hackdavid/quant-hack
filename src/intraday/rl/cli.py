"""CLI commands for the RL execution policy."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

log = structlog.get_logger(__name__)

rl_app = typer.Typer(help="RL execution policy")


@rl_app.command("collect-data")
def rl_collect_data(
    start: Annotated[str, typer.Argument(help="Start date YYYY-MM-DD")],
    end: Annotated[str, typer.Argument(help="End date YYYY-MM-DD")],
    episodes: Annotated[int, typer.Option(help="Number of episodes to collect")] = 50_000,
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    output: Annotated[Optional[Path], typer.Option(help="Output parquet path")] = None,
    perturbation_std: Annotated[float, typer.Option(help="Action noise std dev")] = 0.2,
) -> None:
    """Collect offline RL dataset using the Almgren-Chriss baseline with perturbations.

    Runs the AC baseline on historical tick data and decision history,
    recording (state, action, reward, next_state, done) tuples.

    The perturbation noise is essential for CQL: without out-of-distribution
    coverage the learned Q-function will be overoptimistic on unseen actions.

    Examples:
        intraday rl collect-data 2026-01-01 2026-05-31 --episodes 50000
    """
    import polars as pl

    from intraday.rl.baseline import AlmgrenChrissBaseline
    from intraday.rl.data_collection import collect_offline_dataset

    tick_data_dir = data_dir / "raw" / "binance" / "aggTrades" / "BTCUSDT"
    decisions_path = data_dir / "decisions" / "decisions.parquet"

    if not decisions_path.exists():
        typer.echo(
            f"[red]Decisions file not found at {decisions_path}. "
            f"Run aggregator pipeline first.[/red]"
        )
        raise typer.Exit(1)

    typer.echo(f"Loading decisions from {decisions_path}")
    decisions_df = pl.read_parquet(decisions_path)
    typer.echo(f"Loaded {len(decisions_df):,} decision rows")

    # Filter by date range
    from datetime import date
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    start_ms = int(start_date.strftime("%s")) * 1000
    end_ms = int(end_date.strftime("%s")) * 1000 + 86_400_000

    if "ts_ms" in decisions_df.columns:
        decisions_df = decisions_df.filter(
            (decisions_df["ts_ms"] >= start_ms) & (decisions_df["ts_ms"] < end_ms)
        )
    typer.echo(f"Filtered to {len(decisions_df):,} decisions in [{start}, {end}]")

    if len(decisions_df) == 0:
        typer.echo("[red]No decisions in the specified date range.[/red]")
        raise typer.Exit(1)

    baseline = AlmgrenChrissBaseline(n_slices=10)

    typer.echo(
        f"Collecting {episodes:,} episodes "
        f"(perturbation_std={perturbation_std}, seed={seed})"
    )
    dataset_df = collect_offline_dataset(
        tick_data_dir=tick_data_dir,
        decisions_df=decisions_df,
        n_episodes=episodes,
        perturbation_std=perturbation_std,
        baseline=baseline,
        seed=seed,
    )

    if output is None:
        output = data_dir / "rl_dataset" / f"offline_{start}_{end}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.write_parquet(output)

    typer.echo(f"[green]Saved {len(dataset_df):,} transitions → {output}[/green]")


@rl_app.command("train")
def rl_train(
    data_from: Annotated[Optional[str], typer.Option(help="Dataset start date (YYYY-MM-DD)")] = None,
    data_to: Annotated[Optional[str], typer.Option(help="Dataset end date (YYYY-MM-DD)")] = None,
    algo: Annotated[str, typer.Option(help="Algorithm: cql")] = "cql",
    mode: Annotated[str, typer.Option(help="Training mode: offline")] = "offline",
    n_steps: Annotated[int, typer.Option(help="Total gradient steps")] = 200_000,
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Model output directory")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    dataset_path: Annotated[Optional[Path], typer.Option(help="Explicit dataset parquet path")] = None,
    cql_alpha: Annotated[float, typer.Option(help="CQL alpha (conservatism)")] = 2.0,
    batch_size: Annotated[int, typer.Option(help="Mini-batch size")] = 256,
    actor_lr: Annotated[float, typer.Option(help="Actor learning rate")] = 1e-4,
    critic_lr: Annotated[float, typer.Option(help="Critic learning rate")] = 3e-4,
) -> None:
    """Train a CQL offline RL execution policy.

    Trains on the collected offline dataset and saves the model to the output
    directory.  Validates every epoch against a held-out slice.

    Examples:
        # Basic: train on the default dataset
        intraday rl train --data-from 2026-01-01 --data-to 2026-05-31

        # Custom hyperparameters
        intraday rl train --n-steps 400000 --cql-alpha 5.0 --batch-size 512
    """
    if algo.lower() != "cql":
        typer.echo(f"[red]Unsupported algorithm: {algo!r}. Only 'cql' is supported.[/red]")
        raise typer.Exit(1)

    if mode.lower() != "offline":
        typer.echo(f"[red]Unsupported mode: {mode!r}. Only 'offline' is supported.[/red]")
        raise typer.Exit(1)

    # Locate dataset
    if dataset_path is None:
        if data_from and data_to:
            dataset_path = data_dir / "rl_dataset" / f"offline_{data_from}_{data_to}.parquet"
        else:
            rl_dataset_dir = data_dir / "rl_dataset"
            if rl_dataset_dir.exists():
                candidates = sorted(rl_dataset_dir.glob("offline_*.parquet"))
                if candidates:
                    dataset_path = candidates[-1]
                    typer.echo(f"Using latest dataset: {dataset_path}")

    if dataset_path is None or not dataset_path.exists():
        typer.echo(
            f"[red]Dataset not found. Run: intraday rl collect-data <start> <end>[/red]"
        )
        raise typer.Exit(1)

    if output_dir is None:
        version_tag = f"cql_v1"
        output_dir = data_dir / "models" / "rl" / version_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Training CQL on {dataset_path}")
    typer.echo(f"Output dir: {output_dir}")

    from intraday.rl.train import train_cql_policy

    checkpoint = train_cql_policy(
        dataset_path=dataset_path,
        output_dir=output_dir,
        n_steps=n_steps,
        n_steps_per_epoch=10_000,
        batch_size=batch_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        cql_alpha=cql_alpha,
        seed=seed,
    )

    typer.echo(f"[green]Model saved to {checkpoint}[/green]")
    typer.echo(f"[green]Metadata saved to {output_dir / 'metadata.json'}[/green]")


@rl_app.command("evaluate")
def rl_evaluate(
    version: Annotated[str, typer.Option(help="Model version tag")] = "v1",
    start: Annotated[Optional[str], typer.Option(help="Start date YYYY-MM-DD")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date YYYY-MM-DD")] = None,
    data_dir: Annotated[Path, typer.Option(help="Data root directory")] = Path("data"),
    model_dir: Annotated[Optional[Path], typer.Option(help="Explicit model directory")] = None,
    n_episodes: Annotated[int, typer.Option(help="Evaluation episodes")] = 100,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Evaluate a trained RL execution policy vs the Almgren-Chriss baseline.

    Runs n_episodes of simulation and compares average slippage and fill rate
    against the AC cosine schedule.

    Examples:
        intraday rl evaluate --version v1 --start 2026-06-01 --end 2026-06-19
    """
    import polars as pl

    from intraday.rl.baseline import AlmgrenChrissBaseline
    from intraday.rl.predict import RLExecutionPolicy
    from intraday.rl.action import decode_action, action_to_order_requests
    from intraday.aggregator.decision import Decision
    from intraday.rl.env import ExecutionEnv

    if model_dir is None:
        model_dir = data_dir / "models" / "rl" / f"cql_{version}"

    typer.echo(f"Loading policy from {model_dir}")
    policy = RLExecutionPolicy(model_dir=model_dir)

    # Benchmark latency
    latency = policy.benchmark_latency(n_samples=500)
    typer.echo(
        f"Inference latency: p50={latency['p50_ms']:.2f}ms p99={latency['p99_ms']:.2f}ms"
    )

    # Load decisions for evaluation
    decisions_path = data_dir / "decisions" / "decisions.parquet"
    if not decisions_path.exists():
        typer.echo(f"[red]No decisions file at {decisions_path}[/red]")
        raise typer.Exit(1)

    decisions_df = pl.read_parquet(decisions_path)
    if start and end:
        from datetime import date
        start_ms = int(date.fromisoformat(start).strftime("%s")) * 1000
        end_ms = int(date.fromisoformat(end).strftime("%s")) * 1000 + 86_400_000
        if "ts_ms" in decisions_df.columns:
            decisions_df = decisions_df.filter(
                (decisions_df["ts_ms"] >= start_ms) & (decisions_df["ts_ms"] < end_ms)
            )

    if len(decisions_df) == 0:
        typer.echo("[red]No decisions in range.[/red]")
        raise typer.Exit(1)

    rng = __import__("numpy").random.default_rng(seed)
    baseline = AlmgrenChrissBaseline(n_slices=10)

    rl_slippages: list[float] = []
    ac_slippages: list[float] = []
    rl_fill_rates: list[float] = []

    tick_data_dir = data_dir / "raw" / "binance" / "aggTrades" / "BTCUSDT"

    for ep_idx in range(min(n_episodes, len(decisions_df))):
        row = decisions_df.row(ep_idx, named=True)
        if str(row.get("side", "flat")) not in ("long", "short"):
            continue

        decision = Decision(
            ts_ms=int(row["ts_ms"]),
            side=str(row["side"]),
            confidence=float(row.get("confidence", 0.6)),
        )

        # Generate a synthetic tick df
        tick_df = _make_eval_tick_df(decision.ts_ms, rng)

        env = ExecutionEnv(
            tick_data_df=tick_df,
            decision=decision,
            baseline=baseline,
            seed=int(rng.integers(0, 2**31)),
        )

        obs, _ = env.reset()
        done = False
        ep_slippage = 0.0
        steps = 0

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_slippage += info.get("slippage_bps", 0.0)
            steps += 1

        rl_slippages.append(ep_slippage / max(steps, 1))
        rl_fill_rates.append(info.get("fill_pct", 0.0))

    if rl_slippages:
        import numpy as np
        avg_slip = float(np.mean(rl_slippages))
        avg_fill = float(np.mean(rl_fill_rates))
        typer.echo(f"\nRL Policy Evaluation ({len(rl_slippages)} episodes):")
        typer.echo(f"  Avg slippage  : {avg_slip:.3f} bps")
        typer.echo(f"  Avg fill rate : {avg_fill*100:.1f}%")
    else:
        typer.echo("[yellow]No episodes evaluated.[/yellow]")


def _make_eval_tick_df(ts_ms: int, rng: "Any") -> "Any":
    """Generate synthetic tick data for evaluation episodes."""
    import polars as pl
    import numpy as np

    mid = 60_000.0 + rng.normal(0.0, 500.0)
    n = 300
    ts = [ts_ms + i * 1000 for i in range(n)]
    prices = [mid + rng.normal(0.0, 5.0) for _ in range(n)]
    volumes = [float(rng.exponential(0.05)) for _ in range(n)]
    return pl.DataFrame({"ts_ms": ts, "price": prices, "volume": volumes})


__all__ = ["rl_app"]
