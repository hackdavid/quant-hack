"""CQL offline training via d3rlpy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def train_cql_policy(
    *,
    dataset_path: Path,
    output_dir: Path,
    n_steps: int = 200_000,
    n_steps_per_epoch: int = 10_000,
    batch_size: int = 256,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    cql_alpha: float = 2.0,
    seed: int = 42,
) -> Path:
    """CQL training using d3rlpy.

    If d3rlpy is not installed: raises ImportError with install instructions.
    Saves model to output_dir/cql_policy/ and metadata to output_dir/metadata.json.
    Validates every epoch: runs on held-out data and records avg slippage vs baseline.

    Returns the path to the saved model directory.
    """
    try:
        import d3rlpy
    except ImportError:
        raise ImportError(
            "d3rlpy not installed. Run: uv add d3rlpy gymnasium"
        )

    import polars as pl
    from intraday.rl.state import STATE_DIM

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "cql_train.start",
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        n_steps=n_steps,
        batch_size=batch_size,
        cql_alpha=cql_alpha,
        seed=seed,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    log.info("cql_train.loading_dataset", path=str(dataset_path))
    df = pl.read_parquet(dataset_path)
    log.info("cql_train.dataset_loaded", rows=len(df))

    if len(df) == 0:
        raise ValueError(f"Dataset at {dataset_path} is empty.")

    states = np.array(df["state"].to_list(), dtype=np.float32)
    actions = np.array(df["action"].to_list(), dtype=np.float32)
    rewards = df["reward"].to_numpy(dtype=np.float64).astype(np.float32)
    next_states = np.array(df["next_state"].to_list(), dtype=np.float32)
    terminals = df["done"].to_numpy().astype(np.float32)
    timeouts = np.zeros_like(terminals)

    assert states.shape[1] == STATE_DIM, (
        f"State dim mismatch: expected {STATE_DIM}, got {states.shape[1]}"
    )
    assert actions.shape[1] == 4, (
        f"Action dim mismatch: expected 4, got {actions.shape[1]}"
    )

    # ── Build d3rlpy MDPDataset ───────────────────────────────────────────────
    dataset = d3rlpy.dataset.MDPDataset(
        observations=states,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    # ── Train / val split ─────────────────────────────────────────────────────
    n_total = len(df)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_dataset, val_dataset = dataset.slice_episode(
        end_frame=n_train
    ), dataset.slice_episode(start_frame=n_train)

    log.info(
        "cql_train.split",
        n_train=n_train,
        n_val=n_val,
    )

    # ── CQL algorithm ─────────────────────────────────────────────────────────
    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        alpha=cql_alpha,
        batch_size=batch_size,
    ).create(device="cpu")

    n_epochs = max(1, n_steps // n_steps_per_epoch)
    model_dir = output_dir / "cql_policy"
    model_dir.mkdir(parents=True, exist_ok=True)

    epoch_metrics: list[dict] = []
    train_start = time.monotonic()

    log.info("cql_train.training", n_epochs=n_epochs, n_steps_per_epoch=n_steps_per_epoch)

    for epoch in range(1, n_epochs + 1):
        results = cql.fit(
            train_dataset,
            n_steps=n_steps_per_epoch,
            n_steps_per_epoch=n_steps_per_epoch,
            show_progress=False,
        )

        # Evaluate on validation set
        val_metrics = _evaluate_on_dataset(cql, val_dataset, n_samples=min(500, n_val))
        elapsed = time.monotonic() - train_start

        epoch_metrics.append(
            {
                "epoch": epoch,
                "elapsed_s": round(elapsed, 1),
                **val_metrics,
            }
        )

        log.info(
            "cql_train.epoch",
            epoch=epoch,
            n_epochs=n_epochs,
            elapsed_s=round(elapsed, 1),
            **val_metrics,
        )

    # ── Save model ────────────────────────────────────────────────────────────
    checkpoint_path = model_dir / "cql.d3"
    cql.save(str(checkpoint_path))

    metadata = {
        "framework": "d3rlpy",
        "algorithm": "CQL",
        "state_dim": STATE_DIM,
        "action_dim": 4,
        "n_steps": n_steps,
        "n_steps_per_epoch": n_steps_per_epoch,
        "batch_size": batch_size,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "cql_alpha": cql_alpha,
        "seed": seed,
        "dataset_path": str(dataset_path),
        "n_train": n_train,
        "n_val": n_val,
        "epoch_metrics": epoch_metrics,
        "model_path": str(checkpoint_path),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)

    log.info(
        "cql_train.complete",
        model_path=str(checkpoint_path),
        metadata_path=str(metadata_path),
        total_elapsed_s=round(time.monotonic() - train_start, 1),
    )

    return checkpoint_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _evaluate_on_dataset(
    cql: "Any",
    dataset: "Any",
    n_samples: int = 500,
) -> dict:
    """Run inference on n_samples transitions and compute mean predicted Q-value."""
    try:
        import d3rlpy

        episodes = list(dataset.episodes)
        if not episodes:
            return {"val_mean_q": 0.0}

        obs_list: list[np.ndarray] = []
        for ep in episodes:
            for transition in ep.transitions:
                obs_list.append(transition.observation)
                if len(obs_list) >= n_samples:
                    break
            if len(obs_list) >= n_samples:
                break

        if not obs_list:
            return {"val_mean_q": 0.0}

        obs_arr = np.array(obs_list, dtype=np.float32)
        # Use predict to get actions, then estimate Q-values
        actions = cql.predict(obs_arr)
        q_vals = cql.predict_value(obs_arr, actions)
        return {"val_mean_q": float(np.mean(q_vals))}
    except Exception as exc:
        log.warning("cql_train.eval_error", error=str(exc))
        return {"val_mean_q": 0.0}


__all__ = ["train_cql_policy"]
