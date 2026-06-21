"""Collect offline RL dataset by running the baseline with perturbations."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def collect_offline_dataset(
    *,
    tick_data_dir: Path,
    decisions_df: "Any",   # polars DataFrame of past aggregator decisions
    n_episodes: int = 50_000,
    perturbation_std: float = 0.2,
    baseline: "Any",       # AlmgrenChrissBaseline
    seed: int = 42,
) -> "Any":  # polars DataFrame
    """Run baseline (with action noise ~ Normal(0, perturbation_std)) and record
    (state, action, reward, next_state, done) tuples.

    Returns DataFrame with columns:
      state (list[float]), action (list[float]), reward (float),
      next_state (list[float]), done (bool).

    Perturbation is critical: without it CQL has no out-of-distribution coverage.
    """
    import polars as pl

    from intraday.aggregator.decision import Decision
    from intraday.rl.action import decode_action, action_to_order_requests, ExecutionAction
    from intraday.rl.env import ExecutionEnv
    from intraday.rl.state import STATE_DIM

    rng = np.random.default_rng(seed)

    records: list[dict] = []
    episode_count = 0
    total_transitions = 0

    # Validate decisions_df
    required_cols = {"side", "confidence", "ts_ms"}
    df_cols = set(decisions_df.columns)
    if not required_cols.issubset(df_cols):
        missing = required_cols - df_cols
        raise ValueError(f"decisions_df missing columns: {missing}")

    n_decisions = len(decisions_df)
    if n_decisions == 0:
        log.warning("data_collection.no_decisions")
        return pl.DataFrame(
            {
                "state": pl.Series([], dtype=pl.List(pl.Float32)),
                "action": pl.Series([], dtype=pl.List(pl.Float32)),
                "reward": pl.Series([], dtype=pl.Float64),
                "next_state": pl.Series([], dtype=pl.List(pl.Float32)),
                "done": pl.Series([], dtype=pl.Boolean),
            }
        )

    # Discover available tick data files
    tick_files = sorted(tick_data_dir.glob("**/*.parquet")) if tick_data_dir.exists() else []
    has_tick_files = len(tick_files) > 0

    log.info(
        "data_collection.start",
        n_episodes=n_episodes,
        n_decisions=n_decisions,
        perturbation_std=perturbation_std,
        tick_files=len(tick_files),
    )

    for ep_idx in range(n_episodes):
        # Pick a random decision row
        row_idx = int(rng.integers(0, n_decisions))
        row = decisions_df.row(row_idx, named=True)

        decision = Decision(
            ts_ms=int(row["ts_ms"]),
            side=str(row["side"]),
            confidence=float(row.get("confidence", 0.6)),
            horizon_minutes=int(row.get("horizon_minutes", 15)),
            reason=str(row.get("reason", "")),
        )

        if decision.side not in ("long", "short"):
            continue

        # Load tick data for this episode
        if has_tick_files:
            tick_file = tick_files[int(rng.integers(0, len(tick_files)))]
            try:
                tick_df = pl.read_parquet(tick_file)
            except Exception as exc:
                log.warning("data_collection.tick_load_error", file=str(tick_file), error=str(exc))
                tick_df = _make_synthetic_tick_df(decision.ts_ms, rng)
        else:
            tick_df = _make_synthetic_tick_df(decision.ts_ms, rng)

        env = ExecutionEnv(
            tick_data_df=tick_df,
            decision=decision,
            baseline=baseline,
            seed=int(rng.integers(0, 2**31)),
        )

        obs, _ = env.reset()
        done = False
        ep_transitions = 0

        while not done:
            elapsed_s = ep_transitions * 5.0
            baseline_action_dict = baseline.step(obs, elapsed_s)

            base_action = _dict_to_action_array(baseline_action_dict, rng)
            noise = rng.normal(0.0, perturbation_std, size=4).astype(np.float32)
            perturbed_action = np.clip(base_action + noise, -1.0, 1.0).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(perturbed_action)
            done = terminated or truncated

            records.append(
                {
                    "state": obs.tolist(),
                    "action": perturbed_action.tolist(),
                    "reward": float(reward),
                    "next_state": next_obs.tolist(),
                    "done": done,
                }
            )

            obs = next_obs
            ep_transitions += 1
            total_transitions += 1

        episode_count += 1
        if episode_count % 1000 == 0:
            log.info(
                "data_collection.progress",
                episodes=episode_count,
                total_transitions=total_transitions,
            )

    log.info(
        "data_collection.complete",
        episodes=episode_count,
        total_transitions=total_transitions,
    )

    if not records:
        return _empty_dataset()

    states = [r["state"] for r in records]
    actions = [r["action"] for r in records]
    rewards = [r["reward"] for r in records]
    next_states = [r["next_state"] for r in records]
    dones = [r["done"] for r in records]

    return pl.DataFrame(
        {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
            "done": dones,
        }
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dict_to_action_array(d: dict, rng: np.random.Generator) -> np.ndarray:
    """Convert a baseline action dict to a 4-dim array in [-1, 1]."""
    import math

    order_type = d.get("order_type", "post_only")
    tick_offset = float(d.get("tick_offset", 0.0))
    child_size_pct = float(d.get("child_size_pct", 0.1))
    urgency = float(d.get("urgency", 0.1))

    # Invert the sigmoid for order_type encoding
    order_type_map = {"post_only": 0.16, "limit_ioc": 0.5, "market": 0.84, "cancel_all": 0.16}
    ot_prob = order_type_map.get(order_type, 0.16)
    # logit
    ot_prob = max(1e-6, min(1.0 - 1e-6, ot_prob))
    a0 = math.log(ot_prob / (1.0 - ot_prob))

    # atanh of tick_offset / 5
    t_norm = max(-0.999, min(0.999, tick_offset / 5.0))
    a1 = math.atanh(t_norm)

    # logit of child_size_pct
    cs = max(1e-6, min(1.0 - 1e-6, child_size_pct))
    a2 = math.log(cs / (1.0 - cs))

    # logit of urgency
    urg = max(1e-6, min(1.0 - 1e-6, urgency))
    a3 = math.log(urg / (1.0 - urg))

    a = np.array([a0, a1, a2, a3], dtype=np.float32)
    return np.clip(a, -1.0, 1.0)


def _make_synthetic_tick_df(ts_ms: int, rng: np.random.Generator) -> "Any":
    """Generate a minimal synthetic tick DataFrame for episodes without real data."""
    import polars as pl

    mid = 60_000.0 + rng.normal(0.0, 1000.0)
    n_ticks = 300
    timestamps = [ts_ms + i * 1000 for i in range(n_ticks)]
    prices = [mid + rng.normal(0.0, 10.0) for _ in range(n_ticks)]
    volumes = [float(rng.exponential(0.1)) for _ in range(n_ticks)]

    return pl.DataFrame(
        {
            "ts_ms": timestamps,
            "price": prices,
            "volume": volumes,
        }
    )


def _empty_dataset() -> "Any":
    """Return an empty DataFrame with the correct schema."""
    import polars as pl

    return pl.DataFrame(
        {
            "state": pl.Series([], dtype=pl.List(pl.Float32)),
            "action": pl.Series([], dtype=pl.List(pl.Float32)),
            "reward": pl.Series([], dtype=pl.Float64),
            "next_state": pl.Series([], dtype=pl.List(pl.Float32)),
            "done": pl.Series([], dtype=pl.Boolean),
        }
    )


__all__ = ["collect_offline_dataset"]
