"""Per-step and terminal reward functions."""

from __future__ import annotations


def reward_step(
    *,
    slippage_bps: float,
    spread_bps: float,
    is_taker: bool,
    is_filled: bool,
    cancel_count: int,
    window_overshoot: bool,
    alpha: float = 0.5,   # spread cost weight
    beta: float = 0.1,    # fill bonus weight
    gamma: float = 0.2,   # cancel penalty weight
    delta: float = 2.0,   # window overshoot penalty
) -> float:
    """reward = -slippage_bps - alpha*spread_bps*is_taker + beta*is_filled - gamma*cancel_count - delta*window_overshoot

    Sign convention:
    - Lower slippage is better (negative contribution)
    - Taking liquidity incurs spread cost (negative contribution)
    - Getting filled is rewarded (positive contribution)
    - Cancels are penalised (negative contribution)
    - Exceeding the execution window is severely penalised (negative)
    """
    r = 0.0
    r -= slippage_bps
    r -= alpha * spread_bps * float(is_taker)
    r += beta * float(is_filled)
    r -= gamma * float(cancel_count)
    r -= delta * float(window_overshoot)
    return r


def reward_terminal(
    *,
    episode_pnl_usd: float,
    baseline_pnl_usd: float,
) -> float:
    """Episodic shaping: episode_pnl - baseline_pnl (vs Almgren-Chriss).

    Positive when the RL policy achieves better execution than the
    Almgren-Chriss cosine schedule baseline.  Units: USD.
    """
    return episode_pnl_usd - baseline_pnl_usd


__all__ = ["reward_step", "reward_terminal"]
