"""Gymnasium environment for RL execution training."""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from intraday.rl.action import ExecutionAction, decode_action, action_to_order_requests
from intraday.rl.baseline import AlmgrenChrissBaseline
from intraday.rl.reward import reward_step, reward_terminal
from intraday.rl.state import STATE_DIM, build_state_vector

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

if TYPE_CHECKING:
    import polars as pl
    from intraday.aggregator.decision import Decision
    from intraday.sim.book import LocalOrderBook

log = structlog.get_logger(__name__)

_WINDOW_SECONDS = 300.0  # 5-minute execution window
_TICK_SIZE = 0.1


class ExecutionEnv:
    """Gymnasium-compatible environment for RL execution training.

    Episode lifecycle:
    - Starts when an aggregator Decision arrives.
    - Ends when: fully filled OR 5-minute window expires OR direction changes.

    State:  STATE_DIM-dim float32 vector.
    Action: 4-dim continuous Box([-1,1]^4).
    Reward: step-level slippage-based + terminal AC-relative shaping.
    """

    metadata: dict = {"render_modes": []}

    if GYM_AVAILABLE:
        observation_space: Any = None
        action_space: Any = None

    def __init__(
        self,
        *,
        tick_data_df: "pl.DataFrame",
        decision: "Decision",
        baseline: AlmgrenChrissBaseline,
        tick_size: float = 0.1,
        window_seconds: float = 300.0,
        seed: int = 42,
    ) -> None:
        self._tick_data_df = tick_data_df
        self._decision = decision
        self._baseline = baseline
        self._tick_size = tick_size
        self._window_seconds = window_seconds
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Episode state (initialised in reset)
        self._step_idx: int = 0
        self._window_start_ms: int = 0
        self._window_end_ms: int = 0
        self._target_usd: float = 0.0
        self._filled_usd: float = 0.0
        self._filled_qty_base: float = 0.0
        self._target_qty_base: float = 0.0
        self._equity_usd: float = 100_000.0
        self._recent_fills: list = []
        self._cancel_count: int = 0
        self._episode_id: str = ""
        self._baseline_pnl_usd: float = 0.0
        self._episode_pnl_usd: float = 0.0

        # Build a simulated book from the tick data
        self._book: LocalOrderBook = self._build_book()
        self._mid_price: float = self._book.mid_price()

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(STATE_DIM,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )

        log.debug(
            "execution_env.init",
            decision_side=decision.side,
            window_seconds=window_seconds,
            tick_size=tick_size,
        )

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._episode_id = uuid.uuid4().hex[:12]
        self._step_idx = 0
        self._filled_usd = 0.0
        self._filled_qty_base = 0.0
        self._cancel_count = 0
        self._recent_fills = []
        self._episode_pnl_usd = 0.0

        # Determine start time from tick data
        ts_col = "ts_ms" if "ts_ms" in self._tick_data_df.columns else "timestamp_ms"
        if len(self._tick_data_df) > 0:
            first_ts = int(self._tick_data_df[ts_col][0])
        else:
            import time
            first_ts = int(time.time() * 1000)

        self._window_start_ms = first_ts
        self._window_end_ms = first_ts + int(self._window_seconds * 1000)

        # Infer target from decision — use a notional size based on equity
        self._equity_usd = 100_000.0
        self._mid_price = self._book.mid_price()
        if self._mid_price <= 0.0:
            self._mid_price = 60_000.0

        notional_fraction = 0.01  # 1% of equity per trade
        self._target_usd = self._equity_usd * notional_fraction
        if self._decision.side == "short":
            self._target_usd = -self._target_usd

        self._target_qty_base = abs(self._target_usd) / self._mid_price

        # Plan the AC baseline
        self._baseline.plan(
            target_qty_base=self._target_qty_base,
            side="buy" if self._decision.side == "long" else "sell",
            window_seconds=self._window_seconds,
        )
        self._baseline_pnl_usd = 0.0

        obs = self._compute_obs()
        info = {
            "episode_id": self._episode_id,
            "target_usd": self._target_usd,
            "window_start_ms": self._window_start_ms,
            "window_end_ms": self._window_end_ms,
        }
        log.debug("execution_env.reset", episode_id=self._episode_id)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one execution step.

        Returns: (obs, reward, terminated, truncated, info)
        terminated = True when fully filled or window expired.
        truncated  = True when maximum steps exceeded (safety cap).
        """
        action = np.asarray(action, dtype=np.float32)
        exec_action = decode_action(action)

        current_ts_ms = self._window_start_ms + int(self._step_idx * 5000)  # ~5s steps
        elapsed_s = (current_ts_ms - self._window_start_ms) / 1000.0

        remaining_usd = abs(self._target_usd) - self._filled_usd
        remaining_qty = max(self._target_qty_base - self._filled_qty_base, 0.0)
        side = "buy" if self._decision.side == "long" else "sell"

        # Execute the action against the simulated book
        order_requests = action_to_order_requests(
            exec_action,
            remaining_qty_base=remaining_qty,
            side=side,
            book=self._book,
            tick_size=self._tick_size,
        )

        is_filled = False
        is_taker = False
        step_slippage_bps = 0.0
        fill_usd = 0.0

        for req in order_requests:
            if req.qty_base <= 0.0:
                continue

            mid = self._book.mid_price()
            if mid <= 0.0:
                continue

            fill_qty = req.qty_base
            is_maker = req.type in ("post_only", "limit")

            if req.type == "market":
                fill_price = mid * (1.0 + 0.001 * (1 if side == "buy" else -1))
                is_taker = True
            elif req.type == "ioc":
                fill_price = req.limit_price if req.limit_price else mid
                prob_fill = 0.7
                if self._rng.random() > prob_fill:
                    fill_qty = 0.0
                is_taker = True
            else:
                # post_only: fills with some maker probability
                fill_price = req.limit_price if req.limit_price else (mid - self._tick_size)
                prob_fill = 0.4
                if self._rng.random() > prob_fill:
                    fill_qty = 0.0
                is_taker = False

            if fill_qty > 0.0:
                is_filled = True
                if mid > 0.0:
                    step_slippage_bps = abs(fill_price - mid) / mid * 10_000.0
                fill_usd += fill_qty * fill_price
                self._filled_qty_base += fill_qty
                self._filled_usd += fill_qty * fill_price
                self._episode_pnl_usd -= step_slippage_bps * fill_qty * fill_price / 10_000.0

        if exec_action.urgency > 0.7:
            self._cancel_count += 1

        spread_bps = self._book.spread_bps()
        window_overshoot = current_ts_ms > self._window_end_ms and remaining_qty > 0.0

        r = reward_step(
            slippage_bps=step_slippage_bps,
            spread_bps=spread_bps,
            is_taker=is_taker,
            is_filled=is_filled,
            cancel_count=self._cancel_count,
            window_overshoot=window_overshoot,
        )

        self._step_idx += 1
        terminated = (
            self._filled_qty_base >= self._target_qty_base * 0.999
            or current_ts_ms >= self._window_end_ms
        )

        if terminated:
            terminal_r = reward_terminal(
                episode_pnl_usd=self._episode_pnl_usd,
                baseline_pnl_usd=self._baseline_pnl_usd,
            )
            r += terminal_r

        truncated = self._step_idx > 120  # safety: max 120 steps (~10 min)

        obs = self._compute_obs()
        info = {
            "filled_usd": self._filled_usd,
            "remaining_usd": abs(self._target_usd) - self._filled_usd,
            "fill_pct": self._filled_qty_base / max(self._target_qty_base, 1e-10),
            "slippage_bps": step_slippage_bps,
            "episode_id": self._episode_id,
        }

        return obs, r, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_obs(self) -> np.ndarray:
        current_ts_ms = self._window_start_ms + int(self._step_idx * 5000)
        spread_bps = self._book.spread_bps()

        book_features = {
            "spread_bps": spread_bps,
            "ofi": 0.0,
            "queue_imbalance": self._estimate_queue_imbalance(),
            "vpin": 0.5,
            "microprice_drift_5m_z": 0.0,
            "recent_cancel_rate": min(self._cancel_count / 10.0, 1.0),
        }

        vol_regime_id = 1  # default normal
        if spread_bps < 0.5:
            vol_regime_id = 0
        elif spread_bps > 2.0:
            vol_regime_id = 2

        return build_state_vector(
            ts_ms=current_ts_ms,
            window_start_ms=self._window_start_ms,
            window_end_ms=self._window_end_ms,
            target_usd=self._target_usd,
            filled_usd=self._filled_usd,
            book_features=book_features,
            vol_regime_id=vol_regime_id,
            forecast_confidence=self._decision.confidence,
            recent_fills=self._recent_fills,
            equity_usd=self._equity_usd,
        )

    def _estimate_queue_imbalance(self) -> float:
        """Estimate queue imbalance from the local order book."""
        bb_price, bb_qty = self._book.best_bid()
        ba_price, ba_qty = self._book.best_ask()
        total = bb_qty + ba_qty
        if total <= 0.0:
            return 0.0
        return (bb_qty - ba_qty) / total

    def _build_book(self) -> "LocalOrderBook":
        """Build a minimal LocalOrderBook from tick data for episode simulation."""
        from intraday.sim.book import LocalOrderBook

        book = LocalOrderBook()
        mid = 60_000.0  # default BTC midprice

        # Try to extract a midprice from tick data
        price_cols = ["price", "close", "mid_price"]
        for col in price_cols:
            if col in self._tick_data_df.columns and len(self._tick_data_df) > 0:
                try:
                    mid = float(self._tick_data_df[col][0])
                    break
                except Exception:
                    pass

        half_spread = mid * 0.00005  # 0.5 bps half-spread
        bids = [(round(mid - half_spread, 1), 5.0), (round(mid - 2 * half_spread, 1), 10.0)]
        asks = [(round(mid + half_spread, 1), 5.0), (round(mid + 2 * half_spread, 1), 10.0)]
        book.apply_snapshot(bids, asks)
        return book


__all__ = ["ExecutionEnv"]
