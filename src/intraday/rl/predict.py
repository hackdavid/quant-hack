"""Inference wrapper for the trained CQL policy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import structlog

from intraday.rl.state import STATE_DIM

log = structlog.get_logger(__name__)


class RLExecutionPolicy:
    """Inference wrapper for a trained CQL execution policy.

    Loads a d3rlpy CQL checkpoint and wraps it with a simple act() API.
    Falls back gracefully if d3rlpy is not installed.
    """

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        self._policy: "Any" = None
        self._metadata: dict = {}

        checkpoint_path = self._model_dir / "cql_policy" / "cql.d3"
        metadata_path = self._model_dir / "metadata.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"CQL checkpoint not found at {checkpoint_path}. "
                f"Run: intraday rl train --output-dir {self._model_dir}"
            )

        if metadata_path.exists():
            with metadata_path.open() as fh:
                self._metadata = json.load(fh)

        self._policy = self._load_policy(checkpoint_path)
        log.info(
            "rl_policy.loaded",
            model_dir=str(self._model_dir),
            state_dim=self._metadata.get("state_dim", STATE_DIM),
            action_dim=self._metadata.get("action_dim", 4),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> np.ndarray:
        """Returns 4-dim action vector given state vector.

        Inference latency target: p99 < 5 ms.
        State must be shape (STATE_DIM,) float32.
        Returns action in [-1, 1]^4.
        """
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, :]

        if self._policy is None:
            return np.zeros(4, dtype=np.float32)

        action = self._policy.predict(state)
        return np.clip(action[0].astype(np.float32), -1.0, 1.0)

    def benchmark_latency(self, n_samples: int = 1000) -> dict:
        """Returns {"p50_ms": float, "p99_ms": float} for inference latency."""
        dummy_state = np.zeros((1, STATE_DIM), dtype=np.float32)

        # Warm-up
        for _ in range(10):
            self.act(dummy_state[0])

        latencies_ms: list[float] = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            self.act(dummy_state[0])
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(latencies_ms)
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))

        log.info(
            "rl_policy.latency_benchmark",
            n_samples=n_samples,
            p50_ms=round(p50, 3),
            p99_ms=round(p99, 3),
        )
        return {"p50_ms": p50, "p99_ms": p99}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_policy(self, checkpoint_path: Path) -> "Any":
        """Load the d3rlpy policy object."""
        try:
            import d3rlpy

            policy = d3rlpy.load_learnable(str(checkpoint_path))
            log.debug("rl_policy.d3rlpy_loaded", path=str(checkpoint_path))
            return policy
        except ImportError:
            raise ImportError(
                "d3rlpy not installed. Run: uv add d3rlpy gymnasium\n"
                "Then retrain the policy: intraday rl train"
            )
        except Exception as exc:
            log.error(
                "rl_policy.load_error",
                path=str(checkpoint_path),
                error=str(exc),
            )
            raise


__all__ = ["RLExecutionPolicy"]
