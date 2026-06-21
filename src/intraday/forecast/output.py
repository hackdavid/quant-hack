"""Forecast output schema — a single probabilistic prediction over 11 vol-normalized bins."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from pydantic import BaseModel, field_validator

# 10 edges → 11 bins (index 0 = extreme down, index 10 = extreme up)
# bins represent moves of: <-3, -3..-2, -2..-1, -1..-0.5, -0.5..-0.2,
#                          -0.2..+0.2, +0.2..+0.5, +0.5..1, 1..2, 2..3, >3  sigma
BIN_EDGES: list[float] = [-3.0, -2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0, 3.0]
N_BINS: int = 11

# Representative bin centres (used for E[move])
_BIN_CENTRES: list[float] = [
    -4.0,   # < -3σ
    -2.5,   # -3 .. -2
    -1.5,   # -2 .. -1
    -0.75,  # -1 .. -0.5
    -0.35,  # -0.5 .. -0.2
     0.0,   # -0.2 .. +0.2  (roughly flat)
     0.35,  # +0.2 .. +0.5
     0.75,  # +0.5 .. +1
     1.5,   # +1 .. +2
     2.5,   # +2 .. +3
     4.0,   # > +3σ
]


def _softmax(logits: Sequence[float]) -> list[float]:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    probs = (exp / exp.sum()).tolist()
    return probs


class ForecastOutput(BaseModel):
    """Single probabilistic forecast from the Kronos+TCN ensemble."""

    ts_ms: int
    horizon_minutes: int  # 5 | 15 | 60

    # Probability distribution over 11 vol-normalised bins
    p_bins: list[float]  # len == N_BINS, sums to 1.0

    # Convenience scalars derived from p_bins
    p_up_05sigma: float    # P(move > +0.5σ)
    p_down_05sigma: float  # P(move < -0.5σ)
    expected_move_sigma: float

    # 1 - entropy(p_bins) / log(N_BINS) ∈ [0, 1]; 0 = max uncertainty
    confidence: float

    # Meta-label layer
    meta_act: bool        # whether to act on this forecast
    meta_p_correct: float # calibrated P(direction is correct)

    model_version: str
    inference_ms: float

    @field_validator("p_bins")
    @classmethod
    def _validate_p_bins(cls, v: list[float]) -> list[float]:
        if len(v) != N_BINS:
            raise ValueError(f"p_bins must have {N_BINS} elements, got {len(v)}")
        total = sum(v)
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"p_bins must sum to 1.0, got {total:.6f}")
        return v

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_logits(
        cls,
        logits: Sequence[float],
        *,
        ts_ms: int,
        horizon_minutes: int,
        meta_act: bool,
        meta_p_correct: float,
        model_version: str,
        inference_ms: float,
    ) -> "ForecastOutput":
        """Build a ForecastOutput from raw model logits (length 11).

        Applies softmax, then computes all convenience scalars.
        """
        p_bins = _softmax(logits)

        # P(move > +0.5σ) = bins 7..10 (edges: 0.5, 1, 2, 3, +inf)
        p_up_05sigma = sum(p_bins[7:])

        # P(move < -0.5σ) = bins 0..3 (edges: -inf, -3, -2, -1, -0.5)
        p_down_05sigma = sum(p_bins[:4])

        # Expected move: weighted sum of bin centres
        expected_move_sigma = float(
            sum(p * c for p, c in zip(p_bins, _BIN_CENTRES))
        )

        # Confidence: normalised information gain vs uniform
        entropy = -sum(p * math.log(p) for p in p_bins if p > 1e-12)
        max_entropy = math.log(N_BINS)  # = log(11)
        confidence = float(1.0 - entropy / max_entropy)

        return cls(
            ts_ms=ts_ms,
            horizon_minutes=horizon_minutes,
            p_bins=p_bins,
            p_up_05sigma=p_up_05sigma,
            p_down_05sigma=p_down_05sigma,
            expected_move_sigma=expected_move_sigma,
            confidence=confidence,
            meta_act=meta_act,
            meta_p_correct=meta_p_correct,
            model_version=model_version,
            inference_ms=inference_ms,
        )
