"""Fractional Kelly + CVaR cap position sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from intraday.aggregator.decision import Decision

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KELLY_FRACTION: float = 0.25
# Empirical CVaR multiplier for fat-tailed crypto at the 5% tail
CVAR_MULTIPLIER_5PCT: float = 2.5


class SizingEngine:
    """Fractional Kelly + CVaR-cap position sizing.

    Formula (Phase 6 section 7):
    ::

        edge     = expected_edge_bps / 10_000
        variance = (vol_30m_bps / 10_000) ** 2
        kelly_f  = edge / variance
        target_f = kelly_fraction * kelly_f * confidence
        usd      = target_f * account_equity_usd
        usd      = clamp(usd, -max_position_usd, +max_position_usd)
        usd      = usd * risk_multiplier

        CVaR cap: if |usd| * vol_30m_bps/10_000 * CVAR_MULTIPLIER > cvar_cap_usd
            → scale down to cvar_cap_usd / (vol_30m_bps/10_000 * CVAR_MULTIPLIER)
    """

    def __init__(
        self,
        *,
        kelly_fraction: float = KELLY_FRACTION,
        cvar_cap_usd: float = 50.0,
        max_position_usd: float = 200.0,
    ) -> None:
        self._kelly_fraction = kelly_fraction
        self._cvar_cap_usd = cvar_cap_usd
        self._max_position_usd = max_position_usd
        log.debug(
            "sizing_engine_init",
            kelly_fraction=kelly_fraction,
            cvar_cap_usd=cvar_cap_usd,
            max_position_usd=max_position_usd,
        )

    def size_usd(
        self,
        decision: "Decision",
        *,
        expected_edge_bps: float,
        vol_30m_bps: float,
        risk_multiplier: float,
        account_equity_usd: float,
    ) -> float:
        """Return signed USD notional (+ = long, - = short, 0 = flat)."""
        if decision.side == "flat":
            return 0.0

        # Protect against division-by-zero
        if vol_30m_bps <= 0.0 or account_equity_usd <= 0.0:
            log.warning(
                "sizing.zero_output",
                reason="zero_vol_or_equity",
                vol_30m_bps=vol_30m_bps,
                account_equity_usd=account_equity_usd,
            )
            return 0.0

        edge = expected_edge_bps / 10_000.0
        vol_frac = vol_30m_bps / 10_000.0
        variance = vol_frac ** 2

        if variance <= 0.0 or edge <= 0.0:
            # No positive edge → flat
            return 0.0

        kelly_f = edge / variance
        target_f = self._kelly_fraction * kelly_f * decision.confidence

        # Raw USD notional (unsigned)
        usd_raw = target_f * account_equity_usd

        # Hard clamp before CVaR
        usd_clamped = min(usd_raw, self._max_position_usd)

        # CVaR tail-risk cap
        cvar_loss_estimate = usd_clamped * vol_frac * CVAR_MULTIPLIER_5PCT
        if cvar_loss_estimate > self._cvar_cap_usd:
            usd_clamped = self._cvar_cap_usd / (vol_frac * CVAR_MULTIPLIER_5PCT)

        # Apply risk agent multiplier
        usd_final = usd_clamped * risk_multiplier

        # Direction
        signed_usd = usd_final if decision.side == "long" else -usd_final

        log.debug(
            "sizing.computed",
            side=decision.side,
            edge_bps=round(expected_edge_bps, 4),
            vol_30m_bps=round(vol_30m_bps, 4),
            kelly_f=round(kelly_f, 6),
            target_f=round(target_f, 6),
            usd_raw=round(usd_raw, 2),
            usd_final=round(usd_final, 2),
            signed_usd=round(signed_usd, 2),
            risk_multiplier=risk_multiplier,
        )

        return signed_usd


__all__ = ["SizingEngine", "KELLY_FRACTION", "CVAR_MULTIPLIER_5PCT"]
