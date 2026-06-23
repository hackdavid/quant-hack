"""Specialist agents for the BTC intraday trading system.

Exported names:
    AgentOpinion      — pydantic model returned by every agent
    OrderflowAgent    — OFI + Hawkes + VPIN flow-bias agent (rule-based)
    RegimeAgent       — HMM + LightGBM regime classifier (learnable)
    RiskAgent         — Hard-cap risk gatekeeper (rule-based)
    StayOutDetector   — News-shock / extreme-event detector (rule-based)
    ForecastAgent     — Transformer-based direction forecast (learnable)
"""

from intraday.agents.base import Agent, AgentOpinion
from intraday.agents.orderflow import OrderflowAgent
from intraday.agents.regime import (
    REGIME_LABELS,
    VOL_REGIMES,
    N_HMM_STATES,
    HMM_FEATURES,
    RegimeAgent,
)
from intraday.agents.risk import RiskAgent, RiskConfig
from intraday.agents.stay_out import StayOutDetector
from intraday.agents.forecast import ForecastAgent

__all__ = [
    "Agent",
    "AgentOpinion",
    "OrderflowAgent",
    "RegimeAgent",
    "REGIME_LABELS",
    "VOL_REGIMES",
    "N_HMM_STATES",
    "HMM_FEATURES",
    "RiskAgent",
    "RiskConfig",
    "StayOutDetector",
    "ForecastAgent",
]
