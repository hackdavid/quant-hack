"""Base classes for all specialist agents."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class AgentOpinion(BaseModel):
    """Structured output from any agent."""

    agent: str
    ts_ms: int
    payload: dict[str, Any]
    confidence: float  # 0..1
    inference_ms: float


class Agent(ABC):
    """Abstract base for all specialist agents."""

    name: str

    @abstractmethod
    def predict(self, history_df) -> AgentOpinion:
        """Produce an opinion from a feature dict or DataFrame."""
        ...
