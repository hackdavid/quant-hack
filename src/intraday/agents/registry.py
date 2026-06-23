"""Agent registry: maps agent name strings to agent classes.

Usage:
    from intraday.agents.registry import register, get_agent

    @register
    class MyAgent(Agent): ...

    agent = get_agent("my_agent")
"""

from typing import Any

import structlog

from intraday.agents.base import Agent

log = structlog.get_logger(__name__)

_REGISTRY: dict[str, type[Agent]] = {}


def register(cls: type[Agent]) -> type[Agent]:
    """Class decorator that registers an Agent subclass by its name attribute."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Agent class {cls.__qualname__} must define a non-empty 'name' attribute.")
    _REGISTRY[cls.name] = cls
    log.debug("agent_registered", agent=cls.name)
    return cls


def get_agent(name: str, **kwargs: Any) -> Agent:
    """Instantiate a registered agent by name, forwarding kwargs to __init__.

    Args:
        name: Agent name string (e.g. "orderflow", "regime", "risk", "stay_out").
        **kwargs: Keyword arguments forwarded to the agent constructor.

    Returns:
        Instantiated Agent subclass.

    Raises:
        KeyError: If no agent with the given name has been registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown agent '{name}'. Available agents: {available}"
        )
    cls = _REGISTRY[name]
    log.debug("agent_instantiate", agent=name, kwargs=list(kwargs.keys()))
    return cls(**kwargs)


def _auto_register() -> None:
    """Import all built-in agents so their @register decorators run."""
    # Import here to avoid circular imports; registration is a side-effect
    from intraday.agents.orderflow import OrderflowAgent  # noqa: F401
    from intraday.agents.regime import RegimeAgent  # noqa: F401
    from intraday.agents.risk import RiskAgent  # noqa: F401
    from intraday.agents.stay_out import StayOutDetector  # noqa: F401
    from intraday.agents.forecast import ForecastAgent  # noqa: F401

    # Register built-in agents if not already registered
    for cls in (OrderflowAgent, RegimeAgent, RiskAgent, StayOutDetector, ForecastAgent):
        if cls.name not in _REGISTRY:
            _REGISTRY[cls.name] = cls


# Auto-register on module import
_auto_register()
