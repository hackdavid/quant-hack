"""Strategy registry for named strategy lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intraday.sim.strategies.base import Strategy

_REGISTRY: dict[str, type["Strategy"]] = {}


def register(name: str):
    """Decorator to register a strategy class under a name."""
    def decorator(cls: type["Strategy"]) -> type["Strategy"]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_strategy(name: str) -> type["Strategy"]:
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"Strategy {name!r} not found. Available: {available}")
    return _REGISTRY[name]


__all__ = ["register", "get_strategy", "_REGISTRY"]
