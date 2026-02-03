"""Decorator-based strategy registry with auto-discovery.

New strategies are registered by decorating their class with
``@StrategyRegistry.register``.  At startup, ``discover()`` walks all
strategy sub-packages and imports them, triggering registration.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adversarial_framework.strategies.base import BaseAttackStrategy, StrategyMetadata


class StrategyRegistry:
    """Singleton registry for attack strategies."""

    _strategies: dict[str, type[BaseAttackStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: type[BaseAttackStrategy]) -> type[BaseAttackStrategy]:
        """Class decorator to register a strategy.

        Usage::

            @StrategyRegistry.register
            class PAIRStrategy(BaseAttackStrategy):
                metadata = StrategyMetadata(name="pair", ...)
        """
        name = strategy_class.metadata.name
        if name in cls._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")
        cls._strategies[name] = strategy_class
        return strategy_class

    @classmethod
    def get(cls, name: str) -> type[BaseAttackStrategy]:
        """Retrieve a registered strategy class by name."""
        if name not in cls._strategies:
            available = sorted(cls._strategies.keys())
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        return cls._strategies[name]

    @classmethod
    def list_all(cls) -> list[StrategyMetadata]:
        """Return metadata for every registered strategy."""
        return [s.metadata for s in cls._strategies.values()]

    @classmethod
    def list_by_category(cls, category: str) -> list[StrategyMetadata]:
        """Return metadata filtered by ``AttackCategory``."""
        return [
            s.metadata for s in cls._strategies.values() if s.metadata.category == category
        ]

    @classmethod
    def list_names(cls) -> list[str]:
        """Return all registered strategy names."""
        return sorted(cls._strategies.keys())

    @classmethod
    def discover(cls) -> None:
        """Auto-discover strategies by importing every module under ``strategies/``."""
        import adversarial_framework.strategies as pkg

        for _importer, module_name, _is_pkg in pkgutil.walk_packages(
            pkg.__path__, prefix=f"{pkg.__name__}."
        ):
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass  # Skip optional heavy deps (e.g. torch for GCG)
