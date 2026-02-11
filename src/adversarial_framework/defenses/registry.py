"""Decorator-based defense registry, mirroring the strategy registry pattern.

New defenses are registered by decorating their class with
``@DefenseRegistry.register``.  At startup, ``discover()`` walks all
defense sub-packages and imports them, triggering registration.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adversarial_framework.defenses.base import BaseDefense


class DefenseRegistry:
    """Singleton registry for defense mechanisms."""

    _defenses: dict[str, type[BaseDefense]] = {}

    @classmethod
    def register(cls, defense_class: type[BaseDefense]) -> type[BaseDefense]:
        """Class decorator to register a defense.

        Usage::

            @DefenseRegistry.register
            class RuleBasedDefense(BaseDefense):
                name = "rule_based"
        """
        name = defense_class.name
        if name in cls._defenses:
            raise ValueError(f"Defense '{name}' is already registered")
        cls._defenses[name] = defense_class
        return defense_class

    @classmethod
    def get(cls, name: str) -> type[BaseDefense]:
        """Retrieve a registered defense class by name."""
        if name not in cls._defenses:
            available = sorted(cls._defenses.keys())
            raise KeyError(f"Defense '{name}' not found. Available: {available}")
        return cls._defenses[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """Return names of every registered defense."""
        return sorted(cls._defenses.keys())

    @classmethod
    def list_names(cls) -> list[str]:
        """Return all registered defense names."""
        return sorted(cls._defenses.keys())

    @classmethod
    def discover(cls) -> None:
        """Auto-discover defenses by importing every module under ``defenses/``."""
        import adversarial_framework.defenses as pkg

        for _importer, module_name, _is_pkg in pkgutil.walk_packages(
            pkg.__path__, prefix=f"{pkg.__name__}."
        ):
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)
