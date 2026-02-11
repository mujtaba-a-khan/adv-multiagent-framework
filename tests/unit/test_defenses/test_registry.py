"""Tests for DefenseRegistry."""

from __future__ import annotations

import pytest

from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult
from adversarial_framework.defenses.registry import DefenseRegistry


class _DummyDefense(BaseDefense):
    """Minimal concrete defense for testing."""

    name = "dummy_test_defense"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)


class _DuplicateDefense(BaseDefense):
    """Defense with the same name to test duplicate registration."""

    name = "dummy_test_defense"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)


class TestDefenseRegistry:
    def setup_method(self) -> None:
        """Clear registry before each test."""
        self._backup = dict(DefenseRegistry._defenses)
        DefenseRegistry._defenses.clear()

    def teardown_method(self) -> None:
        """Restore registry after each test."""
        DefenseRegistry._defenses.clear()
        DefenseRegistry._defenses.update(self._backup)

    def test_register_and_get(self) -> None:
        DefenseRegistry.register(_DummyDefense)
        assert DefenseRegistry.get("dummy_test_defense") is _DummyDefense

    def test_register_returns_class(self) -> None:
        result = DefenseRegistry.register(_DummyDefense)
        assert result is _DummyDefense

    def test_duplicate_registration_raises(self) -> None:
        DefenseRegistry.register(_DummyDefense)
        with pytest.raises(ValueError, match="already registered"):
            DefenseRegistry.register(_DuplicateDefense)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            DefenseRegistry.get("nonexistent_defense")

    def test_list_all_empty(self) -> None:
        assert DefenseRegistry.list_all() == []

    def test_list_all_sorted(self) -> None:
        class _AlphaDefense(BaseDefense):
            name = "alpha"

            async def check_input(self, prompt: str) -> DefenseCheckResult:
                return DefenseCheckResult(blocked=False)

            async def check_output(self, response: str) -> DefenseCheckResult:
                return DefenseCheckResult(blocked=False)

        class _BetaDefense(BaseDefense):
            name = "beta"

            async def check_input(self, prompt: str) -> DefenseCheckResult:
                return DefenseCheckResult(blocked=False)

            async def check_output(self, response: str) -> DefenseCheckResult:
                return DefenseCheckResult(blocked=False)

        DefenseRegistry.register(_BetaDefense)
        DefenseRegistry.register(_AlphaDefense)
        assert DefenseRegistry.list_all() == ["alpha", "beta"]

    def test_list_names_alias(self) -> None:
        DefenseRegistry.register(_DummyDefense)
        assert DefenseRegistry.list_names() == DefenseRegistry.list_all()


class TestDefenseDiscovery:
    """Test discovery separately — module-level decorators only fire once
    per interpreter lifetime, so the registry must not be cleared first."""

    def test_discover_populates_registry(self) -> None:
        DefenseRegistry.discover()
        names = DefenseRegistry.list_all()
        # After discovery, some built-in defenses should be registered
        assert len(names) > 0
