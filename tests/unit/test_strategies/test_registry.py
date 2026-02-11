"""Tests for the StrategyRegistry."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.base import (
    AttackResult,
    BaseAttackStrategy,
    StrategyMetadata,
)
from adversarial_framework.strategies.registry import StrategyRegistry


class TestStrategyRegistry:
    def setup_method(self):
        """Save registry state before each test to avoid pollution."""
        self._original = dict(StrategyRegistry._strategies)

    def teardown_method(self):
        """Restore original registry after each test."""
        StrategyRegistry._strategies = self._original

    def test_register_and_get(self):
        @StrategyRegistry.register
        class DummyStrategy(BaseAttackStrategy):
            metadata = StrategyMetadata(
                name="dummy_test",
                display_name="Dummy",
                category=AttackCategory.PROMPT_LEVEL,
                description="Test strategy",
                estimated_asr="0%",
            )

            async def generate(self, objective, context, params):
                return AttackResult(prompt="test", strategy_metadata={})

            async def refine(
                self, objective, previous_prompt, target_response, judge_feedback, params
            ):
                return AttackResult(prompt="refined", strategy_metadata={})

        assert StrategyRegistry.get("dummy_test") is DummyStrategy

    def test_get_nonexistent_raises_key_error(self):
        with pytest.raises(KeyError, match="nonexistent_strategy_xyz"):
            StrategyRegistry.get("nonexistent_strategy_xyz")

    def test_list_all_returns_list(self):
        all_strategies = StrategyRegistry.list_all()
        assert isinstance(all_strategies, list)

    def test_list_names(self):
        names = StrategyRegistry.list_names()
        assert isinstance(names, list)

    def test_register_preserves_class(self):
        @StrategyRegistry.register
        class AnotherDummy(BaseAttackStrategy):
            metadata = StrategyMetadata(
                name="another_dummy",
                display_name="Another Dummy",
                category=AttackCategory.OPTIMIZATION,
                description="Another test",
                estimated_asr="0%",
            )

            async def generate(self, objective, context, params):
                return AttackResult(prompt="test", strategy_metadata={})

            async def refine(
                self, objective, previous_prompt, target_response, judge_feedback, params
            ):
                return AttackResult(prompt="refined", strategy_metadata={})

        assert AnotherDummy.metadata.name == "another_dummy"

    def test_list_by_category(self):
        @StrategyRegistry.register
        class PromptDummy(BaseAttackStrategy):
            metadata = StrategyMetadata(
                name="prompt_dummy",
                display_name="Prompt Dummy",
                category=AttackCategory.PROMPT_LEVEL,
                description="Test",
                estimated_asr="0%",
            )

            async def generate(self, objective, context, params):
                return AttackResult(prompt="test", strategy_metadata={})

            async def refine(
                self, objective, previous_prompt, target_response, judge_feedback, params
            ):
                return AttackResult(prompt="refined", strategy_metadata={})

        prompt_strategies = StrategyRegistry.list_by_category(AttackCategory.PROMPT_LEVEL)
        names = [s.name for s in prompt_strategies]
        assert "prompt_dummy" in names

    def test_duplicate_registration_raises(self):
        @StrategyRegistry.register
        class DupA(BaseAttackStrategy):
            metadata = StrategyMetadata(
                name="dup_test",
                display_name="Dup A",
                category=AttackCategory.PROMPT_LEVEL,
                description="Test",
                estimated_asr="0%",
            )

            async def generate(self, objective, context, params):
                return AttackResult(prompt="test", strategy_metadata={})

            async def refine(
                self, objective, previous_prompt, target_response, judge_feedback, params
            ):
                return AttackResult(prompt="refined", strategy_metadata={})

        with pytest.raises(ValueError, match="already registered"):

            @StrategyRegistry.register
            class DupB(BaseAttackStrategy):
                metadata = StrategyMetadata(
                    name="dup_test",
                    display_name="Dup B",
                    category=AttackCategory.PROMPT_LEVEL,
                    description="Test",
                    estimated_asr="0%",
                )

                async def generate(self, objective, context, params):
                    return AttackResult(prompt="test", strategy_metadata={})

                async def refine(
                    self, objective, previous_prompt, target_response, judge_feedback, params
                ):
                    return AttackResult(prompt="refined", strategy_metadata={})
