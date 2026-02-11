"""Tests for the Mastermind attack strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.multi_turn.mastermind import (
    MastermindStrategy,
)


class TestMastermindStrategy:
    def test_metadata(self):
        assert MastermindStrategy.metadata.name == "mastermind"
        assert MastermindStrategy.metadata.category == AttackCategory.MULTI_TURN
        assert MastermindStrategy.metadata.supports_multi_turn is True

    @pytest.mark.asyncio
    async def test_generate_is_template_based(self, mock_provider_factory):
        provider = mock_provider_factory(["unused"])
        strategy = MastermindStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={},
        )
        # generate() is template-based, no LLM calls
        assert len(provider.calls) == 0
        assert "Test objective" in result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_refine_two_call(self, mock_provider_factory):
        # Mastermind refine: think + attack = 2 calls
        provider = mock_provider_factory(["Analysis", "New prompt"])
        strategy = MastermindStrategy(provider=provider)
        result = await strategy.refine(
            objective="Test objective",
            previous_prompt="Old prompt",
            target_response="Refused",
            judge_feedback="Verdict: refused",
            params={"attacker_model": "mock-model"},
        )
        assert result.prompt == "New prompt"
        assert result.reasoning == "Analysis"
        assert len(provider.calls) == 2


class TestMastermindSingleCallMode:
    """Single-call mode (separate_reasoning=False)."""

    @pytest.mark.asyncio
    async def test_refine_single_call(self, mock_provider_factory):
        provider = mock_provider_factory(["Single-call refined prompt"])
        strategy = MastermindStrategy(provider=provider)
        result = await strategy.refine(
            objective="Test objective",
            previous_prompt="Old prompt",
            target_response="Refused",
            judge_feedback="Verdict: refused",
            params={
                "attacker_model": "mock-model",
                "separate_reasoning": False,
            },
        )
        assert result.prompt == "Single-call refined prompt"
        assert result.reasoning is None
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_single_call_strips_think_tags(
        self,
        mock_provider_factory,
    ):
        provider = mock_provider_factory(["<think>reasoning</think>Clean refined prompt"])
        strategy = MastermindStrategy(provider=provider)
        result = await strategy.refine(
            objective="Test",
            previous_prompt="Old",
            target_response="No",
            judge_feedback="refused",
            params={
                "attacker_model": "mock-model",
                "separate_reasoning": False,
            },
        )
        assert result.prompt == "Clean refined prompt"
        assert "<think>" not in result.prompt

    @pytest.mark.asyncio
    async def test_generate_unaffected_by_toggle(
        self,
        mock_provider_factory,
    ):
        """generate() is template-based so toggle doesn't change behavior."""
        provider = mock_provider_factory(["unused"])
        strategy = MastermindStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test",
            context={"attacker_model": "mock-model"},
            params={"separate_reasoning": False},
        )
        # Still template-based, no LLM calls
        assert len(provider.calls) == 0
        assert result.token_usage == 0
