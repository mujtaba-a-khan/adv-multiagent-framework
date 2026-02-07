"""Tests for the PAIR attack strategy.

PAIR now uses a two-call pattern: think (reasoning) + attack (clean prompt).
Each generate/refine call makes 2 LLM calls via generate_with_reasoning().
"""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.optimization.pair import PAIRStrategy


class TestPAIRStrategy:
    def test_metadata(self):
        assert PAIRStrategy.metadata.name == "pair"
        assert PAIRStrategy.metadata.category == AttackCategory.OPTIMIZATION
        assert PAIRStrategy.metadata.supports_multi_turn is False

    @pytest.mark.asyncio
    async def test_generate(self, mock_provider_factory):
        # PAIR makes 2 calls: think + attack
        provider = mock_provider_factory(["Think reasoning", "Clean attack prompt"])
        strategy = PAIRStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={},
        )
        assert result.prompt == "Clean attack prompt"
        assert result.reasoning == "Think reasoning"
        # 80 tokens per call * 2 = 160
        assert result.token_usage == 160
        assert len(provider.calls) == 2
        # Verify think call includes objective
        think_system = provider.calls[0]["messages"][0]["content"]
        assert "Test objective" in think_system

    @pytest.mark.asyncio
    async def test_refine(self, mock_provider_factory):
        provider = mock_provider_factory(["Refine reasoning", "Refined prompt"])
        strategy = PAIRStrategy(provider=provider)
        result = await strategy.refine(
            objective="Test objective",
            previous_prompt="Old prompt",
            target_response="I can't help with that.",
            judge_feedback="Verdict: refused, Confidence: 0.9",
            params={"attacker_model": "mock-model"},
        )
        assert result.prompt == "Refined prompt"
        assert result.reasoning == "Refine reasoning"
        assert len(provider.calls) == 2
        # Verify think call includes previous attempt info
        think_user = provider.calls[0]["messages"][1]["content"]
        assert "Old prompt" in think_user
        assert "I can't help with that." in think_user

    @pytest.mark.asyncio
    async def test_generate_returns_reasoning_field(self, mock_provider_factory):
        provider = mock_provider_factory(["My strategic analysis", "Final prompt"])
        strategy = PAIRStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={},
        )
        assert result.reasoning is not None
        assert result.reasoning == "My strategic analysis"

    @pytest.mark.asyncio
    async def test_validate_params_defaults(self, mock_provider_factory):
        provider = mock_provider_factory(["r", "p"])
        strategy = PAIRStrategy(provider=provider)
        params = strategy.validate_params({})
        assert "temperature" in params
        assert params["temperature"] == 1.0


class TestPAIRSingleCallMode:
    """Single-call mode (separate_reasoning=False)."""

    @pytest.mark.asyncio
    async def test_generate_single_call(self, mock_provider_factory):
        provider = mock_provider_factory(["Single call prompt"])
        strategy = PAIRStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={"separate_reasoning": False},
        )
        assert result.prompt == "Single call prompt"
        assert result.reasoning is None
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_refine_single_call(self, mock_provider_factory):
        provider = mock_provider_factory(["Refined single prompt"])
        strategy = PAIRStrategy(provider=provider)
        result = await strategy.refine(
            objective="Test objective",
            previous_prompt="Old prompt",
            target_response="I can't help.",
            judge_feedback="Verdict: refused",
            params={
                "attacker_model": "mock-model",
                "separate_reasoning": False,
            },
        )
        assert result.prompt == "Refined single prompt"
        assert result.reasoning is None
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_single_call_strips_think_tags(
        self, mock_provider_factory
    ):
        provider = mock_provider_factory(
            ["<think>reasoning</think>Clean prompt"]
        )
        strategy = PAIRStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test",
            context={"attacker_model": "mock-model"},
            params={"separate_reasoning": False},
        )
        assert result.prompt == "Clean prompt"
        assert "<think>" not in result.prompt
