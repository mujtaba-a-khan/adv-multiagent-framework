"""Tests for the TAP attack strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.optimization.tap import TAPStrategy


class TestTAPStrategy:
    def test_metadata(self):
        assert TAPStrategy.metadata.name == "tap"
        assert TAPStrategy.metadata.category == AttackCategory.OPTIMIZATION
        assert TAPStrategy.metadata.supports_multi_turn is False

    @pytest.mark.asyncio
    async def test_generate_two_call(self, mock_provider_factory):
        # TAP generate: think + attack = 2 calls
        provider = mock_provider_factory(["Think reasoning", "Attack prompt"])
        strategy = TAPStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={},
        )
        assert result.prompt == "Attack prompt"
        assert result.reasoning == "Think reasoning"
        assert len(provider.calls) == 2

    @pytest.mark.asyncio
    async def test_refine_two_call(self, mock_provider_factory):
        # TAP refine with 3 candidates: (think + attack + prune) * 3 = 9 calls
        responses = []
        for i in range(3):
            responses.extend(
                [
                    f"Branch {i} reasoning",
                    f"Branch {i} prompt",
                    "0.8",  # pruning score
                ]
            )
        provider = mock_provider_factory(responses)
        strategy = TAPStrategy(provider=provider)
        result = await strategy.refine(
            objective="Test objective",
            previous_prompt="Old prompt",
            target_response="Refused",
            judge_feedback="Verdict: refused",
            params={"attacker_model": "mock-model"},
        )
        assert result.prompt  # Should be best candidate
        assert result.reasoning is not None
        assert result.strategy_metadata["num_candidates_generated"] == 3


class TestTAPSingleCallMode:
    """Single-call mode (separate_reasoning=False)."""

    @pytest.mark.asyncio
    async def test_generate_single_call(self, mock_provider_factory):
        provider = mock_provider_factory(["Single prompt"])
        strategy = TAPStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={"separate_reasoning": False},
        )
        assert result.prompt == "Single prompt"
        assert result.reasoning is None
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_refine_single_call(self, mock_provider_factory):
        # Single-call refine: (branch + prune) * 3 = 6 calls
        responses = []
        for i in range(3):
            responses.extend([f"Branch {i} prompt", "0.7"])
        provider = mock_provider_factory(responses)
        strategy = TAPStrategy(provider=provider)
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
        assert result.prompt  # Best candidate
        assert result.reasoning is None
        assert result.strategy_metadata["num_candidates_generated"] == 3

    @pytest.mark.asyncio
    async def test_single_call_strips_think_tags(
        self,
        mock_provider_factory,
    ):
        provider = mock_provider_factory(["<think>cot</think>Clean prompt"])
        strategy = TAPStrategy(provider=provider)
        result = await strategy.generate(
            objective="Test",
            context={"attacker_model": "mock-model"},
            params={"separate_reasoning": False},
        )
        assert result.prompt == "Clean prompt"
        assert "<think>" not in result.prompt
