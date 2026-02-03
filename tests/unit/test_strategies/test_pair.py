"""Tests for the PAIR attack strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.strategies.optimization.pair import PAIRStrategy
from adversarial_framework.core.constants import AttackCategory


class TestPAIRStrategy:
    def test_metadata(self):
        assert PAIRStrategy.metadata.name == "pair"
        assert PAIRStrategy.metadata.category == AttackCategory.OPTIMIZATION
        assert PAIRStrategy.metadata.supports_multi_turn is False

    @pytest.mark.asyncio
    async def test_generate(self, mock_provider):
        strategy = PAIRStrategy(provider=mock_provider)
        result = await strategy.generate(
            objective="Test objective",
            context={"attacker_model": "mock-model"},
            params={},
        )
        assert result.prompt == "Mock response"
        assert result.token_usage == 80
        assert len(mock_provider.calls) == 1
        # Verify system prompt includes objective
        system_msg = mock_provider.calls[0]["messages"][0]["content"]
        assert "Test objective" in system_msg

    @pytest.mark.asyncio
    async def test_refine(self, mock_provider):
        strategy = PAIRStrategy(provider=mock_provider)
        result = await strategy.refine(
            objective="Test objective",
            previous_prompt="Old prompt",
            target_response="I can't help with that.",
            judge_feedback="Verdict: refused, Confidence: 0.9",
            params={"attacker_model": "mock-model"},
        )
        assert result.prompt == "Mock response"
        assert len(mock_provider.calls) == 1
        # Verify refine prompt includes previous attempt info
        user_msg = mock_provider.calls[0]["messages"][1]["content"]
        assert "Old prompt" in user_msg
        assert "I can't help with that." in user_msg

    @pytest.mark.asyncio
    async def test_validate_params_defaults(self, mock_provider):
        strategy = PAIRStrategy(provider=mock_provider)
        params = strategy.validate_params({})
        assert "temperature" in params
        assert params["temperature"] == 1.0
