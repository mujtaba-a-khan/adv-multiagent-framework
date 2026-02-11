"""Tests for Log-to-Leak (MCP Exploitation) advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.log_to_leak import (
    _LEAK_TEMPLATES,
    LogToLeakStrategy,
)
from adversarial_framework.strategies.base import AttackResult

_NUM_TEMPLATES = len(_LEAK_TEMPLATES)


class TestLogToLeakMetadata:
    def test_metadata_name(self):
        assert LogToLeakStrategy.metadata.name == "log_to_leak"

    def test_metadata_category(self):
        assert LogToLeakStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert LogToLeakStrategy.metadata.display_name == "Log-to-Leak (MCP Exploitation)"


class TestLogToLeakGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_template(self):
        strategy = LogToLeakStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["template_index"] == 0
        assert result.strategy_metadata["attack_type"] == "tool_exploitation"
        assert result.strategy_metadata["iteration"] == 0
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_specific_template(self):
        strategy = LogToLeakStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 2},
        )
        assert result.strategy_metadata["template_index"] == 2

    @pytest.mark.asyncio
    async def test_generate_template_wraps(self):
        strategy = LogToLeakStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": _NUM_TEMPLATES + 1},
        )
        expected_idx = (_NUM_TEMPLATES + 1) % _NUM_TEMPLATES
        assert result.strategy_metadata["template_index"] == expected_idx

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = LogToLeakStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "Log-to-Leak" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = LogToLeakStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestLogToLeakRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = LogToLeakStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": 0},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt

    @pytest.mark.asyncio
    async def test_refine_cycles_to_next(self):
        strategy = LogToLeakStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": 0},
        )
        assert result.strategy_metadata["template_index"] == 1

    @pytest.mark.asyncio
    async def test_refine_wraps_around(self):
        strategy = LogToLeakStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": _NUM_TEMPLATES - 1},
        )
        assert result.strategy_metadata["template_index"] == 0

    @pytest.mark.asyncio
    async def test_refine_preserves_attack_type(self):
        strategy = LogToLeakStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": 1},
        )
        assert result.strategy_metadata["attack_type"] == "tool_exploitation"


class TestLogToLeakCycling:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Verify cycling through all templates wraps around."""
        strategy = LogToLeakStrategy()
        idx = 0
        for i in range(_NUM_TEMPLATES):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={"template_index": idx},
            )
            expected = (i + 1) % _NUM_TEMPLATES
            assert result.strategy_metadata["template_index"] == expected
            idx = expected


class TestLogToLeakValidateParams:
    def test_validate_params_empty(self):
        strategy = LogToLeakStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["template_index"] == 0

    def test_validate_params_override(self):
        strategy = LogToLeakStrategy()
        result = strategy.validate_params({"template_index": 2})
        assert result["template_index"] == 2
