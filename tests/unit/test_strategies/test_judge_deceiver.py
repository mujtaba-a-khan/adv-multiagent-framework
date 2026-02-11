"""Tests for JudgeDeceiver (Evaluation Evasion) advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.judge_deceiver import (
    _DECEIVER_TEMPLATES,
    JudgeDeceiverStrategy,
)
from adversarial_framework.strategies.base import AttackResult

_NUM_TEMPLATES = len(_DECEIVER_TEMPLATES)


class TestJudgeDeceiverMetadata:
    def test_metadata_name(self):
        assert JudgeDeceiverStrategy.metadata.name == "judge_deceiver"

    def test_metadata_category(self):
        assert JudgeDeceiverStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        expected = "JudgeDeceiver (Evaluation Evasion)"
        assert JudgeDeceiverStrategy.metadata.display_name == expected


class TestJudgeDeceiverGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_template(self):
        strategy = JudgeDeceiverStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["template_index"] == 0
        assert result.strategy_metadata["attack_target"] == "judge"
        assert result.strategy_metadata["iteration"] == 0
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_specific_template(self):
        strategy = JudgeDeceiverStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 2},
        )
        assert result.strategy_metadata["template_index"] == 2

    @pytest.mark.asyncio
    async def test_generate_template_index_wraps(self):
        strategy = JudgeDeceiverStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": _NUM_TEMPLATES + 1},
        )
        expected_idx = (_NUM_TEMPLATES + 1) % _NUM_TEMPLATES
        assert result.strategy_metadata["template_index"] == expected_idx

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = JudgeDeceiverStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "JudgeDeceiver" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = JudgeDeceiverStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestJudgeDeceiverRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = JudgeDeceiverStrategy()
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
        strategy = JudgeDeceiverStrategy()
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
        strategy = JudgeDeceiverStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": _NUM_TEMPLATES - 1},
        )
        assert result.strategy_metadata["template_index"] == 0

    @pytest.mark.asyncio
    async def test_refine_attack_target_is_judge(self):
        strategy = JudgeDeceiverStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": 0},
        )
        assert result.strategy_metadata["attack_target"] == "judge"


class TestJudgeDeceiverCycling:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Verify cycling through all templates wraps around."""
        strategy = JudgeDeceiverStrategy()
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


class TestJudgeDeceiverValidateParams:
    def test_validate_params_empty(self):
        strategy = JudgeDeceiverStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["template_index"] == 0

    def test_validate_params_override(self):
        strategy = JudgeDeceiverStrategy()
        result = strategy.validate_params({"template_index": 3})
        assert result["template_index"] == 3
