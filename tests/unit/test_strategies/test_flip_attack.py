"""Tests for FlipAttack (Character Reordering) advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.flip_attack import (
    _FLIP_MODES,
    FlipAttackStrategy,
)
from adversarial_framework.strategies.base import AttackResult


class TestFlipAttackMetadata:
    def test_metadata_name(self):
        assert FlipAttackStrategy.metadata.name == "flip_attack"

    def test_metadata_category(self):
        assert FlipAttackStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert FlipAttackStrategy.metadata.display_name == "FlipAttack (Character Reordering)"


class TestFlipAttackGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_word_reverse(self):
        strategy = FlipAttackStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["flip_mode"] == "word_reverse"
        assert result.strategy_metadata["iteration"] == 0
        # "test" reversed word-by-word -> "tset evitcejbo"
        assert "tset" in result.strategy_metadata["encoded_text"]

    @pytest.mark.asyncio
    async def test_generate_sentence_reverse(self):
        strategy = FlipAttackStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"flip_mode": "sentence_reverse"},
        )
        assert result.strategy_metadata["flip_mode"] == "sentence_reverse"
        # "test objective" sentence reversed -> "objective test"
        assert result.strategy_metadata["encoded_text"] == "objective test"

    @pytest.mark.asyncio
    async def test_generate_char_reverse(self):
        strategy = FlipAttackStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"flip_mode": "char_reverse"},
        )
        assert result.strategy_metadata["flip_mode"] == "char_reverse"
        assert result.strategy_metadata["encoded_text"] == "evitcejbo tset"

    @pytest.mark.asyncio
    async def test_generate_invalid_mode_defaults(self):
        strategy = FlipAttackStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"flip_mode": "invalid_mode"},
        )
        assert result.strategy_metadata["flip_mode"] == "word_reverse"

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = FlipAttackStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "FlipAttack" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = FlipAttackStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestFlipAttackRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = FlipAttackStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"flip_mode": "word_reverse"},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt

    @pytest.mark.asyncio
    async def test_refine_cycles_word_to_sentence(self):
        strategy = FlipAttackStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"flip_mode": "word_reverse"},
        )
        assert result.strategy_metadata["flip_mode"] == "sentence_reverse"

    @pytest.mark.asyncio
    async def test_refine_cycles_sentence_to_char(self):
        strategy = FlipAttackStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"flip_mode": "sentence_reverse"},
        )
        assert result.strategy_metadata["flip_mode"] == "char_reverse"

    @pytest.mark.asyncio
    async def test_refine_cycles_char_wraps_to_word(self):
        strategy = FlipAttackStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"flip_mode": "char_reverse"},
        )
        assert result.strategy_metadata["flip_mode"] == "word_reverse"


class TestFlipAttackCycling:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Verify cycling through all modes wraps around."""
        strategy = FlipAttackStrategy()
        mode = "word_reverse"
        for expected_next in ["sentence_reverse", "char_reverse", "word_reverse"]:
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={"flip_mode": mode},
            )
            assert result.strategy_metadata["flip_mode"] == expected_next
            mode = expected_next

    @pytest.mark.asyncio
    async def test_invalid_mode_in_refine_resets(self):
        strategy = FlipAttackStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"flip_mode": "nonexistent"},
        )
        # ValueError caught, idx=0, next=(0+1)%3=1 -> sentence_reverse
        assert result.strategy_metadata["flip_mode"] == _FLIP_MODES[1]


class TestFlipAttackValidateParams:
    def test_validate_params_empty(self):
        strategy = FlipAttackStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["flip_mode"] == "word_reverse"

    def test_validate_params_override(self):
        strategy = FlipAttackStrategy()
        result = strategy.validate_params({"flip_mode": "char_reverse"})
        assert result["flip_mode"] == "char_reverse"
