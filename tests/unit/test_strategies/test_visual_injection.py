"""Tests for Visual/Multimodal Injection advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.visual_injection import (
    VisualInjectionStrategy,
    _text_to_ascii_art,
)
from adversarial_framework.strategies.base import AttackResult

_MODES = VisualInjectionStrategy._MODES
_NUM_MODES = len(_MODES)


class TestVisualInjectionMetadata:
    def test_metadata_name(self):
        assert VisualInjectionStrategy.metadata.name == "visual_injection"

    def test_metadata_category(self):
        assert VisualInjectionStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert VisualInjectionStrategy.metadata.display_name == "Visual/Multimodal Injection"


class TestVisualInjectionGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_ascii_art(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["visual_mode"] == "ascii_art"
        assert result.strategy_metadata["iteration"] == 0

    @pytest.mark.asyncio
    async def test_generate_image_desc(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"visual_mode": "image_desc"},
        )
        assert result.strategy_metadata["visual_mode"] == "image_desc"
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_visual_format(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"visual_mode": "visual_format"},
        )
        assert result.strategy_metadata["visual_mode"] == "visual_format"
        assert "PRIORITY INSTRUCTION" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_mind_map(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"visual_mode": "mind_map"},
        )
        assert result.strategy_metadata["visual_mode"] == "mind_map"
        assert "mind map" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_invalid_mode_defaults(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"visual_mode": "invalid"},
        )
        assert result.strategy_metadata["visual_mode"] == "ascii_art"

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "Visual injection" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestVisualInjectionRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"visual_mode": "ascii_art"},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt

    @pytest.mark.asyncio
    async def test_refine_cycles_ascii_to_image(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"visual_mode": "ascii_art"},
        )
        assert result.strategy_metadata["visual_mode"] == "image_desc"

    @pytest.mark.asyncio
    async def test_refine_wraps_around(self):
        strategy = VisualInjectionStrategy()
        last_mode = _MODES[-1]
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"visual_mode": last_mode},
        )
        assert result.strategy_metadata["visual_mode"] == _MODES[0]

    @pytest.mark.asyncio
    async def test_refine_invalid_mode_resets(self):
        strategy = VisualInjectionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"visual_mode": "nonexistent"},
        )
        # ValueError caught, idx=0, next=(0+1)%4=1
        assert result.strategy_metadata["visual_mode"] == _MODES[1]


class TestVisualInjectionCycling:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Verify cycling through all visual modes wraps around."""
        strategy = VisualInjectionStrategy()
        mode = _MODES[0]
        for i in range(_NUM_MODES):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={"visual_mode": mode},
            )
            expected = _MODES[(i + 1) % _NUM_MODES]
            assert result.strategy_metadata["visual_mode"] == expected
            mode = expected


class TestTextToAsciiArt:
    def test_simple_text(self):
        art = _text_to_ascii_art("AB")
        assert "[A]" in art
        assert "[B]" in art

    def test_space_handling(self):
        art = _text_to_ascii_art("A B")
        assert "[A]" in art
        assert "[B]" in art
        assert "   " in art

    def test_lowercase_uppercased(self):
        art = _text_to_ascii_art("abc")
        assert "[A]" in art
        assert "[B]" in art
        assert "[C]" in art

    def test_empty_string(self):
        art = _text_to_ascii_art("")
        assert art == ""


class TestVisualInjectionValidateParams:
    def test_validate_params_empty(self):
        strategy = VisualInjectionStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["visual_mode"] == "ascii_art"

    def test_validate_params_override(self):
        strategy = VisualInjectionStrategy()
        result = strategy.validate_params({"visual_mode": "mind_map"})
        assert result["visual_mode"] == "mind_map"
