"""Tests for System Prompt Extraction advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.prompt_extraction import (
    _EXTRACTION_TEMPLATES,
    PromptExtractionStrategy,
)
from adversarial_framework.strategies.base import AttackResult

_NUM_TEMPLATES = len(_EXTRACTION_TEMPLATES)


class TestPromptExtractionMetadata:
    def test_metadata_name(self):
        assert PromptExtractionStrategy.metadata.name == "prompt_extraction"

    def test_metadata_category(self):
        assert PromptExtractionStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert PromptExtractionStrategy.metadata.display_name == "System Prompt Extraction"


class TestPromptExtractionGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_explicit_index(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"template_index": 0},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["template_index"] == 0
        assert result.strategy_metadata["iteration"] == 0
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_random_when_no_index(self):
        """When template_index is None, a random index is chosen."""
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        idx = result.strategy_metadata["template_index"]
        assert 0 <= idx < _NUM_TEMPLATES

    @pytest.mark.asyncio
    async def test_generate_wraps_large_index(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": _NUM_TEMPLATES + 2},
        )
        expected_idx = (_NUM_TEMPLATES + 2) % _NUM_TEMPLATES
        assert result.strategy_metadata["template_index"] == expected_idx

    @pytest.mark.asyncio
    async def test_generate_has_technique_name(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 1},
        )
        assert result.strategy_metadata["technique"] == "translation_trick"

    @pytest.mark.asyncio
    async def test_generate_tracks_used_indices(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 3},
        )
        used = result.strategy_metadata["_used_indices"]
        assert isinstance(used, list)
        assert 3 in used

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 0},
        )
        assert result.reasoning is not None
        assert "Prompt extraction" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 0},
        )
        assert result.token_usage == 0


class TestPromptExtractionRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "template_index": 0,
                "_used_indices": [0],
            },
        )
        assert isinstance(result, AttackResult)
        assert result.prompt

    @pytest.mark.asyncio
    async def test_refine_picks_next_unused(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "template_index": 0,
                "_used_indices": [0],
            },
        )
        assert result.strategy_metadata["template_index"] == 1
        assert 0 in result.strategy_metadata["_used_indices"]
        assert 1 in result.strategy_metadata["_used_indices"]

    @pytest.mark.asyncio
    async def test_refine_skips_used_indices(self):
        strategy = PromptExtractionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "template_index": 0,
                "_used_indices": [0, 1, 2],
            },
        )
        # Should pick 3 (first unused)
        assert result.strategy_metadata["template_index"] == 3

    @pytest.mark.asyncio
    async def test_refine_exhausted_resets(self):
        strategy = PromptExtractionStrategy()
        all_indices = list(range(_NUM_TEMPLATES))
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "template_index": _NUM_TEMPLATES - 1,
                "_used_indices": all_indices,
            },
        )
        # All exhausted -> reset -> picks first (0)
        assert result.strategy_metadata["template_index"] == 0


class TestPromptExtractionCycling:
    @pytest.mark.asyncio
    async def test_sequential_cycling(self):
        """Verify cycling through all templates without repeats."""
        strategy = PromptExtractionStrategy()
        used: list[int] = [0]
        for _ in range(_NUM_TEMPLATES - 1):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={
                    "template_index": used[-1],
                    "_used_indices": list(used),
                },
            )
            idx = result.strategy_metadata["template_index"]
            assert idx not in used
            used.append(idx)
        assert len(used) == _NUM_TEMPLATES


class TestPromptExtractionValidateParams:
    def test_validate_params_empty(self):
        strategy = PromptExtractionStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["template_index"] is None

    def test_validate_params_override(self):
        strategy = PromptExtractionStrategy()
        result = strategy.validate_params({"template_index": 4})
        assert result["template_index"] == 4
