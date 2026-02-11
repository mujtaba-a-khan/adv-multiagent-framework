"""Tests for RAG Pipeline Poisoning advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.rag_poisoning import (
    _POISON_TEMPLATES,
    RAGPoisoningStrategy,
)
from adversarial_framework.strategies.base import AttackResult

_NUM_TEMPLATES = len(_POISON_TEMPLATES)


class TestRAGPoisoningMetadata:
    def test_metadata_name(self):
        assert RAGPoisoningStrategy.metadata.name == "rag_poisoning"

    def test_metadata_category(self):
        assert RAGPoisoningStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert RAGPoisoningStrategy.metadata.display_name == "RAG Pipeline Poisoning"


class TestRAGPoisoningGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_template(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["template_index"] == 0
        assert result.strategy_metadata["attack_type"] == "context_poisoning"
        assert result.strategy_metadata["iteration"] == 0
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_specific_template(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 1},
        )
        assert result.strategy_metadata["template_index"] == 1

    @pytest.mark.asyncio
    async def test_generate_template_wraps(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": _NUM_TEMPLATES + 1},
        )
        expected_idx = (_NUM_TEMPLATES + 1) % _NUM_TEMPLATES
        assert result.strategy_metadata["template_index"] == expected_idx

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "RAG poisoning" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestRAGPoisoningRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = RAGPoisoningStrategy()
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
        strategy = RAGPoisoningStrategy()
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
        strategy = RAGPoisoningStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": _NUM_TEMPLATES - 1},
        )
        assert result.strategy_metadata["template_index"] == 0

    @pytest.mark.asyncio
    async def test_refine_has_refinement_flag(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": 0},
        )
        assert result.strategy_metadata.get("refinement") is True

    @pytest.mark.asyncio
    async def test_refine_preserves_attack_type(self):
        strategy = RAGPoisoningStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"template_index": 0},
        )
        assert result.strategy_metadata["attack_type"] == "context_poisoning"


class TestRAGPoisoningCycling:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Verify cycling through all templates wraps around."""
        strategy = RAGPoisoningStrategy()
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


class TestRAGPoisoningValidateParams:
    def test_validate_params_empty(self):
        strategy = RAGPoisoningStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["template_index"] == 0

    def test_validate_params_override(self):
        strategy = RAGPoisoningStrategy()
        result = strategy.validate_params({"template_index": 2})
        assert result["template_index"] == 2
