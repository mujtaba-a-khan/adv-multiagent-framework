"""Tests for Invisible PDF Injection advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.pdf_injection import (
    _INJECTION_MODES,
    PDFInjectionStrategy,
)
from adversarial_framework.strategies.base import AttackResult

_MODE_KEYS = list(_INJECTION_MODES.keys())
_NUM_MODES = len(_MODE_KEYS)


class TestPDFInjectionMetadata:
    def test_metadata_name(self):
        assert PDFInjectionStrategy.metadata.name == "pdf_injection"

    def test_metadata_category(self):
        assert PDFInjectionStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert PDFInjectionStrategy.metadata.display_name == "Invisible PDF Injection"


class TestPDFInjectionGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_metadata_mode(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["injection_mode"] == "metadata"
        assert result.strategy_metadata["iteration"] == 0
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_hidden_text_mode(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"injection_mode": "hidden_text"},
        )
        assert result.strategy_metadata["injection_mode"] == "hidden_text"
        assert "Hidden Text Layer" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_annotation_mode(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"injection_mode": "annotation"},
        )
        assert result.strategy_metadata["injection_mode"] == "annotation"
        assert "Annotation" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_invalid_mode_defaults(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"injection_mode": "invalid"},
        )
        assert result.strategy_metadata["injection_mode"] == "metadata"

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "PDF injection" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestPDFInjectionRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"injection_mode": "metadata"},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt

    @pytest.mark.asyncio
    async def test_refine_cycles_metadata_to_hidden(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"injection_mode": "metadata"},
        )
        assert result.strategy_metadata["injection_mode"] == "hidden_text"

    @pytest.mark.asyncio
    async def test_refine_wraps_around(self):
        strategy = PDFInjectionStrategy()
        last_mode = _MODE_KEYS[-1]
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"injection_mode": last_mode},
        )
        assert result.strategy_metadata["injection_mode"] == _MODE_KEYS[0]

    @pytest.mark.asyncio
    async def test_refine_invalid_mode_resets(self):
        strategy = PDFInjectionStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"injection_mode": "nonexistent"},
        )
        # ValueError caught, idx=0, next=(0+1)%3=1
        assert result.strategy_metadata["injection_mode"] == _MODE_KEYS[1]


class TestPDFInjectionCycling:
    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Verify cycling through all injection modes wraps around."""
        strategy = PDFInjectionStrategy()
        mode = _MODE_KEYS[0]
        for i in range(_NUM_MODES):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={"injection_mode": mode},
            )
            expected = _MODE_KEYS[(i + 1) % _NUM_MODES]
            assert result.strategy_metadata["injection_mode"] == expected
            mode = expected


class TestPDFInjectionValidateParams:
    def test_validate_params_empty(self):
        strategy = PDFInjectionStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["injection_mode"] == "metadata"

    def test_validate_params_override(self):
        strategy = PDFInjectionStrategy()
        result = strategy.validate_params({"injection_mode": "annotation"})
        assert result["injection_mode"] == "annotation"
