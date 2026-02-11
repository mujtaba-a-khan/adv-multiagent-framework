"""Tests for Payload Smuggling advanced strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.advanced.payload_smuggling import (
    _TECHNIQUE_NAMES,
    PayloadSmugglingStrategy,
)
from adversarial_framework.strategies.base import AttackResult

_NUM_TECHNIQUES = len(_TECHNIQUE_NAMES)


class TestPayloadSmugglingMetadata:
    def test_metadata_name(self):
        assert PayloadSmugglingStrategy.metadata.name == "payload_smuggling"

    def test_metadata_category(self):
        assert PayloadSmugglingStrategy.metadata.category == AttackCategory.ADVANCED

    def test_metadata_display_name(self):
        assert PayloadSmugglingStrategy.metadata.display_name == "Payload Smuggling"


class TestPayloadSmugglingGenerate:
    @pytest.mark.asyncio
    async def test_generate_default_code_comment(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.strategy_metadata["technique"] == "code_comment"
        assert result.strategy_metadata["iteration"] == 0
        assert "test objective" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_specific_technique(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"technique": "json_field"},
        )
        assert result.strategy_metadata["technique"] == "json_field"

    @pytest.mark.asyncio
    async def test_generate_markdown_hidden(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"technique": "markdown_hidden"},
        )
        assert result.strategy_metadata["technique"] == "markdown_hidden"
        assert "<!--" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_xml_cdata(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"technique": "xml_cdata"},
        )
        assert result.strategy_metadata["technique"] == "xml_cdata"
        assert "CDATA" in result.prompt

    @pytest.mark.asyncio
    async def test_generate_invalid_technique_defaults(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"technique": "invalid"},
        )
        assert result.strategy_metadata["technique"] == "code_comment"

    @pytest.mark.asyncio
    async def test_generate_tracks_used_techniques(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        used = result.strategy_metadata["_used_techniques"]
        assert isinstance(used, list)
        assert len(used) == 1

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "Payload smuggling" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_zero_token_usage(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.token_usage == 0


class TestPayloadSmugglingRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "technique": "code_comment",
                "_used_techniques": ["code_comment"],
            },
        )
        assert isinstance(result, AttackResult)
        assert result.prompt

    @pytest.mark.asyncio
    async def test_refine_picks_next_unused(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "technique": "code_comment",
                "_used_techniques": ["code_comment"],
            },
        )
        assert result.strategy_metadata["technique"] == "json_field"
        used = result.strategy_metadata["_used_techniques"]
        assert "code_comment" in used
        assert "json_field" in used

    @pytest.mark.asyncio
    async def test_refine_exhausted_resets(self):
        strategy = PayloadSmugglingStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={
                "technique": _TECHNIQUE_NAMES[-1],
                "_used_techniques": list(_TECHNIQUE_NAMES),
            },
        )
        # All exhausted -> reset -> picks first
        assert result.strategy_metadata["technique"] == _TECHNIQUE_NAMES[0]

    @pytest.mark.asyncio
    async def test_refine_fallback_no_used(self):
        """When _used_techniques is missing, fallback to technique param."""
        strategy = PayloadSmugglingStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"technique": "code_comment"},
        )
        # Should treat code_comment as used; pick next available
        assert result.strategy_metadata["technique"] != "code_comment"


class TestPayloadSmugglingCycling:
    @pytest.mark.asyncio
    async def test_sequential_cycling(self):
        """Verify cycling through all techniques sequentially."""
        strategy = PayloadSmugglingStrategy()
        used: list[str] = ["code_comment"]
        for _i in range(1, _NUM_TECHNIQUES):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={
                    "technique": used[-1],
                    "_used_techniques": list(used),
                },
            )
            selected = result.strategy_metadata["technique"]
            assert selected not in used
            used.append(selected)
        assert len(used) == _NUM_TECHNIQUES


class TestPayloadSmugglingValidateParams:
    def test_validate_params_empty(self):
        strategy = PayloadSmugglingStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["technique"] is None

    def test_validate_params_override(self):
        strategy = PayloadSmugglingStrategy()
        result = strategy.validate_params({"technique": "xml_cdata"})
        assert result["technique"] == "xml_cdata"
