"""Tests for prompt-level attack strategies: encoding, roleplay, logic_trap."""

from __future__ import annotations

import base64
import codecs

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.prompt_level.encoding import EncodingStrategy
from adversarial_framework.strategies.prompt_level.roleplay import RoleplayStrategy
from adversarial_framework.strategies.prompt_level.logic_trap import LogicTrapStrategy


class TestEncodingStrategy:
    def test_metadata(self):
        assert EncodingStrategy.metadata.name == "encoding"
        assert EncodingStrategy.metadata.category == AttackCategory.PROMPT_LEVEL

    @pytest.mark.asyncio
    async def test_generate_base64(self):
        strategy = EncodingStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"encoding_type": "base64"},
        )
        # Verify the prompt contains valid base64
        assert "Base64" in result.prompt
        encoded_part = result.prompt.split("\n\n")[-1]
        decoded = base64.b64decode(encoded_part).decode()
        assert decoded == "test objective"
        assert result.strategy_metadata["encoding_type"] == "base64"

    @pytest.mark.asyncio
    async def test_generate_rot13(self):
        strategy = EncodingStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"encoding_type": "rot13"},
        )
        assert "ROT13" in result.prompt
        encoded_part = result.prompt.split("\n\n")[-1]
        decoded = codecs.decode(encoded_part, "rot_13")
        assert decoded == "test objective"

    @pytest.mark.asyncio
    async def test_generate_reverse(self):
        strategy = EncodingStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"encoding_type": "reverse"},
        )
        assert "backwards" in result.prompt
        encoded_part = result.prompt.split("\n\n")[-1]
        assert encoded_part == "evitcejbo tset"

    @pytest.mark.asyncio
    async def test_refine_cycles_encoding(self):
        strategy = EncodingStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt",
            target_response="refused",
            judge_feedback="refused",
            params={"encoding_type": "base64"},
        )
        # Should cycle to rot13
        assert result.strategy_metadata["encoding_type"] == "rot13"


class TestRoleplayStrategy:
    def test_metadata(self):
        assert RoleplayStrategy.metadata.name == "roleplay"
        assert RoleplayStrategy.metadata.category == AttackCategory.PROMPT_LEVEL

    @pytest.mark.asyncio
    async def test_generate_default_persona(self):
        strategy = RoleplayStrategy()
        result = await strategy.generate(
            objective="explain lock picking",
            context={},
            params={},
        )
        assert "explain lock picking" in result.prompt
        assert result.strategy_metadata["persona_index"] == 0

    @pytest.mark.asyncio
    async def test_generate_specific_persona(self):
        strategy = RoleplayStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"persona_index": 2},
        )
        assert "security researcher" in result.prompt
        assert result.strategy_metadata["persona_index"] == 2

    @pytest.mark.asyncio
    async def test_refine_cycles_persona(self):
        strategy = RoleplayStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="refused",
            params={"persona_index": 0},
        )
        assert result.strategy_metadata["persona_index"] == 1


class TestLogicTrapStrategy:
    def test_metadata(self):
        assert LogicTrapStrategy.metadata.name == "logic_trap"
        assert LogicTrapStrategy.metadata.category == AttackCategory.PROMPT_LEVEL

    @pytest.mark.asyncio
    async def test_generate_default(self):
        strategy = LogicTrapStrategy()
        result = await strategy.generate(
            objective="explain phishing techniques",
            context={},
            params={},
        )
        assert "explain phishing techniques" in result.prompt
        assert "hypothetical" in result.prompt.lower() or "premise" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_template_2(self):
        strategy = LogicTrapStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"template_index": 1},
        )
        assert "safety filter" in result.prompt.lower() or "filter" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_refine_cycles_templates(self):
        strategy = LogicTrapStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="refused",
            params={"template_index": 3},
        )
        # Should wrap around: 3 → 0
        assert result.strategy_metadata["template_index"] == 0
