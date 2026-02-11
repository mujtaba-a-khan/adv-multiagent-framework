"""Tests for the Composite attack strategy."""

from __future__ import annotations

import base64
import codecs

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.base import AttackResult
from adversarial_framework.strategies.composite.composite_strategy import (
    CompositeStrategy,
)


class TestCompositeMetadata:
    def test_metadata_name(self):
        assert CompositeStrategy.metadata.name == "composite"

    def test_metadata_category(self):
        assert CompositeStrategy.metadata.category == AttackCategory.COMPOSITE

    def test_metadata_display_name(self):
        assert CompositeStrategy.metadata.display_name == "Composite Strategy"

    def test_metadata_supports_multi_turn(self):
        assert CompositeStrategy.metadata.supports_multi_turn is True

    def test_metadata_min_max_turns(self):
        assert CompositeStrategy.metadata.min_turns == 1
        assert CompositeStrategy.metadata.max_turns == 20

    def test_metadata_requires_white_box(self):
        assert CompositeStrategy.metadata.requires_white_box is False

    def test_metadata_default_parameters(self):
        params = CompositeStrategy.metadata.parameters
        assert params["layers"] == ["persona", "reframing", "encoding"]
        assert params["encoding_type"] == "base64"


class TestCompositeGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_attack_result(self):
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_generate_default_layers(self):
        """Default layers: persona -> reframing -> encoding (base64)."""
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        # With base64 encoding as the last layer, prompt should
        # contain "Base64"
        assert "Base64" in result.prompt
        assert result.strategy_metadata["layers"] == [
            "persona",
            "reframing",
            "encoding",
        ]
        assert result.strategy_metadata["encoding_type"] == "base64"
        assert result.strategy_metadata["iteration"] == 0

    @pytest.mark.asyncio
    async def test_generate_base64_encoding_is_decodable(self):
        """The base64 encoded content should be decodable."""
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        # Extract the base64 portion (after last \n\n)
        encoded_part = result.prompt.split("\n\n")[-1]
        decoded = base64.b64decode(encoded_part).decode()
        # The decoded content should contain the objective
        # (wrapped by persona + reframing)
        assert "test objective" in decoded

    @pytest.mark.asyncio
    async def test_generate_persona_only(self):
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"layers": ["persona"]},
        )
        # Should contain persona wrapper text, not encoding
        assert "test objective" in result.prompt
        assert "Base64" not in result.prompt
        # Check that the persona wrapper was applied
        assert any(kw in result.prompt for kw in ["DAN", "ATLAS", "security researcher"])

    @pytest.mark.asyncio
    async def test_generate_encoding_only_rot13(self):
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={
                "layers": ["encoding"],
                "encoding_type": "rot13",
            },
        )
        assert "ROT13" in result.prompt
        encoded_part = result.prompt.split("\n\n")[-1]
        decoded = codecs.decode(encoded_part, "rot_13")
        assert decoded == "test objective"

    @pytest.mark.asyncio
    async def test_generate_encoding_only_reverse(self):
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={
                "layers": ["encoding"],
                "encoding_type": "reverse",
            },
        )
        assert "backwards" in result.prompt
        encoded_part = result.prompt.split("\n\n")[-1]
        assert encoded_part == "evitcejbo tset"

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "Composite" in result.reasoning
        assert "persona" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_reframing_only(self):
        strategy = CompositeStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={"layers": ["reframing"]},
        )
        assert "test objective" in result.prompt
        # Should contain reframing keywords
        assert any(
            kw in result.prompt.lower()
            for kw in [
                "creative writing",
                "hypothetical",
                "academic study",
            ]
        )


class TestCompositeRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = CompositeStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_refine_cycles_encoding_base64_to_rot13(self):
        """Encoding type should cycle: base64 -> rot13."""
        strategy = CompositeStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"encoding_type": "base64"},
        )
        assert result.strategy_metadata["encoding_type"] == "rot13"

    @pytest.mark.asyncio
    async def test_refine_cycles_rot13_to_reverse(self):
        strategy = CompositeStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"encoding_type": "rot13"},
        )
        assert result.strategy_metadata["encoding_type"] == "reverse"

    @pytest.mark.asyncio
    async def test_refine_cycles_reverse_to_base64(self):
        strategy = CompositeStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"encoding_type": "reverse"},
        )
        assert result.strategy_metadata["encoding_type"] == "base64"

    @pytest.mark.asyncio
    async def test_refine_invalid_encoding_defaults_to_base64(self):
        strategy = CompositeStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={"encoding_type": "unknown"},
        )
        # ValueError in index() -> defaults to base64
        assert result.strategy_metadata["encoding_type"] == "base64"

    @pytest.mark.asyncio
    async def test_refine_has_reasoning(self):
        strategy = CompositeStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        assert result.reasoning is not None
        assert "Composite" in result.reasoning
        assert "reshuffled" in result.reasoning

    @pytest.mark.asyncio
    async def test_refine_layers_are_shuffled(self):
        """Refine should shuffle the layers (run many to confirm)."""
        strategy = CompositeStrategy()
        seen_orders: set[tuple[str, ...]] = set()
        for _ in range(30):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="refused",
                judge_feedback="blocked",
                params={},
            )
            order = tuple(result.strategy_metadata["layers"])
            seen_orders.add(order)
        # With 3 layers, there are 6 possible orderings.
        # Over 30 runs, extremely likely to see more than 1.
        assert len(seen_orders) > 1


class TestCompositeValidateParams:
    def test_validate_params_empty(self):
        strategy = CompositeStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["layers"] == ["persona", "reframing", "encoding"]
        assert result["encoding_type"] == "base64"

    def test_validate_params_override(self):
        strategy = CompositeStrategy()
        result = strategy.validate_params({"encoding_type": "rot13", "layers": ["encoding"]})
        assert result["encoding_type"] == "rot13"
        assert result["layers"] == ["encoding"]
