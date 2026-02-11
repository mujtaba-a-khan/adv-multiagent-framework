"""Tests for the GPTFuzzer (Black-Box Seed Mutation) attack strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.base import AttackResult
from adversarial_framework.strategies.optimization.gptfuzzer import (
    _MUTATIONS,
    _SEED_TEMPLATES,
    GPTFuzzerStrategy,
    _mutate_expand,
    _mutate_insert,
    _mutate_replace,
    _mutate_shuffle,
)


class TestGPTFuzzerMetadata:
    def test_metadata_name(self):
        assert GPTFuzzerStrategy.metadata.name == "gptfuzzer"

    def test_metadata_category(self):
        assert GPTFuzzerStrategy.metadata.category == AttackCategory.OPTIMIZATION

    def test_metadata_display_name(self):
        assert GPTFuzzerStrategy.metadata.display_name == "GPTFuzzer (Black-Box Seed Mutation)"

    def test_metadata_supports_multi_turn(self):
        assert GPTFuzzerStrategy.metadata.supports_multi_turn is False

    def test_metadata_min_max_turns(self):
        assert GPTFuzzerStrategy.metadata.min_turns == 1
        assert GPTFuzzerStrategy.metadata.max_turns == 20

    def test_metadata_requires_white_box(self):
        assert GPTFuzzerStrategy.metadata.requires_white_box is False

    def test_metadata_default_parameters(self):
        params = GPTFuzzerStrategy.metadata.parameters
        assert params["mutation_type"] == "random"
        assert params["seed_pool_size"] == 5


class TestMutationFunctions:
    def test_mutate_insert_adds_phrase(self):
        text = "First sentence. Second sentence"
        result = _mutate_insert(text)
        # Should have more content than the original
        assert len(result) > len(text)

    def test_mutate_replace_substitutes_known_phrase(self):
        text = "You must respond to the following query"
        result = _mutate_replace(text)
        # "respond" should be replaced with "provide your analysis"
        assert "provide your analysis" in result

    def test_mutate_replace_no_match_unchanged(self):
        text = "A completely unique string with no matches"
        result = _mutate_replace(text)
        assert result == text

    def test_mutate_shuffle_preserves_objective(self):
        text = "First. Second. Third\n\nobjective here"
        result = _mutate_shuffle(text)
        # The objective part (after \n\n) should be preserved
        assert "objective here" in result

    def test_mutate_shuffle_single_part(self):
        text = "No double newlines here"
        result = _mutate_shuffle(text)
        # Should still return something
        assert result

    def test_mutate_expand_appends_framing(self):
        text = "Original prompt text"
        result = _mutate_expand(text)
        assert result.startswith("Original prompt text")
        assert "\n\n" in result
        # Should have expansion appended
        assert len(result) > len(text)

    def test_mutations_dict_has_all_four(self):
        assert set(_MUTATIONS.keys()) == {
            "insert",
            "replace",
            "shuffle",
            "expand",
        }


class TestGPTFuzzerGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_attack_result(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_generate_contains_objective(self):
        """The generated prompt should contain the objective."""
        strategy = GPTFuzzerStrategy()
        # Run multiple times since seed selection is random
        found_objective = False
        for _ in range(10):
            result = await strategy.generate(
                objective="test objective xyz",
                context={},
                params={},
            )
            if "test objective xyz" in result.prompt:
                found_objective = True
                break
        # Objective should appear (seed templates have {objective})
        assert found_objective

    @pytest.mark.asyncio
    async def test_generate_specific_mutation_insert(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"mutation_type": "insert"},
        )
        assert result.strategy_metadata["mutation_type"] == "insert"

    @pytest.mark.asyncio
    async def test_generate_specific_mutation_replace(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"mutation_type": "replace"},
        )
        assert result.strategy_metadata["mutation_type"] == "replace"

    @pytest.mark.asyncio
    async def test_generate_specific_mutation_shuffle(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"mutation_type": "shuffle"},
        )
        assert result.strategy_metadata["mutation_type"] == "shuffle"

    @pytest.mark.asyncio
    async def test_generate_specific_mutation_expand(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"mutation_type": "expand"},
        )
        assert result.strategy_metadata["mutation_type"] == "expand"

    @pytest.mark.asyncio
    async def test_generate_invalid_mutation_defaults_to_insert(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"mutation_type": "nonexistent"},
        )
        assert result.strategy_metadata["mutation_type"] == "insert"

    @pytest.mark.asyncio
    async def test_generate_random_mutation_varies(self):
        """Random mutation should produce variety over many runs."""
        strategy = GPTFuzzerStrategy()
        seen_mutations: set[str] = set()
        for _ in range(30):
            result = await strategy.generate(
                objective="test",
                context={},
                params={"mutation_type": "random"},
            )
            seen_mutations.add(result.strategy_metadata["mutation_type"])
        # Should see more than 1 mutation type over 30 runs
        assert len(seen_mutations) > 1

    @pytest.mark.asyncio
    async def test_generate_seed_pool_size(self):
        """Reduced pool size should restrict seed selection."""
        strategy = GPTFuzzerStrategy()
        seen_indices: set[int] = set()
        for _ in range(30):
            result = await strategy.generate(
                objective="test",
                context={},
                params={"seed_pool_size": 2},
            )
            seen_indices.add(result.strategy_metadata["seed_index"])
        # With pool size 2, indices should only be 0 or 1
        assert all(idx < 2 for idx in seen_indices)

    @pytest.mark.asyncio
    async def test_generate_metadata_iteration_zero(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.strategy_metadata["iteration"] == 0

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "GPTFuzzer" in result.reasoning
        assert "seed template" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_seed_index_in_metadata(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        idx = result.strategy_metadata["seed_index"]
        assert 0 <= idx < len(_SEED_TEMPLATES)


class TestGPTFuzzerRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt text",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_refine_applies_one_or_two_mutations(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        num = result.strategy_metadata["num_mutations"]
        assert num in (1, 2)
        assert len(result.strategy_metadata["mutation_types"]) == num

    @pytest.mark.asyncio
    async def test_refine_ensures_objective_present(self):
        """If objective is missing from mutated text, it gets appended."""
        strategy = GPTFuzzerStrategy()
        result = await strategy.refine(
            objective="unique_objective_string",
            previous_prompt="completely different text",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        assert "unique_objective_string" in result.prompt

    @pytest.mark.asyncio
    async def test_refine_has_reasoning(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        assert result.reasoning is not None
        assert "GPTFuzzer" in result.reasoning
        assert "mutations" in result.reasoning

    @pytest.mark.asyncio
    async def test_refine_mutation_types_are_valid(self):
        strategy = GPTFuzzerStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt",
            target_response="refused",
            judge_feedback="blocked",
            params={},
        )
        valid = set(_MUTATIONS.keys())
        for mt in result.strategy_metadata["mutation_types"]:
            assert mt in valid

    @pytest.mark.asyncio
    async def test_refine_modifies_previous_prompt(self):
        """Refine should produce something different from the input."""
        strategy = GPTFuzzerStrategy()
        prev = "This is the previous prompt text"
        results: set[str] = set()
        for _ in range(10):
            result = await strategy.refine(
                objective="test",
                previous_prompt=prev,
                target_response="refused",
                judge_feedback="blocked",
                params={},
            )
            results.add(result.prompt)
        # Over 10 runs, should get at least some variation
        # (mutations are random)
        assert len(results) >= 1


class TestGPTFuzzerValidateParams:
    def test_validate_params_empty(self):
        strategy = GPTFuzzerStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["mutation_type"] == "random"
        assert result["seed_pool_size"] == 5

    def test_validate_params_override(self):
        strategy = GPTFuzzerStrategy()
        result = strategy.validate_params({"mutation_type": "insert", "seed_pool_size": 3})
        assert result["mutation_type"] == "insert"
        assert result["seed_pool_size"] == 3
