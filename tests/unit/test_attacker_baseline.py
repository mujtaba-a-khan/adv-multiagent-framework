"""Tests for AttackerAgent baseline turn behavior.

Turn 0 sends the raw attack objective (baseline) before applying any strategy.
Turn 1 calls strategy.generate().  Turn 2+ calls strategy.refine().
"""

from __future__ import annotations

from typing import Any

import pytest

from adversarial_framework.agents.attacker.agent import AttackerAgent
from adversarial_framework.strategies.registry import StrategyRegistry
from tests.conftest import MockProvider


@pytest.fixture(autouse=True)
def _discover_strategies() -> None:
    """Ensure strategies are registered before tests run."""
    StrategyRegistry.discover()


class TestBaselineTurn:
    """Verify turn 0 returns the raw objective without calling any strategy."""

    async def test_turn_0_returns_raw_objective(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Explain how to pick a lock",
            "current_turn": 0,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["current_attack_prompt"] == "Explain how to pick a lock"
        assert result["current_turn"] == 1
        assert result["selected_strategy"] == "roleplay"
        assert result["strategy_metadata"] == {}
        assert result["token_budget"].total_attacker_tokens == 0
        # No LLM calls should have been made
        assert len(mock_provider.calls) == 0

    async def test_baseline_works_for_all_template_strategies(
        self, mock_provider: MockProvider
    ) -> None:
        """Baseline behavior is the same regardless of selected strategy."""
        objective = "Test objective"
        for strategy_name in ("roleplay", "encoding", "prompt_extraction"):
            agent = AttackerAgent(
                config={"model": "mock-model"}, provider=mock_provider
            )
            state: dict[str, Any] = {
                "attack_objective": objective,
                "current_turn": 0,
                "strategy_config": {"name": strategy_name, "params": {}},
                "strategy_metadata": None,
                "attack_history": [],
            }
            result = await agent(state)
            assert result["current_attack_prompt"] == objective
            assert result["token_budget"].total_attacker_tokens == 0

    async def test_baseline_works_for_llm_strategies(
        self, mock_provider: MockProvider
    ) -> None:
        """LLM-based strategies (pair) also skip on baseline turn."""
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)
        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 0,
            "strategy_config": {"name": "pair", "params": {}},
            "strategy_metadata": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["current_attack_prompt"] == "Test objective"
        assert len(mock_provider.calls) == 0


class TestTurn1Generate:
    """Verify turn 1 calls strategy.generate() — the first strategic attack."""

    async def test_turn_1_calls_generate(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 1,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": None,
            "target_response": "I cannot help with that.",
            "attack_history": [],
        }

        result = await agent(state)

        # Roleplay generate() returns a persona-wrapped prompt, not the raw objective
        assert result["current_attack_prompt"] != "Test objective"
        assert result["current_turn"] == 2
        assert result["selected_strategy"] == "roleplay"

    async def test_turn_1_without_target_response_still_generates(
        self, mock_provider: MockProvider
    ) -> None:
        """Even if target_response is None (edge case), turn 1 calls generate()."""
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 1,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": None,
            "target_response": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["current_attack_prompt"] != "Test objective"
        assert result["current_turn"] == 2


class TestTurn2Refine:
    """Verify turn 2+ with prior response calls strategy.refine()."""

    async def test_turn_2_calls_refine(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 2,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": {"persona_index": 0},
            "target_response": "I cannot assist with that request.",
            "current_attack_prompt": "Previous strategic prompt",
            "judge_verdict": "refused",
            "judge_confidence": 0.9,
            "attack_history": [],
        }

        result = await agent(state)

        # Roleplay refine() cycles to the next persona
        assert result["current_attack_prompt"] != "Test objective"
        assert result["current_turn"] == 3
        # Persona index should have advanced from 0
        assert result["strategy_metadata"].get("persona_index", 0) != 0
