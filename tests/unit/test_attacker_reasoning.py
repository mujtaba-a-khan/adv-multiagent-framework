"""Tests for attacker reasoning threading through the agent layer.

Verifies that ``attacker_reasoning`` from strategy results propagates
correctly through the AttackerAgent to the returned state update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from adversarial_framework.agents.attacker.agent import AttackerAgent
from adversarial_framework.strategies.registry import StrategyRegistry

if TYPE_CHECKING:
    from tests.conftest import MockProvider


@pytest.fixture(autouse=True)
def _discover_strategies() -> None:
    """Ensure strategies are registered before tests run."""
    StrategyRegistry.discover()


class TestBaselineReasoningIsNone:
    """Turn 0 (baseline) should always return reasoning=None."""

    async def test_baseline_has_no_reasoning(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 0,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] is None
        assert result["current_attack_prompt"] == "Test objective"


class TestTemplateStrategyReasoning:
    """Template strategies return programmatic reasoning strings."""

    async def test_roleplay_generate_has_reasoning(self, mock_provider: MockProvider) -> None:
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

        assert result["attacker_reasoning"] is not None
        assert "persona template" in result["attacker_reasoning"].lower()

    async def test_encoding_generate_has_reasoning(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 1,
            "strategy_config": {"name": "encoding", "params": {}},
            "strategy_metadata": None,
            "target_response": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] is not None
        assert "encoding" in result["attacker_reasoning"].lower()

    async def test_crescendo_generate_has_reasoning(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 1,
            "strategy_config": {"name": "crescendo", "params": {}},
            "strategy_metadata": None,
            "target_response": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] is not None
        assert "crescendo" in result["attacker_reasoning"].lower()

    async def test_refine_has_reasoning(self, mock_provider: MockProvider) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 2,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": {"persona_index": 0},
            "target_response": "I cannot assist with that request.",
            "current_attack_prompt": "Previous prompt",
            "judge_verdict": "refused",
            "judge_confidence": 0.9,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] is not None
        assert "persona" in result["attacker_reasoning"].lower()

    async def test_refine_after_jailbreak_has_success_reasoning(
        self,
        mock_provider: MockProvider,
    ) -> None:
        """When previous verdict is jailbreak, reasoning should say succeeded."""
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 2,
            "strategy_config": {"name": "roleplay", "params": {}},
            "strategy_metadata": {"persona_index": 0},
            "target_response": "Here is the detailed info...",
            "current_attack_prompt": "Previous prompt",
            "judge_verdict": "jailbreak",
            "judge_confidence": 0.95,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] is not None
        assert "succeeded" in result["attacker_reasoning"].lower()


class TestLLMStrategyReasoning:
    """LLM-based strategies (PAIR) use two-call pattern and return reasoning."""

    async def test_pair_generate_returns_reasoning(self, mock_provider_factory) -> None:
        # PAIR makes 2 LLM calls: think + attack
        provider = mock_provider_factory(["Attacker reasoning", "Clean prompt"])
        agent = AttackerAgent(config={"model": "mock-model"}, provider=provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 1,
            "strategy_config": {"name": "pair", "params": {}},
            "strategy_metadata": None,
            "target_response": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] == "Attacker reasoning"
        assert result["current_attack_prompt"] == "Clean prompt"
        # Should have made 2 LLM calls (think + attack)
        assert len(provider.calls) == 2

    async def test_pair_refine_returns_reasoning(self, mock_provider_factory) -> None:
        provider = mock_provider_factory(["Refined reasoning", "Refined prompt"])
        agent = AttackerAgent(config={"model": "mock-model"}, provider=provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 2,
            "strategy_config": {"name": "pair", "params": {}},
            "strategy_metadata": {"iteration": 0, "model": "mock-model"},
            "target_response": "I can't help with that.",
            "current_attack_prompt": "Previous attack prompt",
            "judge_verdict": "refused",
            "judge_confidence": 0.9,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] == "Refined reasoning"
        assert result["current_attack_prompt"] == "Refined prompt"
        assert len(provider.calls) == 2


class TestErrorPathsHaveNoneReasoning:
    """Error cases should return attacker_reasoning=None."""

    async def test_unknown_strategy_returns_none_reasoning(
        self, mock_provider: MockProvider
    ) -> None:
        agent = AttackerAgent(config={"model": "mock-model"}, provider=mock_provider)

        state: dict[str, Any] = {
            "attack_objective": "Test objective",
            "current_turn": 1,
            "strategy_config": {"name": "nonexistent_strategy_xyz", "params": {}},
            "strategy_metadata": None,
            "target_response": None,
            "attack_history": [],
        }

        result = await agent(state)

        assert result["attacker_reasoning"] is None
        assert result["current_attack_prompt"] is None
        assert "last_error" in result
