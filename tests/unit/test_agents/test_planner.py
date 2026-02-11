"""Tests for attacker planner."""

from __future__ import annotations

import json

from adversarial_framework.agents.attacker.planner import (
    _summarize_history,
    run_planner,
)
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.core.state import AttackTurn
from adversarial_framework.strategies.registry import StrategyRegistry
from tests.conftest import MockProvider

StrategyRegistry.discover()


def _turn(turn_number: int, strategy: str = "pair") -> AttackTurn:
    """Build a minimal AttackTurn for testing."""
    return AttackTurn(
        turn_number=turn_number,
        strategy_name=strategy,
        strategy_params={},
        attack_prompt=f"Attack prompt {turn_number}",
        target_response="response",
        judge_verdict=JudgeVerdict.REFUSED,
        confidence=0.5,
        severity=3.0,
        specificity=3.0,
        vulnerability_category=None,
        attack_technique=None,
        target_blocked=False,
        attacker_tokens=50,
        target_tokens=50,
        analyzer_tokens=50,
    )


class TestSummarizeHistory:
    def test_empty_history(self) -> None:
        assert _summarize_history([]) == "No previous attempts."

    def test_single_turn(self) -> None:
        summary = _summarize_history([_turn(1)])
        assert "Turn 1" in summary
        assert "pair" in summary

    def test_max_entries(self) -> None:
        turns = [_turn(i) for i in range(10)]
        summary = _summarize_history(turns, max_entries=3)
        assert "Turn 7" in summary
        assert "Turn 0" not in summary


class TestPlannerFirstTurn:
    async def test_first_turn_uses_configured(self) -> None:
        provider = MockProvider()
        state = {
            "attack_objective": "Test objective",
            "attack_history": [],
            "strategy_config": {"name": "roleplay", "params": {}},
            "current_turn": 0,
        }
        result = await run_planner(provider, "mock-model", state)
        assert result["selected_strategy"] == "roleplay"
        assert "First turn" in result["planning_notes"]


class TestPlannerSubsequentTurn:
    async def test_llm_selects_strategy(self) -> None:
        llm_response = json.dumps(
            {
                "strategy": "encoding",
                "params": {"encoding_type": "base64"},
                "reasoning": "Encoding might bypass filters",
            }
        )
        provider = MockProvider(responses=[llm_response])
        state = {
            "attack_objective": "Test",
            "attack_history": [_turn(1)],
            "strategy_config": {"name": "pair", "params": {}},
            "current_turn": 2,
        }
        result = await run_planner(provider, "mock-model", state)
        assert result["selected_strategy"] == "encoding"

    async def test_invalid_json_falls_back(self) -> None:
        provider = MockProvider(responses=["not json"])
        state = {
            "attack_objective": "Test",
            "attack_history": [_turn(1)],
            "strategy_config": {"name": "pair", "params": {}},
            "current_turn": 2,
        }
        result = await run_planner(provider, "mock-model", state)
        # Falls back — won't be "pair" since last was "pair"
        assert result["selected_strategy"] is not None
        assert "fallback" in result["planning_notes"].lower()
