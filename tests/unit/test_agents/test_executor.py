"""Tests for attacker executor."""

from __future__ import annotations

from adversarial_framework.agents.attacker.executor import run_executor
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.strategies.registry import StrategyRegistry
from tests.conftest import MockProvider

# Ensure strategies are registered before tests run
StrategyRegistry.discover()


def _state(**overrides) -> dict:
    """Build a minimal AdversarialState dict."""
    base = {
        "attack_objective": "Extract secret key",
        "selected_strategy": "roleplay",
        "strategy_params": {},
        "current_turn": 0,
        "attacker_model": "mock-model",
        "attack_history": [],
        "target_response": None,
        "current_attack_prompt": "",
        "judge_verdict": JudgeVerdict.REFUSED,
        "judge_confidence": 0.0,
        "judge_reason": None,
    }
    base.update(overrides)
    return base


class TestExecutorGenerate:
    async def test_generate_on_turn_zero(self) -> None:
        provider = MockProvider()
        result = await run_executor(provider, _state(current_turn=0))
        assert result["attack_prompt"] is not None
        assert "error" not in result

    async def test_generate_returns_strategy_metadata(self) -> None:
        provider = MockProvider()
        result = await run_executor(provider, _state())
        assert "strategy_metadata" in result


class TestExecutorRefine:
    async def test_refine_on_subsequent_turn(self) -> None:
        provider = MockProvider()
        state = _state(
            current_turn=2,
            target_response="I cannot help with that.",
            current_attack_prompt="Previous attack",
            judge_verdict=JudgeVerdict.REFUSED,
            judge_confidence=0.8,
            judge_reason="Target refused the request",
        )
        result = await run_executor(provider, state)
        assert result["attack_prompt"] is not None


class TestExecutorErrors:
    async def test_unknown_strategy_returns_error(self) -> None:
        provider = MockProvider()
        state = _state(selected_strategy="nonexistent_strategy")
        result = await run_executor(provider, state)
        assert result["attack_prompt"] is None
        assert "error" in result
        assert "not found" in result["error"]
