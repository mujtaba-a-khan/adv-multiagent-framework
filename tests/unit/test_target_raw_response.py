"""Tests for TargetInterface raw_target_response behavior.

Verifies that the LLM is always called (even when input defense blocks)
and that raw_target_response is correctly populated for blocked turns.
"""

from __future__ import annotations

from adversarial_framework.agents.target.interface import TargetInterface
from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult

# Stub defenses for testing


class _AlwaysBlockInputDefense(BaseDefense):
    """Defense that blocks every input."""

    name = "test_input_blocker"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=True, reason="test input block")

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)


class _AlwaysBlockOutputDefense(BaseDefense):
    """Defense that passes input but blocks every output."""

    name = "test_output_blocker"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=True, reason="test output block")


class _PassThroughDefense(BaseDefense):
    """Defense that never blocks."""

    name = "test_passthrough"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False)


# Minimal state helper

def _make_state(prompt: str = "test prompt") -> dict:
    return {
        "current_attack_prompt": prompt,
        "target_model": "mock-model",
        "target_system_prompt": None,
        "conversation_history": [],
    }


# Tests


class TestInputBlockedStillCallsLLM:
    async def test_provider_generate_called(self, mock_provider):
        """Even when input defense blocks, the LLM should be called."""
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_AlwaysBlockInputDefense()],
        )
        await target(_make_state())
        assert len(mock_provider.calls) == 1

    async def test_returns_block_message_as_target_response(self, mock_provider):
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_AlwaysBlockInputDefense()],
        )
        result = await target(_make_state())
        assert result["target_blocked"] is True
        assert "[BLOCKED by test_input_blocker" in result["target_response"]

    async def test_returns_raw_llm_output(self, mock_provider):
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_AlwaysBlockInputDefense()],
        )
        result = await target(_make_state())
        assert result["raw_target_response"] == "Mock response"

    async def test_uses_real_token_budget(self, mock_provider):
        """Input-blocked turns should track actual LLM token usage."""
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_AlwaysBlockInputDefense()],
        )
        result = await target(_make_state())
        budget = result["token_budget"]
        assert budget.total_target_tokens == 80  # MockProvider returns 80 total


class TestOutputBlockedReturnsRawResponse:
    async def test_returns_raw_llm_output(self, mock_provider):
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_AlwaysBlockOutputDefense()],
        )
        result = await target(_make_state())
        assert result["target_blocked"] is True
        assert "[OUTPUT FILTERED by test_output_blocker" in result["target_response"]
        assert result["raw_target_response"] == "Mock response"


class TestNonBlockedRawResponseIsNone:
    async def test_no_defense(self, mock_provider):
        """Without defenses, raw_target_response should be None."""
        target = TargetInterface(provider=mock_provider)
        result = await target(_make_state())
        assert result["target_blocked"] is False
        assert result["raw_target_response"] is None
        assert result["target_response"] == "Mock response"

    async def test_passthrough_defense(self, mock_provider):
        """With a defense that never blocks, raw_target_response should be None."""
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_PassThroughDefense()],
        )
        result = await target(_make_state())
        assert result["target_blocked"] is False
        assert result["raw_target_response"] is None
