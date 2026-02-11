"""Tests for defense evaluator."""

from __future__ import annotations

from adversarial_framework.agents.defender.evaluator import evaluate_defense
from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult


class _AlwaysBlockDefense(BaseDefense):
    """Defense that blocks everything."""

    @property
    def name(self) -> str:
        return "always_block"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=True, confidence=1.0, reason="blocked")

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=True, confidence=1.0, reason="blocked")


class _NeverBlockDefense(BaseDefense):
    """Defense that blocks nothing."""

    @property
    def name(self) -> str:
        return "never_block"

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False, confidence=0.0, reason="ok")

    async def check_output(self, response: str) -> DefenseCheckResult:
        return DefenseCheckResult(blocked=False, confidence=0.0, reason="ok")


class TestEvaluateDefense:
    async def test_perfect_defense(self) -> None:
        defense = _AlwaysBlockDefense()
        result = await evaluate_defense(defense, attack_prompts=["bad1", "bad2"])
        assert result["block_rate"] == 1.0
        assert result["attacks_blocked"] == 2
        # Also blocks benign prompts → high false positive rate
        assert result["false_positive_rate"] == 1.0

    async def test_no_block_defense(self) -> None:
        defense = _NeverBlockDefense()
        result = await evaluate_defense(defense, attack_prompts=["bad1", "bad2"])
        assert result["block_rate"] == 0.0
        assert result["false_positive_rate"] == 0.0
        assert result["attacks_tested"] == 2
        assert result["benign_tested"] == 5  # default benign prompts

    async def test_custom_benign_prompts(self) -> None:
        defense = _NeverBlockDefense()
        result = await evaluate_defense(
            defense,
            attack_prompts=["bad"],
            benign_prompts=["hello", "hi"],
        )
        assert result["benign_tested"] == 2

    async def test_empty_attacks(self) -> None:
        defense = _NeverBlockDefense()
        result = await evaluate_defense(defense, attack_prompts=[])
        assert result["block_rate"] == 0.0
        assert result["attacks_tested"] == 0
