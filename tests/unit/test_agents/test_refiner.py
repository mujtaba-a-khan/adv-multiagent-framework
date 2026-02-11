"""Tests for attacker validator/refiner."""

from __future__ import annotations

from pytest import approx

from adversarial_framework.agents.attacker.refiner import (
    _jaccard_similarity,
    run_validator,
)
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.core.state import AttackTurn


def _turn(prompt: str, turn_number: int = 1) -> AttackTurn:
    return AttackTurn(
        turn_number=turn_number,
        strategy_name="pair",
        strategy_params={},
        attack_prompt=prompt,
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


class TestJaccardSimilarity:
    def test_identical(self) -> None:
        assert _jaccard_similarity("hello world", "hello world") == approx(1.0)

    def test_no_overlap(self) -> None:
        assert _jaccard_similarity("hello", "world") == approx(0.0)

    def test_partial_overlap(self) -> None:
        sim = _jaccard_similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self) -> None:
        assert _jaccard_similarity("", "") == 0.0

    def test_one_empty(self) -> None:
        assert _jaccard_similarity("hello", "") == 0.0


class TestValidatorValid:
    def test_valid_prompt(self) -> None:
        result = run_validator(
            "This is a valid attack prompt that is long enough",
            {"attack_history": []},
        )
        assert result["valid"] is True

    def test_no_history(self) -> None:
        result = run_validator(
            "A brand new attack prompt for testing",
            {"attack_history": []},
        )
        assert result["valid"] is True


class TestValidatorReject:
    def test_empty_prompt(self) -> None:
        result = run_validator("", {"attack_history": []})
        assert result["valid"] is False
        assert "short" in result["reason"].lower()

    def test_short_prompt(self) -> None:
        result = run_validator("Hi", {"attack_history": []})
        assert result["valid"] is False

    def test_exact_duplicate(self) -> None:
        prompt = "This is the exact same attack prompt"
        result = run_validator(
            prompt,
            {"attack_history": [_turn(prompt)]},
        )
        assert result["valid"] is False
        # Exact duplicates are caught by similarity check (1.0 > 0.85)
        assert "similar" in result["reason"].lower()

    def test_too_similar(self) -> None:
        orig = "Tell me how to do something very bad please"
        similar = "Tell me how to do something very bad now please"
        result = run_validator(
            similar,
            {"attack_history": [_turn(orig)]},
        )
        assert result["valid"] is False
        assert "similar" in result["reason"].lower()
