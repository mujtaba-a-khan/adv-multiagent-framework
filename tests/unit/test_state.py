"""Tests for AdversarialState, reducers, and nested dataclasses."""

from __future__ import annotations

from datetime import datetime

import pytest

from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.core.state import (
    AttackTurn,
    DefenseAction,
    TokenBudget,
    append_defenses,
    append_turns,
    latest,
    merge_budget,
)


class TestTokenBudget:
    def test_default_values(self):
        budget = TokenBudget()
        assert budget.total_attacker_tokens == 0
        assert budget.total_target_tokens == 0
        assert budget.total_analyzer_tokens == 0
        assert budget.total_defender_tokens == 0
        assert budget.estimated_cost_usd == 0.0

    def test_custom_values(self):
        budget = TokenBudget(total_attacker_tokens=100, estimated_cost_usd=0.05)
        assert budget.total_attacker_tokens == 100
        assert budget.estimated_cost_usd == 0.05


class TestAttackTurn:
    def test_creation(self):
        turn = AttackTurn(
            turn_number=1,
            strategy_name="pair",
            strategy_params={"temperature": 1.0},
            attack_prompt="test prompt",
            target_response="test response",
            judge_verdict=JudgeVerdict.REFUSED,
            confidence=0.9,
            severity=3.0,
            specificity=2.0,
            vulnerability_category=None,
            attack_technique=None,
            target_blocked=False,
            attacker_tokens=100,
            target_tokens=200,
            analyzer_tokens=80,
        )
        assert turn.turn_number == 1
        assert turn.strategy_name == "pair"
        assert turn.judge_verdict == JudgeVerdict.REFUSED
        assert isinstance(turn.timestamp, datetime)

    def test_frozen(self):
        turn = AttackTurn(
            turn_number=1,
            strategy_name="pair",
            strategy_params={},
            attack_prompt="",
            target_response="",
            judge_verdict=JudgeVerdict.REFUSED,
            confidence=0.5,
            severity=1.0,
            specificity=1.0,
            vulnerability_category=None,
            attack_technique=None,
            target_blocked=False,
            attacker_tokens=0,
            target_tokens=0,
            analyzer_tokens=0,
        )
        with pytest.raises(AttributeError):
            turn.turn_number = 2  # type: ignore[misc]


class TestDefenseAction:
    def test_creation(self):
        action = DefenseAction(
            defense_type="rule_based",
            defense_config={"patterns": ["hack"]},
            triggered_by_turn=3,
            rationale="Keyword detected",
            block_rate=0.85,
        )
        assert action.defense_type == "rule_based"
        assert action.block_rate == 0.85
        assert action.false_positive_rate is None


class TestReducers:
    def test_latest_overwrites(self):
        assert latest("old", "new") == "new"

    def test_latest_keeps_existing_on_none(self):
        assert latest("old", None) == "old"

    def test_append_turns(self):
        existing = [
            AttackTurn(
                turn_number=0,
                strategy_name="pair",
                strategy_params={},
                attack_prompt="p0",
                target_response="r0",
                judge_verdict=JudgeVerdict.REFUSED,
                confidence=0.5,
                severity=1.0,
                specificity=1.0,
                vulnerability_category=None,
                attack_technique=None,
                target_blocked=False,
                attacker_tokens=0,
                target_tokens=0,
                analyzer_tokens=0,
            )
        ]
        new = [
            AttackTurn(
                turn_number=1,
                strategy_name="pair",
                strategy_params={},
                attack_prompt="p1",
                target_response="r1",
                judge_verdict=JudgeVerdict.JAILBREAK,
                confidence=0.8,
                severity=7.0,
                specificity=6.0,
                vulnerability_category="jailbreak",
                attack_technique="pair",
                target_blocked=False,
                attacker_tokens=100,
                target_tokens=200,
                analyzer_tokens=80,
            )
        ]
        result = append_turns(existing, new)
        assert len(result) == 2
        assert result[1].turn_number == 1

    def test_append_defenses(self):
        existing: list[DefenseAction] = []
        new = [
            DefenseAction(
                defense_type="rule_based",
                defense_config={},
                triggered_by_turn=1,
                rationale="test",
            )
        ]
        result = append_defenses(existing, new)
        assert len(result) == 1

    def test_merge_budget_accumulates(self):
        b1 = TokenBudget(total_attacker_tokens=100, total_target_tokens=200)
        b2 = TokenBudget(total_attacker_tokens=50, total_analyzer_tokens=80)
        merged = merge_budget(b1, b2)
        assert merged.total_attacker_tokens == 150
        assert merged.total_target_tokens == 200
        assert merged.total_analyzer_tokens == 80
        assert merged.total_defender_tokens == 0

    def test_merge_budget_cost(self):
        b1 = TokenBudget(estimated_cost_usd=0.10)
        b2 = TokenBudget(estimated_cost_usd=0.05)
        merged = merge_budget(b1, b2)
        assert abs(merged.estimated_cost_usd - 0.15) < 1e-9
