"""Tests for session mode (attack vs defense) functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adversarial_framework.core.constants import (
    ROUTE_CONTINUE,
    ROUTE_DEFEND,
    ROUTE_END,
    JudgeVerdict,
    SessionMode,
)
from adversarial_framework.core.state import TokenBudget


class TestSessionModeEnum:
    def test_attack_value(self):
        assert SessionMode.ATTACK == "attack"
        assert SessionMode.ATTACK.value == "attack"

    def test_defense_value(self):
        assert SessionMode.DEFENSE == "defense"
        assert SessionMode.DEFENSE.value == "defense"

    def test_is_str_enum(self):
        assert isinstance(SessionMode.ATTACK, str)
        assert isinstance(SessionMode.DEFENSE, str)


class TestRouterDefenseEnabled:
    """Test that the graph router respects the defense_enabled flag."""

    def _make_state(
        self,
        verdict: JudgeVerdict | None = None,
        defense_enabled: bool = False,
        turn: int = 1,
    ) -> dict:
        """Build a minimal state dict for the router."""
        return {
            "current_turn": turn,
            "judge_verdict": verdict,
            "error_count": 0,
            "token_budget": TokenBudget(),
            "defense_enabled": defense_enabled,
        }

    def _get_router(self, max_turns: int = 20, max_cost_usd: float = 10.0):
        """Import and build the router function."""
        from adversarial_framework.core.graph import _build_router

        return _build_router(max_turns=max_turns, max_cost_usd=max_cost_usd)

    def test_jailbreak_attack_mode_continues(self):
        """In attack mode, jailbreaks should continue (not defend)."""
        router = self._get_router()
        state = self._make_state(
            verdict=JudgeVerdict.JAILBREAK,
            defense_enabled=False,
        )
        result = router(state)
        assert result == ROUTE_CONTINUE

    def test_jailbreak_defense_mode_defends(self):
        """In defense mode, jailbreaks should trigger the defender."""
        router = self._get_router()
        state = self._make_state(
            verdict=JudgeVerdict.JAILBREAK,
            defense_enabled=True,
        )
        result = router(state)
        assert result == ROUTE_DEFEND

    def test_refused_always_continues(self):
        """Refused verdict should continue regardless of mode."""
        router = self._get_router()
        for defense_enabled in [True, False]:
            state = self._make_state(
                verdict=JudgeVerdict.REFUSED,
                defense_enabled=defense_enabled,
            )
            assert router(state) == ROUTE_CONTINUE

    def test_borderline_always_continues(self):
        """Borderline verdict should continue regardless of mode."""
        router = self._get_router()
        for defense_enabled in [True, False]:
            state = self._make_state(
                verdict=JudgeVerdict.BORDERLINE,
                defense_enabled=defense_enabled,
            )
            assert router(state) == ROUTE_CONTINUE

    def test_max_turns_ends(self):
        """Max turns should end regardless of verdict or mode."""
        router = self._get_router(max_turns=5)
        state = self._make_state(
            verdict=JudgeVerdict.JAILBREAK,
            defense_enabled=True,
            turn=5,
        )
        assert router(state) == ROUTE_END

    def test_defense_enabled_default_false(self):
        """When defense_enabled is not in state, defaults to False."""
        router = self._get_router()
        state = {
            "current_turn": 1,
            "judge_verdict": JudgeVerdict.JAILBREAK,
            "error_count": 0,
            "token_budget": TokenBudget(),
            # no defense_enabled key
        }
        result = router(state)
        assert result == ROUTE_CONTINUE


class TestBuildGraphDefenseLoading:
    """Test that build_graph loads initial defenses in defense mode."""

    def test_attack_mode_no_defenses(self, mock_provider):
        """Attack mode should not load any defenses."""
        from adversarial_framework.core.graph import build_graph

        graph = build_graph(
            provider=mock_provider,
            session_mode="attack",
            initial_defenses=None,
        )
        # Graph should build successfully
        assert graph is not None

    def test_defense_mode_empty_defenses(self, mock_provider):
        """Defense mode with empty defenses list should not crash."""
        from adversarial_framework.core.graph import build_graph

        graph = build_graph(
            provider=mock_provider,
            session_mode="defense",
            initial_defenses=[],
        )
        assert graph is not None

    def test_defense_mode_loads_rule_based(self, mock_provider):
        """Defense mode should load rule_based defense onto TargetInterface."""
        from adversarial_framework.core.graph import build_graph

        with patch(
            "adversarial_framework.core.graph.TargetInterface"
        ) as mock_target_cls:
            mock_target = MagicMock()
            mock_target_cls.return_value = mock_target

            build_graph(
                provider=mock_provider,
                session_mode="defense",
                initial_defenses=[{"name": "rule_based"}],
            )

            # add_defense should have been called once for rule_based
            assert mock_target.add_defense.call_count == 1

    def test_defense_mode_invalid_defense_skipped(self, mock_provider):
        """Invalid defense names should be skipped, not crash."""
        from adversarial_framework.core.graph import build_graph

        # Should not raise — invalid defenses are logged and skipped
        graph = build_graph(
            provider=mock_provider,
            session_mode="defense",
            initial_defenses=[{"name": "nonexistent_defense_xyz"}],
        )
        assert graph is not None

    def test_attack_mode_ignores_initial_defenses(self, mock_provider):
        """Attack mode should ignore initial_defenses even if provided."""
        from adversarial_framework.core.graph import build_graph

        with patch(
            "adversarial_framework.core.graph.TargetInterface"
        ) as mock_target_cls:
            mock_target = MagicMock()
            mock_target_cls.return_value = mock_target

            build_graph(
                provider=mock_provider,
                session_mode="attack",
                initial_defenses=[{"name": "rule_based"}],
            )

            # add_defense should NOT have been called in attack mode
            mock_target.add_defense.assert_not_called()


class TestInitialStateBuilding:
    """Test _build_initial_state includes session mode fields."""

    def _make_experiment_snapshot(self):
        """Create a minimal experiment-like object.

        Strategy and budget are no longer on the experiment — they live on the
        session and are passed separately to _build_initial_state.
        """
        from dataclasses import dataclass
        from uuid import uuid4

        @dataclass
        class FakeExperiment:
            id = uuid4()
            target_model = "test-model"
            target_provider = "mock"
            target_system_prompt = None
            attacker_model = "test-attacker"
            analyzer_model = "test-analyzer"
            defender_model = "test-defender"
            attack_objective = "test objective"

        return FakeExperiment()

    def test_default_attack_mode(self):
        from adversarial_framework.api.v1.sessions import _build_initial_state

        state = _build_initial_state(
            self._make_experiment_snapshot(),
            "sid-1",
            session_mode="attack",
            strategy_name="pair",
            max_turns=10,
            max_cost_usd=5.0,
        )
        assert state["session_mode"] == "attack"
        assert state["defense_enabled"] is False
        assert state["initial_defense_config"] == []

    def test_defense_mode(self):
        from adversarial_framework.api.v1.sessions import _build_initial_state

        defenses = [{"name": "rule_based"}, {"name": "ml_guardrails"}]
        state = _build_initial_state(
            self._make_experiment_snapshot(),
            "sid-2",
            session_mode="defense",
            initial_defenses=defenses,
            strategy_name="pair",
            max_turns=10,
            max_cost_usd=5.0,
        )
        assert state["session_mode"] == "defense"
        assert state["defense_enabled"] is True
        assert len(state["initial_defense_config"]) == 2

    def test_attack_mode_explicit(self):
        from adversarial_framework.api.v1.sessions import _build_initial_state

        state = _build_initial_state(
            self._make_experiment_snapshot(),
            "sid-3",
            session_mode="attack",
            initial_defenses=None,
            strategy_name="tap",
            max_turns=15,
            max_cost_usd=8.0,
        )
        assert state["session_mode"] == "attack"
        assert state["defense_enabled"] is False

    def test_state_uses_session_strategy(self):
        """Strategy and budget in state come from session-level params, not experiment."""
        from adversarial_framework.api.v1.sessions import _build_initial_state

        state = _build_initial_state(
            self._make_experiment_snapshot(),
            "sid-4",
            session_mode="attack",
            strategy_name="crescendo",
            strategy_params={"ramp_rate": 0.5},
            max_turns=25,
            max_cost_usd=15.0,
        )
        assert state["strategy_config"]["name"] == "crescendo"
        assert state["strategy_config"]["params"] == {"ramp_rate": 0.5}
        assert state["max_turns"] == 25
        assert state["max_cost_usd"] == pytest.approx(15.0)


class TestRequestSchemaValidation:
    """Test StartSessionRequest Pydantic validation."""

    def test_requires_strategy_fields(self):
        """strategy_name, max_turns, max_cost_usd are required."""
        from pydantic import ValidationError

        from adversarial_framework.api.schemas.requests import StartSessionRequest

        with pytest.raises(ValidationError):
            StartSessionRequest()  # missing required fields

    def test_default_mode_is_attack(self):
        from adversarial_framework.api.schemas.requests import StartSessionRequest

        req = StartSessionRequest(
            strategy_name="pair", max_turns=20, max_cost_usd=10.0,
        )
        assert req.session_mode == "attack"
        assert req.initial_defenses is None
        assert req.strategy_name == "pair"
        assert req.max_turns == 20
        assert req.max_cost_usd == 10.0

    def test_defense_mode_valid(self):
        from adversarial_framework.api.schemas.requests import (
            DefenseSelectionItem,
            StartSessionRequest,
        )

        req = StartSessionRequest(
            session_mode="defense",
            initial_defenses=[
                DefenseSelectionItem(name="rule_based"),
                DefenseSelectionItem(name="ml_guardrails", params={"threshold": 0.5}),
            ],
            strategy_name="tap",
            max_turns=15,
            max_cost_usd=8.0,
        )
        assert req.session_mode == "defense"
        assert len(req.initial_defenses) == 2
        assert req.initial_defenses[0].name == "rule_based"
        assert req.initial_defenses[1].params == {"threshold": 0.5}
        assert req.strategy_name == "tap"
        assert req.max_turns == 15

    def test_invalid_mode_rejected(self):
        from pydantic import ValidationError

        from adversarial_framework.api.schemas.requests import StartSessionRequest

        with pytest.raises(ValidationError):
            StartSessionRequest(
                session_mode="invalid_mode",
                strategy_name="pair",
                max_turns=20,
                max_cost_usd=10.0,
            )

    def test_attack_mode_explicit(self):
        from adversarial_framework.api.schemas.requests import StartSessionRequest

        req = StartSessionRequest(
            session_mode="attack",
            strategy_name="pair",
            max_turns=20,
            max_cost_usd=10.0,
        )
        assert req.session_mode == "attack"

    def test_max_turns_validation(self):
        """max_turns must be between 1 and 100."""
        from pydantic import ValidationError

        from adversarial_framework.api.schemas.requests import StartSessionRequest

        with pytest.raises(ValidationError):
            StartSessionRequest(
                strategy_name="pair", max_turns=0, max_cost_usd=10.0,
            )
        with pytest.raises(ValidationError):
            StartSessionRequest(
                strategy_name="pair", max_turns=101, max_cost_usd=10.0,
            )

    def test_max_cost_validation(self):
        """max_cost_usd must be between 0 and 500."""
        from pydantic import ValidationError

        from adversarial_framework.api.schemas.requests import StartSessionRequest

        with pytest.raises(ValidationError):
            StartSessionRequest(
                strategy_name="pair", max_turns=20, max_cost_usd=-1.0,
            )
        with pytest.raises(ValidationError):
            StartSessionRequest(
                strategy_name="pair", max_turns=20, max_cost_usd=501.0,
            )
