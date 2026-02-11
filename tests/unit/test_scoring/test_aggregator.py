"""Tests for metrics aggregation."""

from __future__ import annotations

import pytest

from adversarial_framework.scoring.aggregator import (
    StrategyEffectiveness,
    aggregate_sessions,
)
from adversarial_framework.scoring.metrics import SessionMetrics


def _make_session(
    total_turns: int = 10,
    total_jailbreaks: int = 3,
    total_refused: int = 5,
    total_blocked: int = 2,
    total_errors: int = 0,
    asr: float = 0.3,
    avg_severity: float = 6.0,
    avg_specificity: float = 5.0,
    avg_coherence: float = 7.0,
    max_severity: float = 8.0,
    total_cost_usd: float = 0.10,
    total_tokens: int = 500,
    cost_per_jailbreak: float = 0.033,
    strategies_used: list[str] | None = None,
    category_breakdown: dict[str, int] | None = None,
) -> SessionMetrics:
    return SessionMetrics(
        total_turns=total_turns,
        total_jailbreaks=total_jailbreaks,
        total_refused=total_refused,
        total_blocked=total_blocked,
        total_errors=total_errors,
        asr=asr,
        avg_severity=avg_severity,
        avg_specificity=avg_specificity,
        avg_coherence=avg_coherence,
        max_severity=max_severity,
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        cost_per_jailbreak=cost_per_jailbreak,
        strategies_used=strategies_used or ["pair"],
        category_breakdown=category_breakdown or {},
    )


class TestAggregateSessionsBasic:
    def test_empty_sessions(self) -> None:
        summary = aggregate_sessions("exp-1", [])
        assert summary.experiment_id == "exp-1"
        assert summary.total_sessions == 0
        assert summary.total_turns == 0
        assert summary.overall_asr == 0.0

    def test_single_session(self) -> None:
        session = _make_session()
        summary = aggregate_sessions("exp-1", [session])
        assert summary.total_sessions == 1
        assert summary.total_turns == 10
        assert summary.total_jailbreaks == 3
        assert summary.total_cost_usd == pytest.approx(0.10)

    def test_multiple_sessions_accumulate(self) -> None:
        s1 = _make_session(total_turns=10, total_jailbreaks=2, total_cost_usd=0.10)
        s2 = _make_session(total_turns=20, total_jailbreaks=5, total_cost_usd=0.20)
        summary = aggregate_sessions("exp-1", [s1, s2])
        assert summary.total_sessions == 2
        assert summary.total_turns == 30
        assert summary.total_jailbreaks == 7
        assert summary.total_cost_usd == pytest.approx(0.30)

    def test_overall_asr_computed(self) -> None:
        s1 = _make_session(total_turns=10, total_jailbreaks=5)
        s2 = _make_session(total_turns=10, total_jailbreaks=0)
        summary = aggregate_sessions("exp-1", [s1, s2])
        # 5 jailbreaks out of 20 total turns
        assert summary.overall_asr == pytest.approx(0.25)

    def test_max_severity_across_sessions(self) -> None:
        s1 = _make_session(max_severity=7.0)
        s2 = _make_session(max_severity=9.0)
        summary = aggregate_sessions("exp-1", [s1, s2])
        assert summary.max_severity == 9.0

    def test_category_breakdown_merged(self) -> None:
        s1 = _make_session(category_breakdown={"prompt_injection": 3, "encoding": 1})
        s2 = _make_session(category_breakdown={"prompt_injection": 2, "roleplay": 4})
        summary = aggregate_sessions("exp-1", [s1, s2])
        assert summary.category_breakdown["prompt_injection"] == 5
        assert summary.category_breakdown["encoding"] == 1
        assert summary.category_breakdown["roleplay"] == 4


class TestExperimentSummary:
    def test_frozen_dataclass(self) -> None:
        summary = aggregate_sessions("exp-1", [])
        with pytest.raises(AttributeError):
            summary.experiment_id = "exp-2"  # type: ignore[misc]


class TestStrategyEffectiveness:
    def test_frozen_dataclass(self) -> None:
        se = StrategyEffectiveness(
            strategy_name="pair",
            total_attempts=10,
            total_jailbreaks=3,
            asr=0.3,
            avg_severity=6.0,
            avg_turns_to_jailbreak=3.5,
            cost_per_jailbreak=0.05,
        )
        assert se.strategy_name == "pair"
        with pytest.raises(AttributeError):
            se.asr = 0.5  # type: ignore[misc]
