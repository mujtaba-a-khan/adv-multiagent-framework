"""Tests for scoring metrics computation."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.scoring.metrics import (
    SessionMetrics,
    compute_session_metrics,
)


class TestComputeSessionMetrics:
    def test_empty_turns(self) -> None:
        metrics = compute_session_metrics([])
        assert metrics.total_turns == 0
        assert metrics.total_jailbreaks == 0
        assert metrics.asr == 0.0
        assert metrics.avg_severity == 0.0
        assert metrics.cost_per_jailbreak == 0.0
        assert metrics.strategies_used == []
        assert metrics.category_breakdown == {}

    def test_all_refused(self) -> None:
        turns = [
            {
                "judge_verdict": JudgeVerdict.REFUSED,
                "strategy_name": "pair",
                "severity_score": None,
            }
            for _ in range(5)
        ]
        metrics = compute_session_metrics(turns)
        assert metrics.total_turns == 5
        assert metrics.total_refused == 5
        assert metrics.total_jailbreaks == 0
        assert metrics.asr == 0.0

    def test_mixed_verdicts(self) -> None:
        turns = [
            {
                "judge_verdict": JudgeVerdict.JAILBREAK,
                "severity_score": 8.0,
                "specificity_score": 7.0,
                "coherence_score": 9.0,
                "strategy_name": "pair",
                "vulnerability_category": "prompt_injection",
                "target_blocked": False,
            },
            {
                "judge_verdict": JudgeVerdict.REFUSED,
                "severity_score": None,
                "specificity_score": None,
                "coherence_score": None,
                "strategy_name": "pair",
                "vulnerability_category": None,
                "target_blocked": False,
            },
            {
                "judge_verdict": JudgeVerdict.JAILBREAK,
                "severity_score": 6.0,
                "specificity_score": 5.0,
                "coherence_score": 7.0,
                "strategy_name": "encoding",
                "vulnerability_category": "encoding_bypass",
                "target_blocked": False,
            },
        ]
        metrics = compute_session_metrics(turns, total_cost=0.30, total_tokens=1000)

        assert metrics.total_turns == 3
        assert metrics.total_jailbreaks == 2
        assert metrics.total_refused == 1
        assert metrics.asr == pytest.approx(2 / 3)
        assert metrics.avg_severity == pytest.approx(7.0)
        assert metrics.avg_specificity == pytest.approx(6.0)
        assert metrics.avg_coherence == pytest.approx(8.0)
        assert metrics.max_severity == 8.0
        assert metrics.total_cost_usd == 0.30
        assert metrics.total_tokens == 1000
        assert metrics.cost_per_jailbreak == pytest.approx(0.15)

    def test_blocked_turns(self) -> None:
        turns = [
            {
                "judge_verdict": JudgeVerdict.REFUSED,
                "target_blocked": True,
                "strategy_name": "pair",
            },
        ]
        metrics = compute_session_metrics(turns)
        assert metrics.total_blocked == 1

    def test_error_turns(self) -> None:
        turns = [
            {
                "judge_verdict": JudgeVerdict.ERROR,
                "strategy_name": "pair",
            },
        ]
        metrics = compute_session_metrics(turns)
        assert metrics.total_errors == 1

    def test_strategies_used(self) -> None:
        turns = [
            {"judge_verdict": JudgeVerdict.REFUSED, "strategy_name": "pair"},
            {"judge_verdict": JudgeVerdict.REFUSED, "strategy_name": "encoding"},
            {"judge_verdict": JudgeVerdict.REFUSED, "strategy_name": "pair"},
        ]
        metrics = compute_session_metrics(turns)
        assert set(metrics.strategies_used) == {"pair", "encoding"}

    def test_category_breakdown(self) -> None:
        turns = [
            {
                "judge_verdict": JudgeVerdict.JAILBREAK,
                "vulnerability_category": "prompt_injection",
                "severity_score": 5.0,
                "specificity_score": 5.0,
                "coherence_score": 5.0,
                "strategy_name": "pair",
            },
            {
                "judge_verdict": JudgeVerdict.JAILBREAK,
                "vulnerability_category": "prompt_injection",
                "severity_score": 6.0,
                "specificity_score": 6.0,
                "coherence_score": 6.0,
                "strategy_name": "encoding",
            },
            {
                "judge_verdict": JudgeVerdict.JAILBREAK,
                "vulnerability_category": "encoding_bypass",
                "severity_score": 4.0,
                "specificity_score": 4.0,
                "coherence_score": 4.0,
                "strategy_name": "encoding",
            },
        ]
        metrics = compute_session_metrics(turns)
        assert metrics.category_breakdown == {
            "prompt_injection": 2,
            "encoding_bypass": 1,
        }

    def test_verdict_as_string(self) -> None:
        """Test that string verdicts are handled correctly."""
        turns = [
            {
                "judge_verdict": "jailbreak",
                "severity_score": 5.0,
                "specificity_score": 5.0,
                "coherence_score": 5.0,
                "strategy_name": "pair",
            },
        ]
        metrics = compute_session_metrics(turns)
        assert metrics.total_jailbreaks == 1

    def test_zero_jailbreaks_cost_per_jailbreak(self) -> None:
        turns = [
            {"judge_verdict": JudgeVerdict.REFUSED, "strategy_name": "pair"},
        ]
        metrics = compute_session_metrics(turns, total_cost=1.0)
        assert metrics.cost_per_jailbreak == 0.0


class TestSessionMetrics:
    def test_frozen_dataclass(self) -> None:
        metrics = compute_session_metrics([])
        with pytest.raises(AttributeError):
            metrics.total_turns = 99  # type: ignore[misc]
