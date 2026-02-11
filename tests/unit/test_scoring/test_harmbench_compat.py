"""Tests for HarmBench compatibility layer."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.scoring.harmbench_compat import (
    HarmBenchResult,
    compute_harmbench_asr,
    export_harmbench_jsonl,
    to_harmbench_result,
)


class TestToHarmBenchResult:
    def test_jailbreak_turn(self) -> None:
        turn = {
            "attack_prompt": "test attack",
            "target_response": "harmful response",
            "judge_verdict": JudgeVerdict.JAILBREAK,
            "strategy_name": "pair",
            "severity_score": 8.0,
            "specificity_score": 7.0,
            "coherence_score": 9.0,
        }
        result = to_harmbench_result(turn, behavior_id="B001", model="llama3:8b")
        assert result.label == 1
        assert result.behavior_id == "B001"
        assert result.model == "llama3:8b"
        assert result.method == "pair"
        assert result.severity == 8.0
        assert result.test_case == "test attack"
        assert result.generation == "harmful response"

    def test_refused_turn(self) -> None:
        turn = {
            "attack_prompt": "test",
            "target_response": "I cannot help with that.",
            "judge_verdict": JudgeVerdict.REFUSED,
            "strategy_name": "encoding",
        }
        result = to_harmbench_result(turn)
        assert result.label == 0

    def test_missing_scores_default_to_zero(self) -> None:
        turn = {
            "attack_prompt": "test",
            "target_response": "response",
            "judge_verdict": JudgeVerdict.JAILBREAK,
            "strategy_name": "pair",
        }
        result = to_harmbench_result(turn)
        assert result.severity == 0.0
        assert result.specificity == 0.0
        assert result.coherence == 0.0

    def test_default_behavior_id_and_model(self) -> None:
        turn = {
            "attack_prompt": "test",
            "target_response": "resp",
            "judge_verdict": JudgeVerdict.REFUSED,
            "strategy_name": "pair",
        }
        result = to_harmbench_result(turn)
        # Empty behavior_id falls through to turn.get("behavior_id", "unknown")
        assert result.behavior_id == "unknown"
        assert result.model == "unknown"

    def test_frozen_result(self) -> None:
        result = HarmBenchResult(
            behavior_id="B001",
            test_case="tc",
            generation="gen",
            label=1,
            method="pair",
            model="test",
        )
        with pytest.raises(AttributeError):
            result.label = 0  # type: ignore[misc]


class TestComputeHarmBenchASR:
    def test_empty_results(self) -> None:
        asr = compute_harmbench_asr([])
        assert asr["total"] == 0
        assert asr["successful"] == 0
        assert asr["asr"] == 0.0

    def test_all_jailbreaks(self) -> None:
        results = [
            HarmBenchResult(
                behavior_id=f"B{i}",
                test_case="tc",
                generation="gen",
                label=1,
                method="pair",
                model="test",
            )
            for i in range(5)
        ]
        asr = compute_harmbench_asr(results)
        assert asr["total"] == 5
        assert asr["successful"] == 5
        assert asr["asr"] == 1.0

    def test_mixed_results(self) -> None:
        results = [
            HarmBenchResult(
                behavior_id="B1",
                test_case="tc",
                generation="gen",
                label=1,
                method="pair",
                model="m",
            ),
            HarmBenchResult(
                behavior_id="B2",
                test_case="tc",
                generation="gen",
                label=0,
                method="pair",
                model="m",
            ),
            HarmBenchResult(
                behavior_id="B3",
                test_case="tc",
                generation="gen",
                label=1,
                method="encoding",
                model="m",
            ),
        ]
        asr = compute_harmbench_asr(results)
        assert asr["total"] == 3
        assert asr["successful"] == 2
        assert asr["asr"] == pytest.approx(2 / 3)

    def test_per_method_breakdown(self) -> None:
        results = [
            HarmBenchResult(
                behavior_id="B1",
                test_case="tc",
                generation="gen",
                label=1,
                method="pair",
                model="m",
            ),
            HarmBenchResult(
                behavior_id="B2",
                test_case="tc",
                generation="gen",
                label=0,
                method="pair",
                model="m",
            ),
            HarmBenchResult(
                behavior_id="B3",
                test_case="tc",
                generation="gen",
                label=1,
                method="encoding",
                model="m",
            ),
        ]
        asr = compute_harmbench_asr(results)
        assert "per_method" in asr
        per_method = asr["per_method"]
        assert per_method["pair"]["asr"] == pytest.approx(0.5)
        assert per_method["encoding"]["asr"] == pytest.approx(1.0)


class TestExportHarmBenchJsonl:
    def test_export_format(self) -> None:
        results = [
            HarmBenchResult(
                behavior_id="B1",
                test_case="attack prompt",
                generation="model response",
                label=1,
                method="pair",
                model="llama3:8b",
                severity=8.0,
                specificity=7.0,
                coherence=9.0,
            ),
        ]
        exported = export_harmbench_jsonl(results)
        assert len(exported) == 1
        row = exported[0]
        assert row["behavior_id"] == "B1"
        assert row["label"] == 1
        assert row["method"] == "pair"
        assert row["model"] == "llama3:8b"

    def test_export_empty(self) -> None:
        exported = export_harmbench_jsonl([])
        assert exported == []
