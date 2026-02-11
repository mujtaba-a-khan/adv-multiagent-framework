"""Tests for report generation service."""

from __future__ import annotations

import json

from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.services.report_service import (
    Finding,
    export_report_json,
    export_report_summary,
    generate_report,
)


def _make_jailbreak_turn(
    turn_number: int = 1,
    strategy: str = "pair",
    severity: float = 7.0,
    specificity: float = 6.0,
    coherence: float = 8.0,
) -> dict:
    return {
        "turn_number": turn_number,
        "strategy_name": strategy,
        "attack_prompt": f"attack prompt {turn_number}",
        "target_response": f"harmful response {turn_number}",
        "judge_verdict": JudgeVerdict.JAILBREAK,
        "judge_confidence": 0.95,
        "severity_score": severity,
        "specificity_score": specificity,
        "coherence_score": coherence,
        "vulnerability_category": "prompt_injection",
        "attack_technique": strategy,
        "target_blocked": False,
    }


def _make_refused_turn(turn_number: int = 1) -> dict:
    return {
        "turn_number": turn_number,
        "strategy_name": "pair",
        "attack_prompt": f"attack prompt {turn_number}",
        "target_response": "I cannot help with that.",
        "judge_verdict": JudgeVerdict.REFUSED,
        "judge_confidence": 0.99,
        "severity_score": None,
        "specificity_score": None,
        "coherence_score": None,
        "vulnerability_category": None,
        "attack_technique": None,
        "target_blocked": False,
    }


class TestGenerateReport:
    def test_basic_report_structure(self) -> None:
        turns = [_make_refused_turn(1), _make_jailbreak_turn(2)]
        report = generate_report(
            session_id="sess-1",
            experiment_name="Test Experiment",
            target_model="llama3:8b",
            turns=turns,
            total_cost=0.05,
            total_tokens=500,
        )
        assert report.session_id == "sess-1"
        assert report.experiment_name == "Test Experiment"
        assert report.target_model == "llama3:8b"
        assert report.report_id != ""
        assert report.generated_at != ""

    def test_metrics_computed(self) -> None:
        turns = [
            _make_refused_turn(1),
            _make_jailbreak_turn(2),
            _make_refused_turn(3),
        ]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        assert report.metrics.total_turns == 3
        assert report.metrics.total_jailbreaks == 1
        assert report.metrics.total_refused == 2

    def test_findings_extracted(self) -> None:
        turns = [
            _make_refused_turn(1),
            _make_jailbreak_turn(2, severity=8.0),
            _make_jailbreak_turn(3, strategy="encoding", severity=6.0),
        ]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        # Should have 2 findings (only jailbreaks)
        assert len(report.findings) == 2
        assert report.findings[0].turn_number == 2
        assert report.findings[0].severity == 8.0

    def test_recommendations_generated(self) -> None:
        turns = [_make_jailbreak_turn(1)]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        assert len(report.recommendations) > 0

    def test_empty_turns(self) -> None:
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=[],
        )
        assert report.metrics.total_turns == 0
        assert report.findings == []

    def test_strategy_breakdown_populated(self) -> None:
        turns = [
            _make_jailbreak_turn(1, strategy="pair"),
            _make_jailbreak_turn(2, strategy="pair"),
            _make_jailbreak_turn(3, strategy="encoding"),
        ]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        assert "pair" in report.strategy_breakdown
        assert "encoding" in report.strategy_breakdown

    def test_timeline_populated(self) -> None:
        turns = [_make_refused_turn(1), _make_jailbreak_turn(2)]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        assert len(report.timeline) == 2


class TestExportReportJson:
    def test_valid_json_output(self) -> None:
        turns = [_make_jailbreak_turn(1)]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        json_str = export_report_json(report)
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "s1"
        assert "metrics" in parsed
        assert "findings" in parsed

    def test_json_includes_all_fields(self) -> None:
        turns = [_make_refused_turn(1)]
        report = generate_report(
            session_id="s1",
            experiment_name="exp",
            target_model="m",
            turns=turns,
        )
        json_str = export_report_json(report)
        parsed = json.loads(json_str)
        assert "report_id" in parsed
        assert "recommendations" in parsed
        assert "defense_summary" in parsed


class TestExportReportSummary:
    def test_summary_contains_key_info(self) -> None:
        turns = [_make_jailbreak_turn(1)]
        report = generate_report(
            session_id="s1",
            experiment_name="Test Experiment",
            target_model="llama3:8b",
            turns=turns,
        )
        summary = export_report_summary(report)
        assert "Test Experiment" in summary
        assert "llama3:8b" in summary


class TestFinding:
    def test_creation(self) -> None:
        finding = Finding(
            turn_number=1,
            strategy_name="pair",
            vulnerability_category="prompt_injection",
            severity=8.0,
            specificity=7.0,
            attack_prompt_preview="test...",
            response_preview="response...",
        )
        assert finding.turn_number == 1
        assert finding.technique_label == ""
