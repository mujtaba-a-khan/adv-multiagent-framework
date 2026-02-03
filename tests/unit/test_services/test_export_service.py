"""Tests for data export service."""

from __future__ import annotations

import csv
import io
import json

from adversarial_framework.services.export_service import (
    export_findings_csv,
    export_turns_csv,
    export_turns_jsonl,
)


class TestExportTurnsCsv:
    def test_basic_export(self) -> None:
        turns = [
            {
                "turn_number": 1,
                "strategy_name": "pair",
                "attack_prompt": "test prompt",
                "target_response": "test response",
                "target_blocked": False,
                "judge_verdict": "refused",
                "judge_confidence": 0.9,
                "severity_score": None,
                "specificity_score": None,
                "coherence_score": None,
                "vulnerability_category": None,
                "attack_technique": None,
            },
        ]
        result = export_turns_csv(turns)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["turn_number"] == "1"
        assert rows[0]["strategy_name"] == "pair"

    def test_csv_header_fields(self) -> None:
        # Need at least one row — empty list returns empty string
        turns = [
            {
                "turn_number": 1, "strategy_name": "pair",
                "attack_prompt": "p", "target_response": "r",
                "target_blocked": False, "judge_verdict": "refused",
                "judge_confidence": 0.9, "severity_score": None,
                "specificity_score": None, "coherence_score": None,
                "vulnerability_category": None, "attack_technique": None,
            },
        ]
        result = export_turns_csv(turns)
        reader = csv.DictReader(io.StringIO(result))
        expected_fields = {
            "turn_number", "strategy_name", "attack_prompt", "target_response",
            "target_blocked", "judge_verdict", "judge_confidence",
            "severity_score", "specificity_score", "coherence_score",
            "vulnerability_category", "attack_technique",
        }
        assert set(reader.fieldnames or []) == expected_fields

    def test_multiple_turns(self) -> None:
        turns = [
            {
                "turn_number": i,
                "strategy_name": "pair",
                "attack_prompt": f"prompt {i}",
                "target_response": f"response {i}",
                "target_blocked": False,
                "judge_verdict": "refused",
                "judge_confidence": 0.5,
                "severity_score": None,
                "specificity_score": None,
                "coherence_score": None,
                "vulnerability_category": None,
                "attack_technique": None,
            }
            for i in range(5)
        ]
        result = export_turns_csv(turns)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 5

    def test_empty_turns(self) -> None:
        result = export_turns_csv([])
        # Empty input returns empty string (no header)
        assert result == ""


class TestExportTurnsJsonl:
    def test_basic_export(self) -> None:
        turns = [
            {
                "turn_number": 1,
                "strategy_name": "pair",
                "attack_prompt": "test",
                "target_response": "resp",
            },
        ]
        result = export_turns_jsonl(turns)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["turn_number"] == 1

    def test_multiple_lines(self) -> None:
        turns = [
            {"turn_number": i, "data": f"turn-{i}"}
            for i in range(3)
        ]
        result = export_turns_jsonl(turns)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["turn_number"] == i

    def test_empty_turns(self) -> None:
        result = export_turns_jsonl([])
        assert result.strip() == ""

    def test_each_line_is_valid_json(self) -> None:
        turns = [
            {"turn_number": 1, "special_chars": 'quotes "inside" and\nnewlines'},
        ]
        result = export_turns_jsonl(turns)
        for line in result.strip().split("\n"):
            json.loads(line)  # Should not raise


class TestExportFindingsCsv:
    def test_basic_export(self) -> None:
        findings = [
            {
                "turn_number": 2,
                "strategy_name": "pair",
                "vulnerability_category": "prompt_injection",
                "severity": 8.0,
                "specificity": 7.0,
                "technique_label": "pair",
                "attack_prompt_preview": "attack...",
                "response_preview": "response...",
            },
        ]
        result = export_findings_csv(findings)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["severity"] == "8.0"

    def test_findings_header_fields(self) -> None:
        # Need at least one row — empty list returns empty string
        findings = [
            {
                "turn_number": 1, "strategy_name": "pair",
                "vulnerability_category": "test", "severity": 5.0,
                "specificity": 4.0, "technique_label": "pair",
                "attack_prompt_preview": "p", "response_preview": "r",
            },
        ]
        result = export_findings_csv(findings)
        reader = csv.DictReader(io.StringIO(result))
        expected_fields = {
            "turn_number", "strategy_name", "vulnerability_category",
            "severity", "specificity", "technique_label",
            "attack_prompt_preview", "response_preview",
        }
        assert set(reader.fieldnames or []) == expected_fields

    def test_empty_findings(self) -> None:
        result = export_findings_csv([])
        # Empty input returns empty string (no header)
        assert result == ""
