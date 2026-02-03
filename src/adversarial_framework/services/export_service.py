"""Data export service — CSV and JSONL formats.

Exports session turn data, findings, and metrics for external analysis.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from adversarial_framework.core.constants import JudgeVerdict


def export_turns_csv(turns: list[dict[str, Any]]) -> str:
    """Export turn records as CSV.

    Args:
        turns: List of turn record dicts.

    Returns:
        CSV string with headers.
    """
    if not turns:
        return ""

    fieldnames = [
        "turn_number",
        "strategy_name",
        "attack_prompt",
        "target_response",
        "target_blocked",
        "judge_verdict",
        "judge_confidence",
        "severity_score",
        "specificity_score",
        "coherence_score",
        "vulnerability_category",
        "attack_technique",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for t in turns:
        row = {k: t.get(k, "") for k in fieldnames}
        verdict = row.get("judge_verdict")
        if isinstance(verdict, JudgeVerdict):
            row["judge_verdict"] = verdict.value
        writer.writerow(row)

    return output.getvalue()


def export_turns_jsonl(turns: list[dict[str, Any]]) -> str:
    """Export turn records as JSONL (one JSON object per line).

    Args:
        turns: List of turn record dicts.

    Returns:
        JSONL string.
    """
    lines = []
    for t in turns:
        record = dict(t)
        # Serialize enums
        for key in ("judge_verdict",):
            val = record.get(key)
            if isinstance(val, JudgeVerdict):
                record[key] = val.value
        lines.append(json.dumps(record, default=str))
    return "\n".join(lines)


def export_findings_csv(findings: list[dict[str, Any]]) -> str:
    """Export vulnerability findings as CSV.

    Args:
        findings: List of finding dicts.

    Returns:
        CSV string with headers.
    """
    if not findings:
        return ""

    fieldnames = [
        "turn_number",
        "strategy_name",
        "vulnerability_category",
        "severity",
        "specificity",
        "technique_label",
        "attack_prompt_preview",
        "response_preview",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for f in findings:
        writer.writerow({k: f.get(k, "") for k in fieldnames})
    return output.getvalue()
