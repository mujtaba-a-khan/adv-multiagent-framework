"""HarmBench-compatible evaluation layer

Converts internal evaluation results to HarmBench format for
standardized comparison with published benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adversarial_framework.core.constants import JudgeVerdict


@dataclass(frozen=True)
class HarmBenchResult:
    """A single evaluation result in HarmBench-compatible format.

    Follows the HarmBench schema:
    - behavior_id: identifier for the harmful behavior tested
    - test_case: the adversarial prompt used
    - generation: the target model's response
    - label: 1 = jailbreak success, 0 = refusal/safe
    - method: attack strategy used
    - model: target model identifier
    """

    behavior_id: str
    test_case: str
    generation: str
    label: int
    method: str
    model: str
    severity: float = 0.0
    specificity: float = 0.0
    coherence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def to_harmbench_result(
    turn: dict[str, Any],
    behavior_id: str = "",
    model: str = "unknown",
) -> HarmBenchResult:
    """Convert an internal turn record to a HarmBenchResult.

    Args:
        turn: Internal turn dict with verdict, scores, etc.
        behavior_id: HarmBench behavior identifier.
        model: Target model name/identifier.

    Returns:
        HarmBenchResult with mapped fields.
    """
    verdict = turn.get("judge_verdict")
    if isinstance(verdict, JudgeVerdict):
        is_jailbreak = verdict == JudgeVerdict.JAILBREAK
    else:
        is_jailbreak = str(verdict) == JudgeVerdict.JAILBREAK.value

    return HarmBenchResult(
        behavior_id=behavior_id or turn.get("behavior_id", "unknown"),
        test_case=turn.get("attack_prompt", ""),
        generation=turn.get("target_response", ""),
        label=1 if is_jailbreak else 0,
        method=turn.get("strategy_name", "unknown"),
        model=model,
        severity=turn.get("severity_score", 0.0) or 0.0,
        specificity=turn.get("specificity_score", 0.0) or 0.0,
        coherence=turn.get("coherence_score", 0.0) or 0.0,
        metadata={
            k: v
            for k, v in turn.items()
            if k
            not in {
                "attack_prompt",
                "target_response",
                "judge_verdict",
                "strategy_name",
                "severity_score",
                "specificity_score",
                "coherence_score",
                "behavior_id",
            }
        },
    )


def compute_harmbench_asr(results: list[HarmBenchResult]) -> dict[str, Any]:
    """Compute HarmBench-style aggregate metrics.

    Returns:
        Dict with ASR, per-method ASR, per-behavior ASR, and counts.
    """
    if not results:
        return {
            "total": 0,
            "successful": 0,
            "asr": 0.0,
            "per_method": {},
            "per_behavior": {},
        }

    total = len(results)
    successful = sum(r.label for r in results)

    # Per-method breakdown
    method_counts: dict[str, dict[str, int]] = {}
    for r in results:
        if r.method not in method_counts:
            method_counts[r.method] = {"total": 0, "successful": 0}
        method_counts[r.method]["total"] += 1
        method_counts[r.method]["successful"] += r.label

    per_method = {
        method: {
            "total": counts["total"],
            "successful": counts["successful"],
            "asr": counts["successful"] / counts["total"] if counts["total"] > 0 else 0.0,
        }
        for method, counts in method_counts.items()
    }

    # Per-behavior breakdown
    behavior_counts: dict[str, dict[str, int]] = {}
    for r in results:
        if r.behavior_id not in behavior_counts:
            behavior_counts[r.behavior_id] = {"total": 0, "successful": 0}
        behavior_counts[r.behavior_id]["total"] += 1
        behavior_counts[r.behavior_id]["successful"] += r.label

    per_behavior = {
        bid: {
            "total": counts["total"],
            "successful": counts["successful"],
            "asr": counts["successful"] / counts["total"] if counts["total"] > 0 else 0.0,
        }
        for bid, counts in behavior_counts.items()
    }

    return {
        "total": total,
        "successful": successful,
        "asr": successful / total,
        "per_method": per_method,
        "per_behavior": per_behavior,
    }


def export_harmbench_jsonl(results: list[HarmBenchResult]) -> list[dict[str, Any]]:
    """Export results as HarmBench-compatible JSONL records.

    Returns:
        List of dicts ready for JSONL serialization.
    """
    return [
        {
            "behavior_id": r.behavior_id,
            "test_case": r.test_case,
            "generation": r.generation,
            "label": r.label,
            "method": r.method,
            "model": r.model,
            "severity": r.severity,
            "specificity": r.specificity,
            "coherence": r.coherence,
        }
        for r in results
    ]
