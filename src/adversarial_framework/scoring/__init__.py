"""Scoring module for multi-dimensional attack evaluation.

Provides metrics computation, scoring rubrics, HarmBench-compatible
evaluation, and multi-session aggregation.
"""

from adversarial_framework.scoring.aggregator import (
    ExperimentSummary,
    StrategyEffectiveness,
    aggregate_sessions,
)
from adversarial_framework.scoring.harmbench_compat import (
    HarmBenchResult,
    compute_harmbench_asr,
    export_harmbench_jsonl,
    to_harmbench_result,
)
from adversarial_framework.scoring.metrics import (
    SessionMetrics,
    compute_session_metrics,
)
from adversarial_framework.scoring.rubrics import (
    ALL_RUBRICS,
    COHERENCE_RUBRIC,
    SEVERITY_RUBRIC,
    SPECIFICITY_RUBRIC,
    ScoreLevel,
    ScoringRubric,
    format_all_rubrics_for_prompt,
    format_rubric_for_prompt,
)

__all__ = [
    "ALL_RUBRICS",
    "COHERENCE_RUBRIC",
    "ExperimentSummary",
    "HarmBenchResult",
    "SEVERITY_RUBRIC",
    "SPECIFICITY_RUBRIC",
    "ScoreLevel",
    "ScoringRubric",
    "SessionMetrics",
    "StrategyEffectiveness",
    "aggregate_sessions",
    "compute_harmbench_asr",
    "compute_session_metrics",
    "export_harmbench_jsonl",
    "format_all_rubrics_for_prompt",
    "format_rubric_for_prompt",
    "to_harmbench_result",
]
