"""Multi-metric aggregation across sessions and experiments.

Aggregates SessionMetrics into experiment-level summaries with
strategy effectiveness rankings, defense bypass analysis, and
cost efficiency breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adversarial_framework.scoring.metrics import SessionMetrics

@dataclass(frozen=True)
class StrategyEffectiveness:
    """Effectiveness summary for a single attack strategy."""

    strategy_name: str
    total_attempts: int
    total_jailbreaks: int
    asr: float
    avg_severity: float
    avg_turns_to_jailbreak: float
    cost_per_jailbreak: float

@dataclass(frozen=True)
class ExperimentSummary:
    """Aggregate summary across all sessions in an experiment."""

    experiment_id: str
    total_sessions: int
    total_turns: int
    total_jailbreaks: int
    total_refused: int
    total_blocked: int
    total_errors: int
    overall_asr: float
    avg_severity: float
    avg_specificity: float
    avg_coherence: float
    max_severity: float
    total_cost_usd: float
    total_tokens: int
    cost_per_jailbreak: float
    strategy_effectiveness: list[StrategyEffectiveness]
    category_breakdown: dict[str, int]
    defense_bypass_rate: float  # Fraction of attacks that bypassed active defenses
    top_strategies: list[str]  # Ordered by ASR descending

def aggregate_sessions(
    experiment_id: str,
    sessions: list[SessionMetrics],
    session_details: list[dict[str, Any]] | None = None,
) -> ExperimentSummary:
    """Aggregate multiple SessionMetrics into an ExperimentSummary.

    Args:
        experiment_id: Identifier for the experiment.
        sessions: List of SessionMetrics from completed sessions.
        session_details: Optional list of dicts with per-session extra data
            (e.g., turns_to_first_jailbreak, defenses_active).

    Returns:
        ExperimentSummary with all aggregated values.
    """
    if not sessions:
        return ExperimentSummary(
            experiment_id=experiment_id,
            total_sessions=0,
            total_turns=0,
            total_jailbreaks=0,
            total_refused=0,
            total_blocked=0,
            total_errors=0,
            overall_asr=0.0,
            avg_severity=0.0,
            avg_specificity=0.0,
            avg_coherence=0.0,
            max_severity=0.0,
            total_cost_usd=0.0,
            total_tokens=0,
            cost_per_jailbreak=0.0,
            strategy_effectiveness=[],
            category_breakdown={},
            defense_bypass_rate=0.0,
            top_strategies=[],
        )

    total_turns = sum(s.total_turns for s in sessions)
    total_jailbreaks = sum(s.total_jailbreaks for s in sessions)
    total_refused = sum(s.total_refused for s in sessions)
    total_blocked = sum(s.total_blocked for s in sessions)
    total_errors = sum(s.total_errors for s in sessions)
    total_cost = sum(s.total_cost_usd for s in sessions)
    total_tokens = sum(s.total_tokens for s in sessions)

    # Weighted average of scores (weighted by number of turns per session)
    def _weighted_avg(attr: str) -> float:
        numerator = sum(getattr(s, attr) * s.total_turns for s in sessions)
        return numerator / total_turns if total_turns > 0 else 0.0

    # Merge category breakdowns
    merged_categories: dict[str, int] = {}
    for s in sessions:
        for cat, count in s.category_breakdown.items():
            merged_categories[cat] = merged_categories.get(cat, 0) + count

    # Strategy effectiveness (per-strategy aggregation)
    strategy_stats: dict[str, dict[str, Any]] = {}
    for idx, s in enumerate(sessions):
        details = (session_details[idx] if session_details and idx < len(session_details) else {})
        for strat in s.strategies_used:
            if strat not in strategy_stats:
                strategy_stats[strat] = {
                    "attempts": 0,
                    "jailbreaks": 0,
                    "severity_sum": 0.0,
                    "severity_count": 0,
                    "cost": 0.0,
                    "turns_to_jb_sum": 0.0,
                    "turns_to_jb_count": 0,
                }
            stats = strategy_stats[strat]
            stats["attempts"] += s.total_turns
            stats["jailbreaks"] += s.total_jailbreaks
            if s.avg_severity > 0:
                stats["severity_sum"] += s.avg_severity * s.total_turns
                stats["severity_count"] += s.total_turns
            stats["cost"] += s.total_cost_usd
            turns_to_jb = details.get("turns_to_first_jailbreak", 0)
            if turns_to_jb > 0:
                stats["turns_to_jb_sum"] += turns_to_jb
                stats["turns_to_jb_count"] += 1

    effectiveness = []
    for strat, stats in strategy_stats.items():
        asr = stats["jailbreaks"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
        avg_sev = stats["severity_sum"] / stats["severity_count"] if stats["severity_count"] > 0 else 0.0
        avg_turns = stats["turns_to_jb_sum"] / stats["turns_to_jb_count"] if stats["turns_to_jb_count"] > 0 else 0.0
        cpj = stats["cost"] / stats["jailbreaks"] if stats["jailbreaks"] > 0 else 0.0
        effectiveness.append(
            StrategyEffectiveness(
                strategy_name=strat,
                total_attempts=stats["attempts"],
                total_jailbreaks=stats["jailbreaks"],
                asr=asr,
                avg_severity=avg_sev,
                avg_turns_to_jailbreak=avg_turns,
                cost_per_jailbreak=cpj,
            )
        )

    # Sort by ASR descending
    effectiveness.sort(key=lambda e: e.asr, reverse=True)
    top_strategies = [e.strategy_name for e in effectiveness[:5]]

    # Defense bypass rate: jailbreaks / (jailbreaks + blocked)
    defended_total = total_jailbreaks + total_blocked
    defense_bypass_rate = total_jailbreaks / defended_total if defended_total > 0 else 0.0

    return ExperimentSummary(
        experiment_id=experiment_id,
        total_sessions=len(sessions),
        total_turns=total_turns,
        total_jailbreaks=total_jailbreaks,
        total_refused=total_refused,
        total_blocked=total_blocked,
        total_errors=total_errors,
        overall_asr=total_jailbreaks / total_turns if total_turns > 0 else 0.0,
        avg_severity=_weighted_avg("avg_severity"),
        avg_specificity=_weighted_avg("avg_specificity"),
        avg_coherence=_weighted_avg("avg_coherence"),
        max_severity=max(s.max_severity for s in sessions),
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        cost_per_jailbreak=total_cost / total_jailbreaks if total_jailbreaks > 0 else 0.0,
        strategy_effectiveness=effectiveness,
        category_breakdown=merged_categories,
        defense_bypass_rate=defense_bypass_rate,
        top_strategies=top_strategies,
    )
