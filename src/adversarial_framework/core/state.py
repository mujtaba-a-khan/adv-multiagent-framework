"""Central LangGraph state for the adversarial attack cycle.

This module defines the `AdversarialState` TypedDict that flows through every
node in the LangGraph graph.  Fields use `Annotated` types with reducer
functions per LangGraph best practices — every state update creates a new
immutable version that is checkpointable.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from adversarial_framework.core.constants import JudgeVerdict, SessionStatus

# Nested Data Structures


@dataclass(frozen=True)
class AttackTurn:
    """Immutable record of a single attack-response-analysis cycle."""

    turn_number: int
    strategy_name: str
    strategy_params: dict[str, Any]
    attack_prompt: str
    target_response: str
    judge_verdict: JudgeVerdict
    confidence: float  # 0.0 – 1.0
    severity: float  # 1 – 10
    specificity: float  # 1 – 10
    vulnerability_category: str | None
    attack_technique: str | None
    target_blocked: bool
    attacker_tokens: int
    target_tokens: int
    analyzer_tokens: int
    raw_target_response: str | None = None
    attacker_reasoning: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class DefenseAction:
    """Record of a defense generated or applied by the Defender agent."""

    defense_type: str  # "rule_based", "prompt_patch", "ml_classifier", …
    defense_config: dict[str, Any]
    triggered_by_turn: int
    rationale: str
    block_rate: float | None = None
    false_positive_rate: float | None = None


@dataclass
class TokenBudget:
    """Accumulated token usage across all agents."""

    total_attacker_tokens: int = 0
    total_target_tokens: int = 0
    total_analyzer_tokens: int = 0
    total_defender_tokens: int = 0
    estimated_cost_usd: float = 0.0


# Custom Reducers


def append_turns(existing: list[AttackTurn], new: list[AttackTurn]) -> list[AttackTurn]:
    """Append-only reducer: new turns are appended to history."""
    return existing + new


def append_defenses(existing: list[DefenseAction], new: list[DefenseAction]) -> list[DefenseAction]:
    """Append-only reducer: new defenses are appended."""
    return existing + new


def merge_budget(existing: TokenBudget, new: TokenBudget) -> TokenBudget:
    """Accumulate token usage across agents."""
    return TokenBudget(
        total_attacker_tokens=existing.total_attacker_tokens + new.total_attacker_tokens,
        total_target_tokens=existing.total_target_tokens + new.total_target_tokens,
        total_analyzer_tokens=existing.total_analyzer_tokens + new.total_analyzer_tokens,
        total_defender_tokens=existing.total_defender_tokens + new.total_defender_tokens,
        estimated_cost_usd=existing.estimated_cost_usd + new.estimated_cost_usd,
    )


def latest(existing: Any, new: Any) -> Any:  # noqa: ANN401
    """Last-write-wins reducer for scalar fields."""
    return new if new is not None else existing


# Main State TypedDict


class AdversarialState(TypedDict):
    """Central LangGraph state for the adversarial attack cycle.

    Every agent node reads from and writes partial updates to this state.
    Reducers control how concurrent/sequential updates are merged.
    """

    # Identity
    experiment_id: str
    session_id: str

    # Configuration (set once at initialisation, read-only during run)
    target_model: str  # Ollama tag, e.g. "llama3:8b"
    target_provider: str  # "ollama", "openai", "anthropic", …
    target_system_prompt: Annotated[str | None, latest]
    attacker_model: str
    analyzer_model: str
    defender_model: str
    attack_objective: str  # The behaviour we want to elicit
    strategy_config: dict[str, Any]  # Selected strategy + params

    # Defense Configuration (set once at initialisation, read-only during run)
    session_mode: str  # "attack" or "defense"
    initial_defense_config: list[dict[str, Any]]  # defense names + params to load at start
    defense_enabled: bool  # True when session_mode == "defense"

    # Execution Control
    status: Annotated[SessionStatus, latest]
    current_turn: Annotated[int, latest]
    max_turns: int
    max_cost_usd: float

    # Current Turn Data (overwritten each turn via `latest`)
    selected_strategy: Annotated[str | None, latest]
    strategy_params: Annotated[dict[str, Any] | None, latest]
    planning_notes: Annotated[str | None, latest]
    current_attack_prompt: Annotated[str | None, latest]
    attacker_reasoning: Annotated[str | None, latest]
    target_response: Annotated[str | None, latest]
    target_blocked: Annotated[bool, latest]
    raw_target_response: Annotated[str | None, latest]

    # Analysis Results (current turn)
    judge_verdict: Annotated[JudgeVerdict | None, latest]
    judge_confidence: Annotated[float | None, latest]
    judge_reason: Annotated[str | None, latest]
    severity_score: Annotated[float | None, latest]
    specificity_score: Annotated[float | None, latest]
    coherence_score: Annotated[float | None, latest]
    vulnerability_category: Annotated[str | None, latest]
    attack_technique: Annotated[str | None, latest]

    # Accumulated History (append-only via reducers)
    attack_history: Annotated[list[AttackTurn], append_turns]
    defense_actions: Annotated[list[DefenseAction], append_defenses]
    token_budget: Annotated[TokenBudget, merge_budget]

    # Counters
    total_jailbreaks: Annotated[int, operator.add]
    total_borderline: Annotated[int, operator.add]
    total_blocked: Annotated[int, operator.add]
    total_refused: Annotated[int, operator.add]

    # Strategy State (persisted across turns for refinement cycling)
    strategy_metadata: Annotated[dict[str, Any] | None, latest]

    # Multi-Turn Conversation State
    conversation_history: Annotated[list[dict[str, str]], operator.add]

    # Human-in-the-Loop
    human_feedback: Annotated[str | None, latest]
    awaiting_human: Annotated[bool, latest]

    # Error Handling
    last_error: Annotated[str | None, latest]
    error_count: Annotated[int, operator.add]
