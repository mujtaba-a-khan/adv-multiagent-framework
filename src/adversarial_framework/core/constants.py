"""Constants and enums used throughout the adversarial framework."""

from __future__ import annotations

from enum import Enum

# Session & Execution Status


class SessionStatus(str, Enum):
    """Status of an adversarial session."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JudgeVerdict(str, Enum):
    """Verdict from the Analyzer's judge sub-node."""

    JAILBREAK = "jailbreak"
    BORDERLINE = "borderline"
    REFUSED = "refused"
    ERROR = "error"


# Attack Strategy Categories


class AttackCategory(str, Enum):
    """Top-level category for attack strategies."""

    PROMPT_LEVEL = "prompt_level"
    OPTIMIZATION = "optimization"
    MULTI_TURN = "multi_turn"
    ADVANCED = "advanced"
    COMPOSITE = "composite"


# Defense Types


class SessionMode(str, Enum):
    """Mode of a session: attack-only or defense-enabled."""

    ATTACK = "attack"
    DEFENSE = "defense"


class DefenseType(str, Enum):
    """Type of defense mechanism."""

    RULE_BASED = "rule_based"
    PROMPT_PATCH = "prompt_patch"
    ML_CLASSIFIER = "ml_classifier"
    LLM_JUDGE = "llm_judge"
    LAYERED = "layered"
    SEMANTIC_GUARD = "semantic_guard"
    TWO_PASS = "two_pass"


# LangGraph Node Names

INITIALIZE_NODE = "initialize"
PLANNER_NODE = "planner"
ATTACKER_NODE = "attacker"
VALIDATOR_NODE = "validator"
TARGET_NODE = "target"
ANALYZER_NODE = "analyzer"
JUDGE_NODE = "judge"
CLASSIFIER_NODE = "classifier"
SCORER_NODE = "scorer"
ROUTER_NODE = "router"
DEFENDER_NODE = "defender"
HUMAN_REVIEW_NODE = "human_review"
FINALIZE_NODE = "finalize"


# Routing Decisions

ROUTE_CONTINUE = "continue"
ROUTE_REFINE = "refine"
ROUTE_DEFEND = "defend"
ROUTE_HUMAN_REVIEW = "human_review"
ROUTE_END = "end"


# Defaults

DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_COST_USD = 10.0
DEFAULT_MAX_ERRORS = 5
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
