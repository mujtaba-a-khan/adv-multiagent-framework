"""Main LangGraph graph builder for the adversarial attack cycle.

Graph topology::

    START → initialize → attacker → target → analyzer → record_history → router
                 ↑                                                        │
                 │         ┌─ continue ───────────────────────────────────┘
                 │         ├─ defend → defender → apply_defenses ──┐
                 │         ├─ end → finalize → END
                 └─────────┘                                       └──→ attacker
"""

from __future__ import annotations

import structlog
from typing import Any

from langgraph.graph import StateGraph, END

from adversarial_framework.agents.attacker.agent import AttackerAgent
from adversarial_framework.agents.analyzer.agent import AnalyzerAgent
from adversarial_framework.agents.defender.agent import DefenderAgent
from adversarial_framework.agents.target.interface import TargetInterface
from adversarial_framework.agents.target.providers.base import BaseProvider
from adversarial_framework.core.constants import (
    ANALYZER_NODE,
    ATTACKER_NODE,
    DEFENDER_NODE,
    FINALIZE_NODE,
    INITIALIZE_NODE,
    ROUTE_CONTINUE,
    ROUTE_DEFEND,
    ROUTE_END,
    TARGET_NODE,
    DefenseType,
    JudgeVerdict,
    SessionStatus,
    DEFAULT_MAX_ERRORS,
)
from adversarial_framework.core.state import (
    AdversarialState,
    AttackTurn,
    DefenseAction,
    TokenBudget,
)
from adversarial_framework.defenses.registry import DefenseRegistry
from adversarial_framework.strategies.registry import StrategyRegistry

logger = structlog.get_logger(__name__)

APPLY_DEFENSES_NODE = "apply_defenses"

# Defense names that require a provider instance for LLM-based evaluation
_PROVIDER_DEFENSES = {"llm_judge", "two_pass"}


# Node Functions


def _initialize(state: AdversarialState) -> dict[str, Any]:
    """Set the session to running status on the first entry."""
    StrategyRegistry.discover()
    logger.info(
        "session_initialized",
        experiment_id=state.get("experiment_id"),
        session_id=state.get("session_id"),
        target_model=state.get("target_model"),
        objective=state.get("attack_objective", "")[:80],
    )
    return {
        "status": SessionStatus.RUNNING,
        "current_turn": 0,
    }


def _build_router(max_turns: int, max_cost_usd: float) -> Any:
    """Return a routing function that decides the next step after analysis."""

    def _route(state: AdversarialState) -> str:
        turn = state.get("current_turn", 0)
        verdict = state.get("judge_verdict")
        error_count = state.get("error_count", 0)
        budget = state.get("token_budget", TokenBudget())

        # Budget / error guards
        if turn >= max_turns:
            logger.info("route_end_max_turns", turn=turn, max_turns=max_turns)
            return ROUTE_END

        if budget.estimated_cost_usd >= max_cost_usd:
            logger.info("route_end_cost", cost=budget.estimated_cost_usd)
            return ROUTE_END

        if error_count >= DEFAULT_MAX_ERRORS:
            logger.info("route_end_errors", error_count=error_count)
            return ROUTE_END

        # Verdict-based routing
        if verdict == JudgeVerdict.JAILBREAK:
            defense_enabled = state.get("defense_enabled", False)
            if defense_enabled:
                logger.info("route_defend", turn=turn, mode="defense")
                return ROUTE_DEFEND
            else:
                logger.info(
                    "route_continue_attack_mode",
                    turn=turn,
                    verdict=verdict,
                )
                return ROUTE_CONTINUE

        # Continue for refused, borderline, error, or None
        logger.info("route_continue", turn=turn, verdict=verdict)
        return ROUTE_CONTINUE

    return _route


def _finalize(state: AdversarialState) -> dict[str, Any]:
    """Wrap up the session: set status to completed, build final turn record."""
    total_turns = state.get("current_turn", 0)
    total_jailbreaks = state.get("total_jailbreaks", 0)
    total_borderline = state.get("total_borderline", 0)
    total_refused = state.get("total_refused", 0)
    total_blocked = state.get("total_blocked", 0)

    logger.info(
        "session_finalized",
        total_turns=total_turns,
        total_jailbreaks=total_jailbreaks,
        total_borderline=total_borderline,
        total_refused=total_refused,
        total_blocked=total_blocked,
    )

    return {
        "status": SessionStatus.COMPLETED,
    }


def _build_history_recorder() -> Any:
    """Return a node that appends the current turn to attack_history.

    Runs after the analyzer so all fields are populated for the current turn.
    """

    def _record(state: AdversarialState) -> dict[str, Any]:
        turn_number = state.get("current_turn", 0)
        verdict = state.get("judge_verdict", JudgeVerdict.ERROR)
        budget = state.get("token_budget", TokenBudget())

        turn = AttackTurn(
            turn_number=turn_number,
            strategy_name=state.get("selected_strategy", "unknown"),
            strategy_params=state.get("strategy_params") or {},
            attack_prompt=state.get("current_attack_prompt", ""),
            target_response=state.get("target_response", ""),
            judge_verdict=verdict,
            confidence=state.get("judge_confidence", 0.0),
            severity=state.get("severity_score") or 0.0,
            specificity=state.get("specificity_score") or 0.0,
            vulnerability_category=state.get("vulnerability_category"),
            attack_technique=state.get("attack_technique"),
            target_blocked=state.get("target_blocked", False),
            raw_target_response=state.get("raw_target_response"),
            attacker_reasoning=state.get("attacker_reasoning"),
            attacker_tokens=budget.total_attacker_tokens,
            target_tokens=budget.total_target_tokens,
            analyzer_tokens=budget.total_analyzer_tokens,
        )

        return {"attack_history": [turn]}

    return _record


# Defense Application

def _build_defense_applicator(
    target: TargetInterface,
    provider: BaseProvider | None = None,
    defense_model: str | None = None,
) -> Any:
    """Return a node that applies DefenseActions to the TargetInterface.

    Reads the latest defense_actions from state, instantiates live defenses,
    and appends them to the target's active defense pipeline.
    """

    def _apply(state: AdversarialState) -> dict[str, Any]:
        actions: list[DefenseAction] = state.get("defense_actions", [])
        if not actions:
            return {}

        # Only apply defenses generated on the current turn
        current_turn = state.get("current_turn", 0)
        new_actions = [a for a in actions if a.triggered_by_turn == current_turn]

        # Use defender_model from state if available, otherwise fall back
        model = state.get("defender_model") or defense_model

        applied = 0
        for action in new_actions:
            defense = _instantiate_defense(action, provider=provider, defense_model=model)
            if defense is not None:
                target.add_defense(defense)
                applied += 1

        if applied:
            logger.info(
                "defenses_applied",
                turn=current_turn,
                applied=applied,
                total_active=len(target.active_defenses),
            )

        # If a prompt patch was generated, update the system prompt
        for action in new_actions:
            if action.defense_type == DefenseType.PROMPT_PATCH.value:
                patched = action.defense_config.get("patched_prompt")
                if patched:
                    return {"target_system_prompt": patched}

        return {}

    return _apply


def _instantiate_defense(
    action: DefenseAction,
    provider: BaseProvider | None = None,
    defense_model: str | None = None,
) -> Any:
    """Convert a DefenseAction into a live BaseDefense instance."""
    dtype = action.defense_type
    config = dict(action.defense_config)  # copy so we can inject provider/model

    # Prompt patches update state directly — no defense object needed
    if dtype == DefenseType.PROMPT_PATCH.value:
        return None

    # Rule-based needs explicit kwargs rather than **config
    if dtype == DefenseType.RULE_BASED.value:
        DefenseRegistry.discover()
        cls = DefenseRegistry.get("rule_based")
        return cls(
            custom_input_patterns=config.get("input_patterns"),
            custom_output_patterns=config.get("output_patterns"),
            blocked_keywords=config.get("blocked_keywords"),
        )

    # Generic registry lookup for all other types
    registry_name = _DEFENSE_TYPE_TO_REGISTRY.get(dtype)
    if registry_name is None:
        logger.warning("unknown_defense_type", type=dtype)
        return None

    DefenseRegistry.discover()
    try:
        cls = DefenseRegistry.get(registry_name)
        # Inject provider and model for LLM-powered defenses
        if registry_name in _PROVIDER_DEFENSES and provider is not None:
            config["provider"] = provider
            if "model" not in config and defense_model:
                config["model"] = defense_model
        return cls(**config)
    except (KeyError, TypeError):
        logger.warning("defense_instantiation_failed", type=dtype)
        return None


# Maps DefenseType values to DefenseRegistry names for generic instantiation
_DEFENSE_TYPE_TO_REGISTRY: dict[str, str] = {
    DefenseType.LLM_JUDGE.value: "llm_judge",
    DefenseType.LAYERED.value: "layered",
    DefenseType.ML_CLASSIFIER.value: "ml_guardrails",
    DefenseType.SEMANTIC_GUARD.value: "semantic_guard",
    DefenseType.TWO_PASS.value: "two_pass",
}


# Graph Builder

def build_graph(
    provider: BaseProvider,
    attacker_config: dict[str, Any] | None = None,
    analyzer_config: dict[str, Any] | None = None,
    defender_config: dict[str, Any] | None = None,
    target_system_prompt: str | None = None,
    max_turns: int = 20,
    max_cost_usd: float = 10.0,
    session_mode: str = "attack",
    initial_defenses: list[dict[str, Any]] | None = None,
    defense_model: str | None = None,
) -> StateGraph:
    """Construct the main adversarial cycle LangGraph.

    Args:
        provider: LLM provider (Ollama, OpenAI, etc.) shared by all agents.
        attacker_config: Config dict for AttackerAgent.
        analyzer_config: Config dict for AnalyzerAgent.
        defender_config: Config dict for DefenderAgent.
        target_system_prompt: Optional system prompt for the target LLM.
        max_turns: Maximum attack turns before termination.
        max_cost_usd: Maximum cost budget before termination.
        session_mode: ``"attack"`` (no defenses) or ``"defense"`` (arms race).
        initial_defenses: Defense configs to load at start (defense mode only).
        defense_model: Model name for LLM-powered defenses (from user config).

    Returns:
        A compiled LangGraph ``StateGraph``.
    """
    attacker_config = attacker_config or {}
    analyzer_config = analyzer_config or {}
    defender_config = defender_config or {}

    # Instantiate agents
    attacker = AttackerAgent(config=attacker_config, provider=provider)
    analyzer = AnalyzerAgent(config=analyzer_config, provider=provider)
    target = TargetInterface(provider=provider, system_prompt=target_system_prompt)
    defender = DefenderAgent(config=defender_config, provider=provider)

    # Load pre-configured defenses for defense mode
    if session_mode == "defense" and initial_defenses:
        DefenseRegistry.discover()
        for defense_cfg in initial_defenses:
            defense_name = defense_cfg.get("name", "")
            defense_params = dict(defense_cfg.get("params", {}))
            try:
                cls = DefenseRegistry.get(defense_name)
                # Inject provider and model for LLM-powered defenses
                if defense_name in _PROVIDER_DEFENSES:
                    defense_params["provider"] = provider
                    if "model" not in defense_params and defense_model:
                        defense_params["model"] = defense_model
                defense_instance = cls(**defense_params)
                target.add_defense(defense_instance)
                logger.info(
                    "initial_defense_loaded",
                    defense=defense_name,
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "initial_defense_load_failed",
                    name=defense_name,
                    error=str(exc),
                )
    record_history = _build_history_recorder()
    apply_defenses = _build_defense_applicator(
        target, provider=provider, defense_model=defense_model,
    )
    router = _build_router(max_turns=max_turns, max_cost_usd=max_cost_usd)

    # Build graph
    graph = StateGraph(AdversarialState)

    # Nodes
    graph.add_node(INITIALIZE_NODE, _initialize)
    graph.add_node(ATTACKER_NODE, attacker)
    graph.add_node(TARGET_NODE, target)
    graph.add_node(ANALYZER_NODE, analyzer)
    graph.add_node("record_history", record_history)
    graph.add_node(DEFENDER_NODE, defender)
    graph.add_node(APPLY_DEFENSES_NODE, apply_defenses)
    graph.add_node(FINALIZE_NODE, _finalize)

    # Edges: linear pipeline initialize → attacker → target → analyzer → record
    graph.set_entry_point(INITIALIZE_NODE)
    graph.add_edge(INITIALIZE_NODE, ATTACKER_NODE)
    graph.add_edge(ATTACKER_NODE, TARGET_NODE)
    graph.add_edge(TARGET_NODE, ANALYZER_NODE)
    graph.add_edge(ANALYZER_NODE, "record_history")

    # Conditional routing after record_history
    graph.add_conditional_edges(
        "record_history",
        router,
        {
            ROUTE_CONTINUE: ATTACKER_NODE,
            ROUTE_DEFEND: DEFENDER_NODE,
            ROUTE_END: FINALIZE_NODE,
        },
    )

    # Defender → apply_defenses → attacker (arms race loop)
    graph.add_edge(DEFENDER_NODE, APPLY_DEFENSES_NODE)
    graph.add_edge(APPLY_DEFENSES_NODE, ATTACKER_NODE)

    # Finalize → END
    graph.add_edge(FINALIZE_NODE, END)

    return graph
