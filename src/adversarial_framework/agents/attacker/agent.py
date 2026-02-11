"""AttackerAgent generates adversarial prompts via the strategy registry.

This is a single-node implementation that selects a strategy,
generates or refines a prompt, and returns the result.  In Phase 4 this will
be expanded into a hierarchical subgraph (Planner → Executor → Validator).
"""

from __future__ import annotations

from typing import Any

import structlog

from adversarial_framework.agents.base import BaseAgent
from adversarial_framework.agents.target.providers.base import BaseProvider
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.core.state import AdversarialState, TokenBudget
from adversarial_framework.strategies.base import AttackResult
from adversarial_framework.strategies.registry import StrategyRegistry

logger = structlog.get_logger(__name__)


def _build_judge_feedback(
    state: AdversarialState,
) -> tuple[str, JudgeVerdict | str]:
    """Extract judge feedback string and verdict from state."""
    verdict = state.get("judge_verdict", JudgeVerdict.REFUSED)
    confidence = state.get("judge_confidence", 0.0)
    judge_reason = state.get("judge_reason")
    v_str = (
        verdict.value
        if isinstance(verdict, JudgeVerdict)
        else str(verdict)
    )
    parts = [f"Verdict: {v_str}", f"Confidence: {confidence}"]
    if judge_reason:
        parts.append(f"Reason: {judge_reason}")
    return ", ".join(parts), verdict


class AttackerAgent(BaseAgent):
    """LangGraph node that generates adversarial attack prompts.

    On turn 0 it sends the raw attack objective as a baseline (no strategy).
    On turn 1 it calls ``strategy.generate()`` for the first strategic attack.
    On subsequent turns it calls ``strategy.refine()`` using the target's
    previous response and the judge's feedback.
    """

    def __init__(self, config: dict[str, Any], provider: BaseProvider) -> None:
        super().__init__(config)
        self.provider = provider

    async def __call__(self, state: AdversarialState) -> dict[str, Any]:
        """Execute the attacker node and return a partial state update."""
        objective = state.get("attack_objective", "")
        current_turn = state.get("current_turn", 0)
        strategy_config = state.get("strategy_config", {})
        strategy_name = strategy_config.get("name", "pair")
        strategy_params = dict(strategy_config.get("params", {}))

        """ Merge persisted strategy metadata from previous turns 
         Strategy metadata contains cycling state (template indices, encoding
         types, escalation levels, etc.) that must persist across turns so
         that each refinement produces a genuinely different prompt. """

        prev_metadata = state.get("strategy_metadata") or {}
        strategy_params.update(prev_metadata)

        # Resolve strategy from registry
        try:
            strategy_cls = StrategyRegistry.get(strategy_name)
        except KeyError:
            available = StrategyRegistry.list_names()
            logger.error(
                "strategy_not_found",
                strategy=strategy_name,
                available=available,
            )
            return {
                "current_attack_prompt": None,
                "attacker_reasoning": None,
                "last_error": f"Strategy '{strategy_name}' not found. Available: {available}",
                "error_count": 1,
            }

        strategy = strategy_cls(provider=self.provider)

        # Pass attacker model into context/params
        attacker_model = state.get("attacker_model", self.model_name)
        attack_history = state.get("attack_history", [])
        context = {
            "attacker_model": attacker_model,
            "current_turn": current_turn,
            "attack_history": attack_history,
        }
        strategy_params["attacker_model"] = attacker_model

        try:
            if current_turn == 0:
                # Baseline turn: send raw objective to record the model's
                # natural response before any attack strategy is applied.
                logger.info(
                    "attacker_baseline",
                    strategy=strategy_name,
                    turn=current_turn,
                )
                result = AttackResult(
                    prompt=objective,
                    strategy_metadata={},
                    token_usage=0,
                )
            elif current_turn == 1 or state.get("target_response") is None:
                # First strategic turn: generate from scratch
                logger.info(
                    "attacker_generate",
                    strategy=strategy_name,
                    turn=current_turn,
                )
                result = await strategy.generate(objective, context, strategy_params)
            else:
                # Subsequent turns: refine based on feedback
                previous_prompt = state.get("current_attack_prompt", "")
                target_response = state.get("target_response", "")
                judge_feedback, verdict = _build_judge_feedback(state)

                strategy_params["_attack_history"] = attack_history
                strategy_params["_previous_verdict"] = (
                    verdict.value
                    if isinstance(verdict, JudgeVerdict)
                    else str(verdict)
                )

                logger.info(
                    "attacker_refine",
                    strategy=strategy_name,
                    turn=current_turn,
                    prev_verdict=(
                        verdict.value
                        if isinstance(verdict, JudgeVerdict)
                        else verdict
                    ),
                )
                result = await strategy.refine(
                    objective=objective,
                    previous_prompt=previous_prompt,
                    target_response=target_response,
                    judge_feedback=judge_feedback,
                    params=strategy_params,
                )

        except Exception as exc:
            logger.error("attacker_error", error=str(exc), strategy=strategy_name)
            return {
                "current_attack_prompt": None,
                "attacker_reasoning": None,
                "last_error": f"Attacker error: {exc}",
                "error_count": 1,
            }

        logger.info(
            "attacker_result",
            strategy=strategy_name,
            turn=current_turn,
            prompt_len=len(result.prompt),
            tokens=result.token_usage,
        )

        """Persist strategy metadata for next turn
            Merge the new metadata on top of previous so all cycling state
            (template indices, encoding types, used techniques, etc.) carries
            forward across turns. """

        updated_metadata = dict(prev_metadata)
        if result.strategy_metadata:
            updated_metadata.update(result.strategy_metadata)

        """Remove non-serializable keys before writing to state — these are
            only needed during the refine() call, not for DB persistence."""
        strategy_params.pop("_attack_history", None)
        strategy_params.pop("_previous_verdict", None)

        return {
            "current_attack_prompt": result.prompt,
            "attacker_reasoning": (
                result.reasoning
                if strategy_params.get("separate_reasoning", True)
                else None
            ),
            "selected_strategy": strategy_name,
            "strategy_params": strategy_params,
            "strategy_metadata": updated_metadata,
            "current_turn": current_turn + 1,
            "token_budget": TokenBudget(total_attacker_tokens=result.token_usage),
        }

    def get_system_prompt(self, state: AdversarialState) -> str:
        """Not used directly — strategies have their own prompts."""
        return "You are an adversarial red-teaming agent."
