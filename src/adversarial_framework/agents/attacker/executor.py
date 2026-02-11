"""Attacker Executor it generates the adversarial prompt using the selected strategy.

The Executor is the second node in the attacker hierarchical subgraph.
It instantiates the strategy selected by the Planner and calls its
`generate()` or `refine()` method.
"""

from __future__ import annotations

from typing import Any

import structlog

from adversarial_framework.agents.target.providers.base import BaseProvider
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.core.state import AdversarialState
from adversarial_framework.strategies.registry import StrategyRegistry

logger = structlog.get_logger(__name__)


async def run_executor(
    provider: BaseProvider,
    state: AdversarialState,
) -> dict[str, Any]:
    """Generate or refine an adversarial prompt using the selected strategy.

    Returns:
        Dict with `attack_prompt`, `strategy_metadata`, `token_usage`.
    """
    objective = state.get("attack_objective", "")
    strategy_name = state.get("selected_strategy", "pair")
    strategy_params = state.get("strategy_params") or {}
    current_turn = state.get("current_turn", 0)
    attacker_model = state.get("attacker_model", "phi4-reasoning:14b")

    # Resolve strategy
    try:
        strategy_cls = StrategyRegistry.get(strategy_name or "pair")
    except KeyError:
        return {
            "attack_prompt": None,
            "error": f"Strategy '{strategy_name}' not found",
            "token_usage": 0,
        }

    strategy = strategy_cls(provider=provider)

    context = {
        "attacker_model": attacker_model,
        "current_turn": current_turn,
        "attack_history": state.get("attack_history", []),
    }
    strategy_params["attacker_model"] = attacker_model

    try:
        if current_turn == 0 or state.get("target_response") is None:
            result = await strategy.generate(objective, context, strategy_params)
        else:
            previous_prompt = state.get("current_attack_prompt", "")
            target_response = state.get("target_response", "")
            verdict = state.get("judge_verdict", JudgeVerdict.REFUSED)
            confidence = state.get("judge_confidence", 0.0)
            judge_reason = state.get("judge_reason")
            fb_parts = [
                f"Verdict: {verdict.value if isinstance(verdict, JudgeVerdict) else verdict}",
                f"Confidence: {confidence}",
            ]
            if judge_reason:
                fb_parts.append(f"Reason: {judge_reason}")
            judge_feedback = ", ".join(fb_parts)

            strategy_params["_previous_verdict"] = (
                verdict.value if isinstance(verdict, JudgeVerdict) else str(verdict)
            )
            result = await strategy.refine(
                objective=objective,
                previous_prompt=previous_prompt or "",
                target_response=target_response or "",
                judge_feedback=judge_feedback,
                params=strategy_params,
            )

        return {
            "attack_prompt": result.prompt,
            "strategy_metadata": result.strategy_metadata,
            "token_usage": result.token_usage,
        }

    except Exception as exc:
        logger.error("executor_error", error=str(exc), strategy=strategy_name)
        return {
            "attack_prompt": None,
            "error": str(exc),
            "token_usage": 0,
        }
