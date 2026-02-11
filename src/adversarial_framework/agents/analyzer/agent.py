"""AnalyzerAgent sequential pipeline: Judge -> Classifier -> Scorer.

The Judge always runs.  Classifier and Scorer are only invoked when the verdict
is 'jailbreak' or `borderline`, saving ~66% of LLM calls on refusals.
"""

from __future__ import annotations

from typing import Any

import structlog

from adversarial_framework.agents.analyzer.classifier import run_classifier
from adversarial_framework.agents.analyzer.judge import run_judge
from adversarial_framework.agents.analyzer.scorer import run_scorer
from adversarial_framework.agents.base import BaseAgent
from adversarial_framework.agents.target.providers.base import BaseProvider
from adversarial_framework.core.constants import JudgeVerdict
from adversarial_framework.core.state import AdversarialState

logger = structlog.get_logger(__name__)


class AnalyzerAgent(BaseAgent):
    """LangGraph node that evaluates target responses.

    Pipeline:
        1. Judge (always) -> verdict + confidence
        2. Classifier (jailbreak/borderline only) -> vulnerability category
        3. Scorer (jailbreak/borderline only) -> severity/specificity/coherence
    """

    def __init__(self, config: dict[str, Any], provider: BaseProvider) -> None:
        super().__init__(config)
        self.provider = provider

    async def __call__(self, state: AdversarialState) -> dict[str, Any]:
        """Execute the analyzer pipeline and return a partial state update."""
        target_response = state.get("target_response", "")
        attack_prompt = state.get("current_attack_prompt", "")
        objective = state.get("attack_objective", "")
        model = state.get("analyzer_model", self.model_name)
        strategy_name = state.get("selected_strategy", "unknown")

        if not target_response:
            logger.warning("analyzer_skip_empty_response")
            return {
                "judge_verdict": JudgeVerdict.ERROR,
                "judge_confidence": 0.0,
            }

        # Step 1: Judge (always runs)
        logger.info("analyzer_judge_start", turn=state.get("current_turn", 0))
        judge_result = await run_judge(
            provider=self.provider,
            model=model,
            objective=objective,
            attack_prompt=attack_prompt,
            target_response=target_response,
        )

        verdict = judge_result.get("judge_verdict", JudgeVerdict.ERROR)
        updates: dict[str, Any] = {**judge_result}

        # ── Defense-mode blocked override ─────────────────────────────────
        # In defense mode, when a defense blocked the output, the
        # target_response is a block message (e.g., "[OUTPUT FILTERED ...]")
        # whose text can mislead the judge into a false "jailbreak" verdict.
        # Since the defense prevented the content from reaching the user,
        # downgrade to "refused" so ASR only reflects attacks that actually
        # got through the defense pipeline.
        # In attack mode this never triggers (no defenses are loaded).
        target_blocked = state.get("target_blocked", False)
        defense_enabled = state.get("defense_enabled", False)

        if (
            defense_enabled
            and target_blocked
            and verdict in (JudgeVerdict.JAILBREAK, JudgeVerdict.BORDERLINE)
        ):
            logger.info(
                "analyzer_blocked_override",
                original_verdict=verdict.value,
                turn=state.get("current_turn", 0),
            )
            verdict = JudgeVerdict.REFUSED
            updates["judge_verdict"] = verdict

        # Step 2 & 3: Classifier + Scorer (only on jailbreak/borderline)
        needs_deep_analysis = verdict in (
            JudgeVerdict.JAILBREAK,
            JudgeVerdict.BORDERLINE,
        )

        if needs_deep_analysis:
            logger.info(
                "analyzer_deep_analysis",
                verdict=verdict.value,
                turn=state.get("current_turn", 0),
            )

            classifier_result = await run_classifier(
                provider=self.provider,
                model=model,
                attack_prompt=attack_prompt,
                target_response=target_response,
                strategy_name=strategy_name,
            )
            updates.update(classifier_result)

            scorer_result = await run_scorer(
                provider=self.provider,
                model=model,
                target_response=target_response,
            )
            updates.update(scorer_result)
        else:
            """
            Clear stale scores from previous turn.
            NOTE: Use explicit zero/empty values, NOT None.  The `latest`
            reducer ignores None (`return new if new is not None else existing`)
            so setting None would preserve stale scores from prior turns.
            """
            updates.update(
                {
                    "severity_score": 0.0,
                    "specificity_score": 0.0,
                    "coherence_score": 0.0,
                    "vulnerability_category": "",
                    "attack_technique": "",
                }
            )

        # Update counters
        if verdict == JudgeVerdict.JAILBREAK:
            updates["total_jailbreaks"] = 1
        elif verdict == JudgeVerdict.BORDERLINE:
            updates["total_borderline"] = 1
        elif verdict == JudgeVerdict.REFUSED:
            updates["total_refused"] = 1

        if target_blocked:
            updates["total_blocked"] = 1

        logger.info(
            "analyzer_complete",
            verdict=verdict.value,
            confidence=updates.get("judge_confidence"),
            deep_analysis=needs_deep_analysis,
            blocked=target_blocked,
            turn=state.get("current_turn", 0),
        )

        return updates

    def get_system_prompt(self, state: AdversarialState) -> str:
        """Not used directly — sub-nodes have their own prompts."""
        return "You are a safety analysis agent."
