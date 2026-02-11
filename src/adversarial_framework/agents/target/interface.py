"""Target Interface stateless LLM adapter with live defense pipeline.

The Target is *not* an agent.  It receives the current attack prompt from state,
runs it through the active defense pipeline, sends it to the target LLM, then
runs output defenses and returns the result.  Defenses can be appended at
runtime by the Defender agent (live patching / adversarial arms race).
"""

from __future__ import annotations

import structlog
from typing import Any

from adversarial_framework.agents.target.providers.base import BaseProvider
from adversarial_framework.core.state import AdversarialState
from adversarial_framework.defenses.base import BaseDefense

logger = structlog.get_logger(__name__)


class TargetInterface:
    """Stateless adapter between the LangGraph graph and the target LLM."""

    def __init__(
        self,
        provider: BaseProvider,
        system_prompt: str | None = None,
        active_defenses: list[BaseDefense] | None = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.active_defenses: list[BaseDefense] = active_defenses or []

    async def __call__(self, state: AdversarialState) -> dict[str, Any]:
        """LangGraph node function: send attack prompt to target LLM."""
        prompt = state.get("current_attack_prompt", "")
        if not prompt:
            return {
                "target_response": "[ERROR: No attack prompt provided]",
                "raw_target_response": None,
                "target_blocked": True,
                "last_error": "Empty attack prompt",
                "error_count": 1,
            }

        """Input Defense Pipeline 
        effective_prompt may be modified by wrapping defenses (sandwich,
        two_pass) while the original `prompt` is always used for detection."""
        effective_prompt = prompt
        input_block_msg: str | None = None
        for defense in self.active_defenses:
            result = await defense.check_input(prompt)
            if result.blocked:
                logger.info(
                    "input_blocked",
                    defense=defense.name,
                    reason=result.reason,
                )
                input_block_msg = f"[BLOCKED by {defense.name}: {result.reason}]"
                break  # Record but don't return   still call LLM
            # Wrapping defenses (sandwich, two_pass) store a modified prompt
            # in metadata — use it for the LLM call but keep checking the
            # original prompt for accurate jailbreak detection.
            if result.metadata and "wrapped_prompt" in result.metadata:
                effective_prompt = result.metadata["wrapped_prompt"]

        #  Build Messages & Call Target LLM
        messages = self._build_messages(effective_prompt, state)
        target_model = state.get("target_model", "llama3:8b")

        try:
            response = await self.provider.generate(
                messages=messages,
                model=target_model,
                temperature=0.0,
                max_tokens=2048,
            )
        except Exception as exc:
            """Some exceptions (e.g. httpx.ReadTimeout) have empty str()
            representations.  Always include the exception type for
            debuggability."""
            error_type = type(exc).__name__
            error_msg = str(exc) or "(no message)"
            error_detail = f"{error_type}: {error_msg}"
            logger.error("target_llm_error", error=error_detail)
            return {
                "target_response": f"[LLM ERROR: {error_detail}]",
                "raw_target_response": None,
                "target_blocked": False,
                "last_error": error_detail,
                "error_count": 1,
            }

        # Input was blocked but LLM succeeded, return block msg + raw
        
        if input_block_msg is not None:
            return {
                "target_response": input_block_msg,
                "raw_target_response": response.content,
                "target_blocked": True,
                "token_budget": _target_budget(response.usage.total_tokens),
            }

        # Output Defense Pipeline 
        for defense in self.active_defenses:
            result = await defense.check_output(response.content)
            if result.blocked:
                logger.info(
                    "output_filtered",
                    defense=defense.name,
                    reason=result.reason,
                )
                return {
                    "target_response": f"[OUTPUT FILTERED by {defense.name}: {result.reason}]",
                    "raw_target_response": response.content,
                    "target_blocked": True,
                    "token_budget": _target_budget(response.usage.total_tokens),
                }

        return {
            "target_response": response.content,
            "raw_target_response": None,
            "target_blocked": False,
            "token_budget": _target_budget(response.usage.total_tokens),
        }

    # Helpers

    def _build_messages(
        self, prompt: str, state: AdversarialState
    ) -> list[dict[str, str]]:
        """Assemble the message list for the target LLM."""
        messages: list[dict[str, str]] = []

        # System prompt (the one being tested / defended)
        sys_prompt = state.get("target_system_prompt") or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Multi-turn conversation history (for Crescendo, etc.)
        for msg in state.get("conversation_history", []):
            messages.append(msg)

        # Current attack prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def add_defense(self, defense: BaseDefense) -> None:
        """Live-patch: add a new defense to the pipeline (called by Defender agent)."""
        self.active_defenses.append(defense)
        logger.info("defense_added", defense=defense.name, total=len(self.active_defenses))


# Budget helpers
def _zero_budget() -> dict[str, int | float]:
    from adversarial_framework.core.state import TokenBudget
    return TokenBudget()


def _target_budget(tokens: int) -> dict[str, int | float]:
    from adversarial_framework.core.state import TokenBudget
    return TokenBudget(total_target_tokens=tokens)
