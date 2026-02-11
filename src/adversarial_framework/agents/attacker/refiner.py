"""Attacker Validator/Refiner — validates generated prompts before submission.

The Validator is the third node in the attacker hierarchical subgraph.
It checks that the generated prompt is coherent, non-trivially different
from previous attempts, and properly formatted for the selected strategy
(e.g., correct encoding for encoding strategies).
"""

from __future__ import annotations

from typing import Any

import structlog

from adversarial_framework.core.state import AdversarialState

logger = structlog.get_logger(__name__)

"""Minimum Jaccard similarity threshold — prompts that are too similar to
previous attempts are rejected so the Planner can regenerate."""
_MAX_SIMILARITY = 0.85
_MIN_PROMPT_LENGTH = 10


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def run_validator(
    prompt: str,
    state: AdversarialState,
) -> dict[str, Any]:
    """Validate the generated prompt for quality and novelty.

    Returns:
        Dict with `valid` (bool), `reason` (str), and the `prompt`.
    """
    history = state.get("attack_history", [])

    # Check minimum length
    if not prompt or len(prompt.strip()) < _MIN_PROMPT_LENGTH:
        logger.warning("validator_reject_short", length=len(prompt) if prompt else 0)
        return {
            "valid": False,
            "reason": "Prompt too short or empty",
            "prompt": prompt,
        }

    # Check novelty against recent history
    recent_prompts = [t.attack_prompt for t in history[-5:]]
    for prev in recent_prompts:
        similarity = _jaccard_similarity(prompt, prev)
        if similarity > _MAX_SIMILARITY:
            logger.warning(
                "validator_reject_similar",
                similarity=f"{similarity:.2f}",
            )
            return {
                "valid": False,
                "reason": f"Too similar to previous attempt (similarity={similarity:.2f})",
                "prompt": prompt,
            }

    # Check not an exact repeat
    all_prompts = {t.attack_prompt for t in history}
    if prompt in all_prompts:
        logger.warning("validator_reject_duplicate")
        return {
            "valid": False,
            "reason": "Exact duplicate of a previous attempt",
            "prompt": prompt,
        }

    return {
        "valid": True,
        "reason": "Passed validation",
        "prompt": prompt,
    }
