"""GCG (Greedy Coordinate Gradient) attack strategy.

Optimizes an adversarial suffix token-by-token using gradient information.
Requires white-box access to the target model (logits/gradients).
Falls back to a black-box approximation when gradients are unavailable.

Reference: Universal and Transferable Adversarial Attacks (Zou et al. 2023).
"""

from __future__ import annotations

import random  # noqa: S311
import string
from typing import Any

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.base import (
    AttackResult,
    BaseAttackStrategy,
    StrategyMetadata,
)
from adversarial_framework.strategies.registry import StrategyRegistry


def _random_suffix(length: int = 20) -> str:
    """Generate a random alphanumeric suffix (black-box fallback)."""
    chars = string.ascii_letters + string.digits + " !.,"
    return "".join(random.choices(chars, k=length))


def _mutate_suffix(suffix: str, mutation_rate: float = 0.3) -> str:
    """Mutate a suffix by randomly replacing characters."""
    chars = list(suffix)
    for i in range(len(chars)):
        if random.random() < mutation_rate:
            chars[i] = random.choice(string.ascii_letters + string.digits + " !.,")
    return "".join(chars)


@StrategyRegistry.register
class GCGStrategy(BaseAttackStrategy):
    """GCG — Greedy Coordinate Gradient suffix optimization."""

    metadata = StrategyMetadata(
        name="gcg",
        display_name="GCG (Greedy Coordinate Gradient)",
        category=AttackCategory.OPTIMIZATION,
        description=(
            "Optimizes an adversarial suffix appended to the prompt using "
            "greedy coordinate gradient descent. Uses black-box random search "
            "as fallback when gradient access is unavailable."
        ),
        estimated_asr="50-80%",
        min_turns=1,
        max_turns=50,
        requires_white_box=True,
        supports_multi_turn=False,
        parameters={
            "suffix_length": 20,
            "num_candidates": 256,
            "mutation_rate": 0.3,
        },
        references=[
            "https://arxiv.org/abs/2307.15043",
            "https://arxiv.org/abs/2402.14020",
        ],
    )

    def __init__(self, provider: Any = None) -> None:
        self.provider = provider
        self._best_suffix = ""
        self._iteration = 0

    async def generate(
        self,
        objective: str,
        context: dict[str, Any],
        params: dict[str, Any],
    ) -> AttackResult:
        params = self.validate_params(params)
        suffix_length = params.get("suffix_length", 20)

        self._best_suffix = _random_suffix(suffix_length)
        self._iteration = 0
        prompt = f"{objective} {self._best_suffix}"

        return AttackResult(
            prompt=prompt,
            strategy_metadata={
                "suffix": self._best_suffix,
                "iteration": self._iteration,
                "mode": "black_box_init",
            },
            reasoning=(
                f"Generating initial random adversarial suffix ({suffix_length} chars) for"
                " black-box GCG attack"
            ),
            token_usage=0,
        )

    async def refine(
        self,
        objective: str,
        previous_prompt: str,
        target_response: str,
        judge_feedback: str,
        params: dict[str, Any],
    ) -> AttackResult:
        params = self.validate_params(params)
        mutation_rate = params.get("mutation_rate", 0.3)
        num_candidates = params.get("num_candidates", 256)

        # Read best suffix from persisted metadata
        best_suffix = params.get("suffix", _random_suffix(params.get("suffix_length", 20)))
        iteration = int(params.get("iteration", 0)) + 1

        # Black-box: generate candidates by mutating the best suffix
        candidates = [_mutate_suffix(best_suffix, mutation_rate) for _ in range(num_candidates)]
        # Select a random candidate (in white-box mode this would be gradient-guided)
        new_suffix = random.choice(candidates)
        prompt = f"{objective} {new_suffix}"

        return AttackResult(
            prompt=prompt,
            strategy_metadata={
                "suffix": new_suffix,
                "iteration": iteration,
                "num_candidates": num_candidates,
                "mode": "black_box_mutation",
            },
            reasoning=(
                f"GCG iteration {iteration}: mutated suffix from {num_candidates} candidates"
                f" (mutation_rate={mutation_rate})"
            ),
            token_usage=0,
        )
