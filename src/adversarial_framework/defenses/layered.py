"""Layered (defense-in-depth) defense — composes multiple defenses into a pipeline.

Implements the defense-in-depth architecture pattern where multiple defense
mechanisms are applied in sequence.  If any layer blocks, the request is
rejected.  The composition order matters: fast/cheap defenses should run first.
"""

from __future__ import annotations

from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult
from adversarial_framework.defenses.registry import DefenseRegistry


@DefenseRegistry.register
class LayeredDefense(BaseDefense):
    """Composes multiple defenses into a sequential pipeline.

    Defenses are evaluated in order.  The first block wins.  This allows
    building defense-in-depth with rule_based → ml_guardrails → llm_judge.
    """

    name = "layered"

    def __init__(self, layers: list[BaseDefense] | None = None) -> None:
        self.layers: list[BaseDefense] = layers or []

    def add_layer(self, defense: BaseDefense) -> None:
        """Append a defense layer to the pipeline."""
        self.layers.append(defense)

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        """Run input through all defense layers; first block wins."""
        for layer in self.layers:
            result = await layer.check_input(prompt)
            if result.blocked:
                return DefenseCheckResult(
                    blocked=True,
                    reason=f"[{layer.name}] {result.reason}",
                    confidence=result.confidence,
                    metadata={
                        "blocking_layer": layer.name,
                        "original_metadata": result.metadata,
                        "defense": self.name,
                    },
                )
        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        """Run output through all defense layers; first block wins."""
        for layer in self.layers:
            result = await layer.check_output(response)
            if result.blocked:
                return DefenseCheckResult(
                    blocked=True,
                    reason=f"[{layer.name}] {result.reason}",
                    confidence=result.confidence,
                    metadata={
                        "blocking_layer": layer.name,
                        "original_metadata": result.metadata,
                        "defense": self.name,
                    },
                )
        return DefenseCheckResult(blocked=False)
