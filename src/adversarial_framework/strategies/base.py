"""Abstract base class for all attack strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from adversarial_framework.core.constants import AttackCategory


@dataclass(frozen=True)
class StrategyMetadata:
    """Metadata for strategy registration, UI display, and documentation."""

    name: str  # Unique key: "pair", "crescendo", etc.
    display_name: str  # Human-readable: "PAIR (Iterative Refinement)"
    category: AttackCategory
    description: str
    estimated_asr: str  # e.g. "60-89%"
    min_turns: int = 1
    max_turns: int | None = None  # None = unlimited
    requires_white_box: bool = False
    supports_multi_turn: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AttackResult:
    """Output of a strategy's ``generate`` or ``refine`` call."""

    prompt: str
    strategy_metadata: dict[str, Any]
    token_usage: int = 0
    auxiliary_prompts: list[str] | None = None


class BaseAttackStrategy(ABC):
    """Every attack strategy must subclass this and implement ``generate``/``refine``.

    Subclasses are discovered automatically via the ``StrategyRegistry`` decorator.
    """

    metadata: StrategyMetadata  # Must be set by the subclass as a class attribute

    @abstractmethod
    async def generate(
        self,
        objective: str,
        context: dict[str, Any],
        params: dict[str, Any],
    ) -> AttackResult:
        """Generate an adversarial prompt from scratch."""
        ...

    @abstractmethod
    async def refine(
        self,
        objective: str,
        previous_prompt: str,
        target_response: str,
        judge_feedback: str,
        params: dict[str, Any],
    ) -> AttackResult:
        """Refine a failed attack based on judge feedback."""
        ...

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Merge caller-provided params with strategy defaults."""
        defaults = dict(self.metadata.parameters)
        defaults.update(params)
        return defaults
