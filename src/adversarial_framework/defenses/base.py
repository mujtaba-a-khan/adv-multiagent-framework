"""Abstract base class for all defense mechanisms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DefenseCheckResult:
    """Result of a defense check on input or output text."""

    blocked: bool
    reason: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None


class BaseDefense(ABC):
    """All defenses (rule-based, ML, LLM-judge, etc.) implement this interface.

    Defenses are applied by the ``TargetInterface`` as an ordered pipeline.
    """

    name: str

    def __init__(self, **kwargs: Any) -> None:  # noqa: B027
        pass

    @abstractmethod
    async def check_input(self, prompt: str) -> DefenseCheckResult:
        """Check an input prompt before it reaches the target LLM."""
        ...

    @abstractmethod
    async def check_output(self, response: str) -> DefenseCheckResult:
        """Check a target LLM response before it is returned."""
        ...
