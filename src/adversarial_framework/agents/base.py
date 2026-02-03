"""Base abstract class for all LangGraph agent nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adversarial_framework.core.state import AdversarialState


class BaseAgent(ABC):
    """Abstract base for Attacker, Analyzer, and Defender agents.

    Each agent is callable as a LangGraph node — it receives the full state
    and returns a *partial* state update dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_name: str = config.get("model", "phi4-reasoning:14b")
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 4096)

    @abstractmethod
    async def __call__(self, state: AdversarialState) -> dict[str, Any]:
        """Execute the agent node and return a partial state update."""
        ...

    @abstractmethod
    def get_system_prompt(self, state: AdversarialState) -> str:
        """Build a context-aware system prompt for the LLM call."""
        ...
