"""Abstract base class for LLM providers.

Every provider (Ollama, OpenAI, Anthropic, etc.) implements this interface so
the ``TargetInterface`` and agent nodes remain provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Token usage returned by an LLM provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """Standardised response from any LLM provider."""

    content: str
    usage: TokenUsage
    model: str
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract provider interface.

    Concrete providers implement ``generate`` (and optionally ``stream``)
    using the respective SDK's async client.
    """

    name: str  # e.g. "ollama", "openai"

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the LLM and return a standardised response."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return a list of available model identifiers."""
        ...

    @abstractmethod
    async def healthcheck(self) -> bool:
        """Return ``True`` if the provider is reachable."""
        ...
