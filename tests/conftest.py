"""Shared test fixtures for the adversarial framework."""

from __future__ import annotations

from typing import Any

import pytest

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)


class MockProvider(BaseProvider):
    """In-memory mock provider for testing without Ollama."""

    name = "mock"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["Mock response"]
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        content = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1

        return LLMResponse(
            content=content,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
            model=model or "mock-model",
        )

    async def list_models(self) -> list[str]:
        return ["mock-model-a", "mock-model-b"]

    async def healthcheck(self) -> bool:
        return True


@pytest.fixture
def mock_provider() -> MockProvider:
    """Return a basic mock provider with a single response."""
    return MockProvider()


@pytest.fixture
def mock_provider_factory():
    """Factory fixture for creating mock providers with custom responses."""

    def _create(responses: list[str]) -> MockProvider:
        return MockProvider(responses=responses)

    return _create
