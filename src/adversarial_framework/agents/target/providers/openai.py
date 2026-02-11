"""OpenAI provider — commercial API for GPT models."""

from __future__ import annotations

import time
from typing import Any

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]


class OpenAIProvider(BaseProvider):
    """Async provider for the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install with: uv add openai"
            )
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._default_model = default_model

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to OpenAI."""
        model = model or self._default_model

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0] if response.choices else None
        content = choice.message.content or "" if choice else ""
        finish_reason = choice.finish_reason or "stop" if choice else "stop"

        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        return LLMResponse(
            content=content,
            usage=token_usage,
            model=response.model or model,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )

    async def list_models(self) -> list[str]:
        """Fetch available OpenAI models."""
        models = await self._client.models.list()
        return [m.id for m in models.data]

    async def healthcheck(self) -> bool:
        """Check if OpenAI API is reachable."""
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
