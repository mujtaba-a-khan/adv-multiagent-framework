"""Anthropic provider commercial API for Claude models."""

from __future__ import annotations

import time
from typing import Any

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]


class AnthropicProvider(BaseProvider):
    """Async provider for the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install with: uv add anthropic"
            )
        self._client = AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a message request to the Anthropic API.

        Anthropic expects a separate ``system`` parameter rather than a system
        message in the messages list.  This method extracts any leading system
        message automatically.
        """
        model = model or self._default_model

        # Extract system message if present
        system_text: str | None = None
        chat_messages = list(messages)
        if chat_messages and chat_messages[0].get("role") == "system":
            system_text = chat_messages[0].get("content", "")
            chat_messages = chat_messages[1:]

        # Ensure there's at least one user message
        if not chat_messages:
            chat_messages = [{"role": "user", "content": ""}]

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            create_kwargs["system"] = system_text
        create_kwargs.update(kwargs)

        start = time.perf_counter()
        response = await self._client.messages.create(**create_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        # Extract text from content blocks
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        usage = response.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            usage=token_usage,
            model=response.model,
            finish_reason=response.stop_reason or "end_turn",
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )

    async def list_models(self) -> list[str]:
        """Return known Anthropic model identifiers.

        The Anthropic API does not expose a model listing endpoint,
        so we return a static list of currently available models.
        """
        return [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-3-5-20241022",
            "claude-3-5-sonnet-20241022",
        ]

    async def healthcheck(self) -> bool:
        """Check if the Anthropic API is reachable with a minimal request."""
        try:
            await self._client.messages.create(
                model=self._default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
