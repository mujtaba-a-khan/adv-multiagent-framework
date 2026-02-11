"""vLLM provider — high-throughput local serving via OpenAI-compatible API.

vLLM exposes an OpenAI-compatible HTTP API, so this provider uses httpx
to call the /v1/chat/completions endpoint directly without requiring
the openai Python package.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)


class VLLMProvider(BaseProvider):
    """Async provider for vLLM's OpenAI-compatible API."""

    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_model: str = "meta-llama/Llama-3-8B-Instruct",
        api_key: str = "EMPTY",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to vLLM."""
        model = model or self._default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)

        start = time.perf_counter()
        resp = await self._client.post("/v1/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        usage_data = data.get("usage", {})

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            model=data.get("model", model),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw=data,
        )

    async def list_models(self) -> list[str]:
        """Fetch available models from vLLM."""
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    async def healthcheck(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            try:
                resp = await self._client.get("/v1/models")
                return resp.status_code == 200
            except httpx.HTTPError:
                return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
