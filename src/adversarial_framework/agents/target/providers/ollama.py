"""Ollama provider primary LLM backend using locally-hosted open-source models."""

from __future__ import annotations

import time
from typing import Any

import httpx

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)


class OllamaProvider(BaseProvider):
    """Async provider for the Ollama HTTP API (``/api/chat``)."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")
        """Use a generous timeout: Ollama may need to load/switch models
        between agents (attacker → target → analyzer → defender), which
        can take well over 120s on memory-constrained systems."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(300.0, connect=30.0),
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama."""
        model = model or "llama3:8b"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        payload.update(kwargs)

        start = time.perf_counter()
        resp = await self._client.post("/api/chat", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        content = data.get("message", {}).get("content", "")

        # Ollama returns token counts in eval_count / prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=data.get("model", model),
            finish_reason=data.get("done_reason", "stop"),
            latency_ms=latency_ms,
            raw=data,
        )

    async def list_models(self) -> list[str]:
        """Fetch all locally available Ollama model tags."""
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def healthcheck(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def load_model(
        self, model: str, keep_alive: str = "30m"
    ) -> None:
        """Preload a model into Ollama memory."""
        await self._client.post(
            "/api/generate",
            json={"model": model, "keep_alive": keep_alive},
            timeout=300.0,
        )

    async def unload_model(self, model: str) -> None:
        """Unload a model from Ollama memory (free RAM/VRAM)."""
        await self._client.post(
            "/api/generate",
            json={"model": model, "keep_alive": 0},
        )

    async def list_running_models(self) -> list[dict[str, object]]:
        """Return currently loaded models from Ollama ``/api/ps``."""
        resp = await self._client.get("/api/ps")
        resp.raise_for_status()
        return resp.json().get("models", [])  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
