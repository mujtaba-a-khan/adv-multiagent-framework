"""Google Gemini provider commercial API for Gemini models.

Uses httpx to call the Gemini REST API directly, avoiding an extra SDK
dependency.  Set ``GOOGLE_API_KEY`` in the environment or pass it
explicitly.
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

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider(BaseProvider):
    """Async provider for the Google Gemini REST API."""

    name = "google"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-2.0-flash",
    ) -> None:
        if not api_key:
            import os

            api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY or pass api_key="
            )
        self._api_key = api_key
        self._default_model = default_model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a generateContent request to Gemini."""
        model = model or self._default_model

        # Convert OpenAI-style messages to Gemini format
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": text}],
            })

        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        url = f"{_GEMINI_BASE}/models/{model}:generateContent?key={self._api_key}"

        start = time.perf_counter()
        resp = await self._client.post(url, json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()

        # Extract content from response
        candidates = data.get("candidates", [])
        content = ""
        finish_reason = "stop"
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)
            finish_reason = candidate.get("finishReason", "STOP").lower()

        # Extract usage metadata
        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", 0)

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            ),
            model=model,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw=data,
        )

    async def list_models(self) -> list[str]:
        """Fetch available Gemini models."""
        url = f"{_GEMINI_BASE}/models?key={self._api_key}"
        resp = await self._client.get(url)
        resp.raise_for_status()
        data = resp.json()
        return [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]

    async def healthcheck(self) -> bool:
        """Check if the Gemini API is reachable."""
        try:
            url = f"{_GEMINI_BASE}/models?key={self._api_key}"
            resp = await self._client.get(url)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
