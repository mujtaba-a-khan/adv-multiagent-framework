"""Tests for GoogleProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from adversarial_framework.agents.target.providers.google import GoogleProvider


class TestGoogleProviderInit:
    def test_name(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        assert provider.name == "google"

    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            GoogleProvider(api_key="")

    def test_reads_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        provider = GoogleProvider()
        assert provider._api_key == "env-key"


class TestGoogleProviderGenerate:
    async def test_generate_basic(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Paris"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Capital?"}],
            model="gemini-2.0-flash",
        )
        assert result.content == "Paris"
        assert result.usage.total_tokens == 15

    async def test_generate_with_system_message(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "OK"}]},
                }
            ],
            "usageMetadata": {},
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.generate(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        )
        call_args = provider._client.post.call_args
        payload = call_args[1]["json"]
        assert "systemInstruction" in payload

    async def test_generate_empty_candidates(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [],
            "usageMetadata": {},
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == ""

    async def test_generate_assistant_role_mapped(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {},
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.generate(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ],
        )
        payload = provider._client.post.call_args[1]["json"]
        roles = [c["role"] for c in payload["contents"]]
        assert roles == ["user", "model", "user"]


class TestGoogleProviderListModels:
    async def test_list_models(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "supportedGenerationMethods": [
                        "generateContent",
                    ],
                },
                {
                    "name": "models/embedding-001",
                    "supportedGenerationMethods": [
                        "embedContent",
                    ],
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        models = await provider.list_models()
        assert models == ["gemini-2.0-flash"]


class TestGoogleProviderHealthcheck:
    async def test_healthy(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        assert await provider.healthcheck() is True

    async def test_unhealthy(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        assert await provider.healthcheck() is False

    async def test_close(self) -> None:
        provider = GoogleProvider(api_key="test-key")
        provider._client = AsyncMock()

        await provider.close()
        provider._client.aclose.assert_called_once()
