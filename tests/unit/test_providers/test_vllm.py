"""Tests for VLLMProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from adversarial_framework.agents.target.providers.vllm import VLLMProvider


class TestVLLMProviderInit:
    def test_name(self) -> None:
        provider = VLLMProvider()
        assert provider.name == "vllm"

    def test_strips_trailing_slash(self) -> None:
        provider = VLLMProvider(base_url="http://host:8000/")
        assert provider.base_url == "http://host:8000"

    def test_default_model(self) -> None:
        provider = VLLMProvider()
        assert provider._default_model == ("meta-llama/Llama-3-8B-Instruct")


class TestVLLMProviderGenerate:
    async def test_generate_returns_llm_response(self) -> None:
        provider = VLLMProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "model": "llama-3-8b",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama-3-8b",
        )
        assert result.content == "Hello"
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 15
        assert result.model == "llama-3-8b"

    async def test_generate_empty_choices(self) -> None:
        provider = VLLMProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [],
            "model": "llama-3-8b",
            "usage": {},
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == ""


class TestVLLMProviderListModels:
    async def test_list_models(self) -> None:
        provider = VLLMProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "meta-llama/Llama-3-8B-Instruct"},
                {"id": "mistralai/Mistral-7B"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        models = await provider.list_models()
        assert len(models) == 2


class TestVLLMProviderHealthcheck:
    async def test_healthy_via_health_endpoint(self) -> None:
        provider = VLLMProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        assert await provider.healthcheck() is True

    async def test_healthy_via_models_fallback(self) -> None:
        provider = VLLMProvider()
        mock_healthy = MagicMock()
        mock_healthy.status_code = 200

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("health down")
            return mock_healthy

        provider._client = AsyncMock()
        provider._client.get = mock_get

        assert await provider.healthcheck() is True

    async def test_unhealthy(self) -> None:
        provider = VLLMProvider()
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        assert await provider.healthcheck() is False

    async def test_close(self) -> None:
        provider = VLLMProvider()
        provider._client = AsyncMock()

        await provider.close()
        provider._client.aclose.assert_called_once()
