"""Tests for OllamaProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from adversarial_framework.agents.target.providers.ollama import OllamaProvider


class TestOllamaProviderInit:
    def test_name(self) -> None:
        provider = OllamaProvider()
        assert provider.name == "ollama"

    def test_strips_trailing_slash(self) -> None:
        provider = OllamaProvider(base_url="http://host:11434/")
        assert provider.base_url == "http://host:11434"

    def test_default_base_url(self) -> None:
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"


class TestOllamaProviderGenerate:
    async def test_generate_returns_llm_response(self) -> None:
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Hello world"},
            "model": "llama3:8b",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done_reason": "stop",
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama3:8b",
        )
        assert result.content == "Hello world"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.model == "llama3:8b"

    async def test_generate_default_model(self) -> None:
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "response"},
            "model": "llama3:8b",
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.generate(
            messages=[{"role": "user", "content": "test"}],
        )
        call_args = provider._client.post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3:8b"


class TestOllamaProviderListModels:
    async def test_list_models(self) -> None:
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:8b"},
                {"name": "phi4:14b"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        models = await provider.list_models()
        assert models == ["llama3:8b", "phi4:14b"]


class TestOllamaProviderHealthcheck:
    async def test_healthy(self) -> None:
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        assert await provider.healthcheck() is True

    async def test_unhealthy(self) -> None:
        provider = OllamaProvider()
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        assert await provider.healthcheck() is False


class TestOllamaProviderModelManagement:
    async def test_load_model(self) -> None:
        provider = OllamaProvider()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock()

        await provider.load_model("llama3:8b")
        provider._client.post.assert_called_once()
        call_args = provider._client.post.call_args
        assert call_args[1]["json"]["model"] == "llama3:8b"

    async def test_unload_model(self) -> None:
        provider = OllamaProvider()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock()

        await provider.unload_model("llama3:8b")
        call_args = provider._client.post.call_args
        assert call_args[1]["json"]["keep_alive"] == 0

    async def test_list_running_models(self) -> None:
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status = MagicMock()
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        result = await provider.list_running_models()
        assert result == [{"name": "llama3:8b"}]

    async def test_close(self) -> None:
        provider = OllamaProvider()
        provider._client = AsyncMock()

        await provider.close()
        provider._client.aclose.assert_called_once()
