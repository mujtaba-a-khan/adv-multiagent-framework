"""Tests for the provider abstraction using the MockProvider."""

from __future__ import annotations

import pytest

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)


class TestTokenUsage:
    def test_default_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_frozen(self):
        usage = TokenUsage(prompt_tokens=10)
        with pytest.raises(AttributeError):
            usage.prompt_tokens = 20  # type: ignore[misc]


class TestLLMResponse:
    def test_creation(self):
        resp = LLMResponse(
            content="Hello",
            usage=TokenUsage(10, 5, 15),
            model="test-model",
        )
        assert resp.content == "Hello"
        assert resp.usage.total_tokens == 15
        assert resp.finish_reason == "stop"
        assert resp.latency_ms == 0.0

    def test_frozen(self):
        resp = LLMResponse(
            content="Hello",
            usage=TokenUsage(),
            model="test",
        )
        with pytest.raises(AttributeError):
            resp.content = "modified"  # type: ignore[misc]


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_generate(self, mock_provider):
        resp = await mock_provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="test",
        )
        assert resp.content == "Mock response"
        assert resp.usage.total_tokens == 80
        assert len(mock_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_responses(self, mock_provider_factory):
        provider = mock_provider_factory(["First", "Second", "Third"])
        r1 = await provider.generate(messages=[], model="m")
        r2 = await provider.generate(messages=[], model="m")
        r3 = await provider.generate(messages=[], model="m")
        assert r1.content == "First"
        assert r2.content == "Second"
        assert r3.content == "Third"

    @pytest.mark.asyncio
    async def test_response_cycles(self, mock_provider_factory):
        provider = mock_provider_factory(["A", "B"])
        r1 = await provider.generate(messages=[], model="m")
        r2 = await provider.generate(messages=[], model="m")
        r3 = await provider.generate(messages=[], model="m")
        assert r1.content == "A"
        assert r2.content == "B"
        assert r3.content == "A"  # Cycles back

    @pytest.mark.asyncio
    async def test_list_models(self, mock_provider):
        models = await mock_provider.list_models()
        assert "mock-model-a" in models
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_healthcheck(self, mock_provider):
        assert await mock_provider.healthcheck() is True

    @pytest.mark.asyncio
    async def test_calls_tracking(self, mock_provider):
        await mock_provider.generate(
            messages=[{"role": "user", "content": "test"}],
            model="m",
            temperature=0.5,
        )
        assert len(mock_provider.calls) == 1
        assert mock_provider.calls[0]["temperature"] == 0.5
        assert mock_provider.calls[0]["model"] == "m"
