"""Tests for the provider factory function."""

from __future__ import annotations

import pytest

from adversarial_framework.agents.target.providers import get_provider
from adversarial_framework.agents.target.providers.ollama import (
    OllamaProvider,
)


class TestGetProvider:
    def test_ollama_provider(self) -> None:
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_ollama_with_kwargs(self) -> None:
        provider = get_provider("ollama", base_url="http://gpu:11434")
        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://gpu:11434"

    def test_vllm_provider(self) -> None:
        from adversarial_framework.agents.target.providers.vllm import (
            VLLMProvider,
        )

        provider = get_provider("vllm")
        assert isinstance(provider, VLLMProvider)

    def test_google_provider(self) -> None:
        from adversarial_framework.agents.target.providers.google import (
            GoogleProvider,
        )

        provider = get_provider("google", api_key="test-key")
        assert isinstance(provider, GoogleProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_openai_provider(self) -> None:
        try:
            provider = get_provider("openai", api_key="test-key")
            assert provider.name == "openai"
        except ImportError:
            pytest.skip("openai package not installed")

    def test_anthropic_provider(self) -> None:
        try:
            provider = get_provider("anthropic", api_key="test-key")
            assert provider.name == "anthropic"
        except ImportError:
            pytest.skip("anthropic package not installed")
