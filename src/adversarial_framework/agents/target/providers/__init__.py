"""LLM provider adapters for the Target Interface.

All providers implement ``BaseProvider`` and return ``LLMResponse``.
"""

from typing import Any

from adversarial_framework.agents.target.providers.base import (
    BaseProvider,
    LLMResponse,
    TokenUsage,
)
from adversarial_framework.agents.target.providers.ollama import OllamaProvider

__all__ = [
    "BaseProvider",
    "LLMResponse",
    "OllamaProvider",
    "TokenUsage",
]


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Factory to instantiate a provider by name.

    Args:
        name: Provider identifier ("ollama", "openai", "anthropic", "google").
        **kwargs: Provider-specific constructor arguments.

    Returns:
        An instance of the requested provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if name == "ollama":
        return OllamaProvider(**kwargs)
    elif name == "openai":
        from adversarial_framework.agents.target.providers.openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    elif name == "anthropic":
        from adversarial_framework.agents.target.providers.anthropic import (
            AnthropicProvider,
        )

        return AnthropicProvider(**kwargs)
    elif name == "google":
        from adversarial_framework.agents.target.providers.google import GoogleProvider

        return GoogleProvider(**kwargs)
    elif name == "vllm":
        from adversarial_framework.agents.target.providers.vllm import VLLMProvider

        return VLLMProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {name!r}. Choose from: ollama, openai, anthropic, google, vllm"
        )
