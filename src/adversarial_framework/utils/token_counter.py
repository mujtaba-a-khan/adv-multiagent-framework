"""Token counting and cost estimation utilities.

Uses tiktoken for byte-pair encoding token counts.  Cost rates are
approximate and configurable.
"""

from __future__ import annotations

import tiktoken

# Approximate per-token costs (USD) for various providers.
# Local Ollama models are effectively free; these rates are for
# optional commercial providers and cost estimation.
_COST_RATES: dict[str, dict[str, float]] = {
    "ollama": {"input": 0.0, "output": 0.0},
    "openai": {"input": 0.000003, "output": 0.000015},  # GPT-4o approx
    "anthropic": {"input": 0.000003, "output": 0.000015},  # Claude 3.5 approx
    "google": {"input": 0.0000005, "output": 0.0000015},  # Gemini Flash approx
}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in *text* using tiktoken.

    Falls back to ``cl100k_base`` encoding if the model is not recognised
    (which is the case for most Ollama models).
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    provider: str = "ollama",
) -> float:
    """Estimate the USD cost for a single LLM call."""
    rates = _COST_RATES.get(provider, _COST_RATES["ollama"])
    return (prompt_tokens * rates["input"]) + (completion_tokens * rates["output"])


def count_messages_tokens(messages: list[dict[str, str]], model: str = "gpt-4o") -> int:
    """Count total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""), model)
        # Overhead per message (role + separators): ~4 tokens
        total += 4
    return total
