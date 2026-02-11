"""Tests for adversarial_framework.utils.token_counter module."""

from __future__ import annotations

from adversarial_framework.utils.token_counter import (
    count_messages_tokens,
    count_tokens,
    estimate_cost,
)

# count_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0

    def test_known_model(self):
        tokens = count_tokens("Hello", model="gpt-4o")
        assert tokens > 0

    def test_unknown_model_falls_back(self):
        # Ollama models are not in tiktoken; should fallback
        tokens = count_tokens("Hello", model="llama3:8b")
        assert tokens > 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("Hi " * 100)
        assert long > short

    def test_consistent_results(self):
        t1 = count_tokens("test string")
        t2 = count_tokens("test string")
        assert t1 == t2


# estimate_cost


class TestEstimateCost:
    def test_ollama_free(self):
        cost = estimate_cost(1000, 500, provider="ollama")
        assert cost == 0.0

    def test_openai_non_zero(self):
        cost = estimate_cost(1000, 500, provider="openai")
        assert cost > 0.0

    def test_anthropic_non_zero(self):
        cost = estimate_cost(1000, 500, provider="anthropic")
        assert cost > 0.0

    def test_google_non_zero(self):
        cost = estimate_cost(1000, 500, provider="google")
        assert cost > 0.0

    def test_unknown_provider_defaults_to_ollama(self):
        cost = estimate_cost(1000, 500, provider="unknown")
        assert cost == 0.0

    def test_zero_tokens_zero_cost(self):
        cost = estimate_cost(0, 0, provider="openai")
        assert cost == 0.0

    def test_cost_scales_with_tokens(self):
        small = estimate_cost(100, 100, provider="openai")
        large = estimate_cost(1000, 1000, provider="openai")
        assert large > small


# count_messages_tokens


class TestCountMessagesTokens:
    def test_empty_list(self):
        assert count_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        tokens = count_messages_tokens(msgs)
        # At minimum: content tokens + 4 overhead
        assert tokens >= 5

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        tokens = count_messages_tokens(msgs)
        assert tokens > 0

    def test_overhead_per_message(self):
        # Each message adds 4 tokens overhead
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
        tokens = count_messages_tokens(msgs)
        # 2 messages * 4 overhead each = 8
        assert tokens == 8

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        tokens = count_messages_tokens(msgs)
        # Empty content + 4 overhead
        assert tokens == 4

    def test_respects_model_param(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        t1 = count_messages_tokens(msgs, model="gpt-4o")
        t2 = count_messages_tokens(msgs, model="unknown-model")
        # Both should return positive values
        assert t1 > 0
        assert t2 > 0
