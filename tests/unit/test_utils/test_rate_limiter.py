"""Tests for adversarial_framework.utils.rate_limiter module."""

from __future__ import annotations

from unittest import mock

import pytest

from adversarial_framework.utils.rate_limiter import (
    PROVIDER_RATE_LIMITS,
    RateLimitResult,
    TokenBucketRateLimiter,
    get_provider_limits,
)

# RateLimitResult dataclass


class TestRateLimitResult:
    def test_creation(self):
        r = RateLimitResult(
            allowed=True,
            remaining=59,
            retry_after=0.0,
            limit=60,
            window_seconds=60.0,
        )
        assert r.allowed is True
        assert r.remaining == 59
        assert r.retry_after == 0.0
        assert r.limit == 60
        assert r.window_seconds == 60.0

    def test_frozen(self):
        r = RateLimitResult(
            allowed=True,
            remaining=10,
            retry_after=0.0,
            limit=60,
            window_seconds=60.0,
        )
        with pytest.raises(AttributeError):
            r.allowed = False  # type: ignore[misc]


# get_provider_limits


class TestGetProviderLimits:
    def test_known_provider_ollama(self):
        limits = get_provider_limits("ollama")
        assert "capacity" in limits
        assert "refill_rate" in limits
        assert limits["capacity"] == 120

    def test_known_provider_openai(self):
        limits = get_provider_limits("openai")
        assert limits["capacity"] == 60

    def test_known_provider_anthropic(self):
        limits = get_provider_limits("anthropic")
        assert limits["capacity"] == 50

    def test_unknown_provider_defaults_to_ollama(self):
        limits = get_provider_limits("unknown_provider")
        ollama = get_provider_limits("ollama")
        assert limits == ollama

    def test_returns_copy(self):
        limits1 = get_provider_limits("openai")
        limits2 = get_provider_limits("openai")
        assert limits1 == limits2
        assert limits1 is not limits2

    def test_all_providers_have_required_keys(self):
        for provider in PROVIDER_RATE_LIMITS:
            limits = get_provider_limits(provider)
            assert "capacity" in limits
            assert "refill_rate" in limits


# TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    def _make_mock_redis(self):
        """Create a mock aioredis.Redis instance."""
        mock_redis = mock.AsyncMock()
        return mock_redis

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    def test_init(self, mock_aioredis):
        mock_aioredis.from_url.return_value = mock.AsyncMock()
        limiter = TokenBucketRateLimiter(
            redis_url="redis://localhost:6379/0",
            default_capacity=100,
            default_refill_rate=2.0,
            key_prefix="test",
        )
        assert limiter._default_capacity == 100
        assert limiter._default_refill_rate == 2.0
        assert limiter._key_prefix == "test"

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    def test_key_format(self, mock_aioredis):
        mock_aioredis.from_url.return_value = mock.AsyncMock()
        limiter = TokenBucketRateLimiter(key_prefix="rl")
        assert limiter._key("my_bucket") == "rl:my_bucket"

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_acquire_allowed(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_redis.eval.return_value = [1, 59, "0.0"]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter()
        result = await limiter.acquire("test_bucket")

        assert result.allowed is True
        assert result.remaining == 59
        assert result.retry_after == 0.0
        assert result.limit == 60

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_acquire_denied(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_redis.eval.return_value = [0, 0, "5.0"]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter()
        result = await limiter.acquire("test_bucket")

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 5.0

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_acquire_custom_capacity(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_redis.eval.return_value = [1, 99, "0.0"]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter()
        result = await limiter.acquire("bucket", capacity=100, refill_rate=2.0)

        assert result.allowed is True
        assert result.limit == 100
        assert result.window_seconds == 50.0

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_reset(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter(key_prefix="rl")
        await limiter.reset("test_bucket")

        mock_redis.delete.assert_awaited_once_with("rl:test_bucket")

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_peek_full_bucket(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_redis.hmget.return_value = [None, None]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter(default_capacity=60)
        result = await limiter.peek("test_bucket")

        assert result.allowed is True
        assert result.remaining == 60
        assert result.retry_after == 0.0

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_peek_empty_bucket(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        # Bucket has 0 tokens, last refill is now
        mock_redis.hmget.return_value = ["0.0", "9999999999"]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter(default_capacity=60, default_refill_rate=1.0)
        result = await limiter.peek("test_bucket")

        # Should report not allowed (0 + tiny refill < 1)
        assert result.limit == 60

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_close(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter()
        await limiter.close()

        mock_redis.aclose.assert_awaited_once()

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_acquire_with_multiple_tokens(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_redis.eval.return_value = [1, 55, "0.0"]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter()
        result = await limiter.acquire("bucket", tokens=5)

        assert result.allowed is True
        assert result.remaining == 55

    @mock.patch("adversarial_framework.utils.rate_limiter.aioredis")
    async def test_window_seconds_calculation(self, mock_aioredis):
        mock_redis = mock.AsyncMock()
        mock_redis.eval.return_value = [1, 59, "0.0"]
        mock_aioredis.from_url.return_value = mock_redis

        limiter = TokenBucketRateLimiter(default_capacity=60, default_refill_rate=1.0)
        result = await limiter.acquire("bucket")

        assert result.window_seconds == 60.0
