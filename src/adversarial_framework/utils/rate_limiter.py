"""Redis-backed token bucket rate limiter.

Provides per-provider, per-session rate limiting to prevent API abuse
and stay within provider rate limits during adversarial sessions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis.asyncio as aioredis


@dataclass(frozen=True)
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    retry_after: float  # seconds until next token available (0.0 if allowed)
    limit: int
    window_seconds: float


class TokenBucketRateLimiter:
    """Async Redis-backed token bucket rate limiter.

    Each bucket is identified by a key (e.g., ``"provider:openai:session:abc"``).
    Tokens refill at a steady rate up to the configured capacity.

    Args:
        redis_url: Redis connection URL.
        default_capacity: Maximum tokens in the bucket.
        default_refill_rate: Tokens added per second.
        key_prefix: Prefix for all Redis keys.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_capacity: int = 60,
        default_refill_rate: float = 1.0,
        key_prefix: str = "ratelimit",
    ) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._default_capacity = default_capacity
        self._default_refill_rate = default_refill_rate
        self._key_prefix = key_prefix

    def _key(self, bucket_id: str) -> str:
        return f"{self._key_prefix}:{bucket_id}"

    async def acquire(
        self,
        bucket_id: str,
        tokens: int = 1,
        capacity: int | None = None,
        refill_rate: float | None = None,
    ) -> RateLimitResult:
        """Try to consume *tokens* from the bucket.

        Implements a token bucket algorithm atomically via Redis Lua script.

        Args:
            bucket_id: Unique identifier for the bucket.
            tokens: Number of tokens to consume.
            capacity: Override default bucket capacity.
            refill_rate: Override default refill rate (tokens/sec).

        Returns:
            RateLimitResult indicating whether the request is allowed.
        """
        cap = capacity or self._default_capacity
        rate = refill_rate or self._default_refill_rate
        now = time.time()
        key = self._key(bucket_id)

        # Atomic Lua script for token bucket
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested = tonumber(ARGV[4])

        local data = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(data[1])
        local last_refill = tonumber(data[2])

        if tokens == nil then
            -- First request: initialise the bucket
            tokens = capacity
            last_refill = now
        end

        -- Refill tokens based on elapsed time
        local elapsed = now - last_refill
        local refill = elapsed * refill_rate
        tokens = math.min(capacity, tokens + refill)
        last_refill = now

        local allowed = 0
        local remaining = tokens

        if tokens >= requested then
            tokens = tokens - requested
            remaining = tokens
            allowed = 1
        end

        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
        redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 60)

        local retry_after = 0
        if allowed == 0 then
            retry_after = (requested - tokens) / refill_rate
        end

        return {allowed, math.floor(remaining), tostring(retry_after)}
        """

        result = await self._redis.eval(
            lua_script, 1, key, cap, rate, now, tokens
        )
        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = float(result[2])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            retry_after=retry_after if not allowed else 0.0,
            limit=cap,
            window_seconds=cap / rate,
        )

    async def reset(self, bucket_id: str) -> None:
        """Reset a bucket, restoring full capacity."""
        await self._redis.delete(self._key(bucket_id))

    async def peek(
        self,
        bucket_id: str,
        capacity: int | None = None,
        refill_rate: float | None = None,
    ) -> RateLimitResult:
        """Check remaining tokens without consuming any."""
        cap = capacity or self._default_capacity
        rate = refill_rate or self._default_refill_rate
        now = time.time()
        key = self._key(bucket_id)

        data = await self._redis.hmget(key, "tokens", "last_refill")
        tokens = float(data[0]) if data[0] else float(cap)
        last_refill = float(data[1]) if data[1] else now

        elapsed = now - last_refill
        refill = elapsed * rate
        current = min(cap, tokens + refill)

        return RateLimitResult(
            allowed=current >= 1,
            remaining=int(current),
            retry_after=max(0.0, (1 - current) / rate) if current < 1 else 0.0,
            limit=cap,
            window_seconds=cap / rate,
        )

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._redis.aclose()


# Provider-specific rate limit presets (requests per minute)
PROVIDER_RATE_LIMITS: dict[str, dict[str, int]] = {
    "ollama": {"capacity": 120, "refill_rate": 2},  # local, generous
    "openai": {"capacity": 60, "refill_rate": 1},  # Tier 1 default
    "anthropic": {"capacity": 50, "refill_rate": 0.83},  # Conservative
    "google": {"capacity": 60, "refill_rate": 1},
    "vllm": {"capacity": 120, "refill_rate": 2},  # local, generous
}


def get_provider_limits(provider: str) -> dict[str, int | float]:
    """Get rate limit presets for a provider."""
    return dict(PROVIDER_RATE_LIMITS.get(provider, PROVIDER_RATE_LIMITS["ollama"]))
