# ABOUTME: Unit tests for NoOpRateLimiter
# ABOUTME: Tests for no-operation rate limiting implementation

import pytest
import asyncio

from core.implementations.noop.common.rate_limiter import NoOpRateLimiter


class TestNoOpRateLimiter:
    """Test cases for NoOpRateLimiter."""

    @pytest.mark.unit
    @pytest.mark.concurrency
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = NoOpRateLimiter()
        assert limiter is not None

    @pytest.mark.asyncio
    async def test_acquire_always_succeeds(self):
        """Test that acquire always returns True."""
        limiter = NoOpRateLimiter()

        # Should always return True
        assert await limiter.acquire() is True
        assert await limiter.acquire(1) is True
        assert await limiter.acquire(10) is True
        assert await limiter.acquire(1000) is True

    @pytest.mark.asyncio
    async def test_acquire_with_zero_tokens(self):
        """Test acquire with zero tokens."""
        limiter = NoOpRateLimiter()

        # Should still return True (NoOp behavior)
        assert await limiter.acquire(0) is True

    @pytest.mark.asyncio
    async def test_acquire_with_negative_tokens(self):
        """Test acquire with negative tokens."""
        limiter = NoOpRateLimiter()

        # Should still return True (NoOp behavior)
        assert await limiter.acquire(-1) is True
        assert await limiter.acquire(-100) is True

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Test concurrent acquire operations."""
        limiter = NoOpRateLimiter()

        # All should succeed concurrently
        tasks = [limiter.acquire() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        assert all(result is True for result in results)

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_acquire_performance(self, benchmark):
        """Test that acquire is fast (NoOp should be very fast)."""
        limiter = NoOpRateLimiter()

        async def acquire_operations():
            for _ in range(1000):
                await limiter.acquire()
            return 1000

        # Benchmark NoOp rate limiter operations
        result = await benchmark(acquire_operations)
        assert result == 1000
