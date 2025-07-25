# ABOUTME: Unit tests for InMemoryRateLimiter implementation
# ABOUTME: Tests token bucket algorithm, thread safety, and rate limiting behavior

import asyncio
import pytest
import pytest_asyncio
import time_machine

from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter, TokenBucket


class TestTokenBucket:
    """Test the TokenBucket class directly."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10  # Should start full

    @pytest.mark.unit
    def test_initialization_with_custom_tokens(self):
        """Test token bucket initialization with custom initial tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0, initial_tokens=5)
        assert bucket.tokens == 5

    @time_machine.travel("2024-01-01 12:00:00", tick=False)
    @pytest.mark.unit
    def test_consume_tokens_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)

        # Should be able to consume tokens
        assert bucket.try_consume(3) is True
        assert bucket.tokens == 7

        # Should be able to consume more
        assert bucket.try_consume(2) is True
        assert bucket.tokens == 5

    @time_machine.travel("2024-01-01 12:00:00", tick=False)
    @pytest.mark.unit
    def test_consume_tokens_insufficient(self):
        """Test token consumption when insufficient tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0, initial_tokens=3)

        # Should fail when requesting more tokens than available
        assert bucket.try_consume(5) is False
        assert bucket.tokens == 3  # Tokens should remain unchanged

    @time_machine.travel("2024-01-01 12:00:00", tick=False)
    @pytest.mark.unit
    def test_token_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0, initial_tokens=0)

        # Initially no tokens
        assert bucket.tokens == 0

        # Move forward 1 second, should add 2 tokens
        with time_machine.travel("2024-01-01 12:00:01", tick=False):
            bucket._refill()
            assert bucket.tokens == 2.0

        # Move forward another 2.5 seconds, should add 5 more tokens (total 7)
        with time_machine.travel("2024-01-01 12:00:03.5", tick=False):
            bucket._refill()
            assert bucket.tokens == 7.0

    @time_machine.travel("2024-01-01 12:00:00", tick=False)
    @pytest.mark.unit
    def test_token_refill_cap_at_capacity(self):
        """Test that token refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0, initial_tokens=0)

        # Move forward 1 second, would add 10 tokens but should cap at 5
        with time_machine.travel("2024-01-01 12:00:01", tick=False):
            bucket._refill()
            assert bucket.tokens == 5.0


class TestInMemoryRateLimiter:
    """Test the InMemoryRateLimiter class."""

    @pytest_asyncio.fixture
    async def rate_limiter(self):
        """Create a rate limiter for testing."""
        limiter = InMemoryRateLimiter(capacity=10, refill_rate=2.0, cleanup_interval=60.0)
        yield limiter
        # Cleanup
        await limiter.close()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = InMemoryRateLimiter(capacity=100, refill_rate=10.0, cleanup_interval=300.0, identifier_key="test")

        assert limiter.capacity == 100
        assert limiter.refill_rate == 10.0
        assert limiter.cleanup_interval == 300.0
        assert limiter.identifier_key == "test"

        await limiter.close()

    @pytest.mark.asyncio
    async def test_acquire_default_identifier(self, rate_limiter):
        """Test acquiring tokens with default identifier."""
        # Should succeed initially
        assert await rate_limiter.acquire(1) is True
        assert await rate_limiter.acquire(5) is True

        # Should have 4 tokens left
        remaining = await rate_limiter.get_remaining_tokens()
        assert remaining == 4

    @pytest.mark.asyncio
    async def test_acquire_for_specific_identifier(self, rate_limiter):
        """Test acquiring tokens for specific identifiers."""
        # Different identifiers should have separate buckets
        assert await rate_limiter.acquire_for_identifier("user1", 5) is True
        assert await rate_limiter.acquire_for_identifier("user2", 5) is True

        # Each should have 5 tokens remaining
        assert await rate_limiter.get_remaining_tokens("user1") == 5
        assert await rate_limiter.get_remaining_tokens("user2") == 5

    @pytest.mark.asyncio
    async def test_acquire_insufficient_tokens(self, rate_limiter):
        """Test acquiring more tokens than available."""
        # Consume most tokens
        assert await rate_limiter.acquire(8) is True

        # Should fail when requesting more than remaining
        assert await rate_limiter.acquire(5) is False

        # Should still have 2 tokens
        remaining = await rate_limiter.get_remaining_tokens()
        assert remaining == 2

    @pytest.mark.asyncio
    async def test_acquire_invalid_tokens(self, rate_limiter):
        """Test acquiring invalid number of tokens."""
        with pytest.raises(ValueError, match="Number of tokens must be positive"):
            await rate_limiter.acquire_for_identifier("user1", 0)

        with pytest.raises(ValueError, match="Number of tokens must be positive"):
            await rate_limiter.acquire_for_identifier("user1", -1)

    @time_machine.travel("2024-01-01 12:00:00", tick=False)
    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        limiter = InMemoryRateLimiter(capacity=10, refill_rate=2.0)

        # Consume all tokens
        assert await limiter.acquire(10) is True
        assert await limiter.get_remaining_tokens() == 0

        # Move forward 1 second, should have 2 tokens
        with time_machine.travel("2024-01-01 12:00:01", tick=False):
            assert await limiter.acquire(2) is True
            assert await limiter.get_remaining_tokens() == 0

        # Move forward another 2.5 seconds, should have 5 tokens
        with time_machine.travel("2024-01-01 12:00:03.5", tick=False):
            assert await limiter.acquire(5) is True
            assert await limiter.get_remaining_tokens() == 0

        await limiter.close()

    @pytest.mark.asyncio
    async def test_reset_bucket(self, rate_limiter):
        """Test resetting token bucket."""
        # Consume some tokens
        assert await rate_limiter.acquire(7) is True
        assert await rate_limiter.get_remaining_tokens() == 3

        # Reset bucket
        await rate_limiter.reset_bucket()
        assert await rate_limiter.get_remaining_tokens() == 10

    @pytest.mark.asyncio
    async def test_reset_specific_bucket(self, rate_limiter):
        """Test resetting specific identifier's bucket."""
        # Consume tokens for different identifiers
        assert await rate_limiter.acquire_for_identifier("user1", 7) is True
        assert await rate_limiter.acquire_for_identifier("user2", 5) is True

        # Reset only user1's bucket
        await rate_limiter.reset_bucket("user1")

        assert await rate_limiter.get_remaining_tokens("user1") == 10
        assert await rate_limiter.get_remaining_tokens("user2") == 5

    @pytest.mark.asyncio
    async def test_concurrent_access(self, rate_limiter):
        """Test thread safety with concurrent access."""

        async def consume_tokens(identifier: str, tokens: int):
            return await rate_limiter.acquire_for_identifier(identifier, tokens)

        # Create multiple concurrent tasks
        tasks = [
            consume_tokens("user1", 2),
            consume_tokens("user1", 3),
            consume_tokens("user1", 4),
            consume_tokens("user2", 5),
            consume_tokens("user2", 3),
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed since they're using separate buckets or within limits
        assert all(results)

        # Check remaining tokens
        user1_remaining = await rate_limiter.get_remaining_tokens("user1")
        user2_remaining = await rate_limiter.get_remaining_tokens("user2")

        assert user1_remaining == 1  # 10 - 2 - 3 - 4 = 1
        assert user2_remaining == 2  # 10 - 5 - 3 = 2

    @pytest.mark.asyncio
    async def test_cleanup_unused_buckets(self):
        """Test cleanup of unused buckets."""
        import time
        
        limiter = InMemoryRateLimiter(
            capacity=10,
            refill_rate=2.0,
            cleanup_interval=0.1,  # Very short interval for testing
        )

        # Create some buckets
        await limiter.acquire_for_identifier("user1", 1)
        await limiter.acquire_for_identifier("user2", 1)

        # Should have 2 buckets
        assert len(limiter._buckets) == 2

        # Manually set old access times to trigger cleanup
        old_time = time.time() - 1000  # Very old timestamp
        with limiter._lock:
            limiter._last_access["user1"] = old_time
            limiter._last_access["user2"] = old_time

        # Run cleanup
        await limiter._cleanup_unused_buckets()

        # Buckets should be cleaned up
        assert len(limiter._buckets) == 0

        await limiter.close()

    @pytest.mark.asyncio
    async def test_close_cleanup(self, rate_limiter):
        """Test proper cleanup when closing."""
        # Create some buckets
        await rate_limiter.acquire_for_identifier("user1", 1)
        await rate_limiter.acquire_for_identifier("user2", 1)

        assert len(rate_limiter._buckets) == 2

        # Close should clean up
        await rate_limiter.close()

        assert len(rate_limiter._buckets) == 0
        assert rate_limiter._should_cleanup is False


@pytest.mark.asyncio
async def test_rate_limiter_integration():
    """Integration test simulating real-world usage."""
    import time
    
    limiter = InMemoryRateLimiter(capacity=5, refill_rate=1.0)

    # Simulate API requests
    api_calls = []

    async def make_api_call(user_id: str):
        success = await limiter.acquire_for_identifier(user_id, 1)
        api_calls.append((user_id, success, time.time()))
        return success

    # User makes 3 quick requests (should all succeed)
    results = await asyncio.gather(*[make_api_call("user123") for _ in range(3)])
    assert all(results)

    # User makes 3 more requests (should have 2 succeed, 1 fail)
    results = await asyncio.gather(*[make_api_call("user123") for _ in range(3)])
    assert sum(results) == 2  # Only 2 should succeed

    await limiter.close()
