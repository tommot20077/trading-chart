# ABOUTME: In-memory implementation of AbstractRateLimiter using token bucket algorithm
# ABOUTME: Provides thread-safe rate limiting with configurable capacity and refill rate

import asyncio
import math
import time
import threading
from typing import Dict, Optional

from core.interfaces.common import AbstractRateLimiter


class TokenBucket:
    """
    Thread-safe token bucket implementation for rate limiting.

    The token bucket algorithm allows for burst traffic up to the bucket capacity
    while maintaining a steady rate over time.
    """

    def __init__(self, capacity: int, refill_rate: float, initial_tokens: Optional[int] = None):
        """
        Initialize a token bucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Number of tokens added per second
            initial_tokens: Initial number of tokens (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def try_consume(self, tokens: int) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were successfully consumed, False otherwise
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time and refill rate
        tokens_to_add = math.floor(elapsed * self.refill_rate)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class InMemoryRateLimiter(AbstractRateLimiter):
    """
    In-memory implementation of AbstractRateLimiter using token bucket algorithm.

    This implementation provides rate limiting functionality using the token bucket
    algorithm, which allows for burst traffic while maintaining overall rate limits.
    Each identifier (e.g., user ID, IP address) gets its own token bucket.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Per-identifier rate limiting
    - Thread-safe operations
    - Configurable capacity and refill rate
    - Automatic cleanup of unused buckets
    - Support for burst traffic

    The token bucket algorithm works by:
    1. Each identifier has a bucket with a maximum capacity
    2. Tokens are added to the bucket at a steady rate
    3. Requests consume tokens from the bucket
    4. If no tokens are available, the request is rate limited
    """

    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 10.0,
        cleanup_interval: float = 300.0,
        identifier_key: str = "default",
    ):
        """
        Initialize the in-memory rate limiter.

        Args:
            capacity: Maximum number of tokens per bucket (burst capacity)
            refill_rate: Number of tokens added per second to each bucket
            cleanup_interval: Interval in seconds to clean up unused buckets
            identifier_key: Default identifier for rate limiting
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.cleanup_interval = cleanup_interval
        self.identifier_key = identifier_key

        # Store token buckets per identifier
        self._buckets: Dict[str, TokenBucket] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._should_cleanup = True

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the rate limiter.

        Uses the default identifier for rate limiting. For per-user or per-IP
        rate limiting, use acquire_for_identifier instead.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were successfully acquired, False otherwise
        """
        return await self.acquire_for_identifier(self.identifier_key, tokens)

    async def acquire_for_identifier(self, identifier: str, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens for a specific identifier.

        Args:
            identifier: Unique identifier (e.g., user ID, IP address)
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were successfully acquired, False otherwise
        """
        if tokens <= 0:
            raise ValueError("Number of tokens must be positive")

        bucket = self._get_or_create_bucket(identifier)
        success = bucket.try_consume(tokens)

        # Update last access time for cleanup
        with self._lock:
            self._last_access[identifier] = time.time()

        # Start cleanup task if not already running
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        return success

    def _get_or_create_bucket(self, identifier: str) -> TokenBucket:
        """Get existing bucket or create a new one for the identifier."""
        with self._lock:
            if identifier not in self._buckets:
                self._buckets[identifier] = TokenBucket(capacity=self.capacity, refill_rate=self.refill_rate)
                self._last_access[identifier] = time.time()

            return self._buckets[identifier]

    async def _cleanup_loop(self):
        """Periodically clean up unused buckets to prevent memory leaks."""
        while self._should_cleanup:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_unused_buckets()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue cleanup loop even if individual cleanup fails
                pass

    async def _cleanup_unused_buckets(self):
        """Remove buckets that haven't been accessed recently."""
        current_time = time.time()
        cutoff_time = current_time - (self.cleanup_interval * 2)  # Keep buckets for 2x cleanup interval

        with self._lock:
            identifiers_to_remove = [
                identifier for identifier, last_access in self._last_access.items() if last_access < cutoff_time
            ]

            for identifier in identifiers_to_remove:
                self._buckets.pop(identifier, None)
                self._last_access.pop(identifier, None)

    async def get_remaining_tokens(self, identifier: Optional[str] = None) -> int:
        """
        Get the number of remaining tokens for an identifier.

        Args:
            identifier: Identifier to check (uses default if None)

        Returns:
            Number of remaining tokens
        """
        target_identifier = identifier or self.identifier_key

        with self._lock:
            if target_identifier not in self._buckets:
                return self.capacity

            bucket = self._buckets[target_identifier]
            # Trigger refill to get current token count
            bucket._refill()
            return int(bucket.tokens)

    async def reset_bucket(self, identifier: Optional[str] = None):
        """
        Reset the token bucket for an identifier.

        Args:
            identifier: Identifier to reset (uses default if None)
        """
        target_identifier = identifier or self.identifier_key

        with self._lock:
            if target_identifier in self._buckets:
                self._buckets[target_identifier].tokens = self.capacity
                self._buckets[target_identifier].last_refill = time.time()

    async def close(self):
        """Clean up resources and stop background tasks."""
        self._should_cleanup = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        with self._lock:
            self._buckets.clear()
            self._last_access.clear()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        if hasattr(self, "_cleanup_task") and self._cleanup_task and not self._cleanup_task.done():
            try:
                self._cleanup_task.cancel()
            except RuntimeError:
                # Event loop is closed, can't cancel the task
                pass
