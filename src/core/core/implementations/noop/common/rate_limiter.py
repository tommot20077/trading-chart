# ABOUTME: NoOp implementation of AbstractRateLimiter that always allows requests
# ABOUTME: Provides minimal rate limiting functionality for testing scenarios

from core.interfaces.common.rate_limiter import AbstractRateLimiter


class NoOpRateLimiter(AbstractRateLimiter):
    """
    No-operation implementation of AbstractRateLimiter.

    This implementation provides minimal rate limiting functionality that always
    allows requests without performing any actual rate limiting. It's useful
    for testing, performance benchmarking, and scenarios where rate limiting
    is not required.

    Features:
    - Always allows token acquisition (never blocks)
    - No actual rate limiting logic or token tracking
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where rate limiting should be bypassed
    - Performance benchmarking without rate limiting overhead
    - Development environments where rate limiting is not needed
    - Fallback when rate limiting systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation rate limiter."""
        # No initialization needed for NoOp implementation
        pass

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens - always succeeds immediately.

        This implementation always returns True without performing any
        actual rate limiting checks or token bucket operations.

        Args:
            tokens: Number of tokens to acquire (ignored)

        Returns:
            True (always allows token acquisition)
        """
        # Always allow token acquisition in NoOp implementation
        return True
