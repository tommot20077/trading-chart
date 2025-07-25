# ABOUTME: Abstract rate limiter interface for controlling request rates and traffic throttling
# ABOUTME: Defines the contract for components that implement various rate limiting algorithms

from abc import abstractmethod, ABC


class AbstractRateLimiter(ABC):
    """
    [L0] Abstract base class for implementing rate limiting.

    This interface defines the contract for implementing rate limiting functionality.
    Concrete implementations should provide specific rate limiting algorithms
    (token bucket, sliding window, fixed window, etc.).

    Architecture note: This is a [L0] interface that has no dependencies on other
    asset_core modules and provides clean abstractions for [L2] pattern implementations.
    """

    @abstractmethod
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the rate limiter.

        This asynchronous method implements the specific rate limiting logic
        and returns whether the requested tokens could be acquired. It is designed
        to be non-blocking and return immediately, making it suitable for high-concurrency
        environments. For scenarios requiring blocking behavior (i.e., waiting until
        tokens are available), concrete implementations should provide separate wait methods.

        Args:
            tokens (int): The number of tokens to acquire. Defaults to 1.

        Returns:
            bool: True if tokens were successfully acquired, False otherwise (e.g., rate limit exceeded).

        Raises:
            ValueError: If tokens is less than or equal to zero.
            RateLimiterError: If the rate limiter encounters an internal error.
        """
        pass
