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

        This method should implement the specific rate limiting logic
        and return whether the requested tokens could be acquired.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens were successfully acquired, False otherwise

        Note:
            This method should be non-blocking and return immediately.
            For blocking behavior, implementations should provide separate
            wait methods.
        """
        pass
