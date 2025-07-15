# ABOUTME: Common interfaces for cross-cutting concerns
# ABOUTME: Includes rate limiting and other shared interface contracts

from .rate_limiter import AbstractRateLimiter

__all__ = [
    "AbstractRateLimiter",
]
