# ABOUTME: Contract tests for AbstractRateLimiter interface
# ABOUTME: Verifies all ratelimiter implementations comply with the interface contract

import pytest
from typing import Type, List

from core.interfaces.common.rate_limiter import AbstractRateLimiter
from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestRateLimiterContract(ContractTestBase[AbstractRateLimiter], AsyncContractTestMixin):
    """Contract tests for AbstractRateLimiter interface."""

    @property
    def interface_class(self) -> Type[AbstractRateLimiter]:
        return AbstractRateLimiter

    @property
    def implementations(self) -> List[Type[AbstractRateLimiter]]:
        return [
            InMemoryRateLimiter,
        ]

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_acquire_contract(self):
        """Test acquire method contract behavior."""
        for impl_class in self.implementations:
            # Test with default parameters
            limiter = impl_class()

            # Should accept positive token counts
            result = await limiter.acquire(1)
            assert isinstance(result, bool), f"{impl_class.__name__}.acquire should return bool"

            # Should handle default parameter
            result = await limiter.acquire()
            assert isinstance(result, bool), f"{impl_class.__name__}.acquire should handle default parameter"

            # Cleanup
            if hasattr(limiter, "close"):
                await limiter.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_acquire_parameter_validation(self):
        """Test acquire method parameter validation."""
        for impl_class in self.implementations:
            limiter = impl_class()

            # Should reject invalid token counts
            with pytest.raises((ValueError, TypeError)):
                await limiter.acquire(0)

            with pytest.raises((ValueError, TypeError)):
                await limiter.acquire(-1)

            # Cleanup
            if hasattr(limiter, "close"):
                await limiter.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_acquire_return_type_consistency(self):
        """Test that acquire always returns boolean."""
        for impl_class in self.implementations:
            limiter = impl_class(capacity=2, refill_rate=1.0)

            # Multiple calls should all return boolean
            for _ in range(5):  # More than capacity to test both success and failure
                result = await limiter.acquire(1)
                assert isinstance(result, bool), f"{impl_class.__name__}.acquire must always return bool"

            # Cleanup
            if hasattr(limiter, "close"):
                await limiter.close()
