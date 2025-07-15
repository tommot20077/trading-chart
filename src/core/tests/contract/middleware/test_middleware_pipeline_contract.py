# ABOUTME: Contract tests for AbstractMiddlewarePipeline interface implementations
# ABOUTME: Verifies that all middleware pipeline implementations comply with the interface contract

import pytest
from typing import Type, List

from core.interfaces.middleware import AbstractMiddlewarePipeline, AbstractMiddleware
from core.implementations.memory.middleware import InMemoryMiddlewarePipeline
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestMiddlewarePipelineContract(ContractTestBase[AbstractMiddlewarePipeline], AsyncContractTestMixin):
    """
    Contract tests for AbstractMiddlewarePipeline interface.

    This test class verifies that all middleware pipeline implementations:
    - Properly inherit from AbstractMiddlewarePipeline
    - Implement all required abstract methods
    - Follow the correct method signatures
    - Handle async operations correctly
    - Maintain proper middleware ordering
    """

    @property
    def interface_class(self) -> Type[AbstractMiddlewarePipeline]:
        """The AbstractMiddlewarePipeline interface class."""
        return AbstractMiddlewarePipeline

    @property
    def implementations(self) -> List[Type[AbstractMiddlewarePipeline]]:
        """List of concrete middleware pipeline implementations to test."""
        return [InMemoryMiddlewarePipeline]

    def _create_test_middleware(
        self, priority: EventPriority = EventPriority.NORMAL, should_continue: bool = True
    ) -> AbstractMiddleware:
        """Create a test middleware for pipeline testing."""

        class TestMiddleware(AbstractMiddleware):
            def __init__(self, priority: EventPriority = EventPriority.NORMAL):
                super().__init__(priority)
                self.processed_contexts = []

            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                self.processed_contexts.append(context)
                return MiddlewareResult(
                    middleware_name=self.__class__.__name__,
                    status=MiddlewareStatus.SUCCESS,
                    data={"processed": True},
                    should_continue=should_continue,
                )

            def can_process(self, context: MiddlewareContext) -> bool:
                return True

        return TestMiddleware(priority)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_add_middleware_method(self):
        """Test add_middleware method functionality."""
        for impl_class in self.implementations:
            pipeline = impl_class()
            middleware = self._create_test_middleware()

            # Should not raise exception
            await pipeline.add_middleware(middleware)

            # Check count increased
            count = await pipeline.get_middleware_count()
            assert count == 1

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_remove_middleware_method(self):
        """Test remove_middleware method functionality."""
        for impl_class in self.implementations:
            pipeline = impl_class()
            middleware = self._create_test_middleware()

            # Add middleware first
            await pipeline.add_middleware(middleware)
            initial_count = await pipeline.get_middleware_count()

            # Remove middleware
            await pipeline.remove_middleware(middleware)
            final_count = await pipeline.get_middleware_count()

            assert final_count == initial_count - 1

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_remove_nonexistent_middleware_raises_error(self):
        """Test that removing non-existent middleware raises ValueError."""
        for impl_class in self.implementations:
            pipeline = impl_class()
            middleware = self._create_test_middleware()

            # Try to remove middleware that was never added
            with pytest.raises(ValueError):
                await pipeline.remove_middleware(middleware)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_execute_method_returns_middleware_result(self):
        """Test execute method returns MiddlewareResult."""
        for impl_class in self.implementations:
            pipeline = impl_class()
            context = MiddlewareContext(data="test")

            # Test empty pipeline
            result = await pipeline.execute(context)
            assert isinstance(result, MiddlewareResult)
            assert result.middleware_name is not None
            assert isinstance(result.status, MiddlewareStatus)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_execute_with_single_middleware(self):
        """Test execute method with single middleware."""
        for impl_class in self.implementations:
            pipeline = impl_class()
            middleware = self._create_test_middleware()

            await pipeline.add_middleware(middleware)

            context = MiddlewareContext(data="test")
            result = await pipeline.execute(context)

            assert isinstance(result, MiddlewareResult)
            assert result.is_successful()
            assert len(middleware.processed_contexts) == 1

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_execute_with_multiple_middleware(self):
        """Test execute method with multiple middleware."""
        for impl_class in self.implementations:
            pipeline = impl_class()

            # Add middleware in reverse priority order to test sorting
            low_middleware = self._create_test_middleware(EventPriority.LOW)
            high_middleware = self._create_test_middleware(EventPriority.HIGH)
            normal_middleware = self._create_test_middleware(EventPriority.NORMAL)

            await pipeline.add_middleware(low_middleware)
            await pipeline.add_middleware(high_middleware)
            await pipeline.add_middleware(normal_middleware)

            context = MiddlewareContext(data="test")
            result = await pipeline.execute(context)

            assert isinstance(result, MiddlewareResult)
            assert result.is_successful()

            # Verify all middleware were processed
            assert len(high_middleware.processed_contexts) == 1
            assert len(normal_middleware.processed_contexts) == 1
            assert len(low_middleware.processed_contexts) == 1

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_execute_stops_on_should_continue_false(self):
        """Test execute method stops when middleware returns should_continue=False."""

        class StopMiddleware(AbstractMiddleware):
            def __init__(self):
                super().__init__(EventPriority.HIGH)
                self.processed = False

            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                self.processed = True
                return MiddlewareResult(
                    middleware_name=self.__class__.__name__, status=MiddlewareStatus.SUCCESS, should_continue=False
                )

            def can_process(self, context: MiddlewareContext) -> bool:
                return True

        for impl_class in self.implementations:
            pipeline = impl_class()

            stop_middleware = StopMiddleware()
            later_middleware = self._create_test_middleware(EventPriority.LOW)

            await pipeline.add_middleware(stop_middleware)
            await pipeline.add_middleware(later_middleware)

            context = MiddlewareContext(data="test")
            result = await pipeline.execute(context)

            assert isinstance(result, MiddlewareResult)
            assert stop_middleware.processed == True
            assert len(later_middleware.processed_contexts) == 0  # Should not be processed

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_get_middleware_count_method(self):
        """Test get_middleware_count method."""
        for impl_class in self.implementations:
            pipeline = impl_class()

            # Empty pipeline
            count = await pipeline.get_middleware_count()
            assert count == 0

            # Add middleware
            await pipeline.add_middleware(self._create_test_middleware())
            count = await pipeline.get_middleware_count()
            assert count == 1

            # Add another
            await pipeline.add_middleware(self._create_test_middleware())
            count = await pipeline.get_middleware_count()
            assert count == 2

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_clear_method(self):
        """Test clear method functionality."""
        for impl_class in self.implementations:
            pipeline = impl_class()

            # Add some middleware
            await pipeline.add_middleware(self._create_test_middleware())
            await pipeline.add_middleware(self._create_test_middleware())

            initial_count = await pipeline.get_middleware_count()
            assert initial_count > 0

            # Clear pipeline
            await pipeline.clear()

            final_count = await pipeline.get_middleware_count()
            assert final_count == 0

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_get_middleware_by_priority_method(self):
        """Test get_middleware_by_priority method."""
        for impl_class in self.implementations:
            pipeline = impl_class()

            # Add middleware in random order
            low_middleware = self._create_test_middleware(EventPriority.LOW)
            high_middleware = self._create_test_middleware(EventPriority.HIGH)
            normal_middleware = self._create_test_middleware(EventPriority.NORMAL)

            await pipeline.add_middleware(low_middleware)
            await pipeline.add_middleware(high_middleware)
            await pipeline.add_middleware(normal_middleware)

            # Get sorted middleware
            sorted_middleware = await pipeline.get_middleware_by_priority()

            assert len(sorted_middleware) == 3
            assert sorted_middleware[0].priority == EventPriority.HIGH
            assert sorted_middleware[1].priority == EventPriority.NORMAL
            assert sorted_middleware[2].priority == EventPriority.LOW

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_execute_with_cancelled_context(self):
        """Test execute method with cancelled context."""
        for impl_class in self.implementations:
            pipeline = impl_class()
            middleware = self._create_test_middleware()

            await pipeline.add_middleware(middleware)

            # Create cancelled context
            context = MiddlewareContext(data="test", is_cancelled=True)
            result = await pipeline.execute(context)

            assert isinstance(result, MiddlewareResult)
            # Middleware should not be processed when context is cancelled
            assert len(middleware.processed_contexts) == 0

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_execute_with_middleware_exception(self):
        """Test execute method handles middleware exceptions."""

        class ErrorMiddleware(AbstractMiddleware):
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                raise ValueError("Test error")

            def can_process(self, context: MiddlewareContext) -> bool:
                return True

        for impl_class in self.implementations:
            pipeline = impl_class()
            error_middleware = ErrorMiddleware()
            later_middleware = self._create_test_middleware(EventPriority.LOW)

            await pipeline.add_middleware(error_middleware)
            await pipeline.add_middleware(later_middleware)

            context = MiddlewareContext(data="test")
            result = await pipeline.execute(context)

            assert isinstance(result, MiddlewareResult)
            # Pipeline should handle the exception gracefully
            # Later middleware should not be processed
            assert len(later_middleware.processed_contexts) == 0
