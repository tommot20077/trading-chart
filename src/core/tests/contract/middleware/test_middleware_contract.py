# ABOUTME: Contract tests for AbstractMiddleware interface implementations
# ABOUTME: Verifies that all middleware implementations comply with the interface contract

import pytest
from typing import Type, List

from core.interfaces.middleware import AbstractMiddleware
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestMiddlewareContract(ContractTestBase[AbstractMiddleware], AsyncContractTestMixin):
    """
    Contract tests for AbstractMiddleware interface.

    This test class verifies that all middleware implementations:
    - Properly inherit from AbstractMiddleware
    - Implement all required abstract methods
    - Follow the correct method signatures
    - Handle async operations correctly
    - Maintain proper priority ordering
    """

    @property
    def interface_class(self) -> Type[AbstractMiddleware]:
        """The AbstractMiddleware interface class."""
        return AbstractMiddleware

    @property
    def implementations(self) -> List[Type[AbstractMiddleware]]:
        """List of concrete middleware implementations to test."""
        # For now, we'll use a mock implementation
        # In the future, this will include real implementations from libs layer
        return [self._create_test_middleware()]

    def _create_test_middleware(self) -> Type[AbstractMiddleware]:
        """Create a test middleware implementation for contract testing."""

        class TestMiddleware(AbstractMiddleware):
            def __init__(self, priority: EventPriority = EventPriority.NORMAL):
                super().__init__(priority)
                self.processed_contexts = []

            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                """Test implementation of process method."""
                self.processed_contexts.append(context)
                return MiddlewareResult(
                    middleware_name=self.__class__.__name__, status=MiddlewareStatus.SUCCESS, data={"test": "data"}
                )

            def can_process(self, context: MiddlewareContext) -> bool:
                """Test implementation of can_process method."""
                return True

        return TestMiddleware

    @pytest.mark.contract
    def test_middleware_has_priority_attribute(self):
        """Verify middleware instances have priority attribute."""
        for impl_class in self.implementations:
            instance = impl_class()
            assert hasattr(instance, "priority"), f"{impl_class.__name__} missing priority attribute"
            assert isinstance(instance.priority, EventPriority), (
                f"{impl_class.__name__}.priority should be EventPriority"
            )

    @pytest.mark.contract
    def test_middleware_priority_ordering(self):
        """Verify middleware priority ordering works correctly."""
        for impl_class in self.implementations:
            high_priority = impl_class(EventPriority.HIGH)
            normal_priority = impl_class(EventPriority.NORMAL)
            low_priority = impl_class(EventPriority.LOW)

            # Test less than comparison
            assert high_priority < normal_priority
            assert normal_priority < low_priority
            assert high_priority < low_priority

            # Test greater than comparison
            assert normal_priority > high_priority
            assert low_priority > normal_priority
            assert low_priority > high_priority

            # Test equality
            another_high = impl_class(EventPriority.HIGH)
            assert high_priority == another_high

    @pytest.mark.contract
    def test_middleware_hash_and_equality(self):
        """Verify middleware hash and equality implementations."""
        for impl_class in self.implementations:
            middleware1 = impl_class(EventPriority.HIGH)
            middleware2 = impl_class(EventPriority.HIGH)
            middleware3 = impl_class(EventPriority.LOW)

            # Test equality
            assert middleware1 == middleware2
            assert middleware1 != middleware3

            # Test hash consistency
            assert hash(middleware1) == hash(middleware2)
            # Hash inequality not guaranteed, but should be consistent
            assert hash(middleware1) == hash(middleware1)

    @pytest.mark.contract
    def test_middleware_string_representation(self):
        """Verify middleware string representation."""
        for impl_class in self.implementations:
            middleware = impl_class(EventPriority.HIGH)
            repr_str = repr(middleware)

            assert impl_class.__name__ in repr_str
            assert "HIGH" in repr_str
            assert "priority" in repr_str

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_process_returns_middleware_result(self):
        """Verify process method returns MiddlewareResult."""
        for impl_class in self.implementations:
            middleware = impl_class()
            context = MiddlewareContext(data="test")

            result = await middleware.process(context)

            assert isinstance(result, MiddlewareResult)
            assert result.middleware_name is not None
            assert isinstance(result.status, MiddlewareStatus)
            assert isinstance(result.should_continue, bool)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_process_with_various_contexts(self):
        """Test process method with different context types."""
        for impl_class in self.implementations:
            middleware = impl_class()

            # Test with empty context
            empty_context = MiddlewareContext()
            result = await middleware.process(empty_context)
            assert isinstance(result, MiddlewareResult)

            # Test with context containing data
            data_context = MiddlewareContext(data={"key": "value"})
            result = await middleware.process(data_context)
            assert isinstance(result, MiddlewareResult)

            # Test with context containing metadata
            metadata_context = MiddlewareContext(metadata={"auth": "token", "user_id": "123"})
            result = await middleware.process(metadata_context)
            assert isinstance(result, MiddlewareResult)

    @pytest.mark.contract
    def test_can_process_returns_boolean(self):
        """Verify can_process method returns boolean."""
        for impl_class in self.implementations:
            middleware = impl_class()
            context = MiddlewareContext(data="test")

            result = middleware.can_process(context)
            assert isinstance(result, bool)

    @pytest.mark.contract
    def test_can_process_with_various_contexts(self):
        """Test can_process method with different context types."""
        for impl_class in self.implementations:
            middleware = impl_class()

            # Test with empty context
            empty_context = MiddlewareContext()
            result = middleware.can_process(empty_context)
            assert isinstance(result, bool)

            # Test with context containing event type
            event_context = MiddlewareContext(event_type="TRADE")
            result = middleware.can_process(event_context)
            assert isinstance(result, bool)

            # Test with cancelled context
            cancelled_context = MiddlewareContext(is_cancelled=True)
            result = middleware.can_process(cancelled_context)
            assert isinstance(result, bool)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_middleware_error_handling(self):
        """Test middleware error handling behavior."""

        class ErrorMiddleware(AbstractMiddleware):
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                raise ValueError("Test error")

            def can_process(self, context: MiddlewareContext) -> bool:
                return True

        middleware = ErrorMiddleware()
        context = MiddlewareContext(data="test")

        # The process method should raise the exception
        # It's up to the pipeline to handle it properly
        with pytest.raises(ValueError, match="Test error"):
            await middleware.process(context)

    @pytest.mark.contract
    def test_middleware_initialization_with_priority(self):
        """Test middleware initialization with different priorities."""
        for impl_class in self.implementations:
            # Test with default priority
            default_middleware = impl_class()
            assert default_middleware.priority == EventPriority.NORMAL

            # Test with specific priority
            high_priority_middleware = impl_class(EventPriority.HIGH)
            assert high_priority_middleware.priority == EventPriority.HIGH

            # Test with low priority
            low_priority_middleware = impl_class(EventPriority.LOW)
            assert low_priority_middleware.priority == EventPriority.LOW
