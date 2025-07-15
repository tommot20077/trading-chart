# ABOUTME: Unit tests for InMemoryMiddlewarePipeline implementation
# ABOUTME: Tests pipeline functionality, middleware management, and execution flow

import pytest

from core.implementations.memory.middleware import InMemoryMiddlewarePipeline
from core.interfaces.middleware import AbstractMiddleware
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority


class TestInMemoryMiddlewarePipeline:
    """Unit tests for InMemoryMiddlewarePipeline."""

    def _create_test_middleware(
        self,
        priority: EventPriority = EventPriority.NORMAL,
        should_continue: bool = True,
        can_process: bool = True,
        name: str = "TestMiddleware",
    ) -> AbstractMiddleware:
        """Create a test middleware for testing purposes."""

        class TestMiddleware(AbstractMiddleware):
            def __init__(self, priority: EventPriority = EventPriority.NORMAL, name: str = "TestMiddleware"):
                super().__init__(priority)
                self.processed_contexts = []
                self.can_process_calls = []
                self.name = name

            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                self.processed_contexts.append(context)
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={"processed": True},
                    should_continue=should_continue,
                )

            def can_process(self, context: MiddlewareContext) -> bool:
                self.can_process_calls.append(context)
                return can_process

        return TestMiddleware(priority, name)

    @pytest.mark.unit
    def test_pipeline_creation(self):
        """Test creating empty pipeline."""
        pipeline = InMemoryMiddlewarePipeline()

        assert pipeline.name == "InMemoryMiddlewarePipeline"
        assert pipeline._middlewares == []
        assert pipeline._sorted_cache is None
        assert pipeline._cache_dirty is False

    @pytest.mark.unit
    def test_pipeline_creation_with_name(self):
        """Test creating pipeline with custom name."""
        pipeline = InMemoryMiddlewarePipeline("CustomPipeline")

        assert pipeline.name == "CustomPipeline"

    @pytest.mark.asyncio
    async def test_add_middleware(self):
        """Test adding middleware to pipeline."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware = self._create_test_middleware()

        # Initially empty
        count = await pipeline.get_middleware_count()
        assert count == 0

        # Add middleware
        await pipeline.add_middleware(middleware)

        count = await pipeline.get_middleware_count()
        assert count == 1
        assert middleware in pipeline._middlewares
        assert pipeline._cache_dirty is True

    @pytest.mark.asyncio
    async def test_add_multiple_middleware(self):
        """Test adding multiple middleware to pipeline."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware1 = self._create_test_middleware(EventPriority.HIGH)
        middleware2 = self._create_test_middleware(EventPriority.LOW)

        await pipeline.add_middleware(middleware1)
        await pipeline.add_middleware(middleware2)

        count = await pipeline.get_middleware_count()
        assert count == 2
        assert middleware1 in pipeline._middlewares
        assert middleware2 in pipeline._middlewares

    @pytest.mark.asyncio
    async def test_remove_middleware(self):
        """Test removing middleware from pipeline."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware = self._create_test_middleware()

        # Add middleware first
        await pipeline.add_middleware(middleware)
        count = await pipeline.get_middleware_count()
        assert count == 1

        # Remove middleware
        await pipeline.remove_middleware(middleware)

        count = await pipeline.get_middleware_count()
        assert count == 0
        assert middleware not in pipeline._middlewares
        assert pipeline._cache_dirty is True

    @pytest.mark.asyncio
    async def test_remove_nonexistent_middleware(self):
        """Test removing middleware that doesn't exist."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware = self._create_test_middleware()

        # Try to remove middleware that was never added
        with pytest.raises(ValueError, match="not found in pipeline"):
            await pipeline.remove_middleware(middleware)

    @pytest.mark.asyncio
    async def test_execute_empty_pipeline(self):
        """Test executing empty pipeline."""
        pipeline = InMemoryMiddlewarePipeline()
        context = MiddlewareContext(data="test")

        result = await pipeline.execute(context)

        assert isinstance(result, MiddlewareResult)
        assert result.middleware_name == "InMemoryMiddlewarePipeline"
        assert result.status == MiddlewareStatus.SKIPPED
        assert result.metadata["reason"] == "Empty pipeline"

    @pytest.mark.asyncio
    async def test_execute_single_middleware(self):
        """Test executing pipeline with single middleware."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware = self._create_test_middleware()

        await pipeline.add_middleware(middleware)

        context = MiddlewareContext(data="test")
        result = await pipeline.execute(context)

        assert isinstance(result, MiddlewareResult)
        assert result.middleware_name == "InMemoryMiddlewarePipeline"
        assert result.is_successful()
        assert len(middleware.processed_contexts) == 1
        assert middleware.processed_contexts[0] == context
        assert "TestMiddleware" in context.execution_path

    @pytest.mark.asyncio
    async def test_execute_multiple_middleware_priority_order(self):
        """Test executing multiple middleware in priority order."""
        pipeline = InMemoryMiddlewarePipeline()

        # Add middleware in reverse priority order
        low_middleware = self._create_test_middleware(EventPriority.LOW, name="LowMiddleware")
        high_middleware = self._create_test_middleware(EventPriority.HIGH, name="HighMiddleware")
        normal_middleware = self._create_test_middleware(EventPriority.NORMAL, name="NormalMiddleware")

        await pipeline.add_middleware(low_middleware)
        await pipeline.add_middleware(high_middleware)
        await pipeline.add_middleware(normal_middleware)

        context = MiddlewareContext(data="test")
        result = await pipeline.execute(context)

        assert result.is_successful()

        # Check execution order (HIGH -> NORMAL -> LOW)
        assert context.execution_path == ["HighMiddleware", "NormalMiddleware", "LowMiddleware"]

        # All middleware should have been processed
        assert len(high_middleware.processed_contexts) == 1
        assert len(normal_middleware.processed_contexts) == 1
        assert len(low_middleware.processed_contexts) == 1

    @pytest.mark.asyncio
    async def test_execute_stops_on_should_continue_false(self):
        """Test execution stops when middleware returns should_continue=False."""
        pipeline = InMemoryMiddlewarePipeline()

        # First middleware stops execution
        stop_middleware = self._create_test_middleware(EventPriority.HIGH, should_continue=False, name="StopMiddleware")
        later_middleware = self._create_test_middleware(EventPriority.LOW, name="LaterMiddleware")

        await pipeline.add_middleware(stop_middleware)
        await pipeline.add_middleware(later_middleware)

        context = MiddlewareContext(data="test")
        result = await pipeline.execute(context)

        assert result.is_successful()

        # Only first middleware should have been processed
        assert len(stop_middleware.processed_contexts) == 1
        assert len(later_middleware.processed_contexts) == 0
        assert context.execution_path == ["StopMiddleware"]

    @pytest.mark.asyncio
    async def test_execute_skips_middleware_that_cannot_process(self):
        """Test execution skips middleware that cannot process context."""
        pipeline = InMemoryMiddlewarePipeline()

        # First middleware cannot process
        skip_middleware = self._create_test_middleware(EventPriority.HIGH, can_process=False, name="SkipMiddleware")
        process_middleware = self._create_test_middleware(EventPriority.LOW, can_process=True, name="ProcessMiddleware")

        await pipeline.add_middleware(skip_middleware)
        await pipeline.add_middleware(process_middleware)

        context = MiddlewareContext(data="test")
        result = await pipeline.execute(context)

        assert result.is_successful()

        # Only second middleware should have been processed
        assert len(skip_middleware.processed_contexts) == 0
        assert len(process_middleware.processed_contexts) == 1
        assert context.execution_path == ["ProcessMiddleware"]

    @pytest.mark.asyncio
    async def test_execute_with_cancelled_context(self):
        """Test execution with cancelled context."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware = self._create_test_middleware()

        await pipeline.add_middleware(middleware)

        context = MiddlewareContext(data="test", is_cancelled=True)
        result = await pipeline.execute(context)

        assert result.is_successful()

        # No middleware should have been processed
        assert len(middleware.processed_contexts) == 0
        assert context.execution_path == []

    @pytest.mark.asyncio
    async def test_execute_with_middleware_exception(self):
        """Test execution handles middleware exceptions."""

        class ErrorMiddleware(AbstractMiddleware):
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                raise ValueError("Test error")

            def can_process(self, context: MiddlewareContext) -> bool:
                return True

        pipeline = InMemoryMiddlewarePipeline()
        error_middleware = ErrorMiddleware()
        later_middleware = self._create_test_middleware(name="LaterMiddleware")

        await pipeline.add_middleware(error_middleware)
        await pipeline.add_middleware(later_middleware)

        context = MiddlewareContext(data="test")
        result = await pipeline.execute(context)

        # Pipeline should handle the exception
        assert result.is_successful()  # Pipeline itself succeeds

        # Later middleware should not be processed
        assert len(later_middleware.processed_contexts) == 0
        assert context.execution_path == []

    @pytest.mark.asyncio
    async def test_clear_pipeline(self):
        """Test clearing pipeline."""
        pipeline = InMemoryMiddlewarePipeline()
        middleware1 = self._create_test_middleware()
        middleware2 = self._create_test_middleware()

        await pipeline.add_middleware(middleware1)
        await pipeline.add_middleware(middleware2)

        count = await pipeline.get_middleware_count()
        assert count == 2

        # Clear pipeline
        await pipeline.clear()

        count = await pipeline.get_middleware_count()
        assert count == 0
        assert pipeline._middlewares == []
        assert pipeline._cache_dirty is True

    @pytest.mark.asyncio
    async def test_get_middleware_by_priority(self):
        """Test getting middleware sorted by priority."""
        pipeline = InMemoryMiddlewarePipeline()

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

        # Should return a copy
        sorted_middleware.append(self._create_test_middleware())
        count = await pipeline.get_middleware_count()
        assert count == 3

    @pytest.mark.unit
    def test_sorted_cache_functionality(self):
        """Test sorted cache functionality."""
        pipeline = InMemoryMiddlewarePipeline()

        # Initially no cache
        assert pipeline._sorted_cache is None
        assert pipeline._cache_dirty is False

        # Add middleware - should mark cache as dirty
        middleware = self._create_test_middleware()
        pipeline._middlewares.append(middleware)
        pipeline._invalidate_cache()

        assert pipeline._cache_dirty is True
        assert pipeline._sorted_cache is None

        # Get sorted middleware - should create cache
        sorted_middleware = pipeline._get_sorted_middlewares()

        assert pipeline._sorted_cache is not None
        assert pipeline._cache_dirty is False
        assert sorted_middleware == pipeline._sorted_cache

        # Get again - should use cache
        sorted_middleware2 = pipeline._get_sorted_middlewares()
        assert sorted_middleware2 is pipeline._sorted_cache

    @pytest.mark.unit
    def test_get_pipeline_info(self):
        """Test getting pipeline information."""
        pipeline = InMemoryMiddlewarePipeline("TestPipeline")

        # Empty pipeline
        info = pipeline.get_pipeline_info()

        assert info["name"] == "TestPipeline"
        assert info["middleware_count"] == 0
        assert info["priority_distribution"] == {}

        # Add middleware
        high_middleware = self._create_test_middleware(EventPriority.HIGH)
        normal_middleware1 = self._create_test_middleware(EventPriority.NORMAL)
        normal_middleware2 = self._create_test_middleware(EventPriority.NORMAL)

        pipeline._middlewares.extend([high_middleware, normal_middleware1, normal_middleware2])

        info = pipeline.get_pipeline_info()

        assert info["name"] == "TestPipeline"
        assert info["middleware_count"] == 3
        assert info["priority_distribution"] == {"HIGH": 1, "NORMAL": 2}
        assert len(info["middleware_names"]) == 3
