# ABOUTME: Unit tests for NoOpMiddlewarePipeline implementation
# ABOUTME: Tests no-operation middleware pipeline functionality and interface compliance

import pytest
from unittest.mock import Mock
import asyncio

from core.implementations.noop.middleware.pipeline import NoOpMiddlewarePipeline
from core.interfaces.middleware import AbstractMiddleware
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus


class TestNoOpMiddlewarePipeline:
    """Test suite for NoOpMiddlewarePipeline."""

    @pytest.fixture
    def pipeline(self) -> NoOpMiddlewarePipeline:
        """Create a NoOpMiddlewarePipeline instance for testing."""
        return NoOpMiddlewarePipeline(name="TestPipeline")

    @pytest.fixture
    def mock_middleware(self) -> Mock:
        """Create a mock middleware for testing."""
        middleware = Mock(spec=AbstractMiddleware)
        middleware.__class__.__name__ = "MockMiddleware"
        return middleware

    @pytest.fixture
    def sample_context(self) -> MiddlewareContext:
        """Create a sample MiddlewareContext for testing."""
        return MiddlewareContext(
            id="test-context-123",
            event_type="test_event",
            data={"test": "data"},
            metadata={"source": "test"}
        )

    def test_pipeline_initialization(self):
        """Test NoOpMiddlewarePipeline initialization."""
        # Default name
        pipeline = NoOpMiddlewarePipeline()
        assert pipeline.name == "NoOpMiddlewarePipeline"
        assert pipeline._middleware_count == 0

        # Custom name
        custom_pipeline = NoOpMiddlewarePipeline(name="CustomPipeline")
        assert custom_pipeline.name == "CustomPipeline"
        assert custom_pipeline._middleware_count == 0

    @pytest.mark.asyncio
    async def test_add_middleware(self, pipeline, mock_middleware):
        """Test adding middleware to the pipeline (no-op)."""
        initial_count = await pipeline.get_middleware_count()
        assert initial_count == 0

        await pipeline.add_middleware(mock_middleware)

        count_after_add = await pipeline.get_middleware_count()
        assert count_after_add == 1

        # Add more middleware
        await pipeline.add_middleware(mock_middleware)
        count_after_second_add = await pipeline.get_middleware_count()
        assert count_after_second_add == 2

    @pytest.mark.asyncio
    async def test_remove_middleware(self, pipeline, mock_middleware):
        """Test removing middleware from the pipeline (no-op)."""
        # Add some middleware first
        await pipeline.add_middleware(mock_middleware)
        await pipeline.add_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 2

        # Remove middleware
        await pipeline.remove_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 1

        await pipeline.remove_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 0

        # Remove from empty pipeline (should not go negative)
        await pipeline.remove_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 0

    @pytest.mark.asyncio
    async def test_execute(self, pipeline, sample_context):
        """Test executing the middleware pipeline (no-op)."""
        result = await pipeline.execute(sample_context)

        assert isinstance(result, MiddlewareResult)
        assert result.middleware_name == "TestPipeline"
        assert result.status == MiddlewareStatus.SUCCESS
        assert result.data == {"message": "No-op execution completed"}
        assert result.should_continue is True
        assert result.execution_time_ms == 0.0

        # Check metadata
        assert result.metadata["pipeline_type"] == "NoOp"
        assert result.metadata["context_id"] == sample_context.id
        assert result.metadata["execution_mode"] == "bypass"
        assert "middleware_count" in result.metadata

    @pytest.mark.asyncio
    async def test_get_middleware_count(self, pipeline, mock_middleware):
        """Test getting middleware count."""
        assert await pipeline.get_middleware_count() == 0

        await pipeline.add_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 1

        await pipeline.add_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 2

        await pipeline.remove_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 1

    @pytest.mark.asyncio
    async def test_clear(self, pipeline, mock_middleware):
        """Test clearing all middleware from the pipeline (no-op)."""
        # Add some middleware
        await pipeline.add_middleware(mock_middleware)
        await pipeline.add_middleware(mock_middleware)
        assert await pipeline.get_middleware_count() == 2

        # Clear pipeline
        await pipeline.clear()
        assert await pipeline.get_middleware_count() == 0

    @pytest.mark.asyncio
    async def test_get_middleware_by_priority(self, pipeline, mock_middleware):
        """Test getting middleware sorted by priority (no-op)."""
        # Add some middleware
        await pipeline.add_middleware(mock_middleware)
        await pipeline.add_middleware(mock_middleware)

        # Should always return empty list in no-op implementation
        middleware_list = await pipeline.get_middleware_by_priority()
        assert middleware_list == []

    @pytest.mark.asyncio
    async def test_is_empty(self, pipeline, mock_middleware):
        """Test checking if pipeline is empty."""
        assert await pipeline.is_empty() is True

        await pipeline.add_middleware(mock_middleware)
        assert await pipeline.is_empty() is False

        await pipeline.clear()
        assert await pipeline.is_empty() is True

    @pytest.mark.asyncio
    async def test_contains_middleware(self, pipeline, mock_middleware):
        """Test checking if middleware is in pipeline."""
        # Should always return False in no-op implementation
        assert await pipeline.contains_middleware(mock_middleware) is False

        await pipeline.add_middleware(mock_middleware)
        assert await pipeline.contains_middleware(mock_middleware) is False

    def test_get_performance_stats(self, pipeline, mock_middleware):
        """Test getting performance statistics."""
        stats = pipeline.get_performance_stats()

        assert stats["pipeline_name"] == "TestPipeline"
        assert stats["pipeline_type"] == "NoOp"
        assert stats["total_executions"] == 0
        assert stats["total_execution_time_ms"] == 0.0
        assert stats["average_execution_time_ms"] == 0.0
        assert stats["middleware_count"] == 0
        assert stats["performance_impact"] == "none"

    def test_reset_performance_stats(self, pipeline):
        """Test resetting performance statistics (no-op)."""
        # Should complete without error
        pipeline.reset_performance_stats()

    def test_get_pipeline_info(self, pipeline):
        """Test getting pipeline information."""
        info = pipeline.get_pipeline_info()

        assert info["name"] == "TestPipeline"
        assert info["type"] == "NoOp"
        assert info["middleware_count"] == 0
        assert "description" in info
        assert "features" in info
        assert "performance_characteristics" in info

        # Check features
        features = info["features"]
        assert "Interface compliant" in features
        assert "Zero processing overhead" in features
        assert "Always successful execution" in features
        assert "Suitable for testing and development" in features

        # Check performance characteristics
        perf = info["performance_characteristics"]
        assert perf["execution_time"] == "0ms (no processing)"
        assert perf["memory_usage"] == "minimal"
        assert perf["cpu_impact"] == "none"

    def test_string_representations(self, pipeline):
        """Test string representations of the pipeline."""
        # Test __str__
        str_repr = str(pipeline)
        assert "NoOpMiddlewarePipeline" in str_repr
        assert "TestPipeline" in str_repr
        assert "count=0" in str_repr

        # Test __repr__
        repr_str = repr(pipeline)
        assert "NoOpMiddlewarePipeline" in repr_str
        assert "TestPipeline" in repr_str
        assert "middleware_count=0" in repr_str

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, pipeline, mock_middleware):
        """Test concurrent middleware operations."""
        # Test concurrent add operations
        tasks = [
            pipeline.add_middleware(mock_middleware)
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)
        assert await pipeline.get_middleware_count() == 5

        # Test concurrent remove operations
        tasks = [
            pipeline.remove_middleware(mock_middleware)
            for _ in range(3)
        ]
        await asyncio.gather(*tasks)
        assert await pipeline.get_middleware_count() == 2

        # Test concurrent execution
        context = MiddlewareContext(
            id="concurrent-test",
            event_type="test",
            data={},
            metadata={}
        )
        
        tasks = [
            pipeline.execute(context)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        # All executions should succeed
        assert len(results) == 10
        for result in results:
            assert result.status == MiddlewareStatus.SUCCESS
            assert result.execution_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_execution_with_different_contexts(self, pipeline):
        """Test execution with various context types."""
        contexts = [
            MiddlewareContext(
                id=f"context-{i}",
                event_type=f"event_type_{i}",
                data={"index": i},
                metadata={"test": f"metadata_{i}"}
            )
            for i in range(5)
        ]

        for context in contexts:
            result = await pipeline.execute(context)
            
            assert result.status == MiddlewareStatus.SUCCESS
            assert result.metadata["context_id"] == context.id
            assert result.should_continue is True
            assert result.execution_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_middleware_count_tracking(self, pipeline):
        """Test accurate middleware count tracking."""
        mock_middleware_1 = Mock(spec=AbstractMiddleware)
        mock_middleware_1.__class__.__name__ = "MockMiddleware1"
        
        mock_middleware_2 = Mock(spec=AbstractMiddleware)
        mock_middleware_2.__class__.__name__ = "MockMiddleware2"

        # Initial state
        assert await pipeline.get_middleware_count() == 0
        assert await pipeline.is_empty() is True

        # Add different middleware
        await pipeline.add_middleware(mock_middleware_1)
        assert await pipeline.get_middleware_count() == 1
        assert await pipeline.is_empty() is False

        await pipeline.add_middleware(mock_middleware_2)
        assert await pipeline.get_middleware_count() == 2

        # Remove specific middleware
        await pipeline.remove_middleware(mock_middleware_1)
        assert await pipeline.get_middleware_count() == 1

        # Clear all
        await pipeline.clear()
        assert await pipeline.get_middleware_count() == 0
        assert await pipeline.is_empty() is True

    def test_initialization_with_logging(self):
        """Test that initialization completes successfully with logging configured."""
        # Test that pipeline initializes without errors when loguru is being used
        pipeline = NoOpMiddlewarePipeline(name="LogTestPipeline")
        
        # Verify the pipeline was created successfully
        assert pipeline.name == "LogTestPipeline"
        assert hasattr(pipeline, '_logger')
        
        # Verify logger instance is properly bound
        # (loguru logger should be available and bound with the correct name)
        import inspect
        assert hasattr(pipeline._logger, 'info')  # loguru logger methods
        assert hasattr(pipeline._logger, 'debug')
        assert hasattr(pipeline._logger, 'error')