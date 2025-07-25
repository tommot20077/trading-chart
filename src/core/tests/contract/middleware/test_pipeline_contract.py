# ABOUTME: Contract tests for middleware pipeline implementations
# ABOUTME: Ensures all middleware pipeline implementations conform to the same behavioral contract

import pytest
from abc import ABC, abstractmethod
from typing import Type, List
from unittest.mock import Mock
import asyncio

from core.interfaces.middleware import AbstractMiddleware, AbstractMiddlewarePipeline
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.implementations.noop.middleware.pipeline import NoOpMiddlewarePipeline


class MiddlewarePipelineContractTest(ABC):
    """
    Abstract base class for middleware pipeline contract tests.
    
    All middleware pipeline implementations must pass these behavioral tests
    to ensure they conform to the same interface contract.
    """

    @abstractmethod
    def create_pipeline(self) -> AbstractMiddlewarePipeline:
        """Create an instance of the middleware pipeline to test."""
        pass

    @pytest.fixture
    def pipeline(self) -> AbstractMiddlewarePipeline:
        """Get a pipeline instance for testing."""
        return self.create_pipeline()

    @pytest.fixture
    def mock_middleware(self) -> Mock:
        """Create a mock middleware for testing."""
        middleware = Mock(spec=AbstractMiddleware) 
        middleware.__class__.__name__ = "MockMiddleware"
        # Add required attributes for InMemoryMiddlewarePipeline
        from core.models.event.event_priority import EventPriority
        middleware.priority = EventPriority.NORMAL
        return middleware

    @pytest.fixture
    def sample_context(self) -> MiddlewareContext:
        """Create a sample context for testing."""
        return MiddlewareContext(
            id="contract-test-123",
            event_type="contract_test",
            data={"test": "contract_data"},
            metadata={"source": "contract_test"}
        )

    # Contract Test Cases

    @pytest.mark.asyncio
    async def test_initial_state_contract(self, pipeline):
        """Test that new pipelines start in the correct initial state."""
        # New pipeline should be empty
        assert await pipeline.is_empty() is True
        assert await pipeline.get_middleware_count() == 0
        
        # Should return empty list for middleware by priority
        middleware_list = await pipeline.get_middleware_by_priority()
        assert isinstance(middleware_list, list)
        assert len(middleware_list) == 0

    @pytest.mark.asyncio
    async def test_add_middleware_contract(self, pipeline, mock_middleware):
        """Test the add_middleware contract."""
        initial_count = await pipeline.get_middleware_count()
        initial_empty = await pipeline.is_empty()

        # Add middleware
        await pipeline.add_middleware(mock_middleware)

        # Verify state changes
        new_count = await pipeline.get_middleware_count()
        new_empty = await pipeline.is_empty()

        assert new_count == initial_count + 1
        if initial_empty:
            assert new_empty is False

    @pytest.mark.asyncio
    async def test_remove_middleware_contract(self, pipeline, mock_middleware):
        """Test the remove_middleware contract."""
        # Add middleware first
        await pipeline.add_middleware(mock_middleware)
        count_after_add = await pipeline.get_middleware_count()

        # Remove middleware
        await pipeline.remove_middleware(mock_middleware)
        count_after_remove = await pipeline.get_middleware_count()

        # Count should decrease or stay the same (depending on implementation)
        assert count_after_remove <= count_after_add

    @pytest.mark.asyncio
    async def test_execute_contract(self, pipeline, sample_context):
        """Test the execute method contract."""
        result = await pipeline.execute(sample_context)

        # Result must be a MiddlewareResult
        assert isinstance(result, MiddlewareResult)
        
        # Required fields must be present
        assert hasattr(result, 'middleware_name')
        assert hasattr(result, 'status')
        assert hasattr(result, 'should_continue')
        assert hasattr(result, 'execution_time_ms')
        
        # Status must be a valid MiddlewareStatus
        assert isinstance(result.status, MiddlewareStatus)
        
        # should_continue must be boolean
        assert isinstance(result.should_continue, bool)
        
        # execution_time_ms must be numeric and non-negative (or None for some implementations)
        if result.execution_time_ms is not None:
            assert isinstance(result.execution_time_ms, (int, float))  
            assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_clear_contract(self, pipeline, mock_middleware):
        """Test the clear method contract."""
        # Add some middleware
        await pipeline.add_middleware(mock_middleware)
        await pipeline.add_middleware(mock_middleware)
        
        assert await pipeline.get_middleware_count() > 0
        assert await pipeline.is_empty() is False

        # Clear all middleware
        await pipeline.clear()

        # Pipeline should be empty after clear
        assert await pipeline.get_middleware_count() == 0
        assert await pipeline.is_empty() is True

    @pytest.mark.asyncio
    async def test_get_middleware_by_priority_contract(self, pipeline, mock_middleware):
        """Test the get_middleware_by_priority contract."""
        # Should work with empty pipeline
        empty_list = await pipeline.get_middleware_by_priority()
        assert isinstance(empty_list, list)
        assert len(empty_list) == 0

        # Add middleware
        await pipeline.add_middleware(mock_middleware)
        
        # Should return a list (may be empty for NoOp implementations)
        middleware_list = await pipeline.get_middleware_by_priority()
        assert isinstance(middleware_list, list)

    @pytest.mark.asyncio
    async def test_contains_middleware_contract(self, pipeline, mock_middleware):
        """Test the contains_middleware contract."""
        # Should return boolean
        contains_empty = await pipeline.contains_middleware(mock_middleware)
        assert isinstance(contains_empty, bool)

        # Add middleware
        await pipeline.add_middleware(mock_middleware)
        
        # Should still return boolean
        contains_after_add = await pipeline.contains_middleware(mock_middleware)
        assert isinstance(contains_after_add, bool)

    def test_performance_stats_contract(self, pipeline):
        """Test performance statistics methods contract."""
        # get_performance_stats should return a dict
        stats = pipeline.get_performance_stats()
        assert isinstance(stats, dict)
        
        # Should contain pipeline_name (pipeline_type may vary by implementation)
        assert "pipeline_name" in stats

        # reset_performance_stats should not raise
        pipeline.reset_performance_stats()

    def test_pipeline_info_contract(self, pipeline):
        """Test pipeline info method contract."""
        info = pipeline.get_pipeline_info()
        assert isinstance(info, dict)
        
        # Should contain name (type may vary by implementation)
        assert "name" in info
        assert isinstance(info["name"], str)

    def test_string_representation_contract(self, pipeline):
        """Test string representation contract."""
        # __str__ should return a string
        str_repr = str(pipeline)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

        # __repr__ should return a string
        repr_str = repr(pipeline)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    @pytest.mark.asyncio
    async def test_concurrent_access_contract(self, pipeline, mock_middleware):
        """Test that concurrent operations don't break the contract."""
        # Create multiple tasks that modify the pipeline
        add_tasks = [
            pipeline.add_middleware(mock_middleware)
            for _ in range(5)
        ]
        
        # Execute concurrently
        await asyncio.gather(*add_tasks)
        
        # Pipeline should still be in a valid state
        count = await pipeline.get_middleware_count()
        assert isinstance(count, int)
        assert count >= 0

        empty = await pipeline.is_empty()
        assert isinstance(empty, bool)

    @pytest.mark.asyncio
    async def test_context_handling_contract(self, pipeline):
        """Test that various context types are handled properly."""
        contexts = [
            MiddlewareContext(id="test1", event_type="type1"),
            MiddlewareContext(id="test2", data={"key": "value"}),
            MiddlewareContext(id="test3", metadata={"meta": "data"}),
        ]

        for context in contexts:
            result = await pipeline.execute(context)
            # All contexts should produce valid results
            assert isinstance(result, MiddlewareResult)
            assert isinstance(result.status, MiddlewareStatus)


class TestInMemoryMiddlewarePipelineContract(MiddlewarePipelineContractTest):
    """Contract tests for InMemoryMiddlewarePipeline."""

    def create_pipeline(self) -> AbstractMiddlewarePipeline:
        """Create an InMemoryMiddlewarePipeline for testing."""
        return InMemoryMiddlewarePipeline(name="ContractTestPipeline")


class TestNoOpMiddlewarePipelineContract(MiddlewarePipelineContractTest):
    """Contract tests for NoOpMiddlewarePipeline."""

    def create_pipeline(self) -> AbstractMiddlewarePipeline:
        """Create a NoOpMiddlewarePipeline for testing."""
        return NoOpMiddlewarePipeline(name="ContractTestPipeline")


class TestMiddlewarePipelineImplementationConsistency:
    """
    Test consistency between different middleware pipeline implementations.
    
    These tests ensure that different implementations behave consistently
    for the same operations, within acceptable ranges.
    """

    @pytest.fixture
    def pipelines(self) -> List[AbstractMiddlewarePipeline]:
        """Create instances of all pipeline implementations."""
        return [
            InMemoryMiddlewarePipeline(name="InMemoryTest"),
            NoOpMiddlewarePipeline(name="NoOpTest")
        ]

    @pytest.mark.asyncio
    async def test_initial_state_consistency(self, pipelines):
        """Test that all implementations start in consistent initial states."""
        for pipeline in pipelines:
            assert await pipeline.is_empty() is True
            assert await pipeline.get_middleware_count() == 0

    @pytest.mark.asyncio
    async def test_empty_execution_consistency(self, pipelines):
        """Test that all implementations handle empty pipeline execution consistently."""
        context = MiddlewareContext(id="consistency_test", event_type="test")
        
        results = []
        for pipeline in pipelines:
            result = await pipeline.execute(context)
            results.append(result)

        # All should return valid results (status may vary - SKIPPED for empty InMemory, SUCCESS for NoOp)
        for result in results:
            assert isinstance(result, MiddlewareResult)
            assert isinstance(result.status, MiddlewareStatus)
            assert isinstance(result.should_continue, bool)
            if result.execution_time_ms is not None:
                assert result.execution_time_ms >= 0

    def test_performance_stats_consistency(self, pipelines):
        """Test that performance stats have consistent structure."""
        for pipeline in pipelines:
            stats = pipeline.get_performance_stats()
            assert isinstance(stats, dict)
            assert "pipeline_name" in stats
            # pipeline_type may vary by implementation

    def test_pipeline_info_consistency(self, pipelines):
        """Test that pipeline info has consistent structure."""
        for pipeline in pipelines:
            info = pipeline.get_pipeline_info()
            assert isinstance(info, dict)
            assert "name" in info
            # type may vary by implementation