# ABOUTME: Integration tests for Middleware + Pipeline responsibility chain
# ABOUTME: Tests middleware registration, priority sorting, chain execution, and result aggregation

import asyncio
import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from core.interfaces.middleware import AbstractMiddleware, AbstractMiddlewarePipeline
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from core.exceptions.base import BusinessLogicException, ValidationException


class MockMiddleware(AbstractMiddleware):
    """Mock middleware for testing purposes."""
    
    def __init__(self, name: str, priority: EventPriority = EventPriority.NORMAL, 
                 should_continue: bool = True, should_fail: bool = False,
                 can_process_func=None):
        super().__init__(priority)
        self.name = name
        self.should_continue = should_continue
        self.should_fail = should_fail
        self.can_process_func = can_process_func or (lambda ctx: True)
        self.process_count = 0
        self.last_context = None
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process middleware logic."""
        self.process_count += 1
        self.last_context = context
        
        if self.should_fail:
            raise BusinessLogicException(f"Middleware {self.name} failed", "MIDDLEWARE_001")
        
        # Add some data to context
        context.set_data(f"{self.name}_processed", True)
        context.set_metadata(f"{self.name}_timestamp", "2025-01-20T10:30:00Z")
        
        return MiddlewareResult(
            middleware_name=self.name,
            status=MiddlewareStatus.SUCCESS,
            should_continue=self.should_continue,
            data={f"{self.name}_result": "processed"},
            metadata={"processing_order": self.process_count}
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Determine if this middleware can process the context."""
        return self.can_process_func(context)


class TestMiddlewarePipelineResponsibilityChain:
    """Test middleware pipeline responsibility chain integration."""
    
    @pytest.mark.asyncio
    async def test_middleware_registration_priority_sorting_chain_execution_result_aggregation(self):
        """Test complete flow: middleware registration → priority sorting → responsibility chain execution → result aggregation."""
        pipeline = InMemoryMiddlewarePipeline("test_pipeline")
        
        # Create middleware with different priorities
        middlewares = [
            MockMiddleware("high_priority", EventPriority.HIGH),  # Should execute first
            MockMiddleware("normal_priority", EventPriority.NORMAL),  # Should execute second
            MockMiddleware("low_priority", EventPriority.LOW),  # Should execute third
            MockMiddleware("critical_priority", EventPriority.CRITICAL),  # Should execute before high
        ]
        
        # Register middleware in random order to test sorting
        registration_order = [middlewares[1], middlewares[3], middlewares[0], middlewares[2]]
        for middleware in registration_order:
            await pipeline.add_middleware(middleware)
        
        # Verify middleware count
        count = await pipeline.get_middleware_count()
        assert count == 4, "All middleware should be registered"
        
        # Verify priority sorting
        sorted_middlewares = await pipeline.get_middleware_by_priority()
        expected_execution_order = ["critical_priority", "high_priority", "normal_priority", "low_priority"]
        actual_order = [m.name for m in sorted_middlewares]
        assert actual_order == expected_execution_order, f"Middleware should be sorted by priority: {actual_order}"
        
        # Create test context
        context = MiddlewareContext(
            event_id="test_event_001",
            event_type="test_event",
            user_id="user_123",
            timestamp="2025-01-20T10:30:00Z"
        )
        
        # Execute responsibility chain
        pipeline_result = await pipeline.execute(context)
        
        # Verify chain execution
        assert pipeline_result.status == MiddlewareStatus.SUCCESS, "Pipeline should succeed"
        assert pipeline_result.middleware_name == "test_pipeline"
        
        # Verify all middleware were executed in correct order
        for middleware in middlewares:
            assert middleware.process_count == 1, f"Middleware {middleware.name} should be executed once"
        
        # Verify context was passed through the chain correctly
        for middleware in middlewares:
            assert middleware.last_context is context, f"Middleware {middleware.name} should receive the same context"
        
        # Verify context data was accumulated
        assert context.get_data("critical_priority_processed") is True
        assert context.get_data("high_priority_processed") is True
        assert context.get_data("normal_priority_processed") is True
        assert context.get_data("low_priority_processed") is True
        
        # Verify result aggregation
        pipeline_data = pipeline_result.data
        assert "total_middlewares" in pipeline_data
        assert "executed_middlewares" in pipeline_data
        assert "failed_middlewares" in pipeline_data
        assert "successful_middlewares" in pipeline_data
        
        assert pipeline_data["total_middlewares"] == 4
        assert pipeline_data["executed_middlewares"] == 4
        assert pipeline_data["failed_middlewares"] == 0
        assert pipeline_data["successful_middlewares"] == 4

    @pytest.mark.asyncio
    async def test_middleware_chain_early_termination(self):
        """Test middleware chain early termination when should_continue=False."""
        pipeline = InMemoryMiddlewarePipeline("termination_test")
        
        # Create middleware chain with early termination
        middlewares = [
            MockMiddleware("first", EventPriority.HIGH),
            MockMiddleware("terminator", EventPriority.NORMAL, should_continue=False),
            MockMiddleware("should_not_execute", EventPriority.LOW),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Create context and execute
        context = MiddlewareContext(
            event_id="test_termination",
            event_type="termination_test"
        )
        
        result = await pipeline.execute(context)
        
        # Verify execution stopped at terminator
        assert middlewares[0].process_count == 1, "First middleware should execute"
        assert middlewares[1].process_count == 1, "Terminator middleware should execute"
        assert middlewares[2].process_count == 0, "Third middleware should not execute"
        
        # Verify pipeline result
        assert result.status == MiddlewareStatus.SUCCESS
        pipeline_data = result.data
        assert pipeline_data["executed_middlewares"] == 2
        assert pipeline_data["total_middlewares"] == 3

    @pytest.mark.asyncio
    async def test_middleware_selective_processing(self):
        """Test middleware selective processing with can_process logic."""
        pipeline = InMemoryMiddlewarePipeline("selective_test")
        
        # Create middleware with selective processing
        def can_process_event_type_a(ctx):
            return ctx.get_metadata("event_type") == "type_a"
        
        def can_process_event_type_b(ctx):
            return ctx.get_metadata("event_type") == "type_b"
        
        middlewares = [
            MockMiddleware("type_a_processor", EventPriority.HIGH, can_process_func=can_process_event_type_a),
            MockMiddleware("type_b_processor", EventPriority.HIGH, can_process_func=can_process_event_type_b),
            MockMiddleware("universal_processor", EventPriority.LOW),  # Can process anything
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Test with type_a event
        context_a = MiddlewareContext(event_id="test_a", event_type="test")
        context_a.set_metadata("event_type", "type_a")
        
        result_a = await pipeline.execute(context_a)
        
        # Verify selective processing for type_a
        assert middlewares[0].process_count == 1, "Type A processor should execute"
        assert middlewares[1].process_count == 0, "Type B processor should not execute"
        assert middlewares[2].process_count == 1, "Universal processor should execute"
        
        # Reset counts
        for m in middlewares:
            m.process_count = 0
        
        # Test with type_b event
        context_b = MiddlewareContext(event_id="test_b", event_type="test")
        context_b.set_metadata("event_type", "type_b")
        
        result_b = await pipeline.execute(context_b)
        
        # Verify selective processing for type_b
        assert middlewares[0].process_count == 0, "Type A processor should not execute"
        assert middlewares[1].process_count == 1, "Type B processor should execute"
        assert middlewares[2].process_count == 1, "Universal processor should execute"


class TestMiddlewareContextPassing:
    """Test middleware context passing and data flow."""
    
    @pytest.mark.asyncio
    async def test_context_data_passing_through_middleware_chain(self):
        """Test context data passing and modification through middleware chain."""
        pipeline = InMemoryMiddlewarePipeline("context_test")
        
        # Create middleware that modify context
        class DataAccumulatorMiddleware(AbstractMiddleware):
            def __init__(self, name: str, priority: EventPriority):
                super().__init__(priority)
                self.name = name
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                # Get existing counter or start at 0
                counter = context.get_data("counter", 0)
                context.set_data("counter", counter + 1)
                
                # Add middleware-specific data
                context.set_data(f"{self.name}_contribution", counter + 1)
                
                # Add to processing path
                path = context.get_data("processing_path", [])
                path.append(self.name)
                context.set_data("processing_path", path)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={"counter_increment": 1},
                    metadata={"order": counter + 1}
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Add middleware in priority order
        middlewares = [
            DataAccumulatorMiddleware("first", EventPriority.HIGH),
            DataAccumulatorMiddleware("second", EventPriority.NORMAL),
            DataAccumulatorMiddleware("third", EventPriority.LOW),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Create context with initial data
        context = MiddlewareContext(
            event_id="context_test",
            event_type="data_passing"
        )
        context.set_data("initial_value", "test")
        
        # Execute pipeline
        result = await pipeline.execute(context)
        
        # Verify context data accumulation
        assert context.get_data("counter") == 3, "Counter should be incremented by each middleware"
        assert context.get_data("initial_value") == "test", "Initial data should be preserved"
        
        # Verify processing path
        processing_path = context.get_data("processing_path")
        assert processing_path == ["first", "second", "third"], "Processing path should reflect execution order"
        
        # Verify middleware-specific contributions
        assert context.get_data("first_contribution") == 1
        assert context.get_data("second_contribution") == 2
        assert context.get_data("third_contribution") == 3
        
        # Verify execution path tracking
        execution_path = context.get_execution_path()
        assert "first" in execution_path
        assert "second" in execution_path
        assert "third" in execution_path

    @pytest.mark.asyncio
    async def test_context_metadata_propagation(self):
        """Test context metadata propagation and enrichment."""
        pipeline = InMemoryMiddlewarePipeline("metadata_test")
        
        class MetadataEnricherMiddleware(AbstractMiddleware):
            def __init__(self, name: str, priority: EventPriority, metadata_key: str, metadata_value: Any):
                super().__init__(priority)
                self.name = name
                self.metadata_key = metadata_key
                self.metadata_value = metadata_value
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                # Enrich metadata
                context.set_metadata(self.metadata_key, self.metadata_value)
                
                # Also read and modify existing metadata
                existing_tags = context.get_metadata("tags", [])
                existing_tags.append(f"processed_by_{self.name}")
                context.set_metadata("tags", existing_tags)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    metadata={"enriched_key": self.metadata_key}
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Create metadata enriching middleware
        middlewares = [
            MetadataEnricherMiddleware("auth", EventPriority.HIGHEST, "user_role", "admin"),
            MetadataEnricherMiddleware("validation", EventPriority.HIGH, "validation_status", "passed"),
            MetadataEnricherMiddleware("audit", EventPriority.LOW, "audit_timestamp", "2025-01-20T10:30:00Z"),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Create context with initial metadata
        context = MiddlewareContext(
            event_id="metadata_test",
            event_type="enrichment_test"
        )
        context.set_metadata("request_id", "req_12345")
        context.set_metadata("tags", ["initial"])
        
        # Execute pipeline
        result = await pipeline.execute(context)
        
        # Verify metadata enrichment
        assert context.get_metadata("user_role") == "admin"
        assert context.get_metadata("validation_status") == "passed"
        assert context.get_metadata("audit_timestamp") == "2025-01-20T10:30:00Z"
        
        # Verify initial metadata preserved
        assert context.get_metadata("request_id") == "req_12345"
        
        # Verify tags accumulation
        tags = context.get_metadata("tags")
        expected_tags = ["initial", "processed_by_auth", "processed_by_validation", "processed_by_audit"]
        assert tags == expected_tags, f"Tags should accumulate: {tags}"


class TestMiddlewareErrorHandling:
    """Test middleware error handling and pipeline resilience."""
    
    @pytest.mark.asyncio
    async def test_middleware_exception_handling_pipeline_resilience(self):
        """Test middleware exception handling and pipeline resilience."""
        pipeline = InMemoryMiddlewarePipeline("error_test")
        
        # Create middleware with one that fails
        middlewares = [
            MockMiddleware("before_failure", EventPriority.HIGH),
            MockMiddleware("failing_middleware", EventPriority.NORMAL, should_fail=True),
            MockMiddleware("after_failure", EventPriority.LOW),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Create context
        context = MiddlewareContext(
            event_id="error_test",
            event_type="error_handling"
        )
        
        # Execute pipeline
        result = await pipeline.execute(context)
        
        # Verify pipeline handles error gracefully
        assert result.status == MiddlewareStatus.SUCCESS, "Pipeline should succeed despite middleware failure"
        
        # Verify execution stops at failed middleware
        assert middlewares[0].process_count == 1, "First middleware should execute"
        assert middlewares[1].process_count == 1, "Failing middleware should be called but fails during execution"
        assert middlewares[2].process_count == 0, "Third middleware should not execute after failure"
        
        # Verify error is captured in pipeline result
        pipeline_data = result.data
        assert pipeline_data["failed_middlewares"] == 1, "One middleware should be marked as failed"
        assert pipeline_data["successful_middlewares"] == 1, "Only one middleware should complete successfully"
        
        # Verify error details in metadata
        pipeline_results = result.metadata["pipeline_results"]
        error_result = next((r for r in pipeline_results if r["status"] == "failed"), None)
        assert error_result is not None, "Should have error result"
        assert "Middleware failing_middleware failed" in error_result["error"]

    @pytest.mark.asyncio
    async def test_middleware_recovery_and_continuation_strategies(self):
        """Test middleware recovery and continuation strategies."""
        pipeline = InMemoryMiddlewarePipeline("recovery_test")
        
        class ResilientMiddleware(AbstractMiddleware):
            def __init__(self, name: str, priority: EventPriority, fail_on_attempt: int = None):
                super().__init__(priority)
                self.name = name
                self.fail_on_attempt = fail_on_attempt
                self.attempt_count = 0
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                self.attempt_count += 1
                
                # Simulate failure on specific attempt
                if self.fail_on_attempt and self.attempt_count == self.fail_on_attempt:
                    raise ValidationException(f"Simulated failure in {self.name}", "TEST_001")
                
                # Mark successful processing
                context.set_data(f"{self.name}_success", True)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={"attempt": self.attempt_count}
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Create middleware with different recovery behaviors
        middlewares = [
            ResilientMiddleware("always_succeeds", EventPriority.HIGH),
            ResilientMiddleware("fails_first_time", EventPriority.NORMAL, fail_on_attempt=1),
            ResilientMiddleware("cleanup", EventPriority.LOW),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # First execution - should fail
        context1 = MiddlewareContext(event_id="recovery_test_1", event_type="recovery")
        result1 = await pipeline.execute(context1)
        
        # Verify first execution handles failure
        assert result1.status == MiddlewareStatus.SUCCESS, "Pipeline should succeed despite middleware failure"
        assert middlewares[0].attempt_count == 1
        assert middlewares[1].attempt_count == 1  # Failed on first attempt
        assert middlewares[2].attempt_count == 0  # Should not execute after failure
        
        # Second execution - should succeed
        context2 = MiddlewareContext(event_id="recovery_test_2", event_type="recovery")
        result2 = await pipeline.execute(context2)
        
        # Verify second execution succeeds
        assert result2.status == MiddlewareStatus.SUCCESS
        assert middlewares[0].attempt_count == 2
        assert middlewares[1].attempt_count == 2  # Now succeeds on second attempt
        assert middlewares[2].attempt_count == 1  # Should execute after success
        
        # Verify context data for successful execution
        assert context2.get_data("always_succeeds_success") is True
        assert context2.get_data("fails_first_time_success") is True
        assert context2.get_data("cleanup_success") is True


if __name__ == "__main__":
    pytest.main([__file__])