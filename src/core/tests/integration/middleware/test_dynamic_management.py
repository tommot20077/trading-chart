# ABOUTME: Integration tests for dynamic middleware management capabilities
# ABOUTME: Tests runtime middleware manipulation, priority adjustments, and configuration changes

import pytest
import pytest_asyncio
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock
from datetime import datetime, UTC

from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.interfaces.middleware import AbstractMiddleware
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from core.exceptions.base import BusinessLogicException


class TrackingMiddleware(AbstractMiddleware):
    """Middleware that tracks its execution for testing."""
    
    def __init__(self, name: str, priority: EventPriority = EventPriority.NORMAL, 
                 processing_time_ms: float = 1.0, should_fail: bool = False):
        super().__init__(priority)
        self.name = name
        self.processing_time_ms = processing_time_ms
        self.should_fail = should_fail
        self.execution_log = []
        self.call_count = 0
        self.last_context = None
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process with tracking."""
        self.call_count += 1
        self.last_context = context
        
        # Log execution
        execution_entry = {
            "timestamp": datetime.now(UTC),
            "context_id": context.id,
            "event_id": context.event_id,
            "execution_path": context.execution_path.copy(),
            "call_count": self.call_count
        }
        self.execution_log.append(execution_entry)
        
        # Simulate processing time
        if self.processing_time_ms > 0:
            await asyncio.sleep(self.processing_time_ms / 1000)
        
        if self.should_fail:
            raise BusinessLogicException(
                f"Processing failed in {self.name}",
                "PROCESSING_ERROR"
            )
        
        # Enrich context
        context.data[f"{self.name}_processed"] = True
        context.metadata[f"{self.name}_call_count"] = self.call_count
        
        return MiddlewareResult(
            middleware_name=self.name,
            status=MiddlewareStatus.SUCCESS,
            data={f"{self.name}_result": "processed"},
            should_continue=True,
            execution_time_ms=self.processing_time_ms,
            metadata={"processed_at": execution_entry["timestamp"].isoformat()}
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can process all contexts."""
        return True


class ConfigurableMiddleware(AbstractMiddleware):
    """Middleware with configurable behavior."""
    
    def __init__(self, name: str, priority: EventPriority = EventPriority.NORMAL):
        super().__init__(priority)
        self.name = name
        self.config = {
            "enabled": True,
            "processing_delay_ms": 1.0,
            "enrich_data": True,
            "custom_metadata": {}
        }
        self.execution_log = []
        self.config_changes = []
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update middleware configuration."""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        self.config_changes.append({
            "timestamp": datetime.now(UTC),
            "old_config": old_config,
            "new_config": self.config.copy()
        })
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process with configurable behavior."""
        if not self.config["enabled"]:
            return MiddlewareResult(
                middleware_name=self.name,
                status=MiddlewareStatus.SKIPPED,
                should_continue=True,
                execution_time_ms=0.0,
                metadata={"reason": "disabled"}
            )
        
        # Log execution
        execution_entry = {
            "timestamp": datetime.now(UTC),
            "context_id": context.id,
            "config_snapshot": self.config.copy()
        }
        self.execution_log.append(execution_entry)
        
        # Apply processing delay
        delay = self.config.get("processing_delay_ms", 1.0)
        if delay > 0:
            await asyncio.sleep(delay / 1000)
        
        # Enrich data if configured
        if self.config.get("enrich_data", True):
            context.data[f"{self.name}_processed"] = True
            context.data[f"{self.name}_config"] = self.config.copy()
        
        # Add custom metadata
        custom_metadata = self.config.get("custom_metadata", {})
        for key, value in custom_metadata.items():
            context.metadata[f"{self.name}_{key}"] = value
        
        return MiddlewareResult(
            middleware_name=self.name,
            status=MiddlewareStatus.SUCCESS,
            data={f"{self.name}_result": "processed"},
            should_continue=True,
            execution_time_ms=delay,
            metadata={"config_applied": self.config.copy()}
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can process if enabled."""
        return self.config.get("enabled", True)


class TestDynamicMiddlewareManagement:
    """Integration tests for dynamic middleware management."""
    
    @pytest_asyncio.fixture
    async def pipeline(self):
        """Create a clean middleware pipeline."""
        return InMemoryMiddlewarePipeline("TestDynamicPipeline")
    
    @pytest_asyncio.fixture
    def sample_context(self):
        """Create a sample middleware context for testing."""
        return MiddlewareContext(
            event_id="test-event-123",
            data={"message": "test data", "value": 100},
            metadata={"user_id": "test_user", "event_type": "test"}
        )
    
    @pytest.mark.asyncio
    async def test_runtime_middleware_addition(self, pipeline, sample_context):
        """Test adding middleware during runtime."""
        # Start with empty pipeline
        initial_count = await pipeline.get_middleware_count()
        assert initial_count == 0
        
        # Execute empty pipeline
        result1 = await pipeline.execute(sample_context)
        assert result1.status == MiddlewareStatus.SKIPPED
        
        # Add first middleware during runtime
        middleware_a = TrackingMiddleware("MiddlewareA", EventPriority.HIGH, 2.0)
        await pipeline.add_middleware(middleware_a)
        
        # Verify pipeline state
        assert await pipeline.get_middleware_count() == 1
        
        # Execute with first middleware
        context2 = MiddlewareContext(
            event_id="test-event-2",
            data={"message": "after first addition"},
            metadata={"user_id": "test_user"}
        )
        result2 = await pipeline.execute(context2)
        
        assert result2.status == MiddlewareStatus.SUCCESS
        assert len(middleware_a.execution_log) == 1
        assert context2.data["MiddlewareA_processed"] is True
        
        # Add second middleware with higher priority
        middleware_b = TrackingMiddleware("MiddlewareB", EventPriority.CRITICAL, 1.0)
        await pipeline.add_middleware(middleware_b)
        
        # Verify pipeline state
        assert await pipeline.get_middleware_count() == 2
        
        # Execute with both middleware
        context3 = MiddlewareContext(
            event_id="test-event-3",
            data={"message": "after second addition"},
            metadata={"user_id": "test_user"}
        )
        result3 = await pipeline.execute(context3)
        
        assert result3.status == MiddlewareStatus.SUCCESS
        assert len(middleware_a.execution_log) == 2
        assert len(middleware_b.execution_log) == 1
        
        # Verify execution order: MiddlewareB (CRITICAL) -> MiddlewareA (HIGH)
        assert context3.execution_path == ["MiddlewareB", "MiddlewareA"]
    
    @pytest.mark.asyncio
    async def test_runtime_middleware_removal(self, pipeline, sample_context):
        """Test removing middleware during runtime."""
        # Set up initial middleware
        middleware_a = TrackingMiddleware("MiddlewareA", EventPriority.HIGH)
        middleware_b = TrackingMiddleware("MiddlewareB", EventPriority.NORMAL)
        middleware_c = TrackingMiddleware("MiddlewareC", EventPriority.LOW)
        
        await pipeline.add_middleware(middleware_a)
        await pipeline.add_middleware(middleware_b)
        await pipeline.add_middleware(middleware_c)
        
        assert await pipeline.get_middleware_count() == 3
        
        # Execute with all middleware
        result1 = await pipeline.execute(sample_context)
        assert result1.status == MiddlewareStatus.SUCCESS
        assert sample_context.execution_path == ["MiddlewareA", "MiddlewareB", "MiddlewareC"]
        
        # Remove middle middleware during runtime
        await pipeline.remove_middleware(middleware_b)
        assert await pipeline.get_middleware_count() == 2
        
        # Execute after removal
        context2 = MiddlewareContext(
            event_id="test-event-2",
            data={"message": "after removal"},
            metadata={"user_id": "test_user"}
        )
        result2 = await pipeline.execute(context2)
        
        assert result2.status == MiddlewareStatus.SUCCESS
        assert context2.execution_path == ["MiddlewareA", "MiddlewareC"]
        
        # Verify middleware execution counts
        assert len(middleware_a.execution_log) == 2
        assert len(middleware_b.execution_log) == 1  # Only executed once before removal
        assert len(middleware_c.execution_log) == 2
        
        # Remove all remaining middleware
        await pipeline.remove_middleware(middleware_a)
        await pipeline.remove_middleware(middleware_c)
        assert await pipeline.get_middleware_count() == 0
        
        # Execute empty pipeline
        context3 = MiddlewareContext(
            event_id="test-event-3",
            data={"message": "empty pipeline"},
            metadata={"user_id": "test_user"}
        )
        result3 = await pipeline.execute(context3)
        assert result3.status == MiddlewareStatus.SKIPPED
    
    @pytest.mark.asyncio
    async def test_middleware_priority_dynamic_adjustment(self, pipeline, sample_context):
        """Test dynamic priority adjustment of middleware."""
        # Create middleware with initial priorities
        middleware_a = TrackingMiddleware("MiddlewareA", EventPriority.LOW)
        middleware_b = TrackingMiddleware("MiddlewareB", EventPriority.HIGH)
        middleware_c = TrackingMiddleware("MiddlewareC", EventPriority.NORMAL)
        
        await pipeline.add_middleware(middleware_a)
        await pipeline.add_middleware(middleware_b)
        await pipeline.add_middleware(middleware_c)
        
        # Execute with initial priorities
        result1 = await pipeline.execute(sample_context)
        assert result1.status == MiddlewareStatus.SUCCESS
        # Expected order: HIGH -> NORMAL -> LOW
        assert sample_context.execution_path == ["MiddlewareB", "MiddlewareC", "MiddlewareA"]
        
        # Change priorities dynamically
        middleware_a.priority = EventPriority.CRITICAL  # LOW -> CRITICAL
        middleware_b.priority = EventPriority.LOW       # HIGH -> LOW
        
        # Remove and re-add to trigger priority resort
        await pipeline.remove_middleware(middleware_a)
        await pipeline.remove_middleware(middleware_b)
        await pipeline.add_middleware(middleware_a)  # Now CRITICAL
        await pipeline.add_middleware(middleware_b)  # Now LOW
        
        # Execute with new priorities
        context2 = MiddlewareContext(
            event_id="test-event-2",
            data={"message": "after priority change"},
            metadata={"user_id": "test_user"}
        )
        result2 = await pipeline.execute(context2)
        
        assert result2.status == MiddlewareStatus.SUCCESS
        # Expected order: CRITICAL -> NORMAL -> LOW
        assert context2.execution_path == ["MiddlewareA", "MiddlewareC", "MiddlewareB"]
        
        # Verify execution counts
        assert len(middleware_a.execution_log) == 2
        assert len(middleware_b.execution_log) == 2
        assert len(middleware_c.execution_log) == 2
    
    @pytest.mark.asyncio
    async def test_configuration_change_impact(self, pipeline, sample_context):
        """Test impact of configuration changes on middleware behavior."""
        # Set up configurable middleware
        middleware_a = ConfigurableMiddleware("ConfigurableA", EventPriority.HIGH)
        middleware_b = ConfigurableMiddleware("ConfigurableB", EventPriority.LOW)
        
        await pipeline.add_middleware(middleware_a)
        await pipeline.add_middleware(middleware_b)
        
        # Execute with default configuration
        result1 = await pipeline.execute(sample_context)
        assert result1.status == MiddlewareStatus.SUCCESS
        assert sample_context.data["ConfigurableA_processed"] is True
        assert sample_context.data["ConfigurableB_processed"] is True
        
        # Change configuration of middleware A
        middleware_a.update_config({
            "processing_delay_ms": 5.0,
            "custom_metadata": {"custom_key": "custom_value"},
            "enrich_data": True
        })
        
        # Disable middleware B
        middleware_b.update_config({"enabled": False})
        
        # Execute with new configuration
        context2 = MiddlewareContext(
            event_id="test-event-2",
            data={"message": "after config change"},
            metadata={"user_id": "test_user"}
        )
        result2 = await pipeline.execute(context2)
        
        assert result2.status == MiddlewareStatus.SUCCESS
        
        # Verify middleware A applied new configuration
        assert context2.data["ConfigurableA_processed"] is True
        assert context2.metadata["ConfigurableA_custom_key"] == "custom_value"
        assert "ConfigurableA_config" in context2.data
        
        # Verify middleware B was skipped due to disabled configuration
        assert "ConfigurableB_processed" not in context2.data
        assert context2.execution_path == ["ConfigurableA"]  # Only A executed
        
        # Verify execution logs
        assert len(middleware_a.execution_log) == 2
        assert len(middleware_b.execution_log) == 1  # Only executed once before being disabled
        
        # Verify configuration change tracking
        assert len(middleware_a.config_changes) == 1
        assert len(middleware_b.config_changes) == 1
        
        config_change_a = middleware_a.config_changes[0]
        assert config_change_a["new_config"]["processing_delay_ms"] == 5.0
        assert config_change_a["new_config"]["custom_metadata"]["custom_key"] == "custom_value"
        
        config_change_b = middleware_b.config_changes[0]
        assert config_change_b["new_config"]["enabled"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_runtime_modifications(self, pipeline):
        """Test concurrent middleware additions and removals."""
        # Create multiple middleware
        middleware_list = []
        for i in range(5):
            middleware = TrackingMiddleware(f"Middleware{i}", EventPriority.NORMAL, 0.5)
            middleware_list.append(middleware)
        
        # Add middleware concurrently
        add_tasks = []
        for middleware in middleware_list:
            task = asyncio.create_task(pipeline.add_middleware(middleware))
            add_tasks.append(task)
        
        await asyncio.gather(*add_tasks)
        
        # Verify all middleware were added
        assert await pipeline.get_middleware_count() == 5
        
        # Execute pipeline with all middleware
        context = MiddlewareContext(
            event_id="concurrent-test",
            data={"message": "concurrent execution"},
            metadata={"user_id": "test_user"}
        )
        result = await pipeline.execute(context)
        
        assert result.status == MiddlewareStatus.SUCCESS
        assert len(context.execution_path) == 5
        
        # Remove middleware concurrently
        remove_tasks = []
        for middleware in middleware_list[:3]:  # Remove first 3
            task = asyncio.create_task(pipeline.remove_middleware(middleware))
            remove_tasks.append(task)
        
        await asyncio.gather(*remove_tasks)
        
        # Verify remaining middleware
        assert await pipeline.get_middleware_count() == 2
        
        # Execute with remaining middleware
        context2 = MiddlewareContext(
            event_id="concurrent-test-2",
            data={"message": "after concurrent removal"},
            metadata={"user_id": "test_user"}
        )
        result2 = await pipeline.execute(context2)
        
        assert result2.status == MiddlewareStatus.SUCCESS
        assert len(context2.execution_path) == 2
    
    @pytest.mark.asyncio
    async def test_middleware_replacement_during_execution(self, pipeline):
        """Test replacing middleware while pipeline is executing."""
        # Set up initial middleware
        original_middleware = TrackingMiddleware("OriginalMiddleware", EventPriority.NORMAL, 10.0)
        stable_middleware = TrackingMiddleware("StableMiddleware", EventPriority.LOW, 1.0)
        
        await pipeline.add_middleware(original_middleware)
        await pipeline.add_middleware(stable_middleware)
        
        # Start long-running execution
        context1 = MiddlewareContext(
            event_id="long-running-1",
            data={"message": "long execution"},
            metadata={"user_id": "test_user"}
        )
        
        async def long_execution():
            return await pipeline.execute(context1)
        
        # Start execution and replace middleware while it's running
        execution_task = asyncio.create_task(long_execution())
        
        # Wait a bit, then replace middleware
        await asyncio.sleep(0.005)  # Wait 5ms
        
        replacement_middleware = TrackingMiddleware("ReplacementMiddleware", EventPriority.HIGH, 1.0)
        await pipeline.remove_middleware(original_middleware)
        await pipeline.add_middleware(replacement_middleware)
        
        # Wait for execution to complete
        result1 = await execution_task
        
        assert result1.status == MiddlewareStatus.SUCCESS
        
        # Execute new context with replaced middleware
        context2 = MiddlewareContext(
            event_id="after-replacement",
            data={"message": "after replacement"},
            metadata={"user_id": "test_user"}
        )
        result2 = await pipeline.execute(context2)
        
        assert result2.status == MiddlewareStatus.SUCCESS
        # New execution should use replacement middleware
        assert "ReplacementMiddleware" in context2.execution_path
        assert "OriginalMiddleware" not in context2.execution_path
        assert "StableMiddleware" in context2.execution_path
        
        # Verify middleware execution counts
        assert len(original_middleware.execution_log) == 1  # Only executed once before replacement
        assert len(replacement_middleware.execution_log) == 1  # Only executed in second context
        assert len(stable_middleware.execution_log) == 2  # Executed in both contexts
    
    @pytest.mark.asyncio
    async def test_pipeline_state_consistency_during_modifications(self, pipeline):
        """Test pipeline state consistency during frequent modifications."""
        # Create test middleware
        middleware_pool = []
        for i in range(10):
            middleware = TrackingMiddleware(f"PoolMiddleware{i}", EventPriority.NORMAL, 0.1)
            middleware_pool.append(middleware)
        
        # Perform frequent additions and removals
        operations = []
        
        # Add some middleware
        for i in range(5):
            operations.append(("add", middleware_pool[i]))
        
        # Mix of additions and removals
        for i in range(5, 8):
            operations.append(("add", middleware_pool[i]))
            operations.append(("remove", middleware_pool[i-3]))
        
        # Execute operations
        for operation, middleware in operations:
            if operation == "add":
                await pipeline.add_middleware(middleware)
            else:
                await pipeline.remove_middleware(middleware)
            
            # Verify pipeline consistency after each operation
            count = await pipeline.get_middleware_count()
            assert count >= 0
            
            # Execute a test context to ensure pipeline still works
            test_context = MiddlewareContext(
                event_id=f"consistency-test-{len(operations)}",
                data={"operation": operation, "middleware": middleware.name},
                metadata={"user_id": "test_user"}
            )
            
            result = await pipeline.execute(test_context)
            assert result.status in [MiddlewareStatus.SUCCESS, MiddlewareStatus.SKIPPED]
        
        # Final verification
        final_count = await pipeline.get_middleware_count()
        assert final_count == 5  # Should have 5 middleware remaining
        
        # Final execution test
        final_context = MiddlewareContext(
            event_id="final-consistency-test",
            data={"message": "final test"},
            metadata={"user_id": "test_user"}
        )
        final_result = await pipeline.execute(final_context)
        assert final_result.status == MiddlewareStatus.SUCCESS
        assert len(final_context.execution_path) == 5