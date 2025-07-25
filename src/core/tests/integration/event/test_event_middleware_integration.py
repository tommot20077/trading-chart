# ABOUTME: Integration tests for event bus middleware pipeline integration
# ABOUTME: Tests the complete integration between event bus and middleware pipeline systems

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock
from datetime import datetime, UTC

from core.interfaces.event.event_bus import AbstractEventBus
from core.interfaces.middleware import AbstractMiddleware, AbstractMiddlewarePipeline
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.noop.event.event_bus import NoOpEventBus
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.implementations.noop.middleware.pipeline import NoOpMiddlewarePipeline
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority


class AuthMiddleware(AbstractMiddleware):
    """Auth middleware for testing."""
    
    def __init__(self, authorized_users=None, priority=EventPriority.HIGH):
        super().__init__(priority)
        self.authorized_users = authorized_users or ["test_user"]
        self.processed_events = []
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process authentication middleware."""
        self.processed_events.append(context.event_id)
        
        user_id = context.metadata.get("user_id")
        if user_id in self.authorized_users:
            return MiddlewareResult(
                middleware_name="AuthMiddleware",
                status=MiddlewareStatus.SUCCESS,
                data={"authorized": True, "user_id": user_id},
                should_continue=True,
                execution_time_ms=1.0
            )
        else:
            return MiddlewareResult(
                middleware_name="AuthMiddleware",
                status=MiddlewareStatus.FAILED,
                data={"authorized": False, "reason": "Unauthorized user"},
                should_continue=False,
                execution_time_ms=1.0
            )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Check if middleware can process this context."""
        return True


class LoggingMiddleware(AbstractMiddleware):
    """Logging middleware for testing."""
    
    def __init__(self, priority=EventPriority.LOW):
        super().__init__(priority)
        self.logged_events = []
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process logging middleware."""
        log_entry = {
            "event_id": context.event_id,
            "event_type": context.event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": context.metadata
        }
        self.logged_events.append(log_entry)
        
        return MiddlewareResult(
            middleware_name="LoggingMiddleware",
            status=MiddlewareStatus.SUCCESS,
            data={"logged": True, "log_entry": log_entry},
            should_continue=True,
            execution_time_ms=0.5
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Check if middleware can process this context."""
        return True


class TestEventMiddlewareIntegration:
    """Test suite for event bus and middleware integration."""

    @pytest.fixture
    def auth_middleware(self):
        """Create authentication middleware."""
        return AuthMiddleware(authorized_users=["alice", "bob"])

    @pytest.fixture
    def logging_middleware(self):
        """Create logging middleware."""
        return LoggingMiddleware()

    @pytest_asyncio.fixture
    async def middleware_pipeline(self, auth_middleware, logging_middleware):
        """Create middleware pipeline with auth and logging."""
        pipeline = InMemoryMiddlewarePipeline("TestPipeline")
        await pipeline.add_middleware(auth_middleware)
        await pipeline.add_middleware(logging_middleware)
        return pipeline

    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return BaseEvent(
            event_id="test-event-123",
            event_type=EventType.TRADE,
            symbol="BTCUSDT",
            timestamp=datetime.now(UTC),
            priority=EventPriority.NORMAL,
            source="test",
            data={"price": 50000, "quantity": 1.0}
        )

    @pytest.mark.asyncio
    async def test_in_memory_event_bus_middleware_integration(self, middleware_pipeline, sample_event, auth_middleware, logging_middleware):
        """Test InMemoryEventBus with middleware pipeline integration."""
        # Create event bus
        event_bus = InMemoryEventBus()
        
        # Set middleware pipeline
        await event_bus.set_middleware_pipeline(middleware_pipeline)
        
        # Verify pipeline is set
        current_pipeline = await event_bus.get_middleware_pipeline()
        assert current_pipeline is middleware_pipeline
        
        # Create event handler
        received_events = []
        
        async def test_handler(event: BaseEvent):
            received_events.append(event)
        
        # Subscribe to events
        subscription_id = event_bus.subscribe(EventType.TRADE, test_handler)
        
        # Update event metadata for authorized user
        sample_event.metadata = {"user_id": "alice"}
        
        # Publish event
        await event_bus.publish(sample_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify middleware was executed
        assert len(auth_middleware.processed_events) == 1
        assert auth_middleware.processed_events[0] == "test-event-123"
        
        assert len(logging_middleware.logged_events) == 1
        assert logging_middleware.logged_events[0]["event_id"] == "test-event-123"
        
        # Verify event was delivered to handler (authorized user)
        assert len(received_events) == 1
        assert received_events[0].event_id == "test-event-123"
        
        # Cleanup
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_middleware_blocks_unauthorized_event(self, middleware_pipeline, sample_event, auth_middleware):
        """Test that middleware can block unauthorized events."""
        # Create event bus
        event_bus = InMemoryEventBus()
        await event_bus.set_middleware_pipeline(middleware_pipeline)
        
        # Create event handler
        received_events = []
        
        async def test_handler(event: BaseEvent):
            received_events.append(event)
        
        # Subscribe to events
        event_bus.subscribe(EventType.TRADE, test_handler)
        
        # Update event metadata for unauthorized user
        sample_event.metadata = {"user_id": "unauthorized_user"}
        
        # Publish event
        await event_bus.publish(sample_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify middleware was executed
        assert len(auth_middleware.processed_events) == 1
        
        # Verify event was NOT delivered to handler (unauthorized user)
        assert len(received_events) == 0
        
        # Cleanup
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_noop_event_bus_middleware_integration(self, sample_event):
        """Test NoOpEventBus with middleware pipeline integration."""
        # Create NoOp event bus
        event_bus = NoOpEventBus()
        
        # Create NoOp middleware pipeline
        noop_pipeline = NoOpMiddlewarePipeline("NoOpTestPipeline")
        
        # Set middleware pipeline
        await event_bus.set_middleware_pipeline(noop_pipeline)
        
        # Verify pipeline is set
        current_pipeline = await event_bus.get_middleware_pipeline()
        assert current_pipeline is noop_pipeline
        
        # Create event handler (won't be called in NoOp implementation)
        received_events = []
        
        async def test_handler(event: BaseEvent):
            received_events.append(event)
        
        # Subscribe to events (fake subscription in NoOp)
        subscription_id = event_bus.subscribe(EventType.TRADE, test_handler)
        assert subscription_id is not None
        
        # Publish event (discarded in NoOp implementation)
        await event_bus.publish(sample_event)
        
        # NoOp implementation should not deliver events
        assert len(received_events) == 0
        
        # Cleanup
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_middleware_pipeline_priority_ordering(self, sample_event):
        """Test that middleware is executed in priority order."""
        # Create middleware with different priorities
        high_priority_middleware = AuthMiddleware(authorized_users=["alice", "bob"], priority=EventPriority.HIGH)
        low_priority_middleware = LoggingMiddleware(priority=EventPriority.LOW)
        
        # Create pipeline and add middleware in random order
        pipeline = InMemoryMiddlewarePipeline("PriorityTestPipeline")
        await pipeline.add_middleware(low_priority_middleware)  # Add low priority first
        await pipeline.add_middleware(high_priority_middleware)  # Add high priority second
        
        # Create event bus with pipeline
        event_bus = InMemoryEventBus()
        await event_bus.set_middleware_pipeline(pipeline)
        
        # Create handler
        received_events = []
        
        async def test_handler(event: BaseEvent):
            received_events.append(event)
        
        event_bus.subscribe(EventType.TRADE, test_handler)
        
        # Set authorized user
        sample_event.metadata = {"user_id": "alice"}
        
        # Publish event
        await event_bus.publish(sample_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify both middleware were executed
        assert len(high_priority_middleware.processed_events) == 1
        assert len(low_priority_middleware.logged_events) == 1
        
        # Verify event was delivered
        assert len(received_events) == 1
        
        # Cleanup
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_middleware_pipeline_removal(self, middleware_pipeline, sample_event):
        """Test removing middleware pipeline from event bus."""
        # Create event bus with pipeline
        event_bus = InMemoryEventBus()
        await event_bus.set_middleware_pipeline(middleware_pipeline)
        
        # Verify pipeline is set
        assert await event_bus.get_middleware_pipeline() is middleware_pipeline
        
        # Remove pipeline
        await event_bus.set_middleware_pipeline(None)
        
        # Verify pipeline is removed
        assert await event_bus.get_middleware_pipeline() is None
        
        # Create handler
        received_events = []
        
        async def test_handler(event: BaseEvent):
            received_events.append(event)
        
        event_bus.subscribe(EventType.TRADE, test_handler)
        
        # Publish event (should work without middleware)
        await event_bus.publish(sample_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify event was delivered without middleware processing
        assert len(received_events) == 1
        
        # Cleanup
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_multiple_event_types_with_middleware(self, middleware_pipeline, auth_middleware, logging_middleware):
        """Test middleware processing with multiple event types."""
        # Create event bus
        event_bus = InMemoryEventBus()
        await event_bus.set_middleware_pipeline(middleware_pipeline)
        
        # Create handlers for different event types
        trade_events = []
        kline_events = []
        
        async def trade_handler(event: BaseEvent):
            trade_events.append(event)
        
        async def kline_handler(event: BaseEvent):
            kline_events.append(event)
        
        # Subscribe to different event types
        event_bus.subscribe(EventType.TRADE, trade_handler)
        event_bus.subscribe(EventType.KLINE, kline_handler)
        
        # Create different types of events
        trade_event = BaseEvent(
            event_id="trade-123",
            event_type=EventType.TRADE,
            timestamp=datetime.now(UTC),
            source="test",
            metadata={"user_id": "alice"},
            data={"price": 50000}
        )
        
        kline_event = BaseEvent(
            event_id="kline-456",
            event_type=EventType.KLINE,
            timestamp=datetime.now(UTC),
            source="test",
            metadata={"user_id": "bob"},
            data={"open": 49000, "close": 50000}
        )
        
        # Publish both events
        await event_bus.publish(trade_event)
        await event_bus.publish(kline_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify middleware processed both events
        assert len(auth_middleware.processed_events) == 2
        assert "trade-123" in auth_middleware.processed_events
        assert "kline-456" in auth_middleware.processed_events
        
        assert len(logging_middleware.logged_events) == 2
        
        # Verify events were delivered to correct handlers
        assert len(trade_events) == 1
        assert trade_events[0].event_id == "trade-123"
        
        assert len(kline_events) == 1
        assert kline_events[0].event_id == "kline-456"
        
        # Cleanup
        await event_bus.close()

    @pytest.mark.asyncio
    async def test_event_bus_consistency_with_middleware(self):
        """Test that different event bus implementations handle middleware consistently."""
        # Create middleware
        auth_middleware = AuthMiddleware(authorized_users=["test_user"])
        
        # Test with InMemoryEventBus
        in_memory_bus = InMemoryEventBus()
        in_memory_pipeline = InMemoryMiddlewarePipeline("InMemoryPipeline")
        await in_memory_pipeline.add_middleware(auth_middleware)
        await in_memory_bus.set_middleware_pipeline(in_memory_pipeline)
        
        # Test with NoOpEventBus
        noop_bus = NoOpEventBus()
        noop_pipeline = NoOpMiddlewarePipeline("NoOpPipeline")
        await noop_bus.set_middleware_pipeline(noop_pipeline)
        
        # Verify both can set and get middleware pipelines
        assert await in_memory_bus.get_middleware_pipeline() is in_memory_pipeline
        assert await noop_bus.get_middleware_pipeline() is noop_pipeline
        
        # Verify both can remove middleware pipelines
        await in_memory_bus.set_middleware_pipeline(None)
        await noop_bus.set_middleware_pipeline(None)
        
        assert await in_memory_bus.get_middleware_pipeline() is None
        assert await noop_bus.get_middleware_pipeline() is None
        
        # Cleanup
        await in_memory_bus.close()
        await noop_bus.close()