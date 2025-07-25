# ABOUTME: Unit tests for EventMiddlewareBus wrapper implementation
# ABOUTME: Tests middleware integration and event bus decoration functionality

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from core.implementations.memory.event.event_middleware_bus import EventMiddlewareBus
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus


@pytest.fixture
def mock_base_bus():
    """Create a mock base event bus."""
    bus = Mock(spec=InMemoryEventBus)
    bus.publish = AsyncMock()
    bus.subscribe = Mock(return_value="sub_123")
    bus.unsubscribe = Mock(return_value=True)
    bus.unsubscribe_all = Mock(return_value=3)
    bus.close = AsyncMock()
    bus.get_subscription_count = Mock(return_value=5)
    bus.wait_for = AsyncMock()
    bus.is_closed = False
    return bus


@pytest.fixture
def mock_middleware_pipeline():
    """Create a mock middleware pipeline."""
    pipeline = Mock()
    pipeline.execute = AsyncMock()
    return pipeline


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return BaseEvent(
        event_type=EventType.TRADE,
        event_id="test_event_123",
        source="test_exchange",
        symbol="BTCUSDT",
        priority=EventPriority.NORMAL,
        data={"symbol": "BTCUSDT", "price": 50000.0},
        metadata={"timestamp": "2025-01-25T10:00:00Z"}
    )


@pytest.fixture
def event_middleware_bus(mock_base_bus):
    """Create an EventMiddlewareBus instance for testing."""
    return EventMiddlewareBus(mock_base_bus, name="TestMiddlewareBus")


class TestEventMiddlewareBus:
    """Test suite for EventMiddlewareBus implementation."""

    def test_initialization(self, mock_base_bus):
        """Test EventMiddlewareBus initialization."""
        bus = EventMiddlewareBus(mock_base_bus, name="TestBus")
        
        assert bus.base_bus is mock_base_bus
        assert bus.name == "TestBus"
        assert bus.middleware_pipeline is None

    def test_initialization_with_default_name(self, mock_base_bus):
        """Test initialization with default name."""
        bus = EventMiddlewareBus(mock_base_bus)
        
        assert bus.name == "EventMiddlewareBus"

    @pytest.mark.asyncio
    async def test_set_middleware_pipeline(self, event_middleware_bus, mock_middleware_pipeline):
        """Test setting middleware pipeline."""
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        
        assert event_middleware_bus.middleware_pipeline is mock_middleware_pipeline

    @pytest.mark.asyncio
    async def test_set_middleware_pipeline_to_none(self, event_middleware_bus, mock_middleware_pipeline):
        """Test removing middleware pipeline."""
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        await event_middleware_bus.set_middleware_pipeline(None)
        
        assert event_middleware_bus.middleware_pipeline is None

    @pytest.mark.asyncio
    async def test_get_middleware_pipeline(self, event_middleware_bus, mock_middleware_pipeline):
        """Test getting middleware pipeline."""
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        
        result = await event_middleware_bus.get_middleware_pipeline()
        assert result is mock_middleware_pipeline

    @pytest.mark.asyncio
    async def test_publish_without_middleware(self, event_middleware_bus, sample_event):
        """Test event publishing without middleware pipeline."""
        await event_middleware_bus.publish(sample_event)
        
        # Should delegate directly to base bus
        event_middleware_bus.base_bus.publish.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_publish_with_middleware_success(self, event_middleware_bus, mock_middleware_pipeline, sample_event):
        """Test event publishing with successful middleware processing."""
        # Configure middleware to allow continuation
        mock_result = Mock()
        mock_result.should_continue = True
        mock_result.modified_context = None
        mock_result.status = MiddlewareStatus.SUCCESS
        mock_middleware_pipeline.execute.return_value = mock_result
        
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        await event_middleware_bus.publish(sample_event)
        
        # Verify middleware was executed
        mock_middleware_pipeline.execute.assert_called_once()
        
        # Verify event was published to base bus
        event_middleware_bus.base_bus.publish.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_publish_with_middleware_blocked(self, event_middleware_bus, mock_middleware_pipeline, sample_event):
        """Test event publishing blocked by middleware."""
        # Configure middleware to block continuation
        mock_result = Mock()
        mock_result.should_continue = False
        mock_result.status = MiddlewareStatus.BLOCKED
        mock_result.data = {"reason": "Event blocked by security middleware"}
        mock_middleware_pipeline.execute.return_value = mock_result
        
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        await event_middleware_bus.publish(sample_event)
        
        # Verify middleware was executed
        mock_middleware_pipeline.execute.assert_called_once()
        
        # Verify event was NOT published to base bus
        event_middleware_bus.base_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_with_middleware_context_modifications(self, event_middleware_bus, mock_middleware_pipeline, sample_event):
        """Test event publishing with middleware context modifications."""
        # Create modified context
        modified_context = Mock()
        modified_context.data = {"symbol": "ETHUSDT", "price": 3000.0, "modified": True}
        modified_context.metadata = {"processed_by": "middleware", "timestamp": "2025-01-25T11:00:00Z"}
        
        mock_result = Mock()
        mock_result.should_continue = True
        mock_result.modified_context = modified_context
        mock_result.status = MiddlewareStatus.SUCCESS
        mock_middleware_pipeline.execute.return_value = mock_result
        
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        await event_middleware_bus.publish(sample_event)
        
        # Verify middleware was executed
        mock_middleware_pipeline.execute.assert_called_once()
        
        # Verify event was published to base bus
        event_middleware_bus.base_bus.publish.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_publish_with_middleware_error(self, event_middleware_bus, mock_middleware_pipeline, sample_event):
        """Test event publishing with middleware error."""
        # Configure middleware to raise exception
        mock_middleware_pipeline.execute.side_effect = RuntimeError("Middleware error")
        
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        
        with pytest.raises(RuntimeError, match="Middleware error"):
            await event_middleware_bus.publish(sample_event)
        
        # Verify event was NOT published to base bus
        event_middleware_bus.base_bus.publish.assert_not_called()

    def test_subscribe_delegation(self, event_middleware_bus):
        """Test subscription delegation to base bus."""
        mock_handler = Mock()
        
        result = event_middleware_bus.subscribe(EventType.TRADE, mock_handler, filter_symbol="BTCUSDT")
        
        event_middleware_bus.base_bus.subscribe.assert_called_once_with(
            EventType.TRADE, mock_handler, filter_symbol="BTCUSDT"
        )
        assert result == "sub_123"

    def test_unsubscribe_delegation(self, event_middleware_bus):
        """Test unsubscription delegation to base bus."""
        result = event_middleware_bus.unsubscribe("sub_123")
        
        event_middleware_bus.base_bus.unsubscribe.assert_called_once_with("sub_123")
        assert result is True

    def test_unsubscribe_all_delegation(self, event_middleware_bus):
        """Test unsubscribe all delegation to base bus."""
        result = event_middleware_bus.unsubscribe_all(EventType.TRADE)
        
        event_middleware_bus.base_bus.unsubscribe_all.assert_called_once_with(EventType.TRADE)
        assert result == 3

    @pytest.mark.asyncio
    async def test_close_delegation(self, event_middleware_bus, mock_middleware_pipeline):
        """Test close operation delegation and cleanup."""
        await event_middleware_bus.set_middleware_pipeline(mock_middleware_pipeline)
        await event_middleware_bus.close()
        
        # Verify middleware pipeline is cleared
        assert event_middleware_bus.middleware_pipeline is None
        
        # Verify base bus close is called
        event_middleware_bus.base_bus.close.assert_called_once()

    def test_get_subscription_count_delegation(self, event_middleware_bus):
        """Test get subscription count delegation to base bus."""
        result = event_middleware_bus.get_subscription_count(EventType.TRADE)
        
        event_middleware_bus.base_bus.get_subscription_count.assert_called_once_with(EventType.TRADE)
        assert result == 5

    @pytest.mark.asyncio
    async def test_wait_for_delegation(self, event_middleware_bus, sample_event):
        """Test wait for delegation to base bus."""
        event_middleware_bus.base_bus.wait_for.return_value = sample_event
        filter_func = lambda event: event.data.get("symbol") == "BTCUSDT"
        
        result = await event_middleware_bus.wait_for(EventType.TRADE, timeout=10.0, filter_func=filter_func)
        
        event_middleware_bus.base_bus.wait_for.assert_called_once_with(
            EventType.TRADE, timeout=10.0, filter_func=filter_func
        )
        assert result is sample_event

    def test_is_closed_property(self, event_middleware_bus):
        """Test is_closed property delegation."""
        event_middleware_bus.base_bus.is_closed = True
        
        assert event_middleware_bus.is_closed is True

    def test_create_middleware_context(self, event_middleware_bus, sample_event):
        """Test middleware context creation from event."""
        context = event_middleware_bus._create_middleware_context(sample_event)
        
        assert isinstance(context, MiddlewareContext)
        assert context.event_type == EventType.TRADE.value
        assert context.event_id == "test_event_123"
        assert context.symbol == "BTCUSDT"
        assert context.data["symbol"] == "BTCUSDT"
        assert context.data["price"] == 50000.0
        assert context.data["event_metadata"]["event_type"] == EventType.TRADE.value
        assert context.data["event_metadata"]["priority"] == EventPriority.NORMAL.value
        assert context.metadata["processing_stage"] == "event_middleware_processing"
        assert context.metadata["middleware_bus_name"] == "TestMiddlewareBus"

    def test_apply_context_modifications(self, event_middleware_bus, sample_event):
        """Test applying context modifications to event."""
        # Create modifications dict as would be provided by MiddlewareResult.modified_context
        modifications = {
            "data": {"symbol": "ETHUSDT", "price": 3000.0, "modified": True},
            "metadata": {"processed_by": "test_middleware", "custom_field": "test_value"}
        }
        
        event_middleware_bus._apply_context_modifications(sample_event, modifications)
        
        # Verify data modifications (excluding event_metadata)
        assert sample_event.data["symbol"] == "ETHUSDT"
        assert sample_event.data["price"] == 3000.0
        assert sample_event.data["modified"] is True
        
        # Verify metadata modifications (excluding processing stage info)
        assert sample_event.metadata["processed_by"] == "test_middleware"
        assert sample_event.metadata["custom_field"] == "test_value"

    def test_get_bus_info(self, event_middleware_bus, mock_middleware_pipeline):
        """Test getting bus information."""
        # Add middleware pipeline with info method
        mock_middleware_pipeline.get_pipeline_info = Mock(return_value={"middleware_count": 3})
        event_middleware_bus.middleware_pipeline = mock_middleware_pipeline
        
        info = event_middleware_bus.get_bus_info()
        
        assert info["name"] == "TestMiddlewareBus"
        assert info["base_bus_type"] == "Mock"
        assert info["is_closed"] is False
        assert info["has_middleware_pipeline"] is True
        assert info["total_subscriptions"] == 5
        assert info["middleware_pipeline"]["middleware_count"] == 3

    def test_get_bus_info_without_middleware(self, event_middleware_bus):
        """Test getting bus information without middleware."""
        info = event_middleware_bus.get_bus_info()
        
        assert info["name"] == "TestMiddlewareBus"
        assert info["has_middleware_pipeline"] is False
        assert "middleware_pipeline" not in info


class TestEventMiddlewareBusIntegration:
    """Integration tests for EventMiddlewareBus with real components."""

    @pytest.mark.asyncio
    async def test_real_event_bus_integration(self):
        """Test EventMiddlewareBus with real InMemoryEventBus."""
        base_bus = InMemoryEventBus()
        middleware_bus = EventMiddlewareBus(base_bus, name="IntegrationTest")
        
        try:
            # Test subscription
            events_received = []
            
            def handler(event):
                events_received.append(event)
            
            sub_id = middleware_bus.subscribe(EventType.TRADE, handler)
            
            # Test publishing
            test_event = BaseEvent(
                event_type=EventType.TRADE,
                event_id="integration_test",
                source="integration_test",
                priority=EventPriority.HIGH,
                data={"symbol": "BTCUSDT", "price": 50000.0}
            )
            
            await middleware_bus.publish(test_event)
            
            # Wait for event processing
            await asyncio.sleep(0.1)
            
            # Verify event was received
            assert len(events_received) == 1
            assert events_received[0].event_id == "integration_test"
            
            # Test unsubscription
            assert middleware_bus.unsubscribe(sub_id) is True
            assert middleware_bus.get_subscription_count() == 0
            
        finally:
            await middleware_bus.close()

    @pytest.mark.asyncio
    async def test_middleware_pipeline_integration(self):
        """Test EventMiddlewareBus with middleware pipeline."""
        from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
        
        base_bus = InMemoryEventBus()
        pipeline = InMemoryMiddlewarePipeline()
        middleware_bus = EventMiddlewareBus(base_bus, name="MiddlewareIntegration")
        
        try:
            # Create a simple test middleware
            class TestMiddleware:
                def __init__(self):
                    from core.models.event.event_priority import EventPriority
                    self.priority = EventPriority.NORMAL
                
                async def process(self, context):
                    from core.models.middleware import MiddlewareResult, MiddlewareStatus
                    # Simple middleware that allows continuation
                    return MiddlewareResult(
                        middleware_name="TestMiddleware",
                        status=MiddlewareStatus.SUCCESS,
                        should_continue=True,
                        data={"processed": True}
                    )
                
                def can_process(self, context):
                    return True
            
            test_middleware = TestMiddleware()
            await pipeline.add_middleware(test_middleware)
            
            await middleware_bus.set_middleware_pipeline(pipeline)
            
            # Test event publishing with middleware
            test_event = BaseEvent(
                event_type=EventType.TRADE,
                event_id="middleware_test",
                source="middleware_test",
                priority=EventPriority.NORMAL,
                data={"symbol": "ETHUSDT", "price": 3000.0}
            )
            
            events_received = []
            
            def handler(event):
                events_received.append(event)
            
            middleware_bus.subscribe(EventType.TRADE, handler)
            await middleware_bus.publish(test_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Verify event was processed through middleware and received
            assert len(events_received) == 1
            assert events_received[0].event_id == "middleware_test"
            
        finally:
            await middleware_bus.close()