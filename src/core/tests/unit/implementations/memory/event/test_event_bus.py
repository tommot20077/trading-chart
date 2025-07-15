# ABOUTME: Unit tests for InMemoryEventBus implementation
# ABOUTME: Comprehensive tests covering all event bus functionality

import asyncio
import pytest
import pytest_asyncio
import time_machine
from unittest.mock import Mock

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


class TestInMemoryEventBus:
    """Unit tests for InMemoryEventBus implementation."""

    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create a fresh event bus for each test."""
        bus = InMemoryEventBus(max_queue_size=100, handler_timeout=1.0)
        yield bus
        await bus.close()

    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return BaseEvent(
            event_type=EventType.TRADE, source="test_source", symbol="BTCUSDT", data={"price": 50000, "quantity": 1.0}
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, event_bus):
        """Test event bus initialization."""
        assert not event_bus.is_closed
        assert event_bus.get_subscription_count() == 0
        assert event_bus.get_subscription_count(EventType.TRADE) == 0
        assert event_bus.get_queue_size() == 0
        assert event_bus.get_queue_capacity() == 100
        assert not event_bus.is_queue_full()

        # Test statistics
        stats = event_bus.get_statistics()
        assert stats["published_count"] == 0
        assert stats["processed_count"] == 0
        assert stats["error_count"] == 0
        assert stats["timeout_count"] == 0
        assert stats["dropped_count"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_publish_requires_base_event(self, event_bus):
        """Test that publish requires a BaseEvent instance."""
        with pytest.raises(TypeError, match="Event must be an instance of BaseEvent"):
            await event_bus.publish("not an event")

        with pytest.raises(TypeError, match="Event must be an instance of BaseEvent"):
            await event_bus.publish(None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_returns_unique_id(self, event_bus):
        """Test that subscribe returns a unique string ID."""
        handler = Mock()

        id1 = event_bus.subscribe(EventType.TRADE, handler)
        id2 = event_bus.subscribe(EventType.TRADE, handler)

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_validates_parameters(self, event_bus):
        """Test subscription parameter validation."""
        handler = Mock()

        # Invalid event type
        with pytest.raises(ValueError, match="event_type must be an EventType enum member"):
            event_bus.subscribe("invalid", handler)

        # Invalid handler
        with pytest.raises(ValueError, match="handler must be callable"):
            event_bus.subscribe(EventType.TRADE, "not callable")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self, event_bus, sample_event):
        """Test basic publish-subscribe functionality."""
        received_events = []

        def handler(event):
            received_events.append(event)

        # Subscribe to events
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)
        assert event_bus.get_subscription_count() == 1
        assert event_bus.get_subscription_count(EventType.TRADE) == 1

        # Publish event
        await event_bus.publish(sample_event)

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0] == sample_event
        assert received_events[0].symbol == "BTCUSDT"

        # Verify statistics
        stats = event_bus.get_statistics()
        assert stats["published_count"] == 1
        assert stats["processed_count"] == 1
        assert stats["error_count"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_handler(self, event_bus, sample_event):
        """Test asynchronous event handlers."""
        received_events = []

        async def async_handler(event):
            received_events.append(event)

        # Subscribe with async handler
        subscription_id = event_bus.subscribe(EventType.TRADE, async_handler)

        # Publish event
        await event_bus.publish(sample_event)

        # Wait a bit for async processing to complete
        await asyncio.sleep(0.1)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0] == sample_event

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_symbol_filtering(self, event_bus):
        """Test symbol-based event filtering."""
        btc_events = []
        eth_events = []
        all_events = []

        def btc_handler(event):
            btc_events.append(event)

        def eth_handler(event):
            eth_events.append(event)

        def all_handler(event):
            all_events.append(event)

        # Subscribe with symbol filters
        event_bus.subscribe(EventType.TRADE, btc_handler, filter_symbol="BTCUSDT")
        event_bus.subscribe(EventType.TRADE, eth_handler, filter_symbol="ETHUSDT")
        event_bus.subscribe(EventType.TRADE, all_handler)  # No filter

        # Create events with different symbols
        btc_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={"price": 50000})

        eth_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="ETHUSDT", data={"price": 3000})

        # Publish events
        await event_bus.publish(btc_event)
        await event_bus.publish(eth_event)

        # Wait for async processing to complete
        await asyncio.sleep(0.1)

        # Verify filtering
        assert len(btc_events) == 1
        assert btc_events[0].symbol == "BTCUSDT"

        assert len(eth_events) == 1
        assert eth_events[0].symbol == "ETHUSDT"

        assert len(all_events) == 2  # Receives all events

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_type_filtering(self, event_bus):
        """Test event type filtering."""
        trade_events = []
        kline_events = []

        def trade_handler(event):
            trade_events.append(event)

        def kline_handler(event):
            kline_events.append(event)

        # Subscribe to different event types
        event_bus.subscribe(EventType.TRADE, trade_handler)
        event_bus.subscribe(EventType.KLINE, kline_handler)

        # Create events of different types
        trade_event = BaseEvent(event_type=EventType.TRADE, source="test", data={"price": 50000})

        kline_event = BaseEvent(event_type=EventType.KLINE, source="test", data={"open": 49000, "close": 50000})

        # Publish events
        await event_bus.publish(trade_event)
        await event_bus.publish(kline_event)

        # Wait for async processing to complete
        await asyncio.sleep(0.1)

        # Verify type filtering
        assert len(trade_events) == 1
        assert trade_events[0].event_type == EventType.TRADE

        assert len(kline_events) == 1
        assert kline_events[0].event_type == EventType.KLINE

    @pytest.mark.unit
    def test_unsubscribe(self, event_bus):
        """Test unsubscribing handlers."""
        handler = Mock()

        # Subscribe and verify
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)
        assert event_bus.get_subscription_count() == 1

        # Unsubscribe and verify
        result = event_bus.unsubscribe(subscription_id)
        assert result is True
        assert event_bus.get_subscription_count() == 0

        # Try to unsubscribe again
        result = event_bus.unsubscribe(subscription_id)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_unsubscribe_invalid_id(self, event_bus):
        """Test unsubscribing with invalid ID."""
        result = event_bus.unsubscribe("invalid-id")
        assert result is False

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_unsubscribe_all(self, event_bus):
        """Test unsubscribing all handlers."""
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()

        # Subscribe to different event types
        event_bus.subscribe(EventType.TRADE, handler1)
        event_bus.subscribe(EventType.TRADE, handler2)
        event_bus.subscribe(EventType.KLINE, handler3)

        assert event_bus.get_subscription_count() == 3

        # Unsubscribe all for TRADE events
        removed_count = event_bus.unsubscribe_all(EventType.TRADE)
        assert removed_count == 2
        assert event_bus.get_subscription_count() == 1
        assert event_bus.get_subscription_count(EventType.TRADE) == 0
        assert event_bus.get_subscription_count(EventType.KLINE) == 1

        # Unsubscribe all remaining
        removed_count = event_bus.unsubscribe_all()
        assert removed_count == 1
        assert event_bus.get_subscription_count() == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_event(self, event_bus):
        """Test waiting for specific events."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Create event to wait for
            target_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={"price": 50000})

            async def publish_later():
                traveller.shift(0.1)
                await event_bus.publish(target_event)

            # Start publishing task
            publish_task = asyncio.create_task(publish_later())

            # Wait for the event
            received_event = await event_bus.wait_for(EventType.TRADE)

            # Verify
            assert received_event is not None
            assert received_event.event_type == EventType.TRADE
            assert received_event.symbol == "BTCUSDT"

            await publish_task

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_event_with_timeout(self, event_bus):
        """Test waiting for events with timeout."""
        # Wait for event that won't come
        with pytest.raises(asyncio.TimeoutError):
            await event_bus.wait_for(EventType.TRADE, timeout=0.1)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_event_with_filter(self, event_bus):
        """Test waiting for events with custom filter."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Create events
            low_price_event = BaseEvent(event_type=EventType.TRADE, source="test", data={"price": 1000})

            high_price_event = BaseEvent(event_type=EventType.TRADE, source="test", data={"price": 50000})

            async def publish_events():
                traveller.shift(0.05)
                await event_bus.publish(low_price_event)
                traveller.shift(0.05)
                await event_bus.publish(high_price_event)

            # Start publishing task
            publish_task = asyncio.create_task(publish_events())

            # Wait for high-price event only
            def high_price_filter(event):
                return event.data.get("price", 0) > 10000

            received_event = await event_bus.wait_for(EventType.TRADE, filter_func=high_price_filter, timeout=1.0)

            # Should receive the high-price event, not the low-price one
            assert received_event is not None
            assert received_event.data["price"] == 50000

            await publish_task

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handler_error_handling(self, event_bus, sample_event):
        """Test that handler errors don't break the event bus."""
        good_events = []

        def failing_handler(event):
            raise ValueError("Handler error")

        def good_handler(event):
            good_events.append(event)

        # Subscribe both handlers
        event_bus.subscribe(EventType.TRADE, failing_handler)
        event_bus.subscribe(EventType.TRADE, good_handler)

        # Publish event
        await event_bus.publish(sample_event)

        # Wait for async processing to complete
        await asyncio.sleep(0.1)

        # Good handler should still receive the event
        assert len(good_events) == 1
        assert good_events[0] == sample_event

        # Error count should be tracked
        stats = event_bus.get_statistics()
        assert stats["error_count"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_event_bus(self, event_bus):
        """Test closing the event bus."""
        handler = Mock()

        # Subscribe and verify
        event_bus.subscribe(EventType.TRADE, handler)
        assert not event_bus.is_closed
        assert event_bus.get_subscription_count() == 1

        # Close the event bus
        await event_bus.close()

        # Verify closed state
        assert event_bus.is_closed
        assert event_bus.get_subscription_count() == 0

        # Operations should fail after closing
        with pytest.raises(RuntimeError, match="Event bus is closed"):
            event_bus.subscribe(EventType.TRADE, handler)

        with pytest.raises(RuntimeError, match="Event bus is closed"):
            await event_bus.publish(BaseEvent(event_type=EventType.TRADE, source="test", data={}))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_close_calls(self, event_bus):
        """Test that multiple close calls are safe."""
        await event_bus.close()
        assert event_bus.is_closed

        # Second close should not raise error
        await event_bus.close()
        assert event_bus.is_closed

    @pytest.mark.unit
    def test_get_statistics(self, event_bus):
        """Test getting event bus statistics."""
        handler = Mock()

        # Initial statistics
        stats = event_bus.get_statistics()
        assert stats["published_count"] == 0
        assert stats["error_count"] == 0
        assert stats["total_subscriptions"] == 0
        assert stats["is_closed"] is False

        # Add subscription
        event_bus.subscribe(EventType.TRADE, handler)
        event_bus.subscribe(EventType.KLINE, handler)

        stats = event_bus.get_statistics()
        assert stats["total_subscriptions"] == 2
        assert stats["subscriptions_by_type"]["trade"] == 1
        assert stats["subscriptions_by_type"]["kline"] == 1

    @pytest.mark.unit
    def test_get_subscriptions_for_type(self, event_bus):
        """Test getting subscription details for debugging."""

        def named_handler(event):
            pass

        # Subscribe with different configurations
        id1 = event_bus.subscribe(EventType.TRADE, named_handler)
        id2 = event_bus.subscribe(EventType.TRADE, named_handler, filter_symbol="BTCUSDT")

        # Get subscription details
        subscriptions = event_bus.get_subscriptions_for_type(EventType.TRADE)

        assert len(subscriptions) == 2

        # Check first subscription
        sub1 = next(s for s in subscriptions if s["id"] == id1)
        assert sub1["event_type"] == "trade"
        assert sub1["filter_symbol"] is None
        assert sub1["is_async"] is False
        assert sub1["handler_name"] == "named_handler"

        # Check second subscription
        sub2 = next(s for s in subscriptions if s["id"] == id2)
        assert sub2["filter_symbol"] == "BTCUSDT"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, event_bus):
        """Test concurrent publish and subscribe operations."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe
        event_bus.subscribe(EventType.TRADE, handler)

        # Create multiple events
        events = [BaseEvent(event_type=EventType.TRADE, source="test", data={"id": i}) for i in range(10)]

        # Publish events concurrently
        tasks = [event_bus.publish(event) for event in events]
        await asyncio.gather(*tasks)

        # Wait for async processing to complete
        await asyncio.sleep(0.1)

        # All events should be received
        assert len(received_events) == 10
        received_ids = {event.data["id"] for event in received_events}
        expected_ids = {i for i in range(10)}
        assert received_ids == expected_ids


class TestInMemoryEventBusEdgeCases:
    """Test edge cases and error conditions."""

    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create a fresh event bus for each test."""
        bus = InMemoryEventBus(max_queue_size=100, handler_timeout=1.0)
        yield bus
        await bus.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_validates_event_type(self, event_bus):
        """Test that wait_for validates event type parameter."""
        with pytest.raises(ValueError, match="event_type must be an EventType enum member"):
            await event_bus.wait_for("invalid_type")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_cancelled_future(self, event_bus):
        """Test behavior when wait_for future is cancelled."""
        # Start waiting
        wait_task = asyncio.create_task(event_bus.wait_for(EventType.TRADE, timeout=1.0))

        # Cancel the wait
        wait_task.cancel()

        # Should handle cancellation gracefully
        with pytest.raises(asyncio.CancelledError):
            await wait_task

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_publish_after_handler_subscription_removed(self, event_bus):
        """Test publishing after a handler subscription is removed during processing."""
        events_received = []

        def handler(event):
            events_received.append(event)

        # Subscribe and immediately unsubscribe
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)
        event_bus.unsubscribe(subscription_id)

        # Publish event
        event = BaseEvent(event_type=EventType.TRADE, source="test", data={})
        await event_bus.publish(event)

        # No events should be received
        assert len(events_received) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscription_count_edge_cases(self, event_bus):
        """Test subscription count with edge cases."""
        # Count for non-existent event type
        assert event_bus.get_subscription_count(EventType.ERROR) == 0

        # Subscribe and unsubscribe
        handler = Mock()
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)
        assert event_bus.get_subscription_count(EventType.TRADE) == 1

        event_bus.unsubscribe(subscription_id)
        assert event_bus.get_subscription_count(EventType.TRADE) == 0
