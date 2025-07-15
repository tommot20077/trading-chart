# ABOUTME: Unit tests for priority queue functionality in InMemoryEventBus
# ABOUTME: Tests event priority handling, queue management, and concurrency control

import asyncio
import pytest
import pytest_asyncio
import time_machine

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class TestPriorityQueue:
    """Tests for priority queue functionality in InMemoryEventBus."""

    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create a fresh event bus for each test."""
        bus = InMemoryEventBus(max_queue_size=100, handler_timeout=5.0)
        yield bus
        await bus.close()

    @pytest.fixture
    def event_factory(self):
        """Factory function for creating test events."""

        def _create_event(event_type=EventType.TRADE, priority=EventPriority.NORMAL, data=None):
            return BaseEvent(
                event_type=event_type,
                source="test_source",
                symbol="BTCUSDT",
                data=data or {"test": "data"},
                priority=priority,
            )

        return _create_event

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_events_processed_by_priority(self, event_bus, event_factory):
        """Test that events are processed in priority order."""
        processing_order = []

        def handler(event):
            processing_order.append(event.priority.value)

        # Subscribe handler
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Create events with different priorities
        low_event = event_factory(priority=EventPriority.LOW)
        high_event = event_factory(priority=EventPriority.HIGH)
        critical_event = event_factory(priority=EventPriority.CRITICAL)
        normal_event = event_factory(priority=EventPriority.NORMAL)

        # Publish events in non-priority order
        await event_bus.publish(low_event)
        await event_bus.publish(normal_event)
        await event_bus.publish(high_event)
        await event_bus.publish(critical_event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify events were processed in priority order
        assert len(processing_order) == 4

        # Should be processed in order: CRITICAL, HIGH, NORMAL, LOW
        expected_order = [
            EventPriority.CRITICAL.value,
            EventPriority.HIGH.value,
            EventPriority.NORMAL.value,
            EventPriority.LOW.value,
        ]
        assert processing_order == expected_order

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_same_priority_events_fifo_order(self, event_bus, event_factory):
        """Test that events with same priority are processed in FIFO order."""
        processing_order = []

        def handler(event):
            processing_order.append(event.event_id)

        # Subscribe handler
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Create multiple events with same priority
        events = [event_factory(priority=EventPriority.NORMAL) for _ in range(5)]

        # Publish events
        for event in events:
            await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify FIFO order for same priority
        expected_order = [event.event_id for event in events]
        assert processing_order == expected_order

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_priority_values(self, event_bus, event_factory):
        """Test custom priority values work correctly."""
        processing_order = []

        def handler(event):
            processing_order.append(event.priority.value)

        # Subscribe handler
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Create events with custom priorities
        events = [
            event_factory(priority=EventPriority.custom(500)),  # Lower priority
            event_factory(priority=EventPriority.custom(50)),  # Higher priority
            event_factory(priority=EventPriority.custom(150)),  # Medium priority
        ]

        # Publish in reverse priority order
        for event in events:
            await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should be processed in order: 50, 150, 500
        expected_order = [50, 150, 500]
        assert processing_order == expected_order

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_priority_with_before_after_methods(self, event_bus, event_factory):
        """Test EventPriority.before() and after() methods."""
        processing_order = []

        def handler(event):
            processing_order.append(event.priority.value)

        # Subscribe handler
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Create events with relative priorities
        base_event = event_factory(priority=EventPriority.NORMAL)
        before_event = event_factory(priority=EventPriority.before(EventPriority.NORMAL))
        after_event = event_factory(priority=EventPriority.after(EventPriority.NORMAL))

        # Publish in non-priority order
        await event_bus.publish(after_event)
        await event_bus.publish(base_event)
        await event_bus.publish(before_event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify priority relationships
        assert len(processing_order) == 3
        assert processing_order[0] < EventPriority.NORMAL.value  # before_event
        assert processing_order[1] == EventPriority.NORMAL.value  # base_event
        assert processing_order[2] > EventPriority.NORMAL.value  # after_event

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_queue_capacity_management(self, event_bus, event_factory):
        """Test queue capacity and overflow handling."""
        with time_machine.travel("2024-01-01 12:00:00", tick=True) as traveller:
            # Create event bus with small queue
            small_bus = InMemoryEventBus(max_queue_size=3, handler_timeout=1.0)

            try:
                # Slow handler to fill up queue
                async def slow_handler(event):
                    # Simulate slow processing with time machine
                    traveller.shift(0.5)

                # Subscribe slow handler
                subscription_id = small_bus.subscribe(EventType.TRADE, slow_handler)

                # Fill up the queue
                for i in range(3):
                    await small_bus.publish(event_factory())
                    # Allow small time for publishing
                    traveller.shift(0.01)

                # Allow brief moment for async tasks to start
                traveller.shift(0.05)

                # Queue should be full
                assert small_bus.is_queue_full()

                # Next publish should raise exception
                with pytest.raises(RuntimeError, match="Event queue full"):
                    await small_bus.publish(event_factory())

                # Verify dropped count
                stats = small_bus.get_statistics()
                assert stats["dropped_count"] == 1

                # Cleanup
                small_bus.unsubscribe(subscription_id)

            finally:
                await small_bus.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_queue_flush_functionality(self, event_bus, event_factory):
        """Test queue flush functionality."""
        processed_events = []

        async def handler(event):
            processed_events.append(event)

        # Subscribe handler
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Publish multiple events
        events = [event_factory() for _ in range(5)]
        for event in events:
            await event_bus.publish(event)

        # Flush queue
        success = await event_bus.flush_queue(timeout=5.0)
        assert success

        # All events should be processed
        assert len(processed_events) == 5

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_priority_processing(self, event_bus, event_factory):
        """Test concurrent processing respects priority."""
        processing_order = []
        process_lock = asyncio.Lock()

        async def async_handler(event):
            async with process_lock:
                processing_order.append(event.priority.value)

        # Subscribe handler
        subscription_id = event_bus.subscribe(EventType.TRADE, async_handler)

        # Create events with different priorities
        events = [
            event_factory(priority=EventPriority.HIGH),
            event_factory(priority=EventPriority.LOW),
            event_factory(priority=EventPriority.CRITICAL),
            event_factory(priority=EventPriority.NORMAL),
        ]

        # Publish all events quickly
        for event in events:
            await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Verify priority order
        assert len(processing_order) == 4
        expected_order = [
            EventPriority.CRITICAL.value,
            EventPriority.HIGH.value,
            EventPriority.NORMAL.value,
            EventPriority.LOW.value,
        ]
        assert processing_order == expected_order

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_priority_with_mixed_event_types(self, event_bus, event_factory):
        """Test priority handling across different event types."""
        trade_events = []
        kline_events = []

        def trade_handler(event):
            trade_events.append(event.priority.value)

        def kline_handler(event):
            kline_events.append(event.priority.value)

        # Subscribe to different event types
        trade_sub = event_bus.subscribe(EventType.TRADE, trade_handler)
        kline_sub = event_bus.subscribe(EventType.KLINE, kline_handler)

        # Publish events with different priorities and types
        await event_bus.publish(event_factory(event_type=EventType.TRADE, priority=EventPriority.LOW))
        await event_bus.publish(event_factory(event_type=EventType.KLINE, priority=EventPriority.HIGH))
        await event_bus.publish(event_factory(event_type=EventType.TRADE, priority=EventPriority.CRITICAL))

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify events were processed in priority order within each type
        assert len(trade_events) == 2
        assert trade_events == [EventPriority.CRITICAL.value, EventPriority.LOW.value]

        assert len(kline_events) == 1
        assert kline_events == [EventPriority.HIGH.value]

        # Cleanup
        event_bus.unsubscribe(trade_sub)
        event_bus.unsubscribe(kline_sub)
