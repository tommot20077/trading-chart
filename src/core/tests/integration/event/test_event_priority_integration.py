# ABOUTME: Integration tests for event processing order and priority (E2.2)
# ABOUTME: Tests event priority handling and execution order guarantees

import pytest
import asyncio

from core.interfaces.event.event_bus import AbstractEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


@pytest.mark.integration
class TestEventPriorityIntegration:
    """
    Integration tests for event processing order and priority.

    Tests priority-based event processing and execution order guarantees.
    This covers task E2.2 from the integration test plan.
    """

    @pytest.mark.asyncio
    async def test_high_priority_events_processed_first(self, event_bus: AbstractEventBus, event_factory):
        """Test that events are processed in priority order."""
        # Arrange: Track processing order
        processing_order = []

        def priority_handler(event: BaseEvent):
            processing_order.append((event.priority.value, event.event_id))

        subscription_id = event_bus.subscribe(EventType.TRADE, priority_handler)

        # Create events with different priorities
        low_event = event_factory(EventType.TRADE, priority=EventPriority.LOW)
        high_event = event_factory(EventType.TRADE, priority=EventPriority.HIGH)
        critical_event = event_factory(EventType.TRADE, priority=EventPriority.CRITICAL)
        normal_event = event_factory(EventType.TRADE, priority=EventPriority.NORMAL)

        # Act: Publish events in non-priority order
        await event_bus.publish(low_event)
        await event_bus.publish(normal_event)
        await event_bus.publish(high_event)
        await event_bus.publish(critical_event)

        # Allow processing to complete
        await asyncio.sleep(0.5)

        # Assert: Events were processed in priority order (NEW behavior)
        assert len(processing_order) == 4

        # Verify events were processed in priority order: CRITICAL, HIGH, NORMAL, LOW
        expected_priorities = [
            EventPriority.CRITICAL.value,  # Highest priority
            EventPriority.HIGH.value,  # Second highest
            EventPriority.NORMAL.value,  # Third highest
            EventPriority.LOW.value,  # Lowest priority
        ]

        actual_priorities = [item[0] for item in processing_order]
        assert actual_priorities == expected_priorities

        # Verify all events were processed
        assert len(set(item[1] for item in processing_order)) == 4  # All unique event IDs

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_same_priority_events_fifo_order(self, event_bus: AbstractEventBus, event_factory):
        """Test events with same priority are processed in FIFO order."""
        # Arrange: Track processing order
        processing_order = []

        def fifo_handler(event: BaseEvent):
            processing_order.append(event.event_id)

        subscription_id = event_bus.subscribe(EventType.TRADE, fifo_handler)

        # Create multiple events with same priority
        events = []
        for i in range(5):
            event = event_factory(EventType.TRADE, priority=EventPriority.NORMAL)
            events.append(event)

        # Act: Publish events in sequence
        for event in events:
            await event_bus.publish(event)

        # Allow processing to complete
        await asyncio.sleep(0.2)

        # Assert: Events were processed in FIFO order
        expected_order = [event.event_id for event in events]
        assert processing_order == expected_order

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_custom_priority_values(self, event_bus: AbstractEventBus, event_factory):
        """Test custom priority values are processed in correct order."""
        # Arrange: Track processing order
        processing_order = []

        def custom_priority_handler(event: BaseEvent):
            processing_order.append(event.priority.value)

        subscription_id = event_bus.subscribe(EventType.TRADE, custom_priority_handler)

        # Create events with custom priorities
        custom_high = event_factory(EventType.TRADE, priority=EventPriority.custom(50))
        custom_medium = event_factory(EventType.TRADE, priority=EventPriority.custom(150))
        custom_low = event_factory(EventType.TRADE, priority=EventPriority.custom(250))

        # Act: Publish in reverse priority order
        await event_bus.publish(custom_low)
        await event_bus.publish(custom_medium)
        await event_bus.publish(custom_high)

        # Allow processing to complete
        await asyncio.sleep(0.5)

        # Assert: Events were processed in priority order (NEW behavior)
        assert processing_order == [50, 150, 250]  # Priority order

        # Verify custom priorities are preserved
        assert custom_high.priority.value == 50
        assert custom_medium.priority.value == 150
        assert custom_low.priority.value == 250

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_priority_with_before_after_methods(self, event_bus: AbstractEventBus, event_factory):
        """Test EventPriority.before() and after() methods create correct values."""
        # Arrange: Track processing order
        processing_order = []

        def relative_priority_handler(event: BaseEvent):
            processing_order.append(event.priority.value)

        subscription_id = event_bus.subscribe(EventType.TRADE, relative_priority_handler)

        # Create events with relative priorities
        base_event = event_factory(EventType.TRADE, priority=EventPriority.NORMAL)
        before_event = event_factory(EventType.TRADE, priority=EventPriority.before(EventPriority.NORMAL))
        after_event = event_factory(EventType.TRADE, priority=EventPriority.after(EventPriority.NORMAL))

        # Act: Publish in non-priority order
        await event_bus.publish(after_event)
        await event_bus.publish(base_event)
        await event_bus.publish(before_event)

        # Allow processing to complete
        await asyncio.sleep(0.5)

        # Assert: Events were processed in priority order (NEW behavior)
        assert len(processing_order) == 3
        assert processing_order[0] < EventPriority.NORMAL.value  # before_event (highest priority)
        assert processing_order[1] == EventPriority.NORMAL.value  # base_event (normal priority)
        assert processing_order[2] > EventPriority.NORMAL.value  # after_event (lowest priority)

        # Verify priority relationships are correct
        assert before_event.priority.value < EventPriority.NORMAL.value
        assert after_event.priority.value > EventPriority.NORMAL.value
        assert base_event.priority.value == EventPriority.NORMAL.value

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_mixed_event_types_priority_ordering(self, event_bus: AbstractEventBus, event_factory):
        """Test priority values work correctly across different event types."""
        # Arrange: Track processing order for all event types
        processing_order = []

        def universal_priority_handler(event: BaseEvent):
            processing_order.append((event.event_type.value, event.priority.value))

        # Subscribe to multiple event types
        subscriptions = []
        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            sub_id = event_bus.subscribe(event_type, universal_priority_handler)
            subscriptions.append(sub_id)

        # Create mixed events with different priorities
        events = [
            event_factory(EventType.TRADE, priority=EventPriority.LOW),
            event_factory(EventType.KLINE, priority=EventPriority.CRITICAL),
            event_factory(EventType.CONNECTION, priority=EventPriority.HIGH),
            event_factory(EventType.ERROR, priority=EventPriority.NORMAL),
        ]

        # Act: Publish all events
        for event in events:
            await event_bus.publish(event)

        # Allow processing to complete
        await asyncio.sleep(0.5)

        # Assert: Events were processed in priority order (NEW behavior)
        assert len(processing_order) == 4

        # Verify events were processed in priority order (global priority queue)
        expected_order = [
            (EventType.KLINE.value, EventPriority.CRITICAL.value),  # Highest priority
            (EventType.CONNECTION.value, EventPriority.HIGH.value),  # Second highest
            (EventType.ERROR.value, EventPriority.NORMAL.value),  # Third highest
            (EventType.TRADE.value, EventPriority.LOW.value),  # Lowest priority
        ]
        assert processing_order == expected_order

        # Verify priority values are correctly preserved
        priorities = [item[1] for item in processing_order]
        assert priorities == [
            EventPriority.CRITICAL.value,
            EventPriority.HIGH.value,
            EventPriority.NORMAL.value,
            EventPriority.LOW.value,
        ]

        # Cleanup
        for sub_id in subscriptions:
            event_bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_priority_with_concurrent_publishing(self, event_bus: AbstractEventBus, event_factory):
        """Test priority values are preserved with concurrent publishing."""
        # Arrange: Track processing order
        processing_order = []
        processing_lock = asyncio.Lock()

        async def concurrent_priority_handler(event: BaseEvent):
            async with processing_lock:
                processing_order.append(event.priority.value)

        subscription_id = event_bus.subscribe(EventType.TRADE, concurrent_priority_handler)

        # Create events with different priorities
        events = [
            event_factory(EventType.TRADE, priority=EventPriority.custom(i * 10))
            for i in range(10, 0, -1)  # Priorities 100, 90, 80, ..., 10
        ]

        # Act: Publish all events concurrently
        await asyncio.gather(*[event_bus.publish(event) for event in events])

        # Allow processing to complete
        await asyncio.sleep(0.3)

        # Assert: All events were processed
        assert len(processing_order) == 10

        # Verify all expected priority values are present
        expected_priorities = [i * 10 for i in range(10, 0, -1)]  # [100, 90, 80, ..., 10]
        assert set(processing_order) == set(expected_priorities)

        # Note: With concurrent publishing, order may vary, but all events should be processed
        # The important thing is that priority values are preserved correctly
        for priority in processing_order:
            assert priority in expected_priorities

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_priority_edge_cases(self, event_bus: AbstractEventBus, event_factory):
        """Test priority handling edge cases and extreme values."""
        # Arrange: Track processing order
        processing_order = []

        def edge_case_handler(event: BaseEvent):
            processing_order.append(event.priority.value)

        subscription_id = event_bus.subscribe(EventType.TRADE, edge_case_handler)

        # Create events with edge case priorities
        events = [
            event_factory(EventType.TRADE, priority=EventPriority.HIGHEST),
            event_factory(EventType.TRADE, priority=EventPriority.LOWEST),
            event_factory(EventType.TRADE, priority=EventPriority.custom(0)),
            event_factory(EventType.TRADE, priority=EventPriority.custom(-100)),
        ]

        # Act: Publish events
        for event in events:
            await event_bus.publish(event)

        # Allow processing to complete
        await asyncio.sleep(0.5)

        # Assert: Events were processed in priority order (NEW behavior)
        assert len(processing_order) == 4

        # Verify events were processed in priority order (lower value = higher priority)
        expected_order = [
            EventPriority.HIGHEST.value,  # highest priority (-sys.maxsize - 1)
            -100,  # custom(-100) - second highest
            0,  # custom(0) - third highest
            EventPriority.LOWEST.value,  # lowest priority (sys.maxsize)
        ]
        assert processing_order == expected_order

        # Verify extreme priorities are handled correctly
        assert EventPriority.HIGHEST.value in processing_order
        assert EventPriority.LOWEST.value in processing_order
        assert 0 in processing_order
        assert -100 in processing_order

        # Verify priority relationships
        assert EventPriority.HIGHEST.value < EventPriority.LOWEST.value  # Lower value = higher priority

        # Cleanup
        event_bus.unsubscribe(subscription_id)
