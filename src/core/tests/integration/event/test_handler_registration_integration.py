# ABOUTME: Integration tests for event handler registration and management (E4.1-E4.3)
# ABOUTME: Tests static registration, dynamic management, and priority handling of event handlers

import pytest
import asyncio
from typing import Dict, Any

from core.components.event.handler_registry import EventHandlerRegistry
from core.interfaces.event.event_bus import AbstractEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


@pytest.mark.integration
class TestHandlerRegistrationIntegration:
    """
    Integration tests for event handler registration and management.

    Tests static registration, dynamic management, and priority handling.
    This covers tasks E4.1-E4.3 from the integration test plan.
    """

    @pytest.mark.asyncio
    async def test_static_handler_registration_with_event_bus(self, event_bus: AbstractEventBus, event_factory):
        """Test static event handler registration with event bus integration (E4.1)."""
        # Arrange: Create handler registry and tracking
        registry = EventHandlerRegistry()
        call_log = []

        def trade_handler_1(event: BaseEvent):
            call_log.append(f"handler1_{event.event_id[-8:]}")

        def trade_handler_2(event: BaseEvent):
            call_log.append(f"handler2_{event.event_id[-8:]}")

        def kline_handler(event: BaseEvent):
            call_log.append(f"kline_{event.event_id[-8:]}")

        # Act: Register handlers with different priorities
        entry1 = registry.register_handler(EventType.TRADE, trade_handler_1, EventPriority.HIGH, name="trade_handler_1")
        entry2 = registry.register_handler(
            EventType.TRADE, trade_handler_2, EventPriority.NORMAL, name="trade_handler_2"
        )
        entry3 = registry.register_handler(EventType.KLINE, kline_handler, EventPriority.CRITICAL, name="kline_handler")

        # Subscribe handlers to event bus
        for entry in registry.get_handlers(EventType.TRADE):
            event_bus.subscribe(EventType.TRADE, entry.handler)

        for entry in registry.get_handlers(EventType.KLINE):
            event_bus.subscribe(EventType.KLINE, entry.handler)

        # Create and publish events
        trade_event = event_factory(EventType.TRADE)
        kline_event = event_factory(EventType.KLINE)

        await event_bus.publish(trade_event)
        await event_bus.publish(kline_event)

        # Allow processing to complete
        await asyncio.sleep(0.1)

        # Assert: All handlers were called
        assert len(call_log) == 3
        assert f"handler1_{trade_event.event_id[-8:]}" in call_log
        assert f"handler2_{trade_event.event_id[-8:]}" in call_log
        assert f"kline_{kline_event.event_id[-8:]}" in call_log

        # Verify registry state
        assert registry.get_handler_count(EventType.TRADE) == 2
        assert registry.get_handler_count(EventType.KLINE) == 1
        assert registry.get_handler_count() == 3

        # Verify handler entries
        assert entry1.name == "trade_handler_1"
        assert entry1.priority == EventPriority.HIGH
        assert entry2.name == "trade_handler_2"
        assert entry2.priority == EventPriority.NORMAL
        assert entry3.name == "kline_handler"
        assert entry3.priority == EventPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_dynamic_handler_management(self, event_bus: AbstractEventBus, event_factory):
        """Test dynamic handler management during runtime (E4.2)."""
        # Arrange: Create registry and tracking
        registry = EventHandlerRegistry()
        call_log = []
        subscription_ids = []

        def create_handler(name: str):
            def handler(event: BaseEvent):
                call_log.append(f"{name}_{event.event_id[-8:]}")

            handler.__name__ = name
            return handler

        # Act: Start with initial handlers
        handler1 = create_handler("handler1")
        handler2 = create_handler("handler2")

        entry1 = registry.register_handler(EventType.TRADE, handler1, EventPriority.NORMAL, name="handler1")
        entry2 = registry.register_handler(EventType.TRADE, handler2, EventPriority.HIGH, name="handler2")

        # Subscribe initial handlers
        for entry in registry.get_handlers(EventType.TRADE):
            sub_id = event_bus.subscribe(EventType.TRADE, entry.handler)
            subscription_ids.append(sub_id)

        # Publish first event
        event1 = event_factory(EventType.TRADE)
        await event_bus.publish(event1)
        await asyncio.sleep(0.1)

        # Verify initial state
        assert len(call_log) == 2
        assert f"handler1_{event1.event_id[-8:]}" in call_log
        assert f"handler2_{event1.event_id[-8:]}" in call_log

        # Act: Add new handler dynamically
        handler3 = create_handler("handler3")
        entry3 = registry.register_handler(EventType.TRADE, handler3, EventPriority.CRITICAL, name="handler3")
        sub_id3 = event_bus.subscribe(EventType.TRADE, entry3.handler)
        subscription_ids.append(sub_id3)

        # Publish second event
        event2 = event_factory(EventType.TRADE)
        await event_bus.publish(event2)
        await asyncio.sleep(0.1)

        # Verify new handler was called
        assert len(call_log) == 5  # 2 from first event + 3 from second event
        assert f"handler3_{event2.event_id[-8:]}" in call_log

        # Act: Remove handler dynamically
        removed = registry.unregister_handler(EventType.TRADE, "handler2")
        assert removed is True

        # Unsubscribe from event bus (simulate dynamic removal)
        # Note: In real implementation, we'd need to track subscription IDs
        # For this test, we'll verify registry state

        # Verify registry state after removal
        assert registry.get_handler_count(EventType.TRADE) == 2
        assert registry.find_handler(EventType.TRADE, "handler2") is None
        assert registry.find_handler(EventType.TRADE, "handler1") is not None
        assert registry.find_handler(EventType.TRADE, "handler3") is not None

        # Cleanup
        for sub_id in subscription_ids:
            event_bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_handler_priority_management(self, event_bus: AbstractEventBus, event_factory):
        """Test handler priority management and ordering (E4.3)."""
        # Arrange: Create registry and tracking
        registry = EventHandlerRegistry()
        execution_order = []

        def create_priority_handler(name: str):
            def handler(event: BaseEvent):
                execution_order.append(name)

            handler.__name__ = name
            return handler

        # Act: Register handlers with different priorities
        low_handler = create_priority_handler("low_priority")
        normal_handler = create_priority_handler("normal_priority")
        high_handler = create_priority_handler("high_priority")
        critical_handler = create_priority_handler("critical_priority")

        registry.register_handler(EventType.TRADE, low_handler, EventPriority.LOW, name="low_priority")
        registry.register_handler(EventType.TRADE, normal_handler, EventPriority.NORMAL, name="normal_priority")
        registry.register_handler(EventType.TRADE, high_handler, EventPriority.HIGH, name="high_priority")
        registry.register_handler(EventType.TRADE, critical_handler, EventPriority.CRITICAL, name="critical_priority")

        # Get handlers in priority order
        handlers = registry.get_handlers(EventType.TRADE)

        # Verify handlers are sorted by priority (lower value = higher priority)
        priorities = [handler.priority.value for handler in handlers]
        assert priorities == sorted(priorities), f"Handlers not sorted by priority: {priorities}"

        # Subscribe handlers to event bus in priority order
        for entry in handlers:
            event_bus.subscribe(EventType.TRADE, entry.handler)

        # Publish event
        event = event_factory(EventType.TRADE)
        await event_bus.publish(event)
        await asyncio.sleep(0.1)

        # Assert: All handlers were called
        assert len(execution_order) == 4

        # Note: Current EventBus processes handlers in subscription order, not priority order
        # This test verifies that the registry correctly sorts handlers by priority
        expected_priority_order = ["critical_priority", "high_priority", "normal_priority", "low_priority"]
        handler_names = [entry.name for entry in handlers]
        assert handler_names == expected_priority_order

    @pytest.mark.asyncio
    async def test_relative_priority_handlers(self, event_bus: AbstractEventBus, event_factory):
        """Test handlers with relative priorities (before/after)."""
        # Arrange: Create registry
        registry = EventHandlerRegistry()
        execution_order = []

        def create_handler(name: str):
            def handler(event: BaseEvent):
                execution_order.append(name)

            handler.__name__ = name
            return handler

        # Act: Register base handler
        base_handler = create_handler("base_handler")
        registry.register_handler(EventType.TRADE, base_handler, EventPriority.NORMAL, name="base_handler")

        # Register handler that should run before base
        before_handler = create_handler("before_handler")
        before_entry = registry.add_before(EventType.TRADE, "base_handler", before_handler, name="before_handler")

        # Register handler that should run after base
        after_handler = create_handler("after_handler")
        after_entry = registry.add_after(EventType.TRADE, "base_handler", after_handler, name="after_handler")

        # Verify relative priorities
        assert before_entry is not None
        assert after_entry is not None
        assert before_entry.priority.value < EventPriority.NORMAL.value
        assert after_entry.priority.value > EventPriority.NORMAL.value

        # Get handlers in priority order
        handlers = registry.get_handlers(EventType.TRADE)
        handler_names = [entry.name for entry in handlers]

        # Verify order: before -> base -> after
        assert handler_names == ["before_handler", "base_handler", "after_handler"]

    @pytest.mark.asyncio
    async def test_handler_metadata_and_filtering(self, event_bus: AbstractEventBus, event_factory):
        """Test handler metadata and filtering capabilities."""
        # Arrange: Create registry
        registry = EventHandlerRegistry()
        call_log = []

        def create_handler_with_metadata(name: str, metadata: Dict[str, Any]):
            def handler(event: BaseEvent):
                call_log.append(f"{name}_{event.event_id[-8:]}")

            handler.__name__ = name
            return handler

        # Act: Register handlers with metadata
        handler1 = create_handler_with_metadata("handler1", {"category": "trading", "version": "1.0"})
        handler2 = create_handler_with_metadata("handler2", {"category": "analytics", "version": "2.0"})
        handler3 = create_handler_with_metadata("handler3", {"category": "trading", "version": "1.5"})

        entry1 = registry.register_handler(
            EventType.TRADE,
            handler1,
            EventPriority.NORMAL,
            name="handler1",
            metadata={"category": "trading", "version": "1.0"},
        )
        entry2 = registry.register_handler(
            EventType.TRADE,
            handler2,
            EventPriority.NORMAL,
            name="handler2",
            metadata={"category": "analytics", "version": "2.0"},
        )
        entry3 = registry.register_handler(
            EventType.TRADE,
            handler3,
            EventPriority.NORMAL,
            name="handler3",
            metadata={"category": "trading", "version": "1.5"},
        )

        # Verify metadata is stored
        assert entry1.metadata["category"] == "trading"
        assert entry1.metadata["version"] == "1.0"
        assert entry2.metadata["category"] == "analytics"
        assert entry3.metadata["category"] == "trading"

        # Test filtering by metadata (manual filtering for this test)
        trading_handlers = [
            entry for entry in registry.get_handlers(EventType.TRADE) if entry.metadata.get("category") == "trading"
        ]
        assert len(trading_handlers) == 2
        assert all(entry.metadata["category"] == "trading" for entry in trading_handlers)

    @pytest.mark.asyncio
    async def test_handler_registry_state_management(self, event_bus: AbstractEventBus):
        """Test comprehensive handler registry state management."""
        # Arrange: Create registry
        registry = EventHandlerRegistry()

        def dummy_handler(event: BaseEvent):
            pass

        # Act: Test various registry operations

        # Register handlers for different event types
        registry.register_handler(EventType.TRADE, dummy_handler, EventPriority.NORMAL, name="trade1")
        registry.register_handler(EventType.TRADE, dummy_handler, EventPriority.HIGH, name="trade2")
        registry.register_handler(EventType.KLINE, dummy_handler, EventPriority.NORMAL, name="kline1")
        registry.register_handler(EventType.CONNECTION, dummy_handler, EventPriority.CRITICAL, name="conn1")

        # Test state queries
        assert registry.get_handler_count() == 4
        assert registry.get_handler_count(EventType.TRADE) == 2
        assert registry.get_handler_count(EventType.KLINE) == 1
        assert registry.get_handler_count(EventType.CONNECTION) == 1
        assert registry.get_handler_count(EventType.ERROR) == 0

        # Test event type listing
        event_types = registry.get_all_event_types()
        assert EventType.TRADE in event_types
        assert EventType.KLINE in event_types
        assert EventType.CONNECTION in event_types
        assert EventType.ERROR not in event_types

        # Test handler finding
        trade1_entry = registry.find_handler(EventType.TRADE, "trade1")
        assert trade1_entry is not None
        assert trade1_entry.name == "trade1"
        assert trade1_entry.priority == EventPriority.NORMAL

        nonexistent_entry = registry.find_handler(EventType.TRADE, "nonexistent")
        assert nonexistent_entry is None

        # Test selective clearing
        registry.clear_handlers(EventType.TRADE)
        assert registry.get_handler_count(EventType.TRADE) == 0
        assert registry.get_handler_count() == 2  # kline1 + conn1

        # Test complete clearing
        registry.clear_handlers()
        assert registry.get_handler_count() == 0
        assert len(registry.get_all_event_types()) == 0

    @pytest.mark.asyncio
    async def test_handler_error_scenarios(self, event_bus: AbstractEventBus):
        """Test error handling scenarios in handler registration."""
        # Arrange: Create registry
        registry = EventHandlerRegistry()

        def dummy_handler(event: BaseEvent):
            pass

        # Test duplicate handler registration
        entry1 = registry.register_handler(EventType.TRADE, dummy_handler, EventPriority.NORMAL, name="test_handler")
        assert entry1 is not None

        # Registering with same name should replace (or handle appropriately)
        entry2 = registry.register_handler(EventType.TRADE, dummy_handler, EventPriority.HIGH, name="test_handler")
        assert entry2 is not None

        # Should only have one handler with that name
        found_entry = registry.find_handler(EventType.TRADE, "test_handler")
        assert found_entry is not None

        # Test unregistering non-existent handler
        removed = registry.unregister_handler(EventType.TRADE, "nonexistent_handler")
        assert removed is False

        # Test relative priority with non-existent base handler
        before_entry = registry.add_before(
            EventType.TRADE, "nonexistent_base", dummy_handler, name="before_nonexistent"
        )
        assert before_entry is None  # Should fail gracefully

        after_entry = registry.add_after(EventType.TRADE, "nonexistent_base", dummy_handler, name="after_nonexistent")
        assert after_entry is None  # Should fail gracefully
