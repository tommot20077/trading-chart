# ABOUTME: Unit tests for EventHandlerRegistry
# ABOUTME: Tests priority-based event handler registration and execution

import pytest
from unittest.mock import Mock

from core.components.event.handler_registry import EventHandlerRegistry, EventHandlerEntry
from core.models.event.event_priority import EventPriority
from core.models.event.event_type import EventType
from core.models.data.event import BaseEvent


class MockEvent(BaseEvent[dict]):
    """Mock event for testing."""

    def __init__(self, event_type: EventType = EventType.TRADE):
        super().__init__(event_type=event_type, source="test_source", data={"test": "data"})


class TestEventHandlerRegistry:
    """Test cases for EventHandlerRegistry."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test registry initialization."""
        registry = EventHandlerRegistry()
        assert registry is not None
        assert len(registry._handlers) == 0
        assert len(registry._sorted_cache) == 0

    @pytest.mark.unit
    def test_register_handler_with_function(self):
        """Test registering handler with a function."""
        registry = EventHandlerRegistry()

        @pytest.mark.unit
        def test_handler(event):
            pass

        entry = registry.register_handler(event_type=EventType.TRADE, handler=test_handler, priority=EventPriority.HIGH)

        assert isinstance(entry, EventHandlerEntry)
        assert entry.handler == test_handler
        assert entry.priority == EventPriority.HIGH
        assert entry.event_type == EventType.TRADE
        assert entry.name == "test_handler"
        assert len(registry._handlers[EventType.TRADE]) == 1

    @pytest.mark.unit
    def test_register_handler_with_mock(self):
        """Test registering handler with mock (providing name)."""
        registry = EventHandlerRegistry()
        handler = Mock()

        entry = registry.register_handler(
            event_type=EventType.TRADE, handler=handler, priority=EventPriority.HIGH, name="mock_handler"
        )

        assert isinstance(entry, EventHandlerEntry)
        assert entry.handler == handler
        assert entry.name == "mock_handler"
        assert len(registry._handlers[EventType.TRADE]) == 1

    @pytest.mark.unit
    def test_register_handler_with_integer_priority(self):
        """Test registering handler with integer priority."""
        registry = EventHandlerRegistry()

        @pytest.mark.unit
        def test_handler(event):
            pass

        entry = registry.register_handler(event_type=EventType.TRADE, handler=test_handler, priority=100)

        assert entry.priority.value == 100

    @pytest.mark.unit
    def test_get_handlers_sorted_by_priority(self):
        """Test getting handlers sorted by priority."""
        registry = EventHandlerRegistry()

        def high_handler(event):
            pass

        def low_handler(event):
            pass

        def medium_handler(event):
            pass

        # Register in reverse priority order
        registry.register_handler(EventType.TRADE, low_handler, EventPriority.LOW)
        registry.register_handler(EventType.TRADE, high_handler, EventPriority.HIGH)
        registry.register_handler(EventType.TRADE, medium_handler, EventPriority.NORMAL)

        handlers = registry.get_handlers(EventType.TRADE)
        assert len(handlers) == 3

        # Should be sorted by priority (HIGH < NORMAL < LOW)
        assert handlers[0].priority == EventPriority.HIGH
        assert handlers[1].priority == EventPriority.NORMAL
        assert handlers[2].priority == EventPriority.LOW

    @pytest.mark.unit
    def test_get_handlers_empty_event_type(self):
        """Test getting handlers for event type with no handlers."""
        registry = EventHandlerRegistry()
        handlers = registry.get_handlers(EventType.TRADE)
        assert handlers == []

    @pytest.mark.unit
    def test_unregister_handler_by_name(self):
        """Test unregistering handler by name."""
        registry = EventHandlerRegistry()

        @pytest.mark.unit
        def test_handler(event):
            pass

        registry.register_handler(EventType.TRADE, test_handler, EventPriority.HIGH)
        assert len(registry._handlers[EventType.TRADE]) == 1

        success = registry.unregister_handler(EventType.TRADE, "test_handler")
        assert success is True
        assert len(registry._handlers[EventType.TRADE]) == 0

    @pytest.mark.unit
    def test_unregister_nonexistent_handler(self):
        """Test unregistering non-existent handler."""
        registry = EventHandlerRegistry()

        success = registry.unregister_handler(EventType.TRADE, "nonexistent")
        assert success is False

    @pytest.mark.unit
    def test_unregister_handler_wrong_event_type(self):
        """Test unregistering handler with wrong event type."""
        registry = EventHandlerRegistry()

        @pytest.mark.unit
        def test_handler(event):
            pass

        registry.register_handler(EventType.TRADE, test_handler, EventPriority.HIGH)

        success = registry.unregister_handler(EventType.KLINE, "test_handler")
        assert success is False

    @pytest.mark.unit
    def test_find_handler(self):
        """Test finding handler by name."""
        registry = EventHandlerRegistry()

        @pytest.mark.unit
        def test_handler(event):
            pass

        entry = registry.register_handler(EventType.TRADE, test_handler, EventPriority.HIGH)

        found = registry.find_handler(EventType.TRADE, "test_handler")
        assert found == entry

        not_found = registry.find_handler(EventType.TRADE, "nonexistent")
        assert not_found is None

    @pytest.mark.unit
    def test_add_before(self):
        """Test adding handler before existing handler."""
        registry = EventHandlerRegistry()

        def existing_handler(event):
            pass

        def new_handler(event):
            pass

        registry.register_handler(EventType.TRADE, existing_handler, EventPriority.NORMAL)

        new_entry = registry.add_before(EventType.TRADE, "existing_handler", new_handler)
        assert new_entry is not None
        assert new_entry.priority.value == EventPriority.NORMAL.value - 10

        handlers = registry.get_handlers(EventType.TRADE)
        assert handlers[0].name == "new_handler"
        assert handlers[1].name == "existing_handler"

    @pytest.mark.unit
    def test_add_after(self):
        """Test adding handler after existing handler."""
        registry = EventHandlerRegistry()

        def existing_handler(event):
            pass

        def new_handler(event):
            pass

        registry.register_handler(EventType.TRADE, existing_handler, EventPriority.NORMAL)

        new_entry = registry.add_after(EventType.TRADE, "existing_handler", new_handler)
        assert new_entry is not None
        assert new_entry.priority.value == EventPriority.NORMAL.value + 10

        handlers = registry.get_handlers(EventType.TRADE)
        assert handlers[0].name == "existing_handler"
        assert handlers[1].name == "new_handler"

    @pytest.mark.unit
    def test_add_before_nonexistent_handler(self):
        """Test adding handler before non-existent handler."""
        registry = EventHandlerRegistry()

        def new_handler(event):
            pass

        result = registry.add_before(EventType.TRADE, "nonexistent", new_handler)
        assert result is None

    @pytest.mark.unit
    def test_clear_handlers_specific_event_type(self):
        """Test clearing handlers for specific event type."""
        registry = EventHandlerRegistry()

        def handler1(event):
            pass

        def handler2(event):
            pass

        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        registry.register_handler(EventType.KLINE, handler2, EventPriority.LOW)

        assert registry.get_handler_count() == 2

        registry.clear_handlers(EventType.TRADE)
        assert registry.get_handler_count(EventType.TRADE) == 0
        assert registry.get_handler_count(EventType.KLINE) == 1

    @pytest.mark.unit
    def test_clear_all_handlers(self):
        """Test clearing all handlers."""
        registry = EventHandlerRegistry()

        def handler1(event):
            pass

        def handler2(event):
            pass

        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        registry.register_handler(EventType.KLINE, handler2, EventPriority.LOW)

        assert registry.get_handler_count() == 2

        registry.clear_handlers()
        assert registry.get_handler_count() == 0

    @pytest.mark.unit
    def test_get_all_event_types(self):
        """Test getting all event types with handlers."""
        registry = EventHandlerRegistry()

        def handler1(event):
            pass

        def handler2(event):
            pass

        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        registry.register_handler(EventType.KLINE, handler2, EventPriority.LOW)

        event_types = registry.get_all_event_types()
        assert EventType.TRADE in event_types
        assert EventType.KLINE in event_types
        assert len(event_types) == 2

    @pytest.mark.unit
    def test_get_handler_count(self):
        """Test getting handler count."""
        registry = EventHandlerRegistry()

        def handler1(event):
            pass

        def handler2(event):
            pass

        def handler3(event):
            pass

        assert registry.get_handler_count() == 0
        assert registry.get_handler_count(EventType.TRADE) == 0

        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        registry.register_handler(EventType.TRADE, handler2, EventPriority.LOW)
        registry.register_handler(EventType.KLINE, handler3, EventPriority.NORMAL)

        assert registry.get_handler_count() == 3
        assert registry.get_handler_count(EventType.TRADE) == 2
        assert registry.get_handler_count(EventType.KLINE) == 1

    @pytest.mark.unit
    def test_handler_entry_equality(self):
        """Test EventHandlerEntry equality comparison."""

        def handler1(event):
            pass

        def handler2(event):
            pass

        entry1 = EventHandlerEntry(handler1, EventPriority.HIGH, EventType.TRADE)
        entry2 = EventHandlerEntry(handler2, EventPriority.HIGH, EventType.TRADE, name="handler1")
        entry3 = EventHandlerEntry(handler1, EventPriority.LOW, EventType.TRADE)

        assert entry1 == entry2  # Same priority and name
        assert entry1 != entry3  # Different priority
        assert entry1 != "not an entry"  # Different type

    @pytest.mark.unit
    def test_handler_entry_ordering(self):
        """Test EventHandlerEntry ordering by priority."""

        def handler(event):
            pass

        high_entry = EventHandlerEntry(handler, EventPriority.HIGH, EventType.TRADE)
        low_entry = EventHandlerEntry(handler, EventPriority.LOW, EventType.TRADE)

        assert high_entry < low_entry
        assert not (low_entry < high_entry)

    @pytest.mark.unit
    def test_handler_entry_repr(self):
        """Test EventHandlerEntry string representation."""

        @pytest.mark.unit
        def test_handler(event):
            pass

        entry = EventHandlerEntry(test_handler, EventPriority.HIGH, EventType.TRADE)
        repr_str = repr(entry)

        assert "EventHandlerEntry" in repr_str
        assert "test_handler" in repr_str
        assert str(EventPriority.HIGH) in repr_str
        assert str(EventType.TRADE) in repr_str

    @pytest.mark.unit
    def test_sorted_cache_invalidation(self):
        """Test that sorted cache is invalidated when handlers are modified."""
        registry = EventHandlerRegistry()

        def handler1(event):
            pass

        def handler2(event):
            pass

        # Register first handler and get sorted list (populates cache)
        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        handlers1 = registry.get_handlers(EventType.TRADE)
        assert len(handlers1) == 1

        # Register second handler (should invalidate cache)
        registry.register_handler(EventType.TRADE, handler2, EventPriority.LOW)
        handlers2 = registry.get_handlers(EventType.TRADE)
        assert len(handlers2) == 2

        # Unregister handler (should invalidate cache)
        registry.unregister_handler(EventType.TRADE, "handler1")
        handlers3 = registry.get_handlers(EventType.TRADE)
        assert len(handlers3) == 1
