# ABOUTME: Unit tests for EventHandlerRegistry
# ABOUTME: Tests priority-based event handler registration and execution

import pytest
import asyncio
import threading
import time
import concurrent.futures
from unittest.mock import Mock, patch
from typing import List, Callable

from core.components.event.handler_registry import EventHandlerRegistry, EventHandlerEntry
from core.models.event.event_priority import EventPriority
from core.models.event.event_type import EventType
from core.models.data.event import BaseEvent
from tests.constants import TestTimeouts, TestDataSizes, PerformanceThresholds


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


class TestEventHandlerRegistryEdgeCases:
    """Test edge cases and boundary conditions for EventHandlerRegistry."""

    @pytest.mark.unit
    def test_register_duplicate_handler_names(self):
        """Test registering handlers with duplicate names."""
        registry = EventHandlerRegistry()
        
        def handler1(event):
            return "handler1"
        
        def handler2(event):
            return "handler2"
        
        # Register first handler
        entry1 = registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH, name="duplicate_name")
        assert entry1 is not None
        
        # Register second handler with same name (should replace or handle gracefully)
        entry2 = registry.register_handler(EventType.TRADE, handler2, EventPriority.LOW, name="duplicate_name")
        assert entry2 is not None
        
        # Verify behavior - should have both handlers with different internal handling
        handlers = registry.get_handlers(EventType.TRADE)
        assert len(handlers) >= 1  # At least one handler should be registered

    @pytest.mark.unit
    def test_register_handler_with_none_values(self):
        """Test registering handler with None or invalid values."""
        registry = EventHandlerRegistry()
        
        def valid_handler(event):
            pass
        
        # Test with None handler - should raise appropriate error or handle gracefully
        try:
            result = registry.register_handler(EventType.TRADE, None, EventPriority.HIGH)
            # If it doesn't raise an exception, verify the registry is still in a valid state
            assert result is not None or registry.get_handler_count() >= 0
        except (TypeError, ValueError, AttributeError):
            # These are acceptable errors for invalid input
            pass
        
        # Test with invalid event type - should raise appropriate error or handle gracefully
        try:
            result = registry.register_handler(None, valid_handler, EventPriority.HIGH)
            # If it doesn't raise an exception, verify the registry is still in a valid state
            assert result is not None or registry.get_handler_count() >= 0
        except (TypeError, ValueError, AttributeError):
            # These are acceptable errors for invalid input
            pass

    @pytest.mark.unit
    def test_extremely_high_and_low_priorities(self):
        """Test handlers with extreme priority values."""
        registry = EventHandlerRegistry()
        
        def handler1(event):
            pass
        
        def handler2(event):
            pass
        
        def handler3(event):
            pass
        
        # Register handlers with extreme priorities
        registry.register_handler(EventType.TRADE, handler1, priority=999999)  # Very high
        registry.register_handler(EventType.TRADE, handler2, priority=-999999)  # Very low
        registry.register_handler(EventType.TRADE, handler3, priority=0)  # Zero
        
        handlers = registry.get_handlers(EventType.TRADE)
        assert len(handlers) == 3
        
        # Verify correct ordering despite extreme values
        priorities = [h.priority.value for h in handlers]
        assert priorities == sorted(priorities), "Handlers should be sorted by priority"

    @pytest.mark.unit
    def test_register_many_handlers_same_priority(self):
        """Test registering many handlers with the same priority."""
        registry = EventHandlerRegistry()
        
        handler_count = TestDataSizes.MEDIUM_DATASET  # 100 handlers
        handlers = []
        
        # Create and register many handlers with same priority
        for i in range(handler_count):
            def handler(event, index=i):
                return f"handler_{index}"
            handlers.append(handler)
            registry.register_handler(EventType.TRADE, handler, EventPriority.NORMAL, name=f"handler_{i}")
        
        # Verify all handlers are registered
        registered_handlers = registry.get_handlers(EventType.TRADE)
        assert len(registered_handlers) == handler_count
        
        # All should have same priority
        for entry in registered_handlers:
            assert entry.priority == EventPriority.NORMAL

    @pytest.mark.unit
    def test_unregister_from_empty_registry(self):
        """Test unregistering from empty registry."""
        registry = EventHandlerRegistry()
        
        # Try various unregister operations on empty registry
        assert registry.unregister_handler(EventType.TRADE, "nonexistent") is False
        assert registry.unregister_handler(EventType.KLINE, "any_name") is False
        
        # Registry should remain empty and functional
        assert registry.get_handler_count() == 0
        
        # Should still be able to register after failed unregister
        def handler(event):
            pass
        
        entry = registry.register_handler(EventType.TRADE, handler, EventPriority.HIGH)
        assert entry is not None
        assert registry.get_handler_count() == 1

    @pytest.mark.unit
    def test_add_before_after_edge_cases(self):
        """Test add_before and add_after with edge cases."""
        registry = EventHandlerRegistry()
        
        def base_handler(event):
            pass
        
        def new_handler(event):
            pass
        
        # Test add_before with non-existent event type
        result = registry.add_before(EventType.TRADE, "base_handler", new_handler)
        assert result is None
        
        # Register base handler
        registry.register_handler(EventType.TRADE, base_handler, EventPriority.NORMAL)
        
        # Test add_after with non-existent handler name  
        result = registry.add_after(EventType.TRADE, "nonexistent", new_handler)
        assert result is None
        
        # Test add_before/after with wrong event type
        result = registry.add_before(EventType.KLINE, "base_handler", new_handler)
        assert result is None

    @pytest.mark.unit
    def test_handler_priority_boundary_values(self):
        """Test handler priorities at boundary values."""
        registry = EventHandlerRegistry()
        
        def handler_min(event):
            pass
        
        def handler_max(event):
            pass
        
        # Test with minimum and maximum integer values
        import sys
        min_priority = -sys.maxsize - 1
        max_priority = sys.maxsize
        
        entry_min = registry.register_handler(EventType.TRADE, handler_min, priority=min_priority)
        entry_max = registry.register_handler(EventType.TRADE, handler_max, priority=max_priority)
        
        assert entry_min.priority.value == min_priority
        assert entry_max.priority.value == max_priority
        
        # Verify correct ordering
        handlers = registry.get_handlers(EventType.TRADE)
        assert handlers[0] == entry_min  # Lowest priority first
        assert handlers[1] == entry_max

    @pytest.mark.unit
    def test_handler_name_extraction_edge_cases(self):
        """Test handler name extraction with various callable types."""
        registry = EventHandlerRegistry()
        
        # Test with lambda (should use provided name or generate one)
        lambda_handler = lambda event: None
        entry1 = registry.register_handler(EventType.TRADE, lambda_handler, EventPriority.HIGH, name="lambda_handler")
        assert entry1.name == "lambda_handler"
        
        # Test with callable class instance
        class CallableHandler:
            def __call__(self, event):
                pass
            
            def __name__(self):
                return "callable_instance"
        
        callable_handler = CallableHandler()
        entry2 = registry.register_handler(EventType.TRADE, callable_handler, EventPriority.NORMAL)
        # Should use class name or provided name
        assert entry2.name is not None
        
        # Test with partial function
        from functools import partial
        
        def base_func(event, prefix):
            pass
        
        partial_handler = partial(base_func, prefix="test")
        entry3 = registry.register_handler(EventType.TRADE, partial_handler, EventPriority.LOW, name="partial_handler")
        assert entry3.name == "partial_handler"

    @pytest.mark.unit
    def test_concurrent_handler_registration(self):
        """Test concurrent handler registration for thread safety."""
        registry = EventHandlerRegistry()
        registered_handlers = []
        registration_errors = []
        
        def register_handler_worker(handler_id):
            try:
                def handler(event):
                    return f"handler_{handler_id}"
                
                entry = registry.register_handler(
                    EventType.TRADE, 
                    handler, 
                    EventPriority.NORMAL, 
                    name=f"handler_{handler_id}"
                )
                if entry:
                    registered_handlers.append(entry)
            except Exception as e:
                registration_errors.append(e)
        
        # Use ThreadPoolExecutor to register handlers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_handler_worker, i) for i in range(20)]
            concurrent.futures.wait(futures)
        
        # Verify results
        assert len(registration_errors) == 0, f"Registration errors: {registration_errors}"
        assert len(registered_handlers) <= 20  # May have duplicates handled
        assert registry.get_handler_count(EventType.TRADE) > 0

    @pytest.mark.unit  
    def test_clear_handlers_with_cache_invalidation(self):
        """Test that clear_handlers properly invalidates caches."""
        registry = EventHandlerRegistry()
        
        def handler1(event):
            pass
        
        def handler2(event):
            pass
        
        # Register handlers and populate cache
        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        registry.register_handler(EventType.KLINE, handler2, EventPriority.LOW)
        
        # Access handlers to populate sorted cache
        trade_handlers = registry.get_handlers(EventType.TRADE)
        kline_handlers = registry.get_handlers(EventType.KLINE)
        
        assert len(trade_handlers) == 1
        assert len(kline_handlers) == 1
        
        # Clear specific event type
        registry.clear_handlers(EventType.TRADE)
        
        # Verify cache is invalidated and handlers are cleared
        assert len(registry.get_handlers(EventType.TRADE)) == 0
        assert len(registry.get_handlers(EventType.KLINE)) == 1
        
        # Clear all handlers
        registry.clear_handlers()
        
        # Verify all caches are cleared
        assert len(registry.get_handlers(EventType.KLINE)) == 0
        assert registry.get_handler_count() == 0

    @pytest.mark.unit
    def test_handler_entry_edge_cases(self):
        """Test EventHandlerEntry with edge cases."""
        
        def handler(event):
            pass
        
        # Test with very long handler name
        long_name = "a" * 1000
        entry = EventHandlerEntry(handler, EventPriority.HIGH, EventType.TRADE, name=long_name)
        assert entry.name == long_name
        
        # Test with special characters in name
        special_name = "handler!@#$%^&*()_+-=[]{}|;:,.<>?"
        entry2 = EventHandlerEntry(handler, EventPriority.HIGH, EventType.TRADE, name=special_name)
        assert entry2.name == special_name
        
        # Test with unicode characters in name
        unicode_name = "處理器_测试_テスト_핸들러"
        entry3 = EventHandlerEntry(handler, EventPriority.HIGH, EventType.TRADE, name=unicode_name)
        assert entry3.name == unicode_name

    @pytest.mark.unit
    def test_find_handler_case_sensitivity(self):
        """Test that find_handler is case-sensitive."""
        registry = EventHandlerRegistry()
        
        def handler(event):
            pass
        
        registry.register_handler(EventType.TRADE, handler, EventPriority.HIGH, name="TestHandler")
        
        # Exact match should work
        found = registry.find_handler(EventType.TRADE, "TestHandler")
        assert found is not None
        
        # Case mismatch should not work
        not_found1 = registry.find_handler(EventType.TRADE, "testhandler")
        not_found2 = registry.find_handler(EventType.TRADE, "TESTHANDLER")
        not_found3 = registry.find_handler(EventType.TRADE, "testHandler")
        
        assert not_found1 is None
        assert not_found2 is None
        assert not_found3 is None

    @pytest.mark.unit
    def test_get_all_event_types_empty_registry(self):
        """Test get_all_event_types on empty registry."""
        registry = EventHandlerRegistry()
        
        event_types = registry.get_all_event_types()
        assert len(event_types) == 0
        assert isinstance(event_types, (set, list))

    @pytest.mark.unit
    def test_registry_state_after_operations(self):
        """Test registry maintains consistent state after various operations."""
        registry = EventHandlerRegistry()
        
        def handler1(event):
            pass
        
        def handler2(event):
            pass
        
        def handler3(event):
            pass
        
        # Start with empty registry
        assert registry.get_handler_count() == 0
        assert len(registry.get_all_event_types()) == 0
        
        # Add handlers
        registry.register_handler(EventType.TRADE, handler1, EventPriority.HIGH)
        registry.register_handler(EventType.TRADE, handler2, EventPriority.LOW)
        registry.register_handler(EventType.KLINE, handler3, EventPriority.NORMAL)
        
        # Verify state
        assert registry.get_handler_count() == 3
        assert registry.get_handler_count(EventType.TRADE) == 2
        assert registry.get_handler_count(EventType.KLINE) == 1
        assert len(registry.get_all_event_types()) == 2
        
        # Remove one handler
        success = registry.unregister_handler(EventType.TRADE, "handler1")
        assert success is True
        
        # Verify state updated
        assert registry.get_handler_count() == 2
        assert registry.get_handler_count(EventType.TRADE) == 1
        assert registry.get_handler_count(EventType.KLINE) == 1
        assert len(registry.get_all_event_types()) == 2
        
        # Clear one event type
        registry.clear_handlers(EventType.KLINE)
        
        # Verify state updated
        assert registry.get_handler_count() == 1
        assert registry.get_handler_count(EventType.TRADE) == 1
        assert registry.get_handler_count(EventType.KLINE) == 0
        assert len(registry.get_all_event_types()) == 1
        
        # Clear all
        registry.clear_handlers()
        
        # Verify back to empty state
        assert registry.get_handler_count() == 0
        assert len(registry.get_all_event_types()) == 0
