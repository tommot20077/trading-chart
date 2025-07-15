# ABOUTME: Event handler registry with priority-based ordering
# ABOUTME: Manages event handlers and sorts them by priority for execution

from __future__ import annotations
from typing import Any, Callable, Dict, List
from functools import total_ordering

from core.models.event import EventPriority, EventType


@total_ordering
class EventHandlerEntry:
    """
    Entry for an event handler with priority information.

    This class represents a single event handler registration with its
    associated priority and metadata, allowing for ordered execution.
    """

    def __init__(
        self,
        handler: Callable[..., Any],
        priority: int | EventPriority,
        event_type: EventType,
        name: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        """
        Initialize an event handler entry.

        Args:
            handler: The callable handler function
            priority: Priority value (lower = higher priority)
            event_type: Type of event this handler processes
            name: Optional name for the handler
            metadata: Optional metadata dictionary
        """
        self.handler = handler
        self.priority = priority if isinstance(priority, EventPriority) else EventPriority(priority)
        self.event_type = event_type
        self.name = name or handler.__name__
        self.metadata = metadata or {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EventHandlerEntry):
            return False
        return self.priority == other.priority and self.name == other.name

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, EventHandlerEntry):
            return NotImplemented
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"EventHandlerEntry(name='{self.name}', priority={self.priority}, event_type={self.event_type})"


class EventHandlerRegistry:
    """
    Registry for managing event handlers with priority-based ordering.

    This class provides functionality similar to Spring Security's filter
    chain, allowing registration of handlers with priorities and automatic
    sorting for execution order.
    """

    def __init__(self) -> None:
        """Initialize the event handler registry."""
        self._handlers: Dict[EventType, List[EventHandlerEntry]] = {}
        self._sorted_cache: Dict[EventType, List[EventHandlerEntry]] = {}

    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[..., Any],
        priority: int | EventPriority,
        name: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> EventHandlerEntry:
        """
        Register an event handler with the specified priority.

        Args:
            event_type: Type of event this handler processes
            handler: The callable handler function
            priority: Priority value (lower = higher priority)
            name: Optional name for the handler
            metadata: Optional metadata dictionary

        Returns:
            The created EventHandlerEntry
        """
        entry = EventHandlerEntry(handler, priority, event_type, name, metadata)

        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(entry)

        # Clear cache for this event type
        self._sorted_cache.pop(event_type, None)

        return entry

    def get_handlers(self, event_type: EventType) -> List[EventHandlerEntry]:
        """
        Get all handlers for a specific event type, sorted by priority.

        Args:
            event_type: The event type to get handlers for

        Returns:
            List of EventHandlerEntry objects sorted by priority
        """
        if event_type not in self._handlers:
            return []

        # Use cached sorted list if available
        if event_type in self._sorted_cache:
            return self._sorted_cache[event_type]

        # Sort handlers by priority (lower number = higher priority)
        sorted_handlers = sorted(self._handlers[event_type])
        self._sorted_cache[event_type] = sorted_handlers

        return sorted_handlers

    def unregister_handler(self, event_type: EventType, name: str) -> bool:
        """
        Unregister a handler by name.

        Args:
            event_type: The event type
            name: Name of the handler to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if event_type not in self._handlers:
            return False

        handlers = self._handlers[event_type]
        for i, entry in enumerate(handlers):
            if entry.name == name:
                handlers.pop(i)
                self._sorted_cache.pop(event_type, None)
                return True

        return False

    def add_before(
        self,
        event_type: EventType,
        existing_handler_name: str,
        handler: Callable[..., Any],
        name: str | None = None,
        offset: int = 10,
    ) -> EventHandlerEntry | None:
        """
        Add a handler to execute before an existing handler.

        Args:
            event_type: The event type
            existing_handler_name: Name of the existing handler
            handler: The new handler function
            name: Optional name for the new handler
            offset: Priority offset (default: 10)

        Returns:
            The created EventHandlerEntry or None if existing handler not found
        """
        existing_entry = self.find_handler(event_type, existing_handler_name)
        if existing_entry is None:
            return None

        new_priority = EventPriority.before(existing_entry.priority.value, offset)
        return self.register_handler(event_type, handler, new_priority, name)

    def add_after(
        self,
        event_type: EventType,
        existing_handler_name: str,
        handler: Callable[..., Any],
        name: str | None = None,
        offset: int = 10,
    ) -> EventHandlerEntry | None:
        """
        Add a handler to execute after an existing handler.

        Args:
            event_type: The event type
            existing_handler_name: Name of the existing handler
            handler: The new handler function
            name: Optional name for the new handler
            offset: Priority offset (default: 10)

        Returns:
            The created EventHandlerEntry or None if existing handler not found
        """
        existing_entry = self.find_handler(event_type, existing_handler_name)
        if existing_entry is None:
            return None

        new_priority = EventPriority.after(existing_entry.priority.value, offset)
        return self.register_handler(event_type, handler, new_priority, name)

    def find_handler(self, event_type: EventType, name: str) -> EventHandlerEntry | None:
        """
        Find a handler by name.

        Args:
            event_type: The event type
            name: Name of the handler to find

        Returns:
            EventHandlerEntry if found, None otherwise
        """
        if event_type not in self._handlers:
            return None

        for entry in self._handlers[event_type]:
            if entry.name == name:
                return entry

        return None

    def clear_handlers(self, event_type: EventType | None = None) -> None:
        """
        Clear handlers for a specific event type or all handlers.

        Args:
            event_type: Specific event type to clear, or None to clear all
        """
        if event_type is None:
            self._handlers.clear()
            self._sorted_cache.clear()
        else:
            self._handlers.pop(event_type, None)
            self._sorted_cache.pop(event_type, None)

    def get_all_event_types(self) -> List[EventType]:
        """
        Get all event types that have registered handlers.

        Returns:
            List of EventType objects
        """
        return list(self._handlers.keys())

    def get_handler_count(self, event_type: EventType | None = None) -> int:
        """
        Get the number of handlers for a specific event type or all handlers.

        Args:
            event_type: Specific event type, or None for total count

        Returns:
            Number of handlers
        """
        if event_type is None:
            return sum(len(handlers) for handlers in self._handlers.values())
        else:
            return len(self._handlers.get(event_type, []))
