# ABOUTME: NoOp implementation of AbstractEventStorage that performs no actual operations
# ABOUTME: Provides minimal functionality for testing and performance benchmarking scenarios

import uuid
from typing import List, AsyncIterator

from core.interfaces.event.event_storage import AbstractEventStorage
from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.models.data.event import BaseEvent
from core.models.event.event_query import EventQuery
from core.models.event.event_storage_stats import EventStorageStats


class NoOpEventStorage(AbstractEventStorage):
    """
    No-operation implementation of AbstractEventStorage.

    This implementation provides minimal functionality without actually storing or
    processing events. It's useful for testing, performance benchmarking, and
    scenarios where event storage is not required.

    All methods return success indicators but perform no actual operations.
    """

    def __init__(self, serializer: AbstractEventSerializer):
        """
        Initialize the NoOpEventStorage.

        Args:
            serializer: Event serializer (not used in NoOp implementation)
        """
        super().__init__(serializer)
        self._closed = False
        self._event_count = 0  # Track number of "stored" events for stats

    async def store_event(self, event: BaseEvent) -> str:
        """
        Store a single event - returns a fake storage ID without actual storage.

        Args:
            event: The event to "store"

        Returns:
            A fake storage ID
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        self._event_count += 1
        return f"noop-{uuid.uuid4()}"

    async def store_events(self, events: List[BaseEvent]) -> List[str]:
        """
        Store multiple events - returns fake storage IDs without actual storage.

        Args:
            events: List of events to "store"

        Returns:
            List of fake storage IDs
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        self._event_count += len(events)
        return [f"noop-{uuid.uuid4()}" for _ in events]

    async def retrieve_event(self, storage_id: str) -> BaseEvent | None:
        """
        Retrieve a single event - always returns None (no events stored).

        Args:
            storage_id: The storage ID to retrieve

        Returns:
            None (no events are actually stored)
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        return None

    async def query_events(self, query: EventQuery) -> List[BaseEvent]:
        """
        Query events - always returns empty list (no events stored).

        Args:
            query: The query criteria

        Returns:
            Empty list (no events are actually stored)
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        return []

    async def stream_events(self, query: EventQuery) -> AsyncIterator[BaseEvent]:
        """
        Stream events - yields nothing (no events stored).

        Args:
            query: The query criteria

        Yields:
            Nothing (no events are actually stored)
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        # Empty async generator - no events are stored in NoOp implementation
        if False:  # This condition is never true, but makes this an async generator
            yield

    async def delete_event(self, storage_id: str) -> bool:
        """
        Delete a single event - always returns True (pretends success).

        Args:
            storage_id: The storage ID to delete

        Returns:
            True (pretends successful deletion)
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        return True

    async def delete_events(self, query: EventQuery) -> int:
        """
        Delete multiple events - always returns 0 (no events to delete).

        Args:
            query: The query criteria for deletion

        Returns:
            0 (no events are actually stored to delete)
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        return 0

    async def get_stats(self) -> EventStorageStats:
        """
        Get storage statistics - returns fake stats.

        Returns:
            Fake statistics showing the number of events that were "stored"
        """
        if self._closed:
            raise RuntimeError("Event storage is closed")

        return EventStorageStats(
            total_events=self._event_count,
            events_by_type={},
            storage_size_bytes=0,
            oldest_event_time=None,
            newest_event_time=None,
            avg_event_size_bytes=0.0,
        )

    async def health_check(self) -> bool:
        """
        Perform a health check - always returns True (always healthy).

        Returns:
            True (NoOp implementation is always healthy)
        """
        return not self._closed

    async def close(self) -> None:
        """
        Close the event storage - sets closed flag.
        """
        self._closed = True
