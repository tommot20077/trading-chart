# ABOUTME: In-memory implementation of AbstractEventStorage using MetadataRepository as storage backend
# ABOUTME: Provides event storage with indexing, querying, and streaming capabilities

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, AsyncIterator
from collections import defaultdict

from core.interfaces.event.event_storage import AbstractEventStorage
from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.interfaces.storage.metadata_repository import AbstractMetadataRepository
from core.models.data.event import BaseEvent
from core.models.event.event_query import EventQuery
from core.models.event.event_storage_stats import EventStorageStats
from core.exceptions.base import StorageError


class InMemoryEventStorage(AbstractEventStorage):
    """
    In-memory implementation of AbstractEventStorage.

    This implementation uses AbstractMetadataRepository as the underlying storage
    mechanism and provides event-specific indexing and querying capabilities.

    Architecture:
    - Events are stored as serialized data in MetadataRepository
    - Multiple indexes are maintained for efficient querying
    - Supports complex queries, streaming, and statistics
    """

    def __init__(self, serializer: AbstractEventSerializer, metadata_repository: AbstractMetadataRepository):
        """
        Initialize the InMemoryEventStorage.

        Args:
            serializer: Event serializer for converting events to/from bytes
            metadata_repository: Underlying storage repository
        """
        super().__init__(serializer)
        self._metadata_repo = metadata_repository
        self._closed = False
        self._index_lock = asyncio.Lock()  # For transactional index updates
        self._stats_lock = asyncio.Lock()  # For statistics calculations
        self._cleanup_lock = asyncio.Lock()  # For cleanup operations
        self._concurrent_operations = 0  # Track concurrent operations
        self._operation_lock = asyncio.Lock()  # For tracking concurrent operations

    async def _track_operation(self, operation_func, *args, **kwargs):
        """Track concurrent operations to prevent resource contention."""
        async with self._operation_lock:
            self._concurrent_operations += 1

        try:
            return await operation_func(*args, **kwargs)
        finally:
            async with self._operation_lock:
                self._concurrent_operations -= 1

    async def store_event(self, event: BaseEvent) -> str:
        """Store a single event and return its storage ID."""
        result = await self._track_operation(self._store_event_impl, event)
        return str(result)

    async def _store_event_impl(self, event: BaseEvent) -> str:
        """Internal implementation of store_event."""
        if self._closed:
            raise StorageError("Event storage is closed")

        try:
            # Generate unique storage ID
            storage_id = str(uuid.uuid4())

            # Serialize the event
            serialized_data = self.serializer.serialize(event)

            # Store event data
            event_key = f"event:{storage_id}"
            event_data = {
                "data": serialized_data.decode("utf-8"),
                "event_type": event.event_type.value,
                "symbol": event.symbol,
                "source": event.source,
                "timestamp": event.timestamp.isoformat(),
                "priority": event.priority.value,
                "correlation_id": event.correlation_id,
                "metadata": event.metadata,
            }

            await self._metadata_repo.set(event_key, event_data)

            # Update indexes transactionally
            await self._transactional_update_indexes(storage_id, event)

            return storage_id

        except Exception as e:
            if "serialization" in str(e).lower():
                raise StorageError(f"Event serialization failed: {e}")
            else:
                raise StorageError(f"Event storage failed: {e}")

    async def store_events(self, events: List[BaseEvent]) -> List[str]:
        """Store multiple events and return their storage IDs."""
        result = await self._track_operation(self._store_events_impl, events)
        return [str(r) for r in result]

    async def _store_events_impl(self, events: List[BaseEvent]) -> List[str]:
        """Internal implementation of store_events."""
        if self._closed:
            raise StorageError("Event storage is closed")

        if not events:
            return []

        try:
            # Generate all storage IDs at once
            storage_ids = [str(uuid.uuid4()) for _ in events]

            # Prepare all event data for batch storage
            batch_data = {}
            batch_index_updates = []

            for event, storage_id in zip(events, storage_ids):
                # Serialize the event
                serialized_data = self.serializer.serialize(event)

                # Prepare event data
                event_key = f"event:{storage_id}"
                event_data = {
                    "data": serialized_data.decode("utf-8"),
                    "event_type": event.event_type.value,
                    "symbol": event.symbol,
                    "source": event.source,
                    "timestamp": event.timestamp.isoformat(),
                    "priority": event.priority.value,
                    "correlation_id": event.correlation_id,
                    "metadata": event.metadata,
                }

                batch_data[event_key] = event_data
                batch_index_updates.append((storage_id, event))

            # Store all events in batch
            for event_key, event_data in batch_data.items():
                await self._metadata_repo.set(event_key, event_data)

            # Update all indexes in batch transactionally
            await self._transactional_batch_update_indexes(batch_index_updates)

            return [str(sid) for sid in storage_ids]

        except Exception as e:
            if "serialization" in str(e).lower():
                raise StorageError(f"Event serialization failed: {e}")
            else:
                raise StorageError(f"Batch event storage failed: {e}")

    async def retrieve_event(self, storage_id: str) -> BaseEvent | None:
        """Retrieve a single event by its storage ID."""
        if self._closed:
            raise StorageError("Event storage is closed")

        try:
            event_key = f"event:{storage_id}"
            event_data = await self._metadata_repo.get(event_key)

            if event_data is None:
                return None

            # Validate event data structure
            if not isinstance(event_data, dict):
                raise StorageError(f"Corrupted event data: expected dict, got {type(event_data)}")

            # Extract and validate serialized data
            if "data" not in event_data:
                raise StorageError("Corrupted event data: missing 'data' field")

            data = event_data["data"]
            if not isinstance(data, (str, bytes)):
                raise StorageError(f"Corrupted event data: invalid data type {type(data)}")

            # Convert to bytes for deserialization
            if isinstance(data, str):
                try:
                    serialized_data = data.encode("utf-8")
                except UnicodeEncodeError as e:
                    raise StorageError(f"Corrupted event data: invalid UTF-8 encoding: {e}")
            else:
                serialized_data = data

            # Validate serialized data is not empty
            if not serialized_data:
                raise StorageError("Corrupted event data: empty serialized data")

            # Attempt deserialization with error recovery
            try:
                event = self.serializer.deserialize(serialized_data)
            except Exception as deserialize_error:
                # Log the corrupted data for debugging (first 100 chars)
                corrupted_preview = str(serialized_data)[:100] if serialized_data else "empty"
                raise StorageError(
                    f"Event deserialization failed for storage_id '{storage_id}': {deserialize_error}. "
                    f"Corrupted data preview: {corrupted_preview}"
                )

            # Validate the deserialized event
            if event is None:
                raise StorageError(f"Deserialization returned None for storage_id '{storage_id}'")

            # Basic event validation
            if not hasattr(event, "event_id") or not hasattr(event, "event_type"):
                raise StorageError(f"Deserialized event missing required attributes for storage_id '{storage_id}'")

            return event

        except StorageError:
            # Re-raise storage errors as-is
            raise
        except Exception as e:
            raise StorageError(f"Unexpected error during event retrieval for storage_id '{storage_id}': {e}")

    async def query_events(self, query: EventQuery) -> List[BaseEvent]:
        """Query events based on the provided criteria."""
        if self._closed:
            raise StorageError("Event storage is closed")

        try:
            # Get candidate event IDs based on query filters
            candidate_ids = await self._get_candidate_event_ids(query)

            # Retrieve and filter events
            events = []
            for storage_id in candidate_ids:
                event = await self.retrieve_event(storage_id)
                if event and self._matches_query(event, query):
                    events.append(event)

            # Apply ordering
            events = self._apply_ordering(events, query)

            # Apply pagination
            events = self._apply_pagination(events, query)

            return events

        except Exception as e:
            raise StorageError(f"Event query failed: {e}")

    async def stream_events(self, query: EventQuery) -> AsyncIterator[BaseEvent]:
        """Stream events based on the provided criteria."""
        if self._closed:
            raise StorageError("Event storage is closed")

        try:
            # Get candidate event IDs
            candidate_ids = await self._get_candidate_event_ids(query)

            # Stream events one by one (async generator)
            for storage_id in candidate_ids:
                event = await self.retrieve_event(storage_id)
                if event and self._matches_query(event, query):
                    yield event

        except Exception as e:
            raise StorageError(f"Event streaming failed: {e}")

    async def delete_event(self, storage_id: str) -> bool:
        """Delete a single event by its storage ID."""
        if self._closed:
            raise StorageError("Event storage is closed")

        try:
            # Get event data before deletion for index cleanup
            event = await self.retrieve_event(storage_id)
            if event is None:
                return False

            # Delete event data
            event_key = f"event:{storage_id}"
            deleted = await self._metadata_repo.delete(event_key)

            if deleted:
                # Clean up indexes
                await self._remove_from_indexes(storage_id, event)

            return deleted

        except Exception as e:
            raise StorageError(f"Event deletion failed: {e}")

    async def delete_events(self, query: EventQuery) -> int:
        """Delete multiple events based on query criteria."""
        if self._closed:
            raise StorageError("Event storage is closed")

        try:
            # Get candidate event IDs directly from indexes
            candidate_ids = await self._get_candidate_event_ids(query)

            deleted_count = 0
            for storage_id in candidate_ids:
                # Check if event matches query before deleting
                event = await self.retrieve_event(storage_id)
                if event and self._matches_query(event, query):
                    if await self.delete_event(storage_id):
                        deleted_count += 1

            return deleted_count

        except Exception as e:
            raise StorageError(f"Batch event deletion failed: {e}")

    async def get_events(self, query: EventQuery) -> List[BaseEvent]:
        """Get events based on query criteria - alias for query_events."""
        return await self.query_events(query)

    async def get_statistics(self) -> EventStorageStats:
        """Get storage statistics - alias for get_stats."""
        return await self.get_stats()

    async def get_stats(self) -> EventStorageStats:
        """Get storage statistics."""
        result = await self._track_operation(self._get_stats_impl)
        return EventStorageStats(**result.model_dump()) if hasattr(result, "model_dump") else result

    async def _get_stats_impl(self) -> EventStorageStats:
        """Internal implementation of get_stats with concurrency control."""
        if self._closed:
            raise StorageError("Event storage is closed")

        async with self._stats_lock:  # Ensure only one stats calculation at a time
            try:
                # Get all event keys
                event_keys = await self._metadata_repo.list_keys("event:")
                total_events = len(event_keys)

                # Count events by type
                events_by_type: dict[str, int] = defaultdict(int)
                oldest_time = None
                newest_time = None
                total_size = 0

                for event_key in event_keys:
                    event_data = await self._metadata_repo.get(event_key)
                    if event_data:
                        event_type = event_data.get("event_type", "unknown")
                        events_by_type[event_type] += 1

                        # Track timestamps
                        timestamp_str = event_data.get("timestamp")
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            if oldest_time is None or timestamp < oldest_time:
                                oldest_time = timestamp
                            if newest_time is None or timestamp > newest_time:
                                newest_time = timestamp

                        # Estimate size
                        data_size = len(event_data.get("data", ""))
                        total_size += data_size

                avg_size = total_size / total_events if total_events > 0 else 0.0

                stats = EventStorageStats(
                    total_events=total_events,
                    events_by_type=dict(events_by_type),
                    storage_size_bytes=total_size,
                    oldest_event_time=oldest_time,
                    newest_event_time=newest_time,
                    avg_event_size_bytes=avg_size,
                )
                return stats
            except Exception as e:
                raise StorageError(f"Failed to get storage statistics: {e}")

    async def health_check(self) -> bool:
        """Perform a health check on the storage."""
        if self._closed:
            return False

        try:
            # Test basic operations
            test_key = "health_check_test"
            await self._metadata_repo.set(test_key, {"test": "data"})
            result = await self._metadata_repo.get(test_key)
            await self._metadata_repo.delete(test_key)

            return result is not None

        except Exception:
            return False

    async def close(self) -> None:
        """Close the event storage and clean up resources."""
        self._closed = True
        await self._metadata_repo.close()

    # Private helper methods

    async def _update_indexes(self, storage_id: str, event: BaseEvent) -> None:
        """Update indexes for the stored event."""
        # Type index
        type_key = f"index:type:{event.event_type.value}"
        type_index = await self._metadata_repo.get(type_key) or {"event_ids": []}
        type_index["event_ids"].append(storage_id)
        await self._metadata_repo.set(type_key, type_index)

        # Symbol index
        if event.symbol:
            symbol_key = f"index:symbol:{event.symbol}"
            symbol_index = await self._metadata_repo.get(symbol_key) or {"event_ids": []}
            symbol_index["event_ids"].append(storage_id)
            await self._metadata_repo.set(symbol_key, symbol_index)

        # Source index
        source_key = f"index:source:{event.source}"
        source_index = await self._metadata_repo.get(source_key) or {"event_ids": []}
        source_index["event_ids"].append(storage_id)
        await self._metadata_repo.set(source_key, source_index)

    async def _batch_update_indexes(self, batch_updates: List[tuple[str, BaseEvent]]) -> None:
        """Update indexes for multiple events in batch."""
        if not batch_updates:
            return

        # Group updates by index type to minimize repository calls
        type_updates = defaultdict(list)
        symbol_updates = defaultdict(list)
        source_updates = defaultdict(list)

        for storage_id, event in batch_updates:
            # Group by event type
            type_updates[event.event_type.value].append(storage_id)

            # Group by symbol
            if event.symbol:
                symbol_updates[event.symbol].append(storage_id)

            # Group by source
            source_updates[event.source].append(storage_id)

        # Update type indexes
        for event_type, storage_ids in type_updates.items():
            type_key = f"index:type:{event_type}"
            type_index = await self._metadata_repo.get(type_key) or {"event_ids": []}
            type_index["event_ids"].extend(storage_ids)
            await self._metadata_repo.set(type_key, type_index)

        # Update symbol indexes
        for symbol, storage_ids in symbol_updates.items():
            symbol_key = f"index:symbol:{symbol}"
            symbol_index = await self._metadata_repo.get(symbol_key) or {"event_ids": []}
            symbol_index["event_ids"].extend(storage_ids)
            await self._metadata_repo.set(symbol_key, symbol_index)

        # Update source indexes
        for source, storage_ids in source_updates.items():
            source_key = f"index:source:{source}"
            source_index = await self._metadata_repo.get(source_key) or {"event_ids": []}
            source_index["event_ids"].extend(storage_ids)
            await self._metadata_repo.set(source_key, source_index)

    async def _transactional_update_indexes(self, storage_id: str, event: BaseEvent) -> None:
        """Update indexes for a single event with transactional safety."""
        async with self._index_lock:
            # Store original index states for rollback
            original_states = {}

            try:
                # Type index
                type_key = f"index:type:{event.event_type.value}"
                original_states[type_key] = await self._metadata_repo.get(type_key)
                type_index = original_states[type_key] or {"event_ids": []}
                type_index = {"event_ids": type_index["event_ids"].copy()}
                type_index["event_ids"].append(storage_id)
                await self._metadata_repo.set(type_key, type_index)

                # Symbol index
                if event.symbol:
                    symbol_key = f"index:symbol:{event.symbol}"
                    original_states[symbol_key] = await self._metadata_repo.get(symbol_key)
                    symbol_index = original_states[symbol_key] or {"event_ids": []}
                    symbol_index = {"event_ids": symbol_index["event_ids"].copy()}
                    symbol_index["event_ids"].append(storage_id)
                    await self._metadata_repo.set(symbol_key, symbol_index)

                # Source index
                source_key = f"index:source:{event.source}"
                original_states[source_key] = await self._metadata_repo.get(source_key)
                source_index = original_states[source_key] or {"event_ids": []}
                source_index = {"event_ids": source_index["event_ids"].copy()}
                source_index["event_ids"].append(storage_id)
                await self._metadata_repo.set(source_key, source_index)

                # Time-based index (hourly shards)
                time_key = self._get_time_shard_key(event.timestamp)
                original_states[time_key] = await self._metadata_repo.get(time_key)
                time_index = original_states[time_key] or {"event_ids": []}
                time_index = {"event_ids": time_index["event_ids"].copy()}
                time_index["event_ids"].append(storage_id)
                await self._metadata_repo.set(time_key, time_index)

            except Exception as e:
                # Rollback on error
                for key, original_state in original_states.items():
                    if original_state is not None:
                        await self._metadata_repo.set(key, original_state)
                    else:
                        await self._metadata_repo.delete(key)
                raise StorageError(f"Index update failed: {e}")

    async def _transactional_batch_update_indexes(self, batch_updates: list[tuple[str, BaseEvent]]) -> None:
        """Update indexes for multiple events with transactional safety."""
        async with self._index_lock:
            if not batch_updates:
                return

            # Store original index states for rollback
            original_states = {}

            try:
                # Group updates by index type to minimize repository calls
                type_updates = defaultdict(list)
                symbol_updates = defaultdict(list)
                source_updates = defaultdict(list)
                time_updates = defaultdict(list)

                for storage_id, event in batch_updates:
                    # Group by event type
                    type_updates[event.event_type.value].append(storage_id)

                    # Group by symbol
                    if event.symbol:
                        symbol_updates[event.symbol].append(storage_id)

                    # Group by source
                    source_updates[event.source].append(storage_id)

                    # Group by time shard
                    time_shard = self._get_time_shard_key(event.timestamp)
                    time_updates[time_shard].append(storage_id)

                # Update type indexes
                for event_type, storage_ids in type_updates.items():
                    type_key = f"index:type:{event_type}"
                    original_states[type_key] = await self._metadata_repo.get(type_key)
                    type_index = original_states[type_key] or {"event_ids": []}
                    type_index = {"event_ids": type_index["event_ids"].copy()}
                    type_index["event_ids"].extend(storage_ids)
                    await self._metadata_repo.set(type_key, type_index)

                # Update symbol indexes
                for symbol, storage_ids in symbol_updates.items():
                    symbol_key = f"index:symbol:{symbol}"
                    original_states[symbol_key] = await self._metadata_repo.get(symbol_key)
                    symbol_index = original_states[symbol_key] or {"event_ids": []}
                    symbol_index = {"event_ids": symbol_index["event_ids"].copy()}
                    symbol_index["event_ids"].extend(storage_ids)
                    await self._metadata_repo.set(symbol_key, symbol_index)

                # Update source indexes
                for source, storage_ids in source_updates.items():
                    source_key = f"index:source:{source}"
                    original_states[source_key] = await self._metadata_repo.get(source_key)
                    source_index = original_states[source_key] or {"event_ids": []}
                    source_index = {"event_ids": source_index["event_ids"].copy()}
                    source_index["event_ids"].extend(storage_ids)
                    await self._metadata_repo.set(source_key, source_index)

                # Update time indexes
                for time_shard, storage_ids in time_updates.items():
                    original_states[time_shard] = await self._metadata_repo.get(time_shard)
                    time_index = original_states[time_shard] or {"event_ids": []}
                    time_index = {"event_ids": time_index["event_ids"].copy()}
                    time_index["event_ids"].extend(storage_ids)
                    await self._metadata_repo.set(time_shard, time_index)

            except Exception as e:
                # Rollback on error
                for key, original_state in original_states.items():
                    if original_state is not None:
                        await self._metadata_repo.set(key, original_state)
                    else:
                        await self._metadata_repo.delete(key)
                raise StorageError(f"Batch index update failed: {e}")

    async def _remove_from_indexes(self, storage_id: str, event: BaseEvent) -> None:
        """Remove event from indexes."""
        # Type index
        type_key = f"index:type:{event.event_type.value}"
        type_index = await self._metadata_repo.get(type_key)
        if type_index and "event_ids" in type_index:
            if storage_id in type_index["event_ids"]:
                type_index["event_ids"].remove(storage_id)
                await self._metadata_repo.set(type_key, type_index)

        # Symbol index
        if event.symbol:
            symbol_key = f"index:symbol:{event.symbol}"
            symbol_index = await self._metadata_repo.get(symbol_key)
            if symbol_index and "event_ids" in symbol_index:
                if storage_id in symbol_index["event_ids"]:
                    symbol_index["event_ids"].remove(storage_id)
                    await self._metadata_repo.set(symbol_key, symbol_index)

        # Source index
        source_key = f"index:source:{event.source}"
        source_index = await self._metadata_repo.get(source_key)
        if source_index and "event_ids" in source_index:
            if storage_id in source_index["event_ids"]:
                source_index["event_ids"].remove(storage_id)
                await self._metadata_repo.set(source_key, source_index)

        # Time index
        time_key = self._get_time_shard_key(event.timestamp)
        time_index = await self._metadata_repo.get(time_key)
        if time_index and "event_ids" in time_index:
            if storage_id in time_index["event_ids"]:
                time_index["event_ids"].remove(storage_id)
                await self._metadata_repo.set(time_key, time_index)

    async def _get_candidate_event_ids(self, query: EventQuery) -> List[str]:
        """Get candidate event IDs based on query filters."""
        candidate_sets = []

        # Filter by time range first for better performance
        if query.start_time or query.end_time:
            time_candidates = await self._get_time_shard_candidates(query.start_time, query.end_time)
            candidate_sets.append(time_candidates)

        # Filter by event types
        if query.event_types:
            type_candidates = set()
            for event_type in query.event_types:
                type_key = f"index:type:{event_type.value}"
                type_index = await self._metadata_repo.get(type_key)
                if type_index and "event_ids" in type_index:
                    type_candidates.update(type_index["event_ids"])
            candidate_sets.append(type_candidates)

        # Filter by symbols
        if query.symbols:
            symbol_candidates = set()
            for symbol in query.symbols:
                symbol_key = f"index:symbol:{symbol}"
                symbol_index = await self._metadata_repo.get(symbol_key)
                if symbol_index and "event_ids" in symbol_index:
                    symbol_candidates.update(symbol_index["event_ids"])
            candidate_sets.append(symbol_candidates)

        # Filter by sources
        if query.sources:
            source_candidates = set()
            for source in query.sources:
                source_key = f"index:source:{source}"
                source_index = await self._metadata_repo.get(source_key)
                if source_index and "event_ids" in source_index:
                    source_candidates.update(source_index["event_ids"])
            candidate_sets.append(source_candidates)

        # If no specific filters, get all events
        if not candidate_sets:
            all_event_keys = await self._metadata_repo.list_keys("event:")
            all_ids = [key.replace("event:", "") for key in all_event_keys]
            return all_ids

        # Intersect all candidate sets
        result = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            result = result.intersection(candidate_set)

        return list(result)

    def _matches_query(self, event: BaseEvent, query: EventQuery) -> bool:
        """Check if an event matches the query criteria."""
        # Time range filter
        if query.start_time and event.timestamp < query.start_time:
            return False
        if query.end_time and event.timestamp > query.end_time:
            return False

        # Correlation ID filter
        if query.correlation_id and event.correlation_id != query.correlation_id:
            return False

        # Metadata filters
        if query.metadata_filters:
            for key, value in query.metadata_filters.items():
                if key not in event.metadata or event.metadata[key] != value:
                    return False

        return True

    def _apply_ordering(self, events: List[BaseEvent], query: EventQuery) -> List[BaseEvent]:
        """Apply ordering to the events."""
        if query.order_by == "timestamp":
            events.sort(key=lambda e: e.timestamp, reverse=query.order_desc)
        # Add more ordering options as needed

        return events

    def _apply_pagination(self, events: List[BaseEvent], query: EventQuery) -> List[BaseEvent]:
        """Apply pagination to the events."""
        start_idx = query.offset
        end_idx = start_idx + query.limit if query.limit else len(events)

        return events[start_idx:end_idx]

    async def _find_storage_ids_for_event(self, event: BaseEvent) -> List[str]:
        """Find storage IDs for a given event (reverse lookup)."""
        # This is a simplified implementation
        # In a real scenario, we might need a more efficient reverse index
        all_event_keys = await self._metadata_repo.list_keys("event:")
        matching_ids = []

        for event_key in all_event_keys:
            event_data = await self._metadata_repo.get(event_key)
            if event_data and event_data.get("timestamp") == event.timestamp.isoformat():
                storage_id = event_key.replace("event:", "")
                matching_ids.append(storage_id)

        return matching_ids

    def _get_time_shard_key(self, timestamp: datetime) -> str:
        """Generate a time shard key for indexing based on hourly buckets."""
        # Create hourly time shards: e.g., "index:time:2024-01-15T10"
        time_shard = timestamp.strftime("%Y-%m-%dT%H")
        return f"index:time:{time_shard}"

    async def _get_time_shard_candidates(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> set[str]:
        """Get candidate event IDs from time shards within the specified range."""
        if start_time is None and end_time is None:
            # No time filtering, return all events
            all_event_keys = await self._metadata_repo.list_keys("event:")
            return {key.replace("event:", "") for key in all_event_keys}

        candidate_ids = set()

        # Generate all time shards that fall within the time range
        current_time = start_time.replace(minute=0, second=0, microsecond=0) if start_time else None
        end_hour = end_time.replace(minute=0, second=0, microsecond=0) if end_time else None

        if current_time and end_hour:
            # Iterate through each hour in the range
            while current_time <= end_hour:
                time_shard_key = self._get_time_shard_key(current_time)
                time_index = await self._metadata_repo.get(time_shard_key)
                if time_index and "event_ids" in time_index:
                    candidate_ids.update(time_index["event_ids"])
                current_time = current_time + timedelta(hours=1)

        return candidate_ids
