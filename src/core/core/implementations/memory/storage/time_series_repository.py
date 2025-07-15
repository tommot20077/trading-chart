# ABOUTME: In-memory implementation of AbstractTimeSeriesRepository for testing and development
# ABOUTME: Provides zero-dependency time-series data storage using Python standard library only

import bisect
from collections import defaultdict
from datetime import datetime
from typing import Any, AsyncIterator, TypeVar, Generic

from core.interfaces.storage.time_sequence_repository import AbstractTimeSeriesRepository
from core.models.storage.query_option import QueryOptions
from core.models.storage.time_series_data import TimeSeriesData

# Generic type for time-series data
T = TypeVar("T", bound=TimeSeriesData)


class InMemoryTimeSeriesRepository(Generic[T], AbstractTimeSeriesRepository[T]):
    """
    In-memory implementation of AbstractTimeSeriesRepository.

    This implementation provides a complete time-series storage solution using
    only Python standard library data structures. It maintains data sorted by
    timestamp for efficient range queries and provides all required operations.

    Data Structure:
    - _data: Dict[symbol, List[T]] - stores data points sorted by timestamp
    - _timestamp_index: Dict[symbol, List[datetime]] - maintains sorted timestamp index
    - _closed: bool - tracks repository state

    Thread Safety: This implementation is NOT thread-safe. For concurrent access,
    external synchronization is required.
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        # Store data points grouped by symbol, sorted by timestamp
        self._data: dict[str, list[T]] = defaultdict(list)

        # Maintain sorted timestamp index for efficient binary search
        self._timestamp_index: dict[str, list[datetime]] = defaultdict(list)

        # Track repository state
        self._closed = False

    def _ensure_not_closed(self) -> None:
        """Ensure repository is not closed."""
        if self._closed:
            raise RuntimeError("Repository is closed")

    def _insert_sorted(self, symbol: str, item: T) -> None:
        """Insert item maintaining timestamp order."""
        timestamp = item.primary_timestamp
        data_list = self._data[symbol]
        timestamp_list = self._timestamp_index[symbol]

        # Find insertion point using binary search
        insert_pos = bisect.bisect_left(timestamp_list, timestamp)

        # Insert at the correct position
        data_list.insert(insert_pos, item)
        timestamp_list.insert(insert_pos, timestamp)

    def _find_range_indices(self, symbol: str, start_time: datetime, end_time: datetime) -> tuple[int, int]:
        """Find start and end indices for time range query."""
        if symbol not in self._timestamp_index:
            return 0, 0

        timestamp_list = self._timestamp_index[symbol]

        # Find start index (first timestamp >= start_time)
        start_idx = bisect.bisect_left(timestamp_list, start_time)

        # Find end index (first timestamp > end_time)
        end_idx = bisect.bisect_right(timestamp_list, end_time)

        return start_idx, end_idx

    async def save(self, item: T) -> None:
        """Save a single time-series item to the repository."""
        self._ensure_not_closed()

        if not hasattr(item, "symbol") or not hasattr(item, "primary_timestamp"):
            raise ValueError("Item must have 'symbol' and 'primary_timestamp' attributes")

        self._insert_sorted(item.symbol, item)

    async def save_batch(self, items: list[T]) -> int:
        """Save multiple time-series items in batch. Returns number of items saved."""
        self._ensure_not_closed()

        if not items:
            return 0

        saved_count = 0
        for item in items:
            try:
                await self.save(item)
                saved_count += 1
            except Exception:
                # Continue with other items, but don't count failed ones
                continue

        return saved_count

    async def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        *,
        options: QueryOptions | None = None,
    ) -> list[T]:
        """Query time-series items within the specified time range."""
        self._ensure_not_closed()

        if start_time > end_time:
            raise ValueError("start_time must be <= end_time")

        # Get data in time range
        start_idx, end_idx = self._find_range_indices(symbol, start_time, end_time)
        data_list = self._data[symbol]
        result = data_list[start_idx:end_idx]

        # Apply query options if provided
        if options:
            # Handle ordering
            if options.order_desc:
                result = list(reversed(result))

            # Handle pagination
            if options.offset:
                result = result[options.offset :]

            if options.limit:
                result = result[: options.limit]

        return result

    async def stream(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        *,
        batch_size: int = 1000,
    ) -> AsyncIterator[T]:
        """Stream time-series items within the specified time range."""
        self._ensure_not_closed()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        start_idx, end_idx = self._find_range_indices(symbol, start_time, end_time)
        data_list = self._data[symbol]

        # Stream in batches
        current_idx = start_idx
        while current_idx < end_idx:
            batch_end = min(current_idx + batch_size, end_idx)
            for item in data_list[current_idx:batch_end]:
                yield item
            current_idx = batch_end

    async def get_latest(self, symbol: str) -> T | None:
        """Get the most recent time-series item for the symbol."""
        self._ensure_not_closed()

        data_list = self._data[symbol]
        return data_list[-1] if data_list else None

    async def get_oldest(self, symbol: str) -> T | None:
        """Get the oldest time-series item for the symbol."""
        self._ensure_not_closed()

        data_list = self._data[symbol]
        return data_list[0] if data_list else None

    async def count(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count time-series items for the symbol within optional time range."""
        self._ensure_not_closed()

        if start_time is None and end_time is None:
            return len(self._data[symbol])

        # Use earliest/latest times if not specified
        data_list = self._data[symbol]
        if not data_list:
            return 0

        if start_time is None:
            start_time = data_list[0].primary_timestamp
        if end_time is None:
            end_time = data_list[-1].primary_timestamp

        start_idx, end_idx = self._find_range_indices(symbol, start_time, end_time)
        return end_idx - start_idx

    async def delete(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete time-series items within the specified time range. Returns number deleted."""
        self._ensure_not_closed()

        if start_time > end_time:
            raise ValueError("start_time must be <= end_time")

        start_idx, end_idx = self._find_range_indices(symbol, start_time, end_time)

        if start_idx >= end_idx:
            return 0

        # Remove items from both data and timestamp index
        data_list = self._data[symbol]
        timestamp_list = self._timestamp_index[symbol]

        deleted_count = end_idx - start_idx

        # Remove from both lists
        del data_list[start_idx:end_idx]
        del timestamp_list[start_idx:end_idx]

        return deleted_count

    async def get_gaps(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in time-series data within the specified time range."""
        self._ensure_not_closed()

        if start_time > end_time:
            raise ValueError("start_time must be <= end_time")

        timestamp_list = self._timestamp_index[symbol]
        if not timestamp_list:
            return [(start_time, end_time)]

        gaps = []

        # Find first timestamp >= start_time
        start_idx = bisect.bisect_left(timestamp_list, start_time)

        # Check gap before first data point
        if start_idx == 0:
            first_timestamp = timestamp_list[0]
            if start_time < first_timestamp:
                gaps.append((start_time, first_timestamp))
        else:
            # Check if there's a gap from start_time to first relevant data
            if start_idx < len(timestamp_list):
                first_relevant = timestamp_list[start_idx]
                if start_time < first_relevant:
                    gaps.append((start_time, first_relevant))

        # Find gaps between consecutive data points
        for i in range(start_idx, len(timestamp_list) - 1):
            current_time = timestamp_list[i]
            next_time = timestamp_list[i + 1]

            if next_time > end_time:
                break

            # For simplicity, consider any time difference as a potential gap
            # In a real implementation, this would depend on expected interval
            if next_time > current_time:
                gaps.append((current_time, next_time))

        # Check gap after last data point
        if timestamp_list:
            last_timestamp = timestamp_list[-1]
            if last_timestamp < end_time:
                gaps.append((last_timestamp, end_time))

        return gaps

    async def get_statistics(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get statistics for time-series items within optional time range."""
        self._ensure_not_closed()

        data_list = self._data[symbol]

        if not data_list:
            return {
                "count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "time_span_seconds": 0,
            }

        # Filter by time range if specified
        if start_time is not None or end_time is not None:
            if start_time is None:
                start_time = data_list[0].primary_timestamp
            if end_time is None:
                end_time = data_list[-1].primary_timestamp

            start_idx, end_idx = self._find_range_indices(symbol, start_time, end_time)
            filtered_data = data_list[start_idx:end_idx]
        else:
            filtered_data = data_list

        if not filtered_data:
            return {
                "count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "time_span_seconds": 0,
            }

        first_timestamp = filtered_data[0].primary_timestamp
        last_timestamp = filtered_data[-1].primary_timestamp
        time_span = (last_timestamp - first_timestamp).total_seconds()

        return {
            "count": len(filtered_data),
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "time_span_seconds": time_span,
        }

    async def close(self) -> None:
        """Close the repository and clean up resources."""
        if not self._closed:
            self._data.clear()
            self._timestamp_index.clear()
            self._closed = True
