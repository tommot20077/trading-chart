# ABOUTME: NoOp implementation of AbstractTimeSeriesRepository that provides fake storage
# ABOUTME: Provides minimal time series storage functionality for testing scenarios

from typing import Any, AsyncIterator, TypeVar, Generic
from datetime import datetime

from core.interfaces.storage.time_sequence_repository import AbstractTimeSeriesRepository
from core.models.storage.query_option import QueryOptions
from core.models.storage.time_series_data import TimeSeriesData

T = TypeVar("T", bound=TimeSeriesData)


class NoOpTimeSeriesRepository(Generic[T], AbstractTimeSeriesRepository[T]):
    """
    No-operation implementation of AbstractTimeSeriesRepository.

    This implementation provides minimal time series storage functionality that
    simulates storage operations without actually persisting data. It's useful
    for testing, performance benchmarking, and scenarios where time series storage
    is not required.

    Features:
    - Simulates successful storage operations
    - No actual data persistence
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where storage should be bypassed
    - Performance benchmarking without storage overhead
    - Development environments where storage is not needed
    - Fallback when storage systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation time series repository."""
        self._closed = False
        self._item_count = 0  # Track number of "stored" items for stats

    async def save(self, item: T) -> None:
        """
        Save a single item - simulates storage without persistence.

        This implementation simulates successful storage without actually
        persisting the data.

        Args:
            item: The time series item to store (ignored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        self._item_count += 1

    async def save_batch(self, items: list[T]) -> int:
        """
        Save multiple items - simulates batch storage.

        This implementation simulates successful batch storage.

        Args:
            items: List of time series items to store

        Returns:
            Number of items "stored" (length of input list)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        count = len(items)
        self._item_count += count
        return count

    async def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        *,
        options: QueryOptions | None = None,
    ) -> list[T]:
        """
        Query items - always returns empty list (no data stored).

        This implementation always returns an empty list since no data
        is actually stored.

        Args:
            symbol: Trading symbol
            start_time: Start time for query
            end_time: End time for query
            options: Optional query options

        Returns:
            Empty list (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return []

    async def stream(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        *,
        batch_size: int = 1000,
    ) -> AsyncIterator[T]:
        """
        Stream items - yields nothing (no data stored).

        This implementation yields nothing since no data is actually stored.

        Args:
            symbol: Trading symbol
            start_time: Start time for streaming
            end_time: End time for streaming
            batch_size: Batch size for streaming

        Yields:
            Nothing (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        # Empty async generator
        return
        yield  # This line is unreachable but needed for type checking

    async def get_latest(self, symbol: str) -> T | None:
        """
        Get latest item - always returns None (no data stored).

        This implementation always returns None since no data is actually stored.

        Args:
            symbol: Trading symbol

        Returns:
            None (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return None

    async def get_oldest(self, symbol: str) -> T | None:
        """
        Get oldest item - always returns None (no data stored).

        This implementation always returns None since no data is actually stored.

        Args:
            symbol: Trading symbol

        Returns:
            None (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return None

    async def count(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """
        Count items - returns 0 (no data stored).

        This implementation returns 0 since no data is actually stored.

        Args:
            symbol: Trading symbol
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            0 (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return 0

    async def delete(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """
        Delete items - returns 0 (no data to delete).

        This implementation returns 0 since no data is actually stored.

        Args:
            symbol: Trading symbol
            start_time: Start time for deletion
            end_time: End time for deletion

        Returns:
            0 (no data is actually stored to delete)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return 0

    async def get_gaps(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """
        Find gaps - returns empty list (no data stored).

        This implementation returns an empty list since no data is stored.

        Args:
            symbol: Trading symbol
            start_time: Start time for gap analysis
            end_time: End time for gap analysis

        Returns:
            Empty list (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return []

    async def get_statistics(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics - returns fake statistics.

        This implementation returns fake statistics.

        Args:
            symbol: Trading symbol
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Fake statistics dictionary
        """
        if self._closed:
            raise RuntimeError("Time series repository is closed")

        return {
            "total_items": 0,
            "symbol": symbol,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "fake_stat": "noop_implementation",
        }

    async def close(self) -> None:
        """
        Close the repository - sets closed flag and resets counters.
        """
        self._closed = True
        self._item_count = 0
