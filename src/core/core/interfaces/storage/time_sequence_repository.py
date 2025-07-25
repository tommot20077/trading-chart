# ABOUTME: This module defines generic time-series repository interfaces for trading data storage
# ABOUTME: It provides base classes for Kline, Trade, and other time-series data repositories

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, TypeVar, Generic
from types import TracebackType

from core.models.storage.query_option import QueryOptions
from core.models.storage.time_series_data import TimeSeriesData
from core.models.types import StatisticsResult

# Generic type for time-series data
T = TypeVar("T", bound=TimeSeriesData)


class AbstractTimeSeriesRepository(Generic[T], ABC):
    """
    [L0] Abstract base interface for time-series data repositories.

    This generic interface defines the common contract for time-series data storage
    and retrieval operations. It provides base functionality that can be used by
    concrete implementations for different types of time-series data (Kline, Trade, etc.).

    Type Parameters:
        T: The type of time-series data this repository manages.

    Architecture note: This is a [L0] interface that only depends on models
    and provides clean abstractions for [L1] storage implementations.
    """

    @abstractmethod
    async def save(self, item: T) -> None:
        """Save a single time-series item to the repository.

        Args:
            item (T): The time-series data item to save.

        Returns:
            None

        Raises:
            StorageError: If the item cannot be saved due to storage issues.
            ValidationError: If the item data is invalid.
        """
        pass

    @abstractmethod
    async def save_batch(self, items: list[T]) -> int:
        """Save multiple time-series items in batch.

        Args:
            items (list[T]): List of time-series data items to save.

        Returns:
            int: Number of items successfully saved.

        Raises:
            StorageError: If the items cannot be saved due to storage issues.
            ValidationError: If any item data is invalid.
        """
        pass

    @abstractmethod
    async def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        *,
        options: QueryOptions | None = None,
    ) -> list[T]:
        """Query time-series items within the specified time range.

        Args:
            symbol (str): The trading symbol to query for.
            start_time (datetime): Start time for the query range (inclusive).
            end_time (datetime): End time for the query range (inclusive).
            options (QueryOptions | None, optional): Additional query options such as
                ordering, limiting, and filtering. Defaults to None.

        Returns:
            list[T]: List of time-series items matching the query criteria, ordered by timestamp.

        Raises:
            StorageError: If query execution fails due to storage issues.
            ValueError: If start_time is after end_time or parameters are invalid.
        """
        pass

    @abstractmethod
    def stream(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        *,
        batch_size: int = 1000,
    ) -> AsyncIterator[T]:
        """Stream time-series items within the specified time range.

        Args:
            symbol (str): The trading symbol to stream for.
            start_time (datetime): Start time for the stream range (inclusive).
            end_time (datetime): End time for the stream range (inclusive).
            batch_size (int, optional): Number of items to fetch per batch. Defaults to 1000.

        Returns:
            AsyncIterator[T]: Async iterator yielding time-series items in timestamp order.

        Raises:
            StorageError: If streaming fails due to storage issues.
            ValueError: If start_time is after end_time or batch_size is invalid.
        """
        pass

    @abstractmethod
    async def get_latest(self, symbol: str) -> T | None:
        """Get the most recent time-series item for the symbol.

        Args:
            symbol (str): The trading symbol to query for.

        Returns:
            T | None: The most recent time-series item, or None if no items exist.

        Raises:
            StorageError: If query execution fails due to storage issues.
        """
        pass

    @abstractmethod
    async def get_oldest(self, symbol: str) -> T | None:
        """Get the oldest time-series item for the symbol.

        Args:
            symbol (str): The trading symbol to query for.

        Returns:
            T | None: The oldest time-series item, or None if no items exist.

        Raises:
            StorageError: If query execution fails due to storage issues.
        """
        pass

    @abstractmethod
    async def count(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count time-series items for the symbol within optional time range.

        Args:
            symbol (str): The trading symbol to count for.
            start_time (datetime | None, optional): Start time for the count range (inclusive).
                If None, count from the beginning. Defaults to None.
            end_time (datetime | None, optional): End time for the count range (inclusive).
                If None, count to the end. Defaults to None.

        Returns:
            int: Number of time-series items matching the criteria.

        Raises:
            StorageError: If count operation fails due to storage issues.
            ValueError: If start_time is after end_time when both are provided.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete time-series items within the specified time range.

        Args:
            symbol (str): The trading symbol to delete items for.
            start_time (datetime): Start time for the deletion range (inclusive).
            end_time (datetime): End time for the deletion range (inclusive).

        Returns:
            int: Number of time-series items successfully deleted.

        Raises:
            StorageError: If deletion fails due to storage issues.
            ValueError: If start_time is after end_time.
        """
        pass

    @abstractmethod
    async def get_gaps(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in time-series data within the specified time range.

        Args:
            symbol (str): The trading symbol to analyze for gaps.
            start_time (datetime): Start time for the gap analysis range (inclusive).
            end_time (datetime): End time for the gap analysis range (inclusive).

        Returns:
            list[tuple[datetime, datetime]]: List of gaps represented as tuples of
                (gap_start, gap_end) where gap_start and gap_end are the boundaries
                of missing data intervals.

        Raises:
            StorageError: If gap analysis fails due to storage issues.
            ValueError: If start_time is after end_time.
        """
        pass

    @abstractmethod
    async def get_statistics(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> StatisticsResult:
        """Get statistics for time-series items within optional time range.

        Args:
            symbol (str): The trading symbol to get statistics for.
            start_time (datetime | None, optional): Start time for the statistics range (inclusive).
                If None, analyze from the beginning. Defaults to None.
            end_time (datetime | None, optional): End time for the statistics range (inclusive).
                If None, analyze to the end. Defaults to None.

        Returns:
            dict[str, Any]: Dictionary containing statistical information such as:
                - count: Total number of items
                - min_timestamp: Earliest timestamp
                - max_timestamp: Latest timestamp
                - data_statistics: Type-specific statistical information

        Raises:
            StorageError: If statistics calculation fails due to storage issues.
            ValueError: If start_time is after end_time when both are provided.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the repository and clean up resources.

        Returns:
            None

        Raises:
            StorageError: If cleanup fails due to storage issues.
        """
        pass

    async def __aenter__(self) -> "AbstractTimeSeriesRepository[T]":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()
