# ABOUTME: Abstract Kline repository interface for time-series candlestick data storage and retrieval
# ABOUTME: Defines the contract for components that manage Kline data with interval-based operations

from abc import abstractmethod, ABC
from datetime import datetime
from typing import AsyncIterator
from types import TracebackType

from core.models import KlineInterval, Kline
from core.models.storage.query_option import QueryOptions
from core.models.types import StatisticsResult


class AbstractKlineRepository(ABC):
    """
    [L0] Abstract interface for a repository managing Kline data.

    This interface provides Kline-specific operations including interval-based
    queries and operations. Unlike the generic AbstractTimeSeriesRepository,
    this interface is specifically designed for Kline data with interval support.

    Architecture note: This is a [L0] interface that only depends on models
    and provides clean abstractions for [L1] storage implementations.
    """

    @abstractmethod
    async def save(self, kline: Kline) -> None:
        """Save a single kline to the repository.

        Args:
            kline (Kline): The kline data to save.

        Returns:
            None

        Raises:
            StorageError: If the kline cannot be saved due to storage issues.
            ValidationError: If the kline data is invalid.
        """
        pass

    @abstractmethod
    async def save_batch(self, klines: list[Kline]) -> int:
        """Save multiple klines in batch.

        Args:
            klines (list[Kline]): List of kline data to save.

        Returns:
            int: Number of klines successfully saved.

        Raises:
            StorageError: If the klines cannot be saved due to storage issues.
            ValidationError: If any kline data is invalid.
        """
        pass

    @abstractmethod
    async def query(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        options: QueryOptions | None = None,
    ) -> list[Kline]:
        """Query klines within the specified time range for a specific interval.

        Args:
            symbol (str): The trading symbol to query for.
            interval (KlineInterval): The kline interval (e.g., 1m, 5m, 1h).
            start_time (datetime): Start time for the query range (inclusive).
            end_time (datetime): End time for the query range (inclusive).
            options (QueryOptions | None, optional): Additional query options such as
                ordering, limiting, and filtering. Defaults to None.

        Returns:
            list[Kline]: List of klines matching the query criteria, ordered by timestamp.

        Raises:
            StorageError: If query execution fails due to storage issues.
            ValueError: If start_time is after end_time or parameters are invalid.
        """
        pass

    @abstractmethod
    def stream(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        batch_size: int = 1000,
    ) -> AsyncIterator[Kline]:
        """Stream klines within the specified time range for a specific interval.

        Args:
            symbol (str): The trading symbol to stream for.
            interval (KlineInterval): The kline interval (e.g., 1m, 5m, 1h).
            start_time (datetime): Start time for the stream range (inclusive).
            end_time (datetime): End time for the stream range (inclusive).
            batch_size (int, optional): Number of klines to fetch per batch. Defaults to 1000.

        Returns:
            AsyncIterator[Kline]: Async iterator yielding klines in timestamp order.

        Raises:
            StorageError: If streaming fails due to storage issues.
            ValueError: If start_time is after end_time or batch_size is invalid.
        """
        pass

    @abstractmethod
    async def get_latest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the most recent kline for the symbol and interval.

        Args:
            symbol (str): The trading symbol to query for.
            interval (KlineInterval): The kline interval to query for.

        Returns:
            Kline | None: The most recent kline, or None if no klines exist.

        Raises:
            StorageError: If query execution fails due to storage issues.
        """
        pass

    @abstractmethod
    async def get_oldest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the oldest kline for the symbol and interval.

        Args:
            symbol (str): The trading symbol to query for.
            interval (KlineInterval): The kline interval to query for.

        Returns:
            Kline | None: The oldest kline, or None if no klines exist.

        Raises:
            StorageError: If query execution fails due to storage issues.
        """
        pass

    @abstractmethod
    async def count(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count klines for the symbol and interval within optional time range.

        Args:
            symbol (str): The trading symbol to count for.
            interval (KlineInterval): The kline interval to count for.
            start_time (datetime | None, optional): Start time for the count range (inclusive).
                If None, count from the beginning. Defaults to None.
            end_time (datetime | None, optional): End time for the count range (inclusive).
                If None, count to the end. Defaults to None.

        Returns:
            int: Number of klines matching the criteria.

        Raises:
            StorageError: If count operation fails due to storage issues.
            ValueError: If start_time is after end_time when both are provided.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete klines within the specified time range for a specific interval.

        Args:
            symbol (str): The trading symbol to delete klines for.
            interval (KlineInterval): The kline interval to delete for.
            start_time (datetime): Start time for the deletion range (inclusive).
            end_time (datetime): End time for the deletion range (inclusive).

        Returns:
            int: Number of klines successfully deleted.

        Raises:
            StorageError: If deletion fails due to storage issues.
            ValueError: If start_time is after end_time.
        """
        pass

    @abstractmethod
    async def get_gaps(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in kline data within the specified time range for a specific interval.

        Args:
            symbol (str): The trading symbol to analyze for gaps.
            interval (KlineInterval): The kline interval to analyze for gaps.
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
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> StatisticsResult:
        """Get statistics for klines within optional time range for a specific interval.

        Args:
            symbol (str): The trading symbol to get statistics for.
            interval (KlineInterval): The kline interval to get statistics for.
            start_time (datetime | None, optional): Start time for the statistics range (inclusive).
                If None, analyze from the beginning. Defaults to None.
            end_time (datetime | None, optional): End time for the statistics range (inclusive).
                If None, analyze to the end. Defaults to None.

        Returns:
            dict[str, Any]: Dictionary containing statistical information such as:
                - count: Total number of klines
                - min_timestamp: Earliest timestamp
                - max_timestamp: Latest timestamp
                - price_statistics: Min/max/avg price information
                - volume_statistics: Min/max/avg volume information

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

    async def __aenter__(self) -> "AbstractKlineRepository":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()
