# ABOUTME: This module defines generic time-series repository interfaces for trading data storage
# ABOUTME: It provides base classes for Kline, Trade, and other time-series data repositories

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, TypeVar, Generic

from core.models.storage.query_option import QueryOptions
from core.models.storage.time_series_data import TimeSeriesData

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
        """Save a single time-series item to the repository."""
        pass

    @abstractmethod
    async def save_batch(self, items: list[T]) -> int:
        """Save multiple time-series items in batch. Returns number of items saved."""
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
        """Query time-series items within the specified time range."""
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
        """Stream time-series items within the specified time range."""
        pass

    @abstractmethod
    async def get_latest(self, symbol: str) -> T | None:
        """Get the most recent time-series item for the symbol."""
        pass

    @abstractmethod
    async def get_oldest(self, symbol: str) -> T | None:
        """Get the oldest time-series item for the symbol."""
        pass

    @abstractmethod
    async def count(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count time-series items for the symbol within optional time range."""
        pass

    @abstractmethod
    async def delete(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete time-series items within the specified time range. Returns number deleted."""
        pass

    @abstractmethod
    async def get_gaps(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in time-series data within the specified time range."""
        pass

    @abstractmethod
    async def get_statistics(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get statistics for time-series items within optional time range."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the repository and clean up resources."""
        pass

    async def __aenter__(self) -> "AbstractTimeSeriesRepository[T]":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
