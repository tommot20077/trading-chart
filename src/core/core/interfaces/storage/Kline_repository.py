from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, AsyncIterator

from core.models import KlineInterval, Kline
from core.models.storage.query_option import QueryOptions


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
        """Save a single kline to the repository."""
        pass

    @abstractmethod
    async def save_batch(self, klines: list[Kline]) -> int:
        """Save multiple klines in batch. Returns number of klines saved."""
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
        """Query klines within the specified time range for a specific interval."""
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
        """Stream klines within the specified time range for a specific interval."""
        pass

    @abstractmethod
    async def get_latest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the most recent kline for the symbol and interval."""
        pass

    @abstractmethod
    async def get_oldest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the oldest kline for the symbol and interval."""
        pass

    @abstractmethod
    async def count(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count klines for the symbol and interval within optional time range."""
        pass

    @abstractmethod
    async def delete(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete klines within the specified time range for a specific interval. Returns number deleted."""
        pass

    @abstractmethod
    async def get_gaps(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in kline data within the specified time range for a specific interval."""
        pass

    @abstractmethod
    async def get_statistics(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get statistics for klines within optional time range for a specific interval."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the repository and clean up resources."""
        pass

    async def __aenter__(self) -> "AbstractKlineRepository":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
