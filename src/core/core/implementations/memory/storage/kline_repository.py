# ABOUTME: In-memory implementation of AbstractKlineRepository
# ABOUTME: Provides efficient Kline storage with time-based indexing for testing and development

from typing import Dict, List, Any, AsyncIterator
from datetime import datetime
import asyncio
import threading
from collections import defaultdict

from core.interfaces.storage.Kline_repository import AbstractKlineRepository
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval
from core.models.storage.query_option import QueryOptions
from core.models.types import StatisticsResult
from core.exceptions import StorageError, ValidationException


class InMemoryKlineRepository(AbstractKlineRepository):
    """
    In-memory implementation of AbstractKlineRepository.

    This repository provides efficient in-memory storage for Kline data with
    time-based indexing and fast query operations. It's designed for testing
    and development purposes where no external database is available.

    Features:
    - Time-ordered storage with binary search for efficient queries
    - Duplicate detection and prevention
    - Batch operations with performance optimization
    - Streaming support for large datasets
    - Gap detection in time series data
    - Statistical calculations and caching
    - Thread-safe operations
    """

    def __init__(self, max_klines_per_symbol: int = 100000):
        """
        Initialize the in-memory Kline repository.

        Args:
            max_klines_per_symbol: Maximum number of klines to store per symbol-interval pair
        """
        self.max_klines_per_symbol = max_klines_per_symbol

        # Main storage: symbol -> interval -> sorted list of klines
        self._klines: Dict[str, Dict[KlineInterval, List[Kline]]] = defaultdict(lambda: defaultdict(list))

        # Time index for fast lookups: symbol -> interval -> time -> index
        self._time_index: Dict[str, Dict[KlineInterval, Dict[datetime, int]]] = defaultdict(lambda: defaultdict(dict))

        # Statistics cache: symbol -> interval -> stats
        self._stats_cache: Dict[str, Dict[KlineInterval, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

        # Thread safety
        self._lock = threading.RLock()

        # Repository state
        self._closed = False

    async def save(self, kline: Kline) -> None:
        """Save a single kline to the repository."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        await self._validate_kline(kline)

        with self._lock:
            symbol = kline.symbol
            interval = kline.interval

            # Get or create storage for this symbol-interval
            klines_list = self._klines[symbol][interval]
            time_index = self._time_index[symbol][interval]

            # Check for duplicate
            if kline.open_time in time_index:
                existing_index = time_index[kline.open_time]
                existing_kline = klines_list[existing_index]

                # If it's the same kline, update it
                if existing_kline.close_time == kline.close_time:
                    klines_list[existing_index] = kline
                    self._clear_stats_cache(symbol, interval)
                    return
                else:
                    raise ValidationException(
                        f"Duplicate kline open_time {kline.open_time} with different close_time",
                        code="DUPLICATE_KLINE_TIME",
                    )

            # Check storage limits
            if len(klines_list) >= self.max_klines_per_symbol:
                raise StorageError(
                    f"Maximum klines limit ({self.max_klines_per_symbol}) exceeded for {symbol}:{interval}",
                    code="STORAGE_LIMIT_EXCEEDED",
                )

            # Find insertion position to maintain time order
            insert_index = self._find_insert_position(klines_list, kline.open_time)

            # Insert kline and update index
            klines_list.insert(insert_index, kline)

            # Update time index (all indices after insert_index need to be incremented)
            for i in range(insert_index, len(klines_list)):
                time_index[klines_list[i].open_time] = i

            # Clear stats cache
            self._clear_stats_cache(symbol, interval)

    async def save_batch(self, klines: List[Kline]) -> int:
        """Save multiple klines in batch. Returns number of klines saved."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if not klines:
            return 0

        # Validate all klines first
        for kline in klines:
            await self._validate_kline(kline)

        # Group by symbol and interval for efficient batch processing
        grouped_klines: dict[str, dict[KlineInterval, list[Kline]]] = defaultdict(lambda: defaultdict(list))
        for kline in klines:
            grouped_klines[kline.symbol][kline.interval].append(kline)

        saved_count = 0

        with self._lock:
            for symbol, intervals in grouped_klines.items():
                for interval, klines_to_save in intervals.items():
                    # Sort klines by open_time for efficient insertion
                    klines_to_save.sort(key=lambda k: k.open_time)

                    # Check for duplicates within the batch
                    seen_times = set()
                    for kline in klines_to_save:
                        if kline.open_time in seen_times:
                            raise ValidationException(
                                f"Duplicate kline open_time {kline.open_time} in batch",
                                code="DUPLICATE_KLINE_TIME_IN_BATCH",
                            )
                        seen_times.add(kline.open_time)

                    # Get storage for this symbol-interval
                    klines_list = self._klines[symbol][interval]
                    time_index = self._time_index[symbol][interval]

                    # Check storage limits
                    if len(klines_list) + len(klines_to_save) > self.max_klines_per_symbol:
                        raise StorageError(
                            f"Batch would exceed maximum klines limit ({self.max_klines_per_symbol}) for {symbol}:{interval}",
                            code="STORAGE_LIMIT_EXCEEDED",
                        )

                    # Process each kline
                    for kline in klines_to_save:
                        # Check for existing kline
                        if kline.open_time in time_index:
                            existing_index = time_index[kline.open_time]
                            existing_kline = klines_list[existing_index]

                            # Update if same close_time
                            if existing_kline.close_time == kline.close_time:
                                klines_list[existing_index] = kline
                                saved_count += 1
                                continue
                            else:
                                raise ValidationException(
                                    f"Duplicate kline open_time {kline.open_time} with different close_time",
                                    code="DUPLICATE_KLINE_TIME",
                                )

                        # Insert new kline
                        insert_index = self._find_insert_position(klines_list, kline.open_time)
                        klines_list.insert(insert_index, kline)
                        saved_count += 1

                    # Rebuild time index for this symbol-interval
                    self._rebuild_time_index(symbol, interval)

                    # Clear stats cache
                    self._clear_stats_cache(symbol, interval)

        return saved_count

    async def query(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        options: QueryOptions | None = None,
    ) -> List[Kline]:
        """Query klines within the specified time range for a specific interval."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if start_time >= end_time:
            raise ValidationException("start_time must be before end_time", code="INVALID_TIME_RANGE")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return []

            # Find start and end indices using binary search
            start_index = self._find_start_position(klines_list, start_time)
            end_index = self._find_end_position(klines_list, end_time)

            # Extract klines in range
            result = klines_list[start_index:end_index]

            # Apply query options if provided
            if options:
                result = self._apply_query_options(result, options)

            return result

    async def stream(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        batch_size: int = 1000,
    ) -> AsyncIterator[Kline]:
        """Stream klines within the specified time range for a specific interval."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if batch_size <= 0:
            raise ValidationException("batch_size must be positive", code="INVALID_BATCH_SIZE")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return

            # Find start and end indices
            start_index = self._find_start_position(klines_list, start_time)
            end_index = self._find_end_position(klines_list, end_time)

            # Stream klines in batches
            for i in range(start_index, end_index, batch_size):
                batch_end = min(i + batch_size, end_index)
                batch = klines_list[i:batch_end]

                for kline in batch:
                    yield kline

                # Yield control to allow other coroutines to run
                await asyncio.sleep(0)

    async def get_latest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the most recent kline for the symbol and interval."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return None

            # List is sorted by open_time, so last element is most recent
            return klines_list[-1]

    async def get_oldest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the oldest kline for the symbol and interval."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return None

            # List is sorted by open_time, so first element is oldest
            return klines_list[0]

    async def count(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count klines for the symbol and interval within optional time range."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return 0

            # If no time range specified, return total count
            if start_time is None and end_time is None:
                return len(klines_list)

            # Handle partial ranges
            if start_time is None:
                start_time = klines_list[0].open_time
            if end_time is None:
                end_time = klines_list[-1].open_time

            if start_time >= end_time:
                return 0

            # Find range using binary search
            start_index = self._find_start_position(klines_list, start_time)
            end_index = self._find_end_position(klines_list, end_time)

            return end_index - start_index

    async def delete(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete klines within the specified time range for a specific interval. Returns number deleted."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if start_time >= end_time:
            raise ValidationException("start_time must be before end_time", code="INVALID_TIME_RANGE")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return 0

            # Find range to delete
            start_index = self._find_start_position(klines_list, start_time)
            end_index = self._find_end_position(klines_list, end_time)

            # Delete klines
            deleted_count = end_index - start_index
            if deleted_count > 0:
                del klines_list[start_index:end_index]

                # Rebuild time index
                self._rebuild_time_index(symbol, interval)

                # Clear stats cache
                self._clear_stats_cache(symbol, interval)

            return deleted_count

    async def get_gaps(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> List[tuple[datetime, datetime]]:
        """Find gaps in kline data within the specified time range for a specific interval."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if start_time >= end_time:
            raise ValidationException("start_time must be before end_time", code="INVALID_TIME_RANGE")

        with self._lock:
            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return [(start_time, end_time)]

            gaps = []
            interval_duration = KlineInterval.to_timedelta(interval)

            # Find klines in the specified range
            start_index = self._find_start_position(klines_list, start_time)
            end_index = self._find_end_position(klines_list, end_time)

            range_klines = klines_list[start_index:end_index]

            if not range_klines:
                return [(start_time, end_time)]

            # Check for gap at the beginning
            first_kline = range_klines[0]
            if first_kline.open_time > start_time:
                gaps.append((start_time, first_kline.open_time))

            # Check for gaps between consecutive klines
            for i in range(len(range_klines) - 1):
                current_kline = range_klines[i]
                next_kline = range_klines[i + 1]

                expected_next_time = current_kline.open_time + interval_duration
                if next_kline.open_time > expected_next_time:
                    gaps.append((expected_next_time, next_kline.open_time))

            # Check for gap at the end
            last_kline = range_klines[-1]
            expected_end_time = last_kline.open_time + interval_duration
            if expected_end_time < end_time:
                gaps.append((expected_end_time, end_time))

            return gaps

    async def get_statistics(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> StatisticsResult:
        """Get statistics for klines within optional time range for a specific interval."""
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        with self._lock:
            # Check cache first
            cache_key = f"{start_time}:{end_time}"
            if cache_key in self._stats_cache[symbol][interval]:
                cached_stats = self._stats_cache[symbol][interval][cache_key]
                return cached_stats  # type: ignore[no-any-return]

            klines_list = self._klines[symbol][interval]
            if not klines_list:
                return self._empty_statistics()

            # Get klines in range
            if start_time is None and end_time is None:
                range_klines = klines_list
            else:
                if start_time is None:
                    start_time = klines_list[0].open_time
                if end_time is None:
                    end_time = klines_list[-1].open_time

                start_index = self._find_start_position(klines_list, start_time)
                end_index = self._find_end_position(klines_list, end_time)

                range_klines = klines_list[start_index:end_index]

            if not range_klines:
                return self._empty_statistics()

            # Calculate statistics
            stats = self._calculate_statistics(range_klines)

            # Cache the result
            self._stats_cache[symbol][interval][cache_key] = stats

            return stats

    async def close(self) -> None:
        """Close the repository and clean up resources."""
        with self._lock:
            self._closed = True
            self._klines.clear()
            self._time_index.clear()
            self._stats_cache.clear()

    # Helper methods

    def _find_insert_position(self, klines_list: List[Kline], target_time: datetime) -> int:
        """Find the position where a kline with target_time should be inserted."""
        left, right = 0, len(klines_list)
        while left < right:
            mid = (left + right) // 2
            if klines_list[mid].open_time < target_time:
                left = mid + 1
            else:
                right = mid
        return left

    def _find_start_position(self, klines_list: List[Kline], target_time: datetime) -> int:
        """Find the first position where kline.open_time >= target_time."""
        left, right = 0, len(klines_list)
        while left < right:
            mid = (left + right) // 2
            if klines_list[mid].open_time < target_time:
                left = mid + 1
            else:
                right = mid
        return left

    def _find_end_position(self, klines_list: List[Kline], target_time: datetime) -> int:
        """Find the first position where kline.open_time > target_time."""
        left, right = 0, len(klines_list)
        while left < right:
            mid = (left + right) // 2
            if klines_list[mid].open_time <= target_time:
                left = mid + 1
            else:
                right = mid
        return left

    async def _validate_kline(self, kline: Kline) -> None:
        """Validate a kline object."""
        if not isinstance(kline, Kline):
            raise ValidationException("kline must be a Kline object", code="INVALID_KLINE_TYPE")

    def _rebuild_time_index(self, symbol: str, interval: KlineInterval) -> None:
        """Rebuild the time index for a symbol-interval pair."""
        klines_list = self._klines[symbol][interval]
        time_index = self._time_index[symbol][interval]

        time_index.clear()
        for i, kline in enumerate(klines_list):
            time_index[kline.open_time] = i

    def _clear_stats_cache(self, symbol: str, interval: KlineInterval) -> None:
        """Clear the statistics cache for a symbol-interval pair."""
        if symbol in self._stats_cache and interval in self._stats_cache[symbol]:
            self._stats_cache[symbol][interval].clear()

    def _apply_query_options(self, klines: List[Kline], options: QueryOptions) -> List[Kline]:
        """Apply query options to filter and sort klines."""
        result = klines

        # Apply ordering
        if options.order_by == "timestamp":
            result = sorted(result, key=lambda k: k.open_time, reverse=options.order_desc)
        elif options.order_by == "volume":
            result = sorted(result, key=lambda k: k.volume, reverse=options.order_desc)
        elif options.order_by == "price":
            result = sorted(result, key=lambda k: k.close_price, reverse=options.order_desc)

        # Apply offset and limit
        if options.offset:
            result = result[options.offset :]
        if options.limit:
            result = result[: options.limit]

        return result

    def _empty_statistics(self) -> StatisticsResult:
        """Return empty statistics result."""
        return StatisticsResult(
            count=0,
            min_timestamp=None,
            max_timestamp=None,
            earliest_timestamp=None,
            latest_timestamp=None,
            first_timestamp=None,
            last_timestamp=None,
            storage_size_bytes=0,
            avg_size_bytes=0.0,
            price_statistics={},
            volume_statistics={},
            data_statistics={},
            # Flat fields expected by tests
            volume_total=0.0,
            quote_volume_total=0.0,
            price_high=None,
            price_low=None,
            avg_price=None,
            avg_volume=None,
        )

    def _calculate_statistics(self, klines: List[Kline]) -> StatisticsResult:
        """Calculate statistics for a list of klines."""
        if not klines:
            return self._empty_statistics()

        count = len(klines)
        first_kline = klines[0]
        last_kline = klines[-1]

        # Price statistics
        high_prices = [k.high_price for k in klines]
        low_prices = [k.low_price for k in klines]
        close_prices = [k.close_price for k in klines]

        price_high = max(high_prices)
        price_low = min(low_prices)
        avg_price = sum(close_prices) / count

        # Volume statistics
        volumes = [k.volume for k in klines]
        quote_volumes = [k.quote_volume for k in klines]
        trades_counts = [k.trades_count for k in klines]

        volume_total = sum(volumes)
        quote_volume_total = sum(quote_volumes)
        trades_count_total = sum(trades_counts)
        avg_volume = volume_total / count

        return StatisticsResult(
            count=count,
            min_timestamp=first_kline.open_time,
            max_timestamp=last_kline.open_time,
            earliest_timestamp=first_kline.open_time,
            latest_timestamp=last_kline.open_time,
            first_timestamp=first_kline.open_time,
            last_timestamp=last_kline.open_time,
            storage_size_bytes=0,  # TODO: Calculate actual size
            avg_size_bytes=0.0,  # TODO: Calculate actual size
            price_statistics={
                "high": float(price_high),
                "low": float(price_low),
                "avg": float(avg_price),
            },
            volume_statistics={
                "total": float(volume_total),
                "quote_total": float(quote_volume_total),
                "avg": float(avg_volume),
            },
            data_statistics={
                "trades_count_total": trades_count_total,
            },
            # Flat fields expected by tests
            volume_total=float(volume_total),
            quote_volume_total=float(quote_volume_total),
            price_high=float(price_high),
            price_low=float(price_low),
            avg_price=float(avg_price),
            avg_volume=float(avg_volume),
        )

    # Additional helper methods for testing

    async def get_stored_symbols(self) -> List[str]:
        """Get all symbols stored in the repository."""
        with self._lock:
            return list(self._klines.keys())

    async def get_stored_intervals(self, symbol: str) -> List[KlineInterval]:
        """Get all intervals for a specific symbol."""
        with self._lock:
            if symbol not in self._klines:
                return []
            return list(self._klines[symbol].keys())

    async def clear_all(self) -> None:
        """Clear all data from the repository."""
        with self._lock:
            self._klines.clear()
            self._time_index.clear()
            self._stats_cache.clear()

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            total_klines = 0
            symbols_count = len(self._klines)
            intervals_count = 0

            for symbol, intervals in self._klines.items():
                intervals_count += len(intervals)
                for interval, klines_list in intervals.items():
                    total_klines += len(klines_list)

            return {
                "total_klines": total_klines,
                "symbols_count": symbols_count,
                "intervals_count": intervals_count,
                "max_klines_per_symbol": self.max_klines_per_symbol,
                "is_closed": self._closed,
            }
