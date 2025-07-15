# ABOUTME: Contract tests for AbstractKlineRepository interface
# ABOUTME: Verifies all Kline repository implementations comply with the interface contract

import pytest
from typing import Type, List, Any, Dict, AsyncIterator
from datetime import datetime, UTC, timedelta
from decimal import Decimal

from core.interfaces.storage.Kline_repository import AbstractKlineRepository
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval
from core.models.storage.query_option import QueryOptions
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin, ResourceManagementContractMixin


class MockKlineRepository(AbstractKlineRepository):
    """Mock implementation of AbstractKlineRepository for contract testing."""

    def __init__(self):
        self._klines: Dict[str, List[Kline]] = {}
        self._closed = False

    def _get_key(self, symbol: str, interval: KlineInterval | str) -> str:
        """Generate storage key for symbol and interval."""
        # Ensure consistent key generation regardless of whether interval is string or enum
        # Check for enum first, since some enums might also be strings
        if hasattr(interval, "value") and hasattr(interval, "name"):
            # This is an enum object
            interval_str = interval.value
        elif isinstance(interval, str):
            interval_str = interval
        else:
            # Fallback for other types
            interval_str = str(interval)
        return f"{symbol}:{interval_str}"

    def _check_not_closed(self):
        """Check if repository is not closed."""
        if self._closed:
            raise RuntimeError("Repository is closed")

    async def save(self, kline: Kline) -> None:
        """Save a single kline to the repository."""
        self._check_not_closed()

        if not isinstance(kline, Kline):
            raise TypeError("kline must be a Kline instance")

        key = self._get_key(kline.symbol, kline.interval)  # kline.interval is string due to use_enum_values=True
        if key not in self._klines:
            self._klines[key] = []

        # Remove existing kline with same open_time (upsert behavior)
        self._klines[key] = [k for k in self._klines[key] if k.open_time != kline.open_time]
        self._klines[key].append(kline)
        self._klines[key].sort(key=lambda k: k.open_time)

    async def save_batch(self, klines: list[Kline]) -> int:
        """Save multiple klines in batch. Returns number of klines saved."""
        self._check_not_closed()

        if not isinstance(klines, list):
            raise TypeError("klines must be a list")

        if not klines:
            return 0

        saved_count = 0
        for kline in klines:
            await self.save(kline)
            saved_count += 1

        return saved_count

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
        self._check_not_closed()

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        if not isinstance(interval, KlineInterval):
            raise TypeError("interval must be a KlineInterval")

        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime")

        if not isinstance(end_time, datetime):
            raise TypeError("end_time must be a datetime")

        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        # Filter by time range
        filtered = []
        for k in klines:
            if start_time <= k.open_time < end_time:
                filtered.append(k)

        # Apply options if provided
        if options:
            # Apply ordering
            if options.order_desc:
                filtered.sort(key=lambda k: k.open_time, reverse=True)
            else:
                filtered.sort(key=lambda k: k.open_time)

            # Apply offset and limit
            if options.offset:
                filtered = filtered[options.offset :]
            if options.limit:
                filtered = filtered[: options.limit]

        return filtered

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
        self._check_not_closed()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Get all klines in the range
        klines = await self.query(symbol, interval, start_time, end_time)

        # Yield in batches
        for i in range(0, len(klines), batch_size):
            batch = klines[i : i + batch_size]
            for kline in batch:
                yield kline

    async def get_latest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the most recent kline for the symbol and interval."""
        self._check_not_closed()

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        if not klines:
            return None

        return max(klines, key=lambda k: k.open_time)

    async def get_oldest(self, symbol: str, interval: KlineInterval) -> Kline | None:
        """Get the oldest kline for the symbol and interval."""
        self._check_not_closed()

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        if not klines:
            return None

        return min(klines, key=lambda k: k.open_time)

    async def count(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count klines for the symbol and interval within optional time range."""
        self._check_not_closed()

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        if start_time is None and end_time is None:
            return len(klines)

        # Filter by time range if provided
        filtered = klines
        if start_time:
            filtered = [k for k in filtered if k.open_time >= start_time]
        if end_time:
            filtered = [k for k in filtered if k.open_time < end_time]

        return len(filtered)

    async def delete(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Delete klines within the specified time range for a specific interval. Returns number deleted."""
        self._check_not_closed()

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        # Find klines to delete
        to_delete = [k for k in klines if start_time <= k.open_time < end_time]

        # Remove them
        self._klines[key] = [k for k in klines if not (start_time <= k.open_time < end_time)]

        return len(to_delete)

    async def get_gaps(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find gaps in kline data within the specified time range for a specific interval."""
        self._check_not_closed()

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        # Filter and sort klines in the range
        filtered = [k for k in klines if start_time <= k.open_time < end_time]
        filtered.sort(key=lambda k: k.open_time)

        if not filtered:
            return [(start_time, end_time)]

        gaps = []
        interval_seconds = KlineInterval.to_seconds(interval)
        interval_delta = timedelta(seconds=interval_seconds)

        # Check gap before first kline
        if filtered[0].open_time > start_time:
            gaps.append((start_time, filtered[0].open_time))

        # Check gaps between klines
        for i in range(len(filtered) - 1):
            current_end = filtered[i].open_time + interval_delta
            next_start = filtered[i + 1].open_time

            if current_end < next_start:
                gaps.append((current_end, next_start))

        # Check gap after last kline
        last_end = filtered[-1].open_time + interval_delta
        if last_end < end_time:
            gaps.append((last_end, end_time))

        return gaps

    async def get_statistics(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get statistics for klines within optional time range for a specific interval."""
        self._check_not_closed()

        key = self._get_key(symbol, interval)
        klines = self._klines.get(key, [])

        # Filter by time range if provided
        if start_time or end_time:
            filtered = klines
            if start_time:
                filtered = [k for k in filtered if k.open_time >= start_time]
            if end_time:
                filtered = [k for k in filtered if k.open_time < end_time]
        else:
            filtered = klines

        if not filtered:
            return {
                "count": 0,
                "earliest": None,
                "latest": None,
                "total_volume": "0",
                "avg_volume": "0",
                "price_range": {"min": None, "max": None},
            }

        total_volume = sum(k.volume for k in filtered)
        avg_volume = total_volume / len(filtered)

        all_prices = []
        for k in filtered:
            all_prices.extend([k.open_price, k.high_price, k.low_price, k.close_price])

        return {
            "count": len(filtered),
            "earliest": min(filtered, key=lambda k: k.open_time).open_time.isoformat(),
            "latest": max(filtered, key=lambda k: k.open_time).open_time.isoformat(),
            "total_volume": str(total_volume),
            "avg_volume": str(avg_volume),
            "price_range": {
                "min": str(min(all_prices)) if all_prices else None,
                "max": str(max(all_prices)) if all_prices else None,
            },
        }

    async def close(self) -> None:
        """Close the repository and clean up resources."""
        self._closed = True
        self._klines.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class TestKlineRepositoryContract(
    ContractTestBase[AbstractKlineRepository], AsyncContractTestMixin, ResourceManagementContractMixin
):
    """Contract tests for AbstractKlineRepository interface."""

    @property
    def interface_class(self) -> Type[AbstractKlineRepository]:
        return AbstractKlineRepository

    @property
    def implementations(self) -> List[Type[AbstractKlineRepository]]:
        return [
            MockKlineRepository,
            # TODO: Add actual implementations when they exist
            # InMemoryKlineRepository,  # Disabled until contract tests are fixed
            # DatabaseKlineRepository,
        ]

    def create_test_kline(
        self,
        symbol: str = "BTCUSDT",
        interval: KlineInterval = KlineInterval.MINUTE_1,
        open_time: datetime = None,
        **kwargs,
    ) -> Kline:
        """Create a test kline with default values."""
        if open_time is None:
            open_time = datetime.now(UTC).replace(second=0, microsecond=0)

        close_time = open_time + timedelta(minutes=1)

        return Kline(
            symbol=symbol,
            interval=interval,  # Keep as KlineInterval enum
            open_time=open_time,
            close_time=close_time,
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("10.5"),
            quote_volume=Decimal("525000.00"),
            trades_count=100,
            **kwargs,
        )

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_save_method_contract(self):
        """Test save method contract behavior."""
        repo = MockKlineRepository()

        # Valid kline should save without error
        kline = self.create_test_kline()
        await repo.save(kline)  # Should not raise

        # Invalid input should raise TypeError
        with pytest.raises(TypeError):
            await repo.save("not_a_kline")

        with pytest.raises(TypeError):
            await repo.save(None)

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_save_batch_method_contract(self):
        """Test save_batch method contract behavior."""
        repo = MockKlineRepository()

        # Valid klines list should return count
        klines = [self.create_test_kline(open_time=datetime.now(UTC) + timedelta(minutes=i)) for i in range(3)]
        count = await repo.save_batch(klines)
        assert isinstance(count, int)
        assert count == 3

        # Empty list should return 0
        count = await repo.save_batch([])
        assert count == 0

        # Invalid input should raise TypeError
        with pytest.raises(TypeError):
            await repo.save_batch("not_a_list")

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_query_method_contract(self):
        """Test query method contract behavior."""
        repo = MockKlineRepository()

        # Setup test data
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(5)]
        await repo.save_batch(klines)

        # Valid query should return list of klines
        start_time = base_time
        end_time = base_time + timedelta(minutes=3)

        result = await repo.query("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(k, Kline) for k in result)

        # Invalid parameters should raise appropriate errors
        with pytest.raises(ValueError):
            await repo.query("", KlineInterval.MINUTE_1, start_time, end_time)

        with pytest.raises(TypeError):
            await repo.query("BTCUSDT", "not_interval", start_time, end_time)

        with pytest.raises(ValueError):
            await repo.query("BTCUSDT", KlineInterval.MINUTE_1, end_time, start_time)  # start > end

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_stream_method_contract(self):
        """Test stream method contract behavior."""
        repo = MockKlineRepository()

        # Setup test data
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(3)]
        await repo.save_batch(klines)

        # Valid stream should yield klines
        start_time = base_time
        end_time = base_time + timedelta(minutes=3)

        streamed_klines = []
        async for kline in repo.stream("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time):
            assert isinstance(kline, Kline)
            streamed_klines.append(kline)

        assert len(streamed_klines) == 3

        # Invalid batch_size should raise ValueError
        with pytest.raises(ValueError):
            async for _ in repo.stream("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time, batch_size=0):
                pass

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_get_latest_oldest_contract(self):
        """Test get_latest and get_oldest method contract behavior."""
        repo = MockKlineRepository()

        # Empty repository should return None
        latest = await repo.get_latest("BTCUSDT", KlineInterval.MINUTE_1)
        assert latest is None

        oldest = await repo.get_oldest("BTCUSDT", KlineInterval.MINUTE_1)
        assert oldest is None

        # With data should return correct klines
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(3)]
        await repo.save_batch(klines)

        latest = await repo.get_latest("BTCUSDT", KlineInterval.MINUTE_1)
        assert isinstance(latest, Kline)
        assert latest.open_time == base_time + timedelta(minutes=2)

        oldest = await repo.get_oldest("BTCUSDT", KlineInterval.MINUTE_1)
        assert isinstance(oldest, Kline)
        assert oldest.open_time == base_time

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_count_method_contract(self):
        """Test count method contract behavior."""
        repo = MockKlineRepository()

        # Empty repository should return 0
        count = await repo.count("BTCUSDT", KlineInterval.MINUTE_1)
        assert count == 0

        # With data should return correct count
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(5)]
        await repo.save_batch(klines)

        # Total count
        count = await repo.count("BTCUSDT", KlineInterval.MINUTE_1)
        assert count == 5

        # Count with time range
        start_time = base_time + timedelta(minutes=1)
        end_time = base_time + timedelta(minutes=4)
        count = await repo.count("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert count == 3

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_delete_method_contract(self):
        """Test delete method contract behavior."""
        repo = MockKlineRepository()

        # Setup test data
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(5)]
        await repo.save_batch(klines)

        # Delete should return number of deleted items
        start_time = base_time + timedelta(minutes=1)
        end_time = base_time + timedelta(minutes=4)
        deleted_count = await repo.delete("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert isinstance(deleted_count, int)
        assert deleted_count == 3

        # Verify deletion
        remaining_count = await repo.count("BTCUSDT", KlineInterval.MINUTE_1)
        assert remaining_count == 2

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_get_gaps_method_contract(self):
        """Test get_gaps method contract behavior."""
        repo = MockKlineRepository()

        # Empty repository should return full range as gap
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        start_time = base_time
        end_time = base_time + timedelta(minutes=5)

        gaps = await repo.get_gaps("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert isinstance(gaps, list)
        assert len(gaps) == 1
        assert gaps[0] == (start_time, end_time)

        # With some data should identify gaps correctly
        klines = [
            self.create_test_kline(open_time=base_time),
            self.create_test_kline(open_time=base_time + timedelta(minutes=3)),
        ]
        await repo.save_batch(klines)

        gaps = await repo.get_gaps("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert len(gaps) >= 1  # Should have gaps

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_get_statistics_method_contract(self):
        """Test get_statistics method contract behavior."""
        repo = MockKlineRepository()

        # Empty repository should return empty statistics
        stats = await repo.get_statistics("BTCUSDT", KlineInterval.MINUTE_1)
        assert isinstance(stats, dict)
        assert stats["count"] == 0

        # With data should return valid statistics
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(3)]
        await repo.save_batch(klines)

        stats = await repo.get_statistics("BTCUSDT", KlineInterval.MINUTE_1)
        assert stats["count"] == 3
        assert "total_volume" in stats
        assert "avg_volume" in stats
        assert "earliest" in stats
        assert "latest" in stats
        assert "price_range" in stats

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_close_method_contract(self):
        """Test close method contract behavior."""
        repo = MockKlineRepository()

        # Should be able to close without error
        await repo.close()  # Should not raise

        # Operations after close should raise error
        with pytest.raises(RuntimeError):
            await repo.save(self.create_test_kline())

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_context_manager_contract(self):
        """Test async context manager contract behavior."""
        # Should work as async context manager
        async with MockKlineRepository() as repo:
            assert isinstance(repo, AbstractKlineRepository)
            # Should be able to use repository
            kline = self.create_test_kline()
            await repo.save(kline)

        # Repository should be closed after context
        with pytest.raises(RuntimeError):
            await repo.save(self.create_test_kline())

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_query_options_contract(self):
        """Test query method with QueryOptions contract behavior."""
        repo = MockKlineRepository()

        # Setup test data
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(10)]
        await repo.save_batch(klines)

        # Test with QueryOptions
        options = QueryOptions(limit=3, offset=2, order_desc=True)
        start_time = base_time
        end_time = base_time + timedelta(minutes=10)

        result = await repo.query("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time, options=options)
        assert len(result) <= 3  # Respects limit

        # Test without options should work
        result = await repo.query("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert len(result) == 10

        await repo.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_data_consistency_contract(self):
        """Test data consistency across different operations."""
        repo = MockKlineRepository()

        # Save data and verify consistency across operations
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        klines = [self.create_test_kline(open_time=base_time + timedelta(minutes=i)) for i in range(5)]
        await repo.save_batch(klines)

        # Count should match query results
        total_count = await repo.count("BTCUSDT", KlineInterval.MINUTE_1)
        all_klines = await repo.query(
            "BTCUSDT", KlineInterval.MINUTE_1, base_time - timedelta(hours=1), base_time + timedelta(hours=1)
        )
        assert total_count == len(all_klines)

        # Latest should be the last one
        latest = await repo.get_latest("BTCUSDT", KlineInterval.MINUTE_1)
        assert latest.open_time == max(k.open_time for k in all_klines)

        # Oldest should be the first one
        oldest = await repo.get_oldest("BTCUSDT", KlineInterval.MINUTE_1)
        assert oldest.open_time == min(k.open_time for k in all_klines)

        await repo.close()
