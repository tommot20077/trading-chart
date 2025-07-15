# ABOUTME: Unit tests for InMemoryKlineRepository implementation
# ABOUTME: Tests all functionality including storage, querying, statistics, and gap detection

import pytest
from datetime import datetime, timedelta, UTC
from decimal import Decimal

from core.implementations.memory.storage.kline_repository import InMemoryKlineRepository
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval
from core.models.storage.query_option import QueryOptions
from core.exceptions import StorageError, ValidationException


@pytest.fixture
def repository():
    """Create a repository instance for testing."""
    return InMemoryKlineRepository(max_klines_per_symbol=100)


@pytest.fixture
def sample_kline():
    """Create a sample kline for testing."""
    return Kline(
        symbol="BTC/USDT",
        interval=KlineInterval.MINUTE_1,
        open_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        close_time=datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC),
        open_price=Decimal("45000.00"),
        high_price=Decimal("45100.00"),
        low_price=Decimal("44900.00"),
        close_price=Decimal("45050.00"),
        volume=Decimal("100.5"),
        quote_volume=Decimal("4525000.0"),
        trades_count=150,
        taker_buy_volume=Decimal("60.3"),
        taker_buy_quote_volume=Decimal("2715000.0"),
    )


@pytest.fixture
def sample_klines():
    """Create multiple sample klines for testing."""
    klines = []
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    base_price = Decimal("45000.00")

    for i in range(5):
        kline = Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=base_time + timedelta(minutes=i),
            close_time=base_time + timedelta(minutes=i + 1),
            open_price=base_price + Decimal(i * 10),
            high_price=base_price + Decimal(i * 10 + 50),
            low_price=base_price + Decimal(i * 10 - 50),
            close_price=base_price + Decimal(i * 10 + 25),
            volume=Decimal("100.0") + Decimal(i * 10),
            quote_volume=Decimal("4500000.0") + Decimal(i * 450000),
            trades_count=100 + i * 10,
            taker_buy_volume=Decimal("60.0") + Decimal(i * 6),
            taker_buy_quote_volume=Decimal("2700000.0") + Decimal(i * 270000),
        )
        klines.append(kline)

    return klines


class TestInMemoryKlineRepository:
    """Test suite for InMemoryKlineRepository."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repository_initialization(self, repository):
        """Test repository initialization."""
        assert repository.max_klines_per_symbol == 100
        assert not repository._closed
        assert len(repository._klines) == 0

        # Test memory usage
        memory_usage = await repository.get_memory_usage()
        assert memory_usage["total_klines"] == 0
        assert memory_usage["symbols_count"] == 0
        assert memory_usage["max_klines_per_symbol"] == 100
        assert not memory_usage["is_closed"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_single_kline(self, repository, sample_kline):
        """Test saving a single kline."""
        await repository.save(sample_kline)

        # Check if kline was stored
        stored_symbols = await repository.get_stored_symbols()
        assert "BTC/USDT" in stored_symbols

        stored_intervals = await repository.get_stored_intervals("BTC/USDT")
        assert KlineInterval.MINUTE_1 in stored_intervals

        # Check count
        count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_duplicate_kline_same_close_time(self, repository, sample_kline):
        """Test saving duplicate kline with same close time (should update)."""
        await repository.save(sample_kline)

        # Save the same kline again (should update)
        updated_kline = Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=sample_kline.open_time,
            close_time=sample_kline.close_time,
            open_price=Decimal("46000.00"),  # Different price
            high_price=Decimal("46100.00"),
            low_price=Decimal("45900.00"),
            close_price=Decimal("46050.00"),
            volume=sample_kline.volume,
            quote_volume=sample_kline.quote_volume,
            trades_count=sample_kline.trades_count,
            taker_buy_volume=sample_kline.taker_buy_volume,
            taker_buy_quote_volume=sample_kline.taker_buy_quote_volume,
        )

        await repository.save(updated_kline)

        # Should still have only one kline
        count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert count == 1

        # Should be the updated kline
        latest = await repository.get_latest("BTC/USDT", KlineInterval.MINUTE_1)
        assert latest.open_price == Decimal("46000.00")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_duplicate_kline_different_close_time(self, repository, sample_kline):
        """Test saving duplicate kline with different close time (should raise error)."""
        await repository.save(sample_kline)

        # Try to save a kline with same open time but different close time
        duplicate_kline = Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=sample_kline.open_time,
            close_time=sample_kline.close_time + timedelta(minutes=1),  # Different close time
            open_price=sample_kline.open_price,
            high_price=sample_kline.high_price,
            low_price=sample_kline.low_price,
            close_price=sample_kline.close_price,
            volume=sample_kline.volume,
            quote_volume=sample_kline.quote_volume,
            trades_count=sample_kline.trades_count,
            taker_buy_volume=sample_kline.taker_buy_volume,
            taker_buy_quote_volume=sample_kline.taker_buy_quote_volume,
        )

        with pytest.raises(ValidationException) as exc_info:
            await repository.save(duplicate_kline)

        assert exc_info.value.code == "DUPLICATE_KLINE_TIME"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_batch_klines(self, repository, sample_klines):
        """Test saving multiple klines in batch."""
        saved_count = await repository.save_batch(sample_klines)

        assert saved_count == len(sample_klines)

        count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert count == len(sample_klines)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_batch_empty_list(self, repository):
        """Test saving empty batch."""
        saved_count = await repository.save_batch([])
        assert saved_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_batch_with_duplicates_in_batch(self, repository, sample_kline):
        """Test saving batch with duplicates within the batch."""
        duplicate_klines = [sample_kline, sample_kline]

        with pytest.raises(ValidationException) as exc_info:
            await repository.save_batch(duplicate_klines)

        assert exc_info.value.code == "DUPLICATE_KLINE_TIME_IN_BATCH"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_klines_in_range(self, repository, sample_klines):
        """Test querying klines within a time range."""
        await repository.save_batch(sample_klines)

        # Query middle 3 klines
        start_time = sample_klines[1].open_time
        end_time = sample_klines[3].open_time

        results = await repository.query("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert len(results) == 3
        assert results[0].open_time == sample_klines[1].open_time
        assert results[2].open_time == sample_klines[3].open_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_options(self, repository, sample_klines):
        """Test querying with QueryOptions."""
        await repository.save_batch(sample_klines)

        options = QueryOptions(limit=3, offset=1, order_by="timestamp", order_desc=True)

        start_time = sample_klines[0].open_time
        end_time = sample_klines[-1].open_time

        results = await repository.query("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time, options=options)

        assert len(results) == 3
        # Should be in descending order with offset 1
        assert results[0].open_time == sample_klines[3].open_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_invalid_time_range(self, repository):
        """Test querying with invalid time range."""
        start_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)  # Before start

        with pytest.raises(ValidationException) as exc_info:
            await repository.query("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert exc_info.value.code == "INVALID_TIME_RANGE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_klines(self, repository, sample_klines):
        """Test streaming klines."""
        await repository.save_batch(sample_klines)

        start_time = sample_klines[0].open_time
        end_time = sample_klines[-1].open_time

        streamed_klines = []
        async for kline in repository.stream("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time, batch_size=2):
            streamed_klines.append(kline)

        assert len(streamed_klines) == len(sample_klines)
        assert streamed_klines[0].open_time == sample_klines[0].open_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_invalid_batch_size(self, repository):
        """Test streaming with invalid batch size."""
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)

        with pytest.raises(ValidationException) as exc_info:
            async for _ in repository.stream("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time, batch_size=0):
                break

        assert exc_info.value.code == "INVALID_BATCH_SIZE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_latest_kline(self, repository, sample_klines):
        """Test getting the latest kline."""
        await repository.save_batch(sample_klines)

        latest = await repository.get_latest("BTC/USDT", KlineInterval.MINUTE_1)
        assert latest is not None
        assert latest.open_time == sample_klines[-1].open_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_latest_kline_no_data(self, repository):
        """Test getting latest kline when no data exists."""
        latest = await repository.get_latest("BTC/USDT", KlineInterval.MINUTE_1)
        assert latest is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_oldest_kline(self, repository, sample_klines):
        """Test getting the oldest kline."""
        await repository.save_batch(sample_klines)

        oldest = await repository.get_oldest("BTC/USDT", KlineInterval.MINUTE_1)
        assert oldest is not None
        assert oldest.open_time == sample_klines[0].open_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_oldest_kline_no_data(self, repository):
        """Test getting oldest kline when no data exists."""
        oldest = await repository.get_oldest("BTC/USDT", KlineInterval.MINUTE_1)
        assert oldest is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_all_klines(self, repository, sample_klines):
        """Test counting all klines."""
        await repository.save_batch(sample_klines)

        count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert count == len(sample_klines)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_with_time_range(self, repository, sample_klines):
        """Test counting klines within a time range."""
        await repository.save_batch(sample_klines)

        start_time = sample_klines[1].open_time
        end_time = sample_klines[3].open_time

        count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count_invalid_range(self, repository):
        """Test counting with invalid time range."""
        start_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)
        assert count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_klines(self, repository, sample_klines):
        """Test deleting klines within a time range."""
        await repository.save_batch(sample_klines)

        # Delete middle 3 klines
        start_time = sample_klines[1].open_time
        end_time = sample_klines[3].open_time

        deleted_count = await repository.delete("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert deleted_count == 3

        # Check remaining count
        remaining_count = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert remaining_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_invalid_time_range(self, repository):
        """Test deleting with invalid time range."""
        start_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(ValidationException) as exc_info:
            await repository.delete("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert exc_info.value.code == "INVALID_TIME_RANGE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_gaps_no_data(self, repository):
        """Test getting gaps when no data exists."""
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)

        gaps = await repository.get_gaps("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert len(gaps) == 1
        assert gaps[0] == (start_time, end_time)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_gaps_with_missing_data(self, repository):
        """Test getting gaps when data has missing intervals."""
        # Create klines with gaps
        klines = []
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # First kline
        klines.append(
            Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time,
                close_time=base_time + timedelta(minutes=1),
                open_price=Decimal("45000.00"),
                high_price=Decimal("45100.00"),
                low_price=Decimal("44900.00"),
                close_price=Decimal("45050.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("4500000.0"),
                trades_count=100,
                taker_buy_volume=Decimal("60.0"),
                taker_buy_quote_volume=Decimal("2700000.0"),
            )
        )

        # Third kline (missing second kline)
        klines.append(
            Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time + timedelta(minutes=2),
                close_time=base_time + timedelta(minutes=3),
                open_price=Decimal("45020.00"),
                high_price=Decimal("45120.00"),
                low_price=Decimal("44920.00"),
                close_price=Decimal("45070.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("4500000.0"),
                trades_count=100,
                taker_buy_volume=Decimal("60.0"),
                taker_buy_quote_volume=Decimal("2700000.0"),
            )
        )

        await repository.save_batch(klines)

        start_time = base_time
        end_time = base_time + timedelta(minutes=5)

        gaps = await repository.get_gaps("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        # Should have gap for minute 1 and gap from minute 3 to minute 5
        assert len(gaps) == 2
        assert gaps[0] == (base_time + timedelta(minutes=1), base_time + timedelta(minutes=2))
        assert gaps[1] == (base_time + timedelta(minutes=3), base_time + timedelta(minutes=5))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_statistics(self, repository, sample_klines):
        """Test getting statistics."""
        await repository.save_batch(sample_klines)

        stats = await repository.get_statistics("BTC/USDT", KlineInterval.MINUTE_1)

        assert stats["count"] == len(sample_klines)
        assert stats["first_timestamp"] == sample_klines[0].open_time
        assert stats["last_timestamp"] == sample_klines[-1].open_time
        assert stats["price_high"] == Decimal("45090.00")  # Highest high price
        assert stats["price_low"] == Decimal("44950.00")  # Lowest low price
        assert stats["volume_total"] > 0
        assert stats["avg_price"] > 0
        assert stats["avg_volume"] > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, repository):
        """Test getting statistics when no data exists."""
        stats = await repository.get_statistics("BTC/USDT", KlineInterval.MINUTE_1)

        assert stats["count"] == 0
        assert stats["first_timestamp"] is None
        assert stats["last_timestamp"] is None
        assert stats["price_high"] is None
        assert stats["price_low"] is None
        assert stats["volume_total"] == 0
        assert stats["avg_price"] is None
        assert stats["avg_volume"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_statistics_with_time_range(self, repository, sample_klines):
        """Test getting statistics within a time range."""
        await repository.save_batch(sample_klines)

        start_time = sample_klines[1].open_time
        end_time = sample_klines[3].open_time

        stats = await repository.get_statistics("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert stats["count"] == 3
        assert stats["first_timestamp"] == sample_klines[1].open_time
        assert stats["last_timestamp"] == sample_klines[3].open_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_storage_limit_exceeded(self, repository):
        """Test storage limit exceeded."""
        # Create more klines than the limit
        klines = []
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        for i in range(repository.max_klines_per_symbol + 1):
            kline = Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time + timedelta(minutes=i),
                close_time=base_time + timedelta(minutes=i + 1),
                open_price=Decimal("45000.00"),
                high_price=Decimal("45100.00"),
                low_price=Decimal("44900.00"),
                close_price=Decimal("45050.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("4500000.0"),
                trades_count=100,
                taker_buy_volume=Decimal("60.0"),
                taker_buy_quote_volume=Decimal("2700000.0"),
            )
            klines.append(kline)

        with pytest.raises(StorageError) as exc_info:
            await repository.save_batch(klines)

        assert exc_info.value.code == "STORAGE_LIMIT_EXCEEDED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repository_closed_operations(self, repository, sample_kline):
        """Test operations on closed repository."""
        await repository.close()

        # Test various operations fail when closed
        with pytest.raises(StorageError) as exc_info:
            await repository.save(sample_kline)
        assert exc_info.value.code == "REPOSITORY_CLOSED"

        with pytest.raises(StorageError) as exc_info:
            await repository.save_batch([sample_kline])
        assert exc_info.value.code == "REPOSITORY_CLOSED"

        with pytest.raises(StorageError) as exc_info:
            await repository.query("BTC/USDT", KlineInterval.MINUTE_1, datetime.now(UTC), datetime.now(UTC))
        assert exc_info.value.code == "REPOSITORY_CLOSED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self, repository, sample_kline):
        """Test repository as async context manager."""
        async with repository as repo:
            await repo.save(sample_kline)
            count = await repo.count("BTC/USDT", KlineInterval.MINUTE_1)
            assert count == 1

        # Repository should be closed
        assert repository._closed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_all(self, repository, sample_klines):
        """Test clearing all data."""
        await repository.save_batch(sample_klines)

        count_before = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert count_before > 0

        await repository.clear_all()

        count_after = await repository.count("BTC/USDT", KlineInterval.MINUTE_1)
        assert count_after == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_memory_usage(self, repository, sample_klines):
        """Test getting memory usage statistics."""
        await repository.save_batch(sample_klines)

        usage = await repository.get_memory_usage()

        assert usage["total_klines"] == len(sample_klines)
        assert usage["symbols_count"] == 1
        assert usage["intervals_count"] == 1
        assert usage["max_klines_per_symbol"] == repository.max_klines_per_symbol
        assert not usage["is_closed"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stored_symbols_and_intervals(self, repository, sample_klines):
        """Test getting stored symbols and intervals."""
        await repository.save_batch(sample_klines)

        symbols = await repository.get_stored_symbols()
        assert "BTC/USDT" in symbols

        intervals = await repository.get_stored_intervals("BTC/USDT")
        assert KlineInterval.MINUTE_1 in intervals

        # Test non-existent symbol
        intervals_empty = await repository.get_stored_intervals("ETH/USDT")
        assert intervals_empty == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_statistics_caching(self, repository, sample_klines):
        """Test that statistics are cached properly."""
        await repository.save_batch(sample_klines)

        # First call should calculate and cache
        stats1 = await repository.get_statistics("BTC/USDT", KlineInterval.MINUTE_1)

        # Second call should return cached result
        stats2 = await repository.get_statistics("BTC/USDT", KlineInterval.MINUTE_1)

        assert stats1 == stats2

        # Add new kline should clear cache
        new_kline = Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 10, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 11, 0, tzinfo=UTC),
            open_price=Decimal("46000.00"),
            high_price=Decimal("46100.00"),
            low_price=Decimal("45900.00"),
            close_price=Decimal("46050.00"),
            volume=Decimal("100.0"),
            quote_volume=Decimal("4600000.0"),
            trades_count=100,
            taker_buy_volume=Decimal("60.0"),
            taker_buy_quote_volume=Decimal("2760000.0"),
        )

        await repository.save(new_kline)

        # Stats should be recalculated
        stats3 = await repository.get_statistics("BTC/USDT", KlineInterval.MINUTE_1)
        assert stats3["count"] == stats1["count"] + 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_kline_type(self, repository):
        """Test saving invalid kline type."""
        with pytest.raises(ValidationException) as exc_info:
            await repository.save("not_a_kline")

        assert exc_info.value.code == "INVALID_KLINE_TYPE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_time_index_integrity(self, repository, sample_klines):
        """Test that time index is maintained correctly."""
        await repository.save_batch(sample_klines)

        # Test that time index is consistent
        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1

        klines_list = repository._klines[symbol][interval]
        time_index = repository._time_index[symbol][interval]

        # Check that every kline has correct index
        for i, kline in enumerate(klines_list):
            assert time_index[kline.open_time] == i

        # Check that time index size matches klines list size
        assert len(time_index) == len(klines_list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_binary_search_efficiency(self, repository):
        """Test that binary search is working correctly for large datasets."""
        # Create a large dataset
        klines = []
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

        for i in range(1000):
            kline = Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time + timedelta(minutes=i),
                close_time=base_time + timedelta(minutes=i + 1),
                open_price=Decimal("45000.00"),
                high_price=Decimal("45100.00"),
                low_price=Decimal("44900.00"),
                close_price=Decimal("45050.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("4500000.0"),
                trades_count=100,
                taker_buy_volume=Decimal("60.0"),
                taker_buy_quote_volume=Decimal("2700000.0"),
            )
            klines.append(kline)

        # Save in batches to avoid storage limit
        repository.max_klines_per_symbol = 2000
        await repository.save_batch(klines)

        # Query a small range in the middle
        start_time = base_time + timedelta(minutes=500)
        end_time = base_time + timedelta(minutes=510)

        results = await repository.query("BTC/USDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert len(results) == 11  # Inclusive range: 500, 501, ..., 510
        assert results[0].open_time == start_time
        assert results[-1].open_time == base_time + timedelta(minutes=510)
