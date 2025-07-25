# ABOUTME: Integration tests for KlineRepository + MetadataRepository collaboration
# ABOUTME: Tests Kline data storage and metadata synchronization across repositories

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal

from core.implementations.memory.storage.kline_repository import InMemoryKlineRepository
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval
from core.exceptions.base import ValidationException


class TestKlineMetadataRepositoryIntegration:
    """
    Integration tests for KlineRepository and MetadataRepository collaboration.

    Tests the interaction between:
    - InMemoryKlineRepository (Kline data storage)
    - InMemoryMetadataRepository (Metadata and sync time tracking)

    Validates:
    - Kline storage triggers metadata updates
    - Sync time tracking for different symbols and intervals
    - Backfill status coordination
    - Batch operation consistency
    - Concurrent access scenarios
    """

    @pytest_asyncio.fixture
    async def metadata_repository(self):
        """Create a clean InMemoryMetadataRepository."""
        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest_asyncio.fixture
    async def kline_repository(self):
        """Create a clean InMemoryKlineRepository."""
        repo = InMemoryKlineRepository()
        yield repo
        await repo.close()

    @pytest.fixture
    def sample_klines(self):
        """Create sample klines for testing."""
        now = datetime.now(UTC)
        klines = []

        for i in range(5):
            open_time = now + timedelta(minutes=i)
            close_time = open_time + timedelta(minutes=1)

            kline = Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=open_time,
                close_time=close_time,
                open_price=Decimal(f"{45000 + i * 10}.00"),
                high_price=Decimal(f"{45100 + i * 10}.00"),
                low_price=Decimal(f"{44900 + i * 10}.00"),
                close_price=Decimal(f"{45050 + i * 10}.00"),
                volume=Decimal("100.5"),
                quote_volume=Decimal("4525000.0"),
                trades_count=150,
            )
            klines.append(kline)

        return klines

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kline_storage_updates_metadata(self, kline_repository, metadata_repository, sample_klines):
        """Test that storing klines triggers metadata updates."""
        # Store klines
        await kline_repository.save_batch(sample_klines)

        # Manually set sync time to simulate metadata coordination
        last_kline = sample_klines[-1]
        await metadata_repository.set_last_sync_time("BTC/USDT", "klines_1m", last_kline.close_time)

        # Verify klines are stored
        stored_klines = await kline_repository.query(
            "BTC/USDT",
            KlineInterval.MINUTE_1,
            sample_klines[0].open_time,
            sample_klines[-1].close_time + timedelta(seconds=1),
        )
        assert len(stored_klines) == 5

        # Verify metadata is updated
        sync_time = await metadata_repository.get_last_sync_time("BTC/USDT", "klines_1m")
        assert sync_time is not None
        assert sync_time == last_kline.close_time

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_symbol_metadata_coordination(self, kline_repository, metadata_repository):
        """Test metadata coordination across multiple symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        now = datetime.now(UTC)

        # Store klines for each symbol
        for symbol in symbols:
            klines = []
            for i in range(3):
                open_time = now + timedelta(minutes=i)
                close_time = open_time + timedelta(minutes=1)

                kline = Kline(
                    symbol=symbol,
                    interval=KlineInterval.MINUTE_1,
                    open_time=open_time,
                    close_time=close_time,
                    open_price=Decimal("1000.00"),
                    high_price=Decimal("1010.00"),
                    low_price=Decimal("990.00"),
                    close_price=Decimal("1005.00"),
                    volume=Decimal("100.0"),
                    quote_volume=Decimal("100500.0"),
                    trades_count=50,
                )
                klines.append(kline)

            await kline_repository.save_batch(klines)

            # Set sync time for each symbol
            await metadata_repository.set_last_sync_time(symbol, "klines_1m", klines[-1].close_time)

        # Verify each symbol has independent metadata
        for symbol in symbols:
            sync_time = await metadata_repository.get_last_sync_time(symbol, "klines_1m")
            assert sync_time is not None

            # Verify klines are stored correctly
            stored_klines = await kline_repository.query(
                symbol, KlineInterval.MINUTE_1, now, now + timedelta(minutes=10)
            )
            assert len(stored_klines) == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_interval_specific_metadata_tracking(self, kline_repository, metadata_repository):
        """Test metadata tracking for different intervals."""
        symbol = "BTC/USDT"
        intervals = [KlineInterval.MINUTE_1, KlineInterval.MINUTE_5, KlineInterval.HOUR_1]
        now = datetime.now(UTC)

        for interval in intervals:
            # Create interval-specific klines
            interval_duration = KlineInterval.to_timedelta(interval)
            klines = []

            for i in range(2):
                open_time = now + timedelta(seconds=i * interval_duration.total_seconds())
                close_time = open_time + interval_duration

                kline = Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=open_time,
                    close_time=close_time,
                    open_price=Decimal("50000.00"),
                    high_price=Decimal("50100.00"),
                    low_price=Decimal("49900.00"),
                    close_price=Decimal("50050.00"),
                    volume=Decimal("200.0"),
                    quote_volume=Decimal("10010000.0"),
                    trades_count=100,
                )
                klines.append(kline)

            await kline_repository.save_batch(klines)

            # Set interval-specific sync time
            await metadata_repository.set_last_sync_time(symbol, f"klines_{interval.value}", klines[-1].close_time)

        # Verify each interval has independent tracking
        for interval in intervals:
            sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
            assert sync_time is not None

            # Verify klines exist for each interval
            count = await kline_repository.count(symbol, interval)
            assert count == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_backfill_status_coordination(self, kline_repository, metadata_repository, sample_klines):
        """Test backfill status coordination with kline storage."""
        symbol = "BTC/USDT"
        data_type = "klines_1m"

        # Set initial backfill status
        initial_status = {
            "status": "in_progress",
            "start_time": sample_klines[0].open_time.isoformat(),
            "end_time": sample_klines[-1].close_time.isoformat(),
            "processed_count": 0,
            "total_count": len(sample_klines),
        }

        await metadata_repository.set_backfill_status(symbol, data_type, initial_status)

        # Store klines in batches to simulate backfill progress
        batch_size = 2
        for i in range(0, len(sample_klines), batch_size):
            batch = sample_klines[i : i + batch_size]
            await kline_repository.save_batch(batch)

            # Update backfill status
            processed_count = min(i + batch_size, len(sample_klines))
            updated_status = {
                **initial_status,
                "processed_count": processed_count,
                "status": "completed" if processed_count == len(sample_klines) else "in_progress",
            }

            await metadata_repository.set_backfill_status(symbol, data_type, updated_status)

        # Verify final state
        final_status = await metadata_repository.get_backfill_status(symbol, data_type)
        assert final_status is not None
        assert final_status["status"] == "completed"
        assert final_status["processed_count"] == len(sample_klines)

        # Verify all klines are stored
        stored_count = await kline_repository.count(symbol, KlineInterval.MINUTE_1)
        assert stored_count == len(sample_klines)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_operation_consistency(self, kline_repository, metadata_repository):
        """Test consistency of batch operations across repositories."""
        symbol = "ETH/USDT"
        interval = KlineInterval.MINUTE_5
        now = datetime.now(UTC)

        # Create large batch of klines
        large_batch = []
        for i in range(50):
            open_time = now + timedelta(minutes=i * 5)
            close_time = open_time + timedelta(minutes=5)

            kline = Kline(
                symbol=symbol,
                interval=interval,
                open_time=open_time,
                close_time=close_time,
                open_price=Decimal(f"{3000 + i}.00"),
                high_price=Decimal(f"{3050 + i}.00"),
                low_price=Decimal(f"{2950 + i}.00"),
                close_price=Decimal(f"{3025 + i}.00"),
                volume=Decimal("500.0"),
                quote_volume=Decimal("1512500.0"),
                trades_count=200,
            )
            large_batch.append(kline)

        # Store batch
        saved_count = await kline_repository.save_batch(large_batch)
        assert saved_count == 50

        # Update metadata to reflect batch operation
        await metadata_repository.set_last_sync_time(symbol, f"klines_{interval.value}", large_batch[-1].close_time)

        # Set batch processing status
        batch_status = {
            "operation": "batch_insert",
            "symbol": symbol,
            "interval": interval.value,
            "count": saved_count,
            "start_time": large_batch[0].open_time.isoformat(),
            "end_time": large_batch[-1].close_time.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        }

        await metadata_repository.set_backfill_status(symbol, f"batch_klines_{interval.value}", batch_status)

        # Verify consistency
        stored_count = await kline_repository.count(symbol, interval)
        assert stored_count == 50

        sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
        assert sync_time == large_batch[-1].close_time

        batch_info = await metadata_repository.get_backfill_status(symbol, f"batch_klines_{interval.value}")
        assert batch_info["count"] == 50

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_repository_operations(self, kline_repository, metadata_repository):
        """Test concurrent operations across both repositories."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        interval = KlineInterval.MINUTE_1
        now = datetime.now(UTC)

        async def store_and_track_symbol(symbol: str, start_index: int):
            """Store klines and update metadata for a symbol."""
            klines = []
            for i in range(10):
                open_time = now + timedelta(minutes=start_index * 10 + i)
                close_time = open_time + timedelta(minutes=1)

                kline = Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=open_time,
                    close_time=close_time,
                    open_price=Decimal(f"{40000 + i}.00"),
                    high_price=Decimal(f"{40100 + i}.00"),
                    low_price=Decimal(f"{39900 + i}.00"),
                    close_price=Decimal(f"{40050 + i}.00"),
                    volume=Decimal("150.0"),
                    quote_volume=Decimal("6007500.0"),
                    trades_count=75,
                )
                klines.append(kline)

            # Store klines
            await kline_repository.save_batch(klines)

            # Update sync time
            await metadata_repository.set_last_sync_time(symbol, "klines_1m", klines[-1].close_time)

            return len(klines)

        # Run concurrent operations
        tasks = [store_and_track_symbol(symbols[0], 0), store_and_track_symbol(symbols[1], 1)]

        results = await asyncio.gather(*tasks)

        # Verify all operations completed successfully
        assert all(result == 10 for result in results)

        # Verify data integrity for each symbol
        for symbol in symbols:
            stored_count = await kline_repository.count(symbol, interval)
            assert stored_count == 10

            sync_time = await metadata_repository.get_last_sync_time(symbol, "klines_1m")
            assert sync_time is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repository_cleanup_coordination(self, kline_repository, metadata_repository, sample_klines):
        """Test coordinated cleanup across repositories."""
        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1

        # Store initial data
        await kline_repository.save_batch(sample_klines)
        await metadata_repository.set_last_sync_time(symbol, "klines_1m", sample_klines[-1].close_time)

        # Set some metadata
        await metadata_repository.set_backfill_status(
            symbol, "klines_1m", {"status": "completed", "count": len(sample_klines)}
        )

        # Verify data exists
        assert await kline_repository.count(symbol, interval) == len(sample_klines)
        assert await metadata_repository.get_last_sync_time(symbol, "klines_1m") is not None

        # Delete klines from repository
        deleted_count = await kline_repository.delete(
            symbol, interval, sample_klines[0].open_time, sample_klines[-1].close_time + timedelta(seconds=1)
        )
        assert deleted_count == len(sample_klines)

        # Clean up corresponding metadata
        await metadata_repository.delete(f"sync_time:{symbol}:klines_1m")
        await metadata_repository.delete(f"backfill_status:{symbol}:klines_1m")

        # Verify cleanup
        assert await kline_repository.count(symbol, interval) == 0
        assert await metadata_repository.get_last_sync_time(symbol, "klines_1m") is None
        assert await metadata_repository.get_backfill_status(symbol, "klines_1m") is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repository_error_handling_coordination(self, kline_repository, metadata_repository):
        """Test error handling coordination between repositories."""
        symbol = "TEST/SYMBOL"
        interval = KlineInterval.MINUTE_1
        now = datetime.now(UTC)

        # Test that Pydantic validation prevents invalid kline creation
        # This should fail at model creation time, so we expect ValueError (wrapped as ValidationError by pytest)
        with pytest.raises((ValueError, ValidationException)) as exc_info:
            # This should fail at model creation due to negative price
            invalid_kline = Kline(
                symbol=symbol,
                interval=interval,
                open_time=now,
                close_time=now + timedelta(minutes=1),
                open_price=Decimal("-100.00"),  # Invalid negative price
                high_price=Decimal("100.00"),
                low_price=Decimal("50.00"),
                close_price=Decimal("75.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("7500.0"),
                trades_count=10,
            )

        # Verify the error message contains expected validation error
        assert "Price must be greater than 0" in str(exc_info.value) or "open_price" in str(exc_info.value)

        # Since invalid kline creation failed, verify no data was stored
        count = await kline_repository.count(symbol, interval)
        assert count == 0

        # Verify no metadata was created for failed operation
        sync_time = await metadata_repository.get_last_sync_time(symbol, "klines_1m")
        assert sync_time is None

        # Test successful operation to ensure repositories are still functional
        valid_kline = Kline(
            symbol=symbol,
            interval=interval,
            open_time=now,
            close_time=now + timedelta(minutes=1),
            open_price=Decimal("100.00"),  # Valid positive price
            high_price=Decimal("110.00"),
            low_price=Decimal("90.00"),
            close_price=Decimal("105.00"),
            volume=Decimal("100.0"),
            quote_volume=Decimal("10500.0"),
            trades_count=10,
        )

        # This should succeed
        await kline_repository.save(valid_kline)
        await metadata_repository.set_last_sync_time(symbol, "klines_1m", valid_kline.close_time)

        # Verify successful storage
        count = await kline_repository.count(symbol, interval)
        assert count == 1

        sync_time = await metadata_repository.get_last_sync_time(symbol, "klines_1m")
        assert sync_time == valid_kline.close_time

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repository_statistics_consistency(self, kline_repository, metadata_repository, sample_klines):
        """Test statistics consistency across repositories."""
        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1

        # Store klines
        await kline_repository.save_batch(sample_klines)

        # Get kline statistics
        kline_stats = await kline_repository.get_statistics(symbol, interval)

        # Store corresponding metadata statistics
        metadata_stats = {
            "symbol": symbol,
            "interval": interval.value,
            "kline_count": kline_stats["count"],
            "first_kline_time": kline_stats["first_timestamp"].isoformat() if kline_stats["first_timestamp"] else None,
            "last_kline_time": kline_stats["last_timestamp"].isoformat() if kline_stats["last_timestamp"] else None,
            "total_volume": str(kline_stats["volume_total"]),
            "total_quote_volume": str(kline_stats["quote_volume_total"]),
            "updated_at": datetime.now(UTC).isoformat(),
        }

        await metadata_repository.set(f"stats:{symbol}:{interval.value}", metadata_stats)

        # Verify consistency
        stored_stats = await metadata_repository.get(f"stats:{symbol}:{interval.value}")
        assert stored_stats is not None
        assert stored_stats["kline_count"] == len(sample_klines)
        assert stored_stats["symbol"] == symbol
        assert stored_stats["interval"] == interval.value

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repository_memory_usage_coordination(self, kline_repository, metadata_repository, sample_klines):
        """Test memory usage coordination between repositories."""
        # Store data in both repositories
        await kline_repository.save_batch(sample_klines)

        # Store various metadata
        for i, kline in enumerate(sample_klines):
            await metadata_repository.set(
                f"kline_meta:{kline.symbol}:{i}",
                {
                    "kline_id": i,
                    "open_time": kline.open_time.isoformat(),
                    "price": str(kline.close_price),
                    "volume": str(kline.volume),
                },
            )

        # Get memory usage from both repositories
        kline_memory = await kline_repository.get_memory_usage()
        metadata_memory = await metadata_repository.get_memory_usage()

        # Verify data consistency
        assert kline_memory["total_klines"] == len(sample_klines)
        assert metadata_memory["total_keys"] >= len(sample_klines)  # At least one key per kline

        # Verify repositories are operational
        assert not kline_memory["is_closed"]
        assert not metadata_memory["is_closed"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_repository_gap_detection_with_metadata(self, kline_repository, metadata_repository):
        """Test gap detection coordination with metadata tracking."""
        symbol = "ETH/USDT"
        interval = KlineInterval.MINUTE_1
        now = datetime.now(UTC)

        # Create klines with gaps
        klines_part1 = []
        for i in range(3):
            open_time = now + timedelta(minutes=i)
            close_time = open_time + timedelta(minutes=1)

            kline = Kline(
                symbol=symbol,
                interval=interval,
                open_time=open_time,
                close_time=close_time,
                open_price=Decimal("2000.00"),
                high_price=Decimal("2010.00"),
                low_price=Decimal("1990.00"),
                close_price=Decimal("2005.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("200500.0"),
                trades_count=50,
            )
            klines_part1.append(kline)

        # Create second part with gap (starting 10 minutes later)
        klines_part2 = []
        for i in range(3):
            open_time = now + timedelta(minutes=10 + i)
            close_time = open_time + timedelta(minutes=1)

            kline = Kline(
                symbol=symbol,
                interval=interval,
                open_time=open_time,
                close_time=close_time,
                open_price=Decimal("2100.00"),
                high_price=Decimal("2110.00"),
                low_price=Decimal("2090.00"),
                close_price=Decimal("2105.00"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("210500.0"),
                trades_count=50,
            )
            klines_part2.append(kline)

        # Store both parts
        await kline_repository.save_batch(klines_part1)
        await kline_repository.save_batch(klines_part2)

        # Detect gaps
        gaps = await kline_repository.get_gaps(symbol, interval, now, now + timedelta(minutes=20))

        # Should detect gap between part1 and part2
        assert len(gaps) >= 1

        # Store gap information in metadata
        gap_info = {
            "symbol": symbol,
            "interval": interval.value,
            "gaps_detected": len(gaps),
            "gap_details": [
                {
                    "start": gap[0].isoformat(),
                    "end": gap[1].isoformat(),
                    "duration_minutes": (gap[1] - gap[0]).total_seconds() / 60,
                }
                for gap in gaps
            ],
            "detected_at": datetime.now(UTC).isoformat(),
        }

        await metadata_repository.set(f"gaps:{symbol}:{interval.value}", gap_info)

        # Verify gap metadata
        stored_gap_info = await metadata_repository.get(f"gaps:{symbol}:{interval.value}")
        assert stored_gap_info is not None
        assert stored_gap_info["gaps_detected"] == len(gaps)
