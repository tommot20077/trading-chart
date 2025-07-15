# ABOUTME: Unit tests for InMemoryTimeSeriesRepository implementation
# ABOUTME: Comprehensive tests covering all repository operations and edge cases

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal

from core.implementations.memory.storage.time_series_repository import InMemoryTimeSeriesRepository
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import TradeSide, KlineInterval, AssetClass
from core.models.storage.query_option import QueryOptions


class TestInMemoryTimeSeriesRepository:
    """Test suite for InMemoryTimeSeriesRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryTimeSeriesRepository[Trade]:
        """Create a fresh repository instance for each test."""
        return InMemoryTimeSeriesRepository[Trade]()

    @pytest.fixture
    def sample_trades(self) -> list[Trade]:
        """Create sample trade data for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        return [
            Trade(
                symbol="BTCUSDT",
                trade_id="1",
                price=Decimal("50000.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY,
                timestamp=base_time,
                asset_class=AssetClass.DIGITAL,
            ),
            Trade(
                symbol="BTCUSDT",
                trade_id="2",
                price=Decimal("50100.00"),
                quantity=Decimal("0.2"),
                side=TradeSide.SELL,
                timestamp=base_time + timedelta(minutes=1),
                asset_class=AssetClass.DIGITAL,
            ),
            Trade(
                symbol="BTCUSDT",
                trade_id="3",
                price=Decimal("50200.00"),
                quantity=Decimal("0.15"),
                side=TradeSide.BUY,
                timestamp=base_time + timedelta(minutes=2),
                asset_class=AssetClass.DIGITAL,
            ),
            Trade(
                symbol="ETHUSDT",
                trade_id="4",
                price=Decimal("3000.00"),
                quantity=Decimal("1.0"),
                side=TradeSide.BUY,
                timestamp=base_time + timedelta(minutes=1),
                asset_class=AssetClass.DIGITAL,
            ),
        ]

    @pytest.fixture
    def sample_klines(self) -> list[Kline]:
        """Create sample kline data for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        return [
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time,
                close_time=base_time + timedelta(minutes=1),
                open_price=Decimal("50000.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49900.00"),
                close_price=Decimal("50050.00"),
                volume=Decimal("10.0"),
                quote_volume=Decimal("500000.00"),
                trades_count=100,
                asset_class=AssetClass.DIGITAL,
            ),
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=base_time + timedelta(minutes=1),
                close_time=base_time + timedelta(minutes=2),
                open_price=Decimal("50050.00"),
                high_price=Decimal("50200.00"),
                low_price=Decimal("50000.00"),
                close_price=Decimal("50150.00"),
                volume=Decimal("15.0"),
                quote_volume=Decimal("752250.00"),
                trades_count=150,
                asset_class=AssetClass.DIGITAL,
            ),
        ]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_single_item(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test saving a single item."""
        trade = sample_trades[0]
        await repository.save(trade)

        # Verify item was saved
        result = await repository.query(
            trade.symbol,
            trade.timestamp - timedelta(seconds=1),
            trade.timestamp + timedelta(seconds=1),
        )
        assert len(result) == 1
        assert result[0] == trade

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_batch(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test saving multiple items in batch."""
        btc_trades = [t for t in sample_trades if t.symbol == "BTCUSDT"]
        saved_count = await repository.save_batch(btc_trades)

        assert saved_count == len(btc_trades)

        # Verify all items were saved
        result = await repository.query(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
        )
        assert len(result) == len(btc_trades)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_batch_empty_list(self, repository: InMemoryTimeSeriesRepository[Trade]):
        """Test saving empty batch."""
        saved_count = await repository.save_batch([])
        assert saved_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_time_range(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test querying items within time range."""
        await repository.save_batch(sample_trades)

        # Query specific time range
        start_time = datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 2, 30, tzinfo=UTC)

        result = await repository.query("BTCUSDT", start_time, end_time)

        # Should get trades 2 and 3 (at 1min and 2min)
        assert len(result) == 2
        assert all(start_time <= trade.timestamp <= end_time for trade in result)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_options(
        self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]
    ):
        """Test querying with various options."""
        btc_trades = [t for t in sample_trades if t.symbol == "BTCUSDT"]
        await repository.save_batch(btc_trades)

        # Test with limit
        options = QueryOptions(limit=2)
        result = await repository.query(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
            options=options,
        )
        assert len(result) == 2

        # Test with offset
        options = QueryOptions(offset=1, limit=1)
        result = await repository.query(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
            options=options,
        )
        assert len(result) == 1
        assert result[0].trade_id == "2"  # Second trade

        # Test with descending order
        options = QueryOptions(order_desc=True)
        result = await repository.query(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
            options=options,
        )
        assert len(result) == 3
        assert result[0].trade_id == "3"  # Latest trade first

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_invalid_time_range(self, repository: InMemoryTimeSeriesRepository[Trade]):
        """Test querying with invalid time range."""
        start_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            await repository.query("BTCUSDT", start_time, end_time)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test streaming items."""
        await repository.save_batch(sample_trades)

        # Stream all BTCUSDT trades
        result = []
        async for trade in repository.stream(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
            batch_size=2,
        ):
            result.append(trade)

        btc_trades = [t for t in sample_trades if t.symbol == "BTCUSDT"]
        assert len(result) == len(btc_trades)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_invalid_batch_size(self, repository: InMemoryTimeSeriesRepository[Trade]):
        """Test streaming with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            async for _ in repository.stream(
                "BTCUSDT",
                datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
                batch_size=0,
            ):
                pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_latest(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test getting latest item."""
        await repository.save_batch(sample_trades)

        latest = await repository.get_latest("BTCUSDT")
        assert latest is not None
        assert latest.trade_id == "3"  # Latest BTCUSDT trade

        # Test non-existent symbol
        latest = await repository.get_latest("NONEXISTENT")
        assert latest is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_oldest(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test getting oldest item."""
        await repository.save_batch(sample_trades)

        oldest = await repository.get_oldest("BTCUSDT")
        assert oldest is not None
        assert oldest.trade_id == "1"  # Oldest BTCUSDT trade

        # Test non-existent symbol
        oldest = await repository.get_oldest("NONEXISTENT")
        assert oldest is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_count(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test counting items."""
        await repository.save_batch(sample_trades)

        # Count all BTCUSDT trades
        count = await repository.count("BTCUSDT")
        assert count == 3

        # Count with time range
        count = await repository.count(
            "BTCUSDT",
            datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 2, 30, tzinfo=UTC),
        )
        assert count == 2

        # Count non-existent symbol
        count = await repository.count("NONEXISTENT")
        assert count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test deleting items."""
        await repository.save_batch(sample_trades)

        # Delete middle trade
        deleted_count = await repository.delete(
            "BTCUSDT",
            datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 1, 30, tzinfo=UTC),
        )
        assert deleted_count == 1

        # Verify deletion
        remaining = await repository.count("BTCUSDT")
        assert remaining == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_invalid_time_range(self, repository: InMemoryTimeSeriesRepository[Trade]):
        """Test deleting with invalid time range."""
        start_time = datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            await repository.delete("BTCUSDT", start_time, end_time)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_gaps(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test finding gaps in data."""
        # Save only first and third trades (creating a gap)
        trades_with_gap = [sample_trades[0], sample_trades[2]]
        await repository.save_batch(trades_with_gap)

        gaps = await repository.get_gaps(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 59, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 3, 0, tzinfo=UTC),
        )

        # Should find gaps before first trade, between trades, and after last trade
        assert len(gaps) >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_statistics(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test getting statistics."""
        btc_trades = [t for t in sample_trades if t.symbol == "BTCUSDT"]
        await repository.save_batch(btc_trades)

        stats = await repository.get_statistics("BTCUSDT")

        assert stats["count"] == 3
        assert stats["first_timestamp"] == btc_trades[0].timestamp
        assert stats["last_timestamp"] == btc_trades[-1].timestamp
        assert stats["time_span_seconds"] == 120  # 2 minutes

        # Test empty symbol
        empty_stats = await repository.get_statistics("NONEXISTENT")
        assert empty_stats["count"] == 0
        assert empty_stats["first_timestamp"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close(self, repository: InMemoryTimeSeriesRepository[Trade], sample_trades: list[Trade]):
        """Test closing repository."""
        await repository.save_batch(sample_trades)

        # Close repository
        await repository.close()

        # Operations should fail after closing
        with pytest.raises(RuntimeError, match="Repository is closed"):
            await repository.save(sample_trades[0])

        with pytest.raises(RuntimeError, match="Repository is closed"):
            await repository.query("BTCUSDT", datetime.now(UTC), datetime.now(UTC))

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self, sample_trades: list[Trade]):
        """Test using repository as context manager."""
        async with InMemoryTimeSeriesRepository[Trade]() as repo:
            await repo.save_batch(sample_trades)
            count = await repo.count("BTCUSDT")
            assert count == 3

        # Repository should be closed after context exit
        with pytest.raises(RuntimeError, match="Repository is closed"):
            await repo.save(sample_trades[0])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_with_kline_data(self, sample_klines: list[Kline]):
        """Test repository with Kline data type."""
        repo = InMemoryTimeSeriesRepository[Kline]()

        await repo.save_batch(sample_klines)

        # Query klines
        result = await repo.query(
            "BTCUSDT",
            datetime(2024, 1, 1, 11, 59, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 3, 0, tzinfo=UTC),
        )

        assert len(result) == 2
        assert all(isinstance(kline, Kline) for kline in result)

        await repo.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_invalid_item(self, repository: InMemoryTimeSeriesRepository[Trade]):
        """Test saving item without required attributes."""

        # Create a mock object without required attributes
        class InvalidItem:
            pass

        invalid_item = InvalidItem()

        with pytest.raises(ValueError, match="Item must have 'symbol' and 'primary_timestamp' attributes"):
            await repository.save(invalid_item)  # type: ignore

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ordering_maintained(self, repository: InMemoryTimeSeriesRepository[Trade]):
        """Test that items are maintained in timestamp order."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Save trades out of order
        trades = [
            Trade(
                symbol="BTCUSDT",
                trade_id="3",
                price=Decimal("50200.00"),
                quantity=Decimal("0.15"),
                side=TradeSide.BUY,
                timestamp=base_time + timedelta(minutes=2),
                asset_class=AssetClass.DIGITAL,
            ),
            Trade(
                symbol="BTCUSDT",
                trade_id="1",
                price=Decimal("50000.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY,
                timestamp=base_time,
                asset_class=AssetClass.DIGITAL,
            ),
            Trade(
                symbol="BTCUSDT",
                trade_id="2",
                price=Decimal("50100.00"),
                quantity=Decimal("0.2"),
                side=TradeSide.SELL,
                timestamp=base_time + timedelta(minutes=1),
                asset_class=AssetClass.DIGITAL,
            ),
        ]

        for trade in trades:
            await repository.save(trade)

        # Query should return in timestamp order
        result = await repository.query(
            "BTCUSDT",
            base_time - timedelta(minutes=1),
            base_time + timedelta(minutes=3),
        )

        assert len(result) == 3
        assert result[0].trade_id == "1"
        assert result[1].trade_id == "2"
        assert result[2].trade_id == "3"
