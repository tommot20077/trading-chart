# ABOUTME: Integration tests for EventStorage implementations with real dependencies
# ABOUTME: Tests complete layered architecture functionality

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal

from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.event.event_storage_stats import EventStorageStats
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval, TradeSide
from core.models.network.enum import ConnectionStatus
from core.exceptions.base import StorageError


class TestEventStorageLayeredArchitecture:
    """
    Integration tests for EventStorage layered architecture.

    Tests the complete interaction between:
    - EventStorage (Business Layer)
    - MetadataRepository (Storage Layer)

    """

    @pytest_asyncio.fixture
    async def metadata_repository(self):
        """Create a real InMemoryMetadataRepository."""
        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest.fixture
    def event_serializer(self):
        """Create a real MemoryEventSerializer."""
        return MemoryEventSerializer()

    @pytest_asyncio.fixture
    async def event_storage(self, event_serializer, metadata_repository):
        """Create InMemoryEventStorage with real dependencies."""
        storage = InMemoryEventStorage(serializer=event_serializer, metadata_repository=metadata_repository)
        yield storage
        await storage.close()

    @pytest.fixture
    def sample_trade_event(self):
        """Create a sample TradeEvent for testing."""
        trade = Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.50"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )
        return TradeEvent(source="binance", symbol="BTC/USDT", data=trade, priority=EventPriority.NORMAL)

    @pytest.fixture
    def sample_kline_event(self):
        """Create a sample KlineEvent for testing."""
        now = datetime.now(UTC)
        open_time = now.replace(second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=1)

        kline = Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=open_time,
            close_time=close_time,
            open_price=Decimal("45000.00"),
            high_price=Decimal("45100.00"),
            low_price=Decimal("44900.00"),
            close_price=Decimal("45050.00"),
            volume=Decimal("100.5"),
            quote_volume=Decimal("4525000.0"),
            trades_count=150,
        )
        return KlineEvent(source="binance", symbol="BTC/USDT", data=kline, priority=EventPriority.NORMAL)

    @pytest.fixture
    def sample_connection_event(self):
        """Create a sample ConnectionEvent for testing."""
        return ConnectionEvent(
            source="binance",
            symbol=None,
            data={"status": ConnectionStatus.CONNECTED, "message": "Connected successfully"},
            priority=EventPriority.HIGH,
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_event_lifecycle(self, event_storage, sample_trade_event):
        """Test complete event lifecycle through layered architecture."""
        # Store event
        storage_id = await event_storage.store_event(sample_trade_event)
        assert storage_id is not None
        assert isinstance(storage_id, str)

        # Retrieve event
        retrieved_event = await event_storage.retrieve_event(storage_id)
        assert retrieved_event is not None
        assert retrieved_event.symbol == sample_trade_event.symbol
        assert retrieved_event.source == sample_trade_event.source
        assert retrieved_event.event_type == EventType.TRADE

        # Query events
        query = EventQuery(event_types=[EventType.TRADE])
        events = await event_storage.query_events(query)
        assert len(events) >= 1
        assert any(e.event_id == retrieved_event.event_id for e in events)

        # Delete event
        deleted = await event_storage.delete_event(storage_id)
        assert deleted is True

        # Verify deletion
        retrieved_after_delete = await event_storage.retrieve_event(storage_id)
        assert retrieved_after_delete is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_layered_storage_interaction(self, event_storage, metadata_repository, sample_trade_event):
        """Test interaction between EventStorage and MetadataRepository layers."""
        # Store event through EventStorage
        storage_id = await event_storage.store_event(sample_trade_event)

        # Verify data exists in underlying MetadataRepository
        event_key = f"event:{storage_id}"
        raw_data = await metadata_repository.get(event_key)
        assert raw_data is not None
        assert "data" in raw_data
        assert "event_type" in raw_data
        assert raw_data["event_type"] == EventType.TRADE.value

        # Verify indexes are created in MetadataRepository
        type_index_key = f"index:type:{EventType.TRADE.value}"
        type_index = await metadata_repository.get(type_index_key)
        assert type_index is not None
        assert "event_ids" in type_index
        assert storage_id in type_index["event_ids"]

        # Verify symbol index
        symbol_index_key = f"index:symbol:{sample_trade_event.symbol}"
        symbol_index = await metadata_repository.get(symbol_index_key)
        assert symbol_index is not None
        assert storage_id in symbol_index["event_ids"]

        # Verify source index
        source_index_key = f"index:source:{sample_trade_event.source}"
        source_index = await metadata_repository.get(source_index_key)
        assert source_index is not None
        assert storage_id in source_index["event_ids"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_operations_with_indexes(self, event_storage, metadata_repository):
        """Test batch operations maintain index consistency."""
        # Create multiple events
        events = [
            TradeEvent(
                source="binance",
                symbol="BTC/USDT",
                data=Trade(
                    symbol="BTC/USDT",
                    trade_id=f"trade_{i}",
                    price=Decimal("45000.00"),
                    quantity=Decimal("0.1"),
                    side=TradeSide.BUY,
                    timestamp=datetime.now(UTC),
                    is_buyer_maker=True,
                ),
                priority=EventPriority.NORMAL,
            )
            for i in range(5)
        ]

        # Store events in batch
        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 5

        # Verify all events are indexed correctly
        type_index_key = f"index:type:{EventType.TRADE.value}"
        type_index = await metadata_repository.get(type_index_key)
        assert type_index is not None
        assert len(type_index["event_ids"]) >= 5

        # Verify all storage_ids are in the index
        for storage_id in storage_ids:
            assert storage_id in type_index["event_ids"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_query_scenarios(self, event_storage, sample_trade_event, sample_kline_event):
        """Test complex query scenarios with real data."""
        # Store different types of events
        trade_id = await event_storage.store_event(sample_trade_event)
        kline_id = await event_storage.store_event(sample_kline_event)

        # Query by type
        trade_query = EventQuery(event_types=[EventType.TRADE])
        trade_events = await event_storage.query_events(trade_query)
        assert len(trade_events) >= 1
        assert all(e.event_type == EventType.TRADE for e in trade_events)

        kline_query = EventQuery(event_types=[EventType.KLINE])
        kline_events = await event_storage.query_events(kline_query)
        assert len(kline_events) >= 1
        assert all(e.event_type == EventType.KLINE for e in kline_events)

        # Query by symbol
        symbol_query = EventQuery(symbols=["BTC/USDT"])
        symbol_events = await event_storage.query_events(symbol_query)
        assert len(symbol_events) >= 2

        # Query by source
        source_query = EventQuery(sources=["binance"])
        source_events = await event_storage.query_events(source_query)
        assert len(source_events) >= 2

        # Complex query with multiple criteria
        complex_query = EventQuery(
            event_types=[EventType.TRADE, EventType.KLINE], symbols=["BTC/USDT"], sources=["binance"], limit=10
        )
        complex_events = await event_storage.query_events(complex_query)
        assert len(complex_events) >= 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_time_based_queries(self, event_storage):
        """Test time-based queries with time shard indexing."""
        now = datetime.now(UTC)

        # Create events with different timestamps
        events = []
        for i in range(5):
            timestamp = now - timedelta(hours=i)
            event = TradeEvent(
                source="binance",
                symbol="BTC/USDT",
                data=Trade(
                    symbol="BTC/USDT",
                    trade_id=f"trade_{i}",
                    price=Decimal("45000.00"),
                    quantity=Decimal("0.1"),
                    side=TradeSide.BUY,
                    timestamp=timestamp,
                    is_buyer_maker=True,
                ),
                priority=EventPriority.NORMAL,
            )
            # Override timestamp for testing
            event.timestamp = timestamp
            events.append(event)

        # Store events
        await event_storage.store_events(events)

        # Query by time range
        start_time = now - timedelta(hours=2)
        end_time = now + timedelta(hours=1)

        time_query = EventQuery(start_time=start_time, end_time=end_time)
        time_events = await event_storage.query_events(time_query)

        # Should return events within the time range
        assert len(time_events) >= 2
        for event in time_events:
            assert start_time <= event.timestamp <= end_time

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_large_dataset(self, event_storage):
        """Test streaming functionality with larger dataset."""
        # Store multiple events
        events = []
        for i in range(100):
            event = TradeEvent(
                source=f"exchange_{i % 3}",
                symbol="BTC/USDT",
                data=Trade(
                    symbol="BTC/USDT",
                    trade_id=f"trade_{i}",
                    price=Decimal("45000.00"),
                    quantity=Decimal("0.1"),
                    side=TradeSide.BUY,
                    timestamp=datetime.now(UTC),
                    is_buyer_maker=True,
                ),
                priority=EventPriority.NORMAL,
            )
            events.append(event)

        await event_storage.store_events(events)

        # Stream events
        query = EventQuery(event_types=[EventType.TRADE])
        streamed_count = 0
        async for event in event_storage.stream_events(query):
            streamed_count += 1
            assert isinstance(event, TradeEvent)
            # Prevent infinite loop
            if streamed_count >= 100:
                break

        assert streamed_count >= 100

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_statistics_accuracy(self, event_storage, sample_trade_event, sample_kline_event):
        """Test that statistics accurately reflect stored data."""
        # Store events
        await event_storage.store_event(sample_trade_event)
        await event_storage.store_event(sample_kline_event)

        # Get statistics
        stats = await event_storage.get_stats()

        # Verify statistics
        assert isinstance(stats, EventStorageStats)
        assert stats.total_events >= 2
        assert EventType.TRADE.value in stats.events_by_type
        assert EventType.KLINE.value in stats.events_by_type
        assert stats.events_by_type[EventType.TRADE.value] >= 1
        assert stats.events_by_type[EventType.KLINE.value] >= 1
        assert stats.storage_size_bytes > 0
        assert stats.avg_event_size_bytes > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, event_storage, metadata_repository):
        """Test error handling and recovery scenarios."""
        # Test with corrupted data
        corrupted_key = "event:corrupted"
        await metadata_repository.set(corrupted_key, {"invalid": "data"})

        # Should handle gracefully by raising StorageError
        with pytest.raises(StorageError, match="Corrupted event data: missing 'data' field"):
            await event_storage.retrieve_event("corrupted")

        # Test health check
        health = await event_storage.health_check()
        assert health is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_access_patterns(self, event_storage):
        """Test concurrent access patterns common in real applications."""

        # Simulate concurrent read/write operations
        async def write_events(prefix: str, count: int):
            events = []
            for i in range(count):
                event = TradeEvent(
                    source="binance",
                    symbol="BTC/USDT",
                    data=Trade(
                        symbol="BTC/USDT",
                        trade_id=f"{prefix}_{i}",
                        price=Decimal("45000.00"),
                        quantity=Decimal("0.1"),
                        side=TradeSide.BUY,
                        timestamp=datetime.now(UTC),
                        is_buyer_maker=True,
                    ),
                    priority=EventPriority.NORMAL,
                )
                events.append(event)
            return await event_storage.store_events(events)

        async def read_events():
            query = EventQuery(event_types=[EventType.TRADE])
            return await event_storage.query_events(query)

        # Run concurrent operations
        tasks = [write_events("writer1", 10), write_events("writer2", 10), read_events(), read_events()]

        results = await asyncio.gather(*tasks)

        # Verify all operations completed successfully
        assert len(results[0]) == 10  # writer1 results
        assert len(results[1]) == 10  # writer2 results
        assert isinstance(results[2], list)  # read results
        assert isinstance(results[3], list)  # read results

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cleanup_and_resource_management(self, event_serializer, metadata_repository):
        """Test proper cleanup and resource management."""
        # Create storage
        storage = InMemoryEventStorage(serializer=event_serializer, metadata_repository=metadata_repository)

        # Use storage
        event = TradeEvent(
            source="binance",
            symbol="BTC/USDT",
            data=Trade(
                symbol="BTC/USDT",
                trade_id="test",
                price=Decimal("45000.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY,
                timestamp=datetime.now(UTC),
                is_buyer_maker=True,
            ),
            priority=EventPriority.NORMAL,
        )

        storage_id = await storage.store_event(event)
        assert storage_id is not None

        # Close storage
        await storage.close()

        # Verify operations after close raise errors
        with pytest.raises(StorageError):
            await storage.store_event(event)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_integrity_across_layers(self, event_storage, metadata_repository, sample_trade_event):
        """Test data integrity across storage layers."""
        # Store event
        storage_id = await event_storage.store_event(sample_trade_event)

        # Verify data integrity at storage layer
        event_key = f"event:{storage_id}"
        raw_data = await metadata_repository.get(event_key)
        assert raw_data is not None

        # Verify serialized data can be deserialized
        serialized_data = raw_data["data"].encode("utf-8")
        event_storage.serializer.deserialize(serialized_data)

        # Verify event can be retrieved through business layer
        retrieved_event = await event_storage.retrieve_event(storage_id)
        assert retrieved_event is not None
        assert retrieved_event.symbol == sample_trade_event.symbol
        assert retrieved_event.source == sample_trade_event.source
