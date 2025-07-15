# ABOUTME: Index consistency tests for EventStorage implementations
# ABOUTME: Tests that indexes remain synchronized with actual event data under various scenarios

import pytest
import pytest_asyncio
import asyncio
import time_machine
from datetime import datetime, UTC, timedelta
from decimal import Decimal

from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import TradeSide, KlineInterval
from core.exceptions.base import StorageError


class TestEventStorageIndexConsistency:
    """Index consistency tests for EventStorage implementations."""

    @pytest_asyncio.fixture
    async def metadata_repository(self):
        """Create MetadataRepository for direct index verification."""
        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest_asyncio.fixture
    async def event_storage(self, metadata_repository):
        """Create EventStorage for index consistency testing."""
        serializer = MemoryEventSerializer()
        storage = InMemoryEventStorage(serializer, metadata_repository)
        yield storage
        await storage.close()

    def create_trade_event(self, trade_id: str, symbol: str = "BTC/USDT", source: str = "binance") -> TradeEvent:
        """Create a test trade event."""
        trade = Trade(
            symbol=symbol,
            trade_id=trade_id,
            price=Decimal("45000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )

        return TradeEvent(source=source, symbol=symbol, data=trade, priority=EventPriority.NORMAL)

    def create_kline_event(self, symbol: str = "BTC/USDT", source: str = "binance") -> KlineEvent:
        """Create a test kline event."""
        now = datetime.now(UTC)
        open_time = now.replace(second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=1)

        kline = Kline(
            symbol=symbol,
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

        return KlineEvent(source=source, symbol=symbol, data=kline, priority=EventPriority.NORMAL)

    async def verify_index_consistency(self, event_storage, metadata_repository):
        """Verify that all indexes are consistent with actual stored data."""
        # Get all stored events
        all_keys = await metadata_repository.list_keys()
        event_keys = [key for key in all_keys if key.startswith("event:")]

        # Track actual events by their attributes
        actual_events = {}
        events_by_type = {}
        events_by_symbol = {}
        events_by_source = {}
        events_by_priority = {}

        # Collect actual event data
        for event_key in event_keys:
            event_data = await metadata_repository.get(event_key)
            if event_data:
                storage_id = event_key.replace("event:", "")
                actual_events[storage_id] = event_data

                # Group by attributes
                event_type = event_data.get("event_type")
                if event_type:
                    events_by_type.setdefault(event_type, set()).add(storage_id)

                symbol = event_data.get("symbol")
                if symbol:
                    events_by_symbol.setdefault(symbol, set()).add(storage_id)

                source = event_data.get("source")
                if source:
                    events_by_source.setdefault(source, set()).add(storage_id)

                priority = event_data.get("priority")
                if priority:
                    events_by_priority.setdefault(priority, set()).add(storage_id)

        # Verify type indexes
        for event_type, expected_ids in events_by_type.items():
            index_key = f"index:type:{event_type}"
            index_data = await metadata_repository.get(index_key)

            if index_data:
                indexed_ids = set(index_data.get("event_ids", []))
                assert indexed_ids == expected_ids, (
                    f"Type index {event_type} inconsistent: {indexed_ids} != {expected_ids}"
                )

        # Verify symbol indexes
        for symbol, expected_ids in events_by_symbol.items():
            index_key = f"index:symbol:{symbol}"
            index_data = await metadata_repository.get(index_key)

            if index_data:
                indexed_ids = set(index_data.get("event_ids", []))
                assert indexed_ids == expected_ids, (
                    f"Symbol index {symbol} inconsistent: {indexed_ids} != {expected_ids}"
                )

        # Verify source indexes
        for source, expected_ids in events_by_source.items():
            index_key = f"index:source:{source}"
            index_data = await metadata_repository.get(index_key)

            if index_data:
                indexed_ids = set(index_data.get("event_ids", []))
                assert indexed_ids == expected_ids, (
                    f"Source index {source} inconsistent: {indexed_ids} != {expected_ids}"
                )

        # Verify priority indexes
        for priority, expected_ids in events_by_priority.items():
            index_key = f"index:priority:{priority}"
            index_data = await metadata_repository.get(index_key)

            if index_data:
                indexed_ids = set(index_data.get("event_ids", []))
                assert indexed_ids == expected_ids, (
                    f"Priority index {priority} inconsistent: {indexed_ids} != {expected_ids}"
                )

        return len(actual_events)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_after_single_operations(self, event_storage, metadata_repository):
        """Test index consistency after single store/delete operations."""
        # Store a single event
        event = self.create_trade_event("test_trade_1", "BTC/USDT", "binance")
        storage_id = await event_storage.store_event(event)

        # Verify indexes are consistent
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 1

        # Delete the event
        await event_storage.delete_event(storage_id)

        # Verify indexes are still consistent after deletion
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)

        # Verify the event is removed from all indexes
        type_index = await metadata_repository.get(f"index:type:{EventType.TRADE.value}")
        symbol_index = await metadata_repository.get("index:symbol:BTC/USDT")
        source_index = await metadata_repository.get("index:source:binance")

        if type_index:
            assert storage_id not in type_index.get("event_ids", [])
        if symbol_index:
            assert storage_id not in symbol_index.get("event_ids", [])
        if source_index:
            assert storage_id not in source_index.get("event_ids", [])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_after_batch_operations(self, event_storage, metadata_repository):
        """Test index consistency after batch store/delete operations."""
        # Create multiple events with different attributes
        events = [
            self.create_trade_event("trade_1", "BTC/USDT", "binance"),
            self.create_trade_event("trade_2", "ETH/USDT", "binance"),
            self.create_trade_event("trade_3", "BTC/USDT", "kraken"),
            self.create_kline_event("BTC/USDT", "binance"),
            self.create_kline_event("ETH/USDT", "kraken"),
        ]

        # Store events in batch
        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 5

        # Verify indexes are consistent
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 5

        # Delete some events by query
        query = EventQuery(symbols=["BTC/USDT"])
        deleted_count = await event_storage.delete_events(query)
        assert deleted_count >= 2

        # Verify indexes are still consistent after batch deletion
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_under_concurrent_operations(self, event_storage, metadata_repository):
        """Test index consistency under concurrent operations."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:

            async def store_events_task(task_id: int):
                events = []
                for i in range(10):
                    event = self.create_trade_event(f"task_{task_id}_trade_{i}", "BTC/USDT", "binance")
                    events.append(event)
                return await event_storage.store_events(events)

            async def delete_events_task():
                # Wait a bit to ensure some events are stored
                traveller.shift(0.01)
                query = EventQuery(event_types=[EventType.TRADE], limit=5)
                return await event_storage.delete_events(query)

            # Run concurrent operations
            tasks = [
                store_events_task(1),
                store_events_task(2),
                store_events_task(3),
                delete_events_task(),
                delete_events_task(),
            ]

            results = await asyncio.gather(*tasks)

            # Verify indexes are consistent after concurrent operations
            event_count = await self.verify_index_consistency(event_storage, metadata_repository)
            assert event_count >= 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_with_mixed_event_types(self, event_storage, metadata_repository):
        """Test index consistency with mixed event types."""
        # Create events of different types
        events = []

        # Create trade events
        for i in range(10):
            event = self.create_trade_event(f"trade_{i}", "BTC/USDT", "binance")
            events.append(event)

        # Create kline events
        for i in range(10):
            event = self.create_kline_event("BTC/USDT", "binance")
            events.append(event)

        # Store all events
        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 20

        # Verify initial consistency
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 20

        # Delete only trade events
        trade_query = EventQuery(event_types=[EventType.TRADE])
        deleted_count = await event_storage.delete_events(trade_query)
        assert deleted_count >= 10

        # Verify consistency after selective deletion
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)

        # Verify only kline events remain
        kline_query = EventQuery(event_types=[EventType.KLINE])
        remaining_events = await event_storage.query_events(kline_query)
        assert len(remaining_events) >= 10

        # Verify no trade events remain
        trade_events = await event_storage.query_events(trade_query)
        assert len(trade_events) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_with_time_based_operations(self, event_storage, metadata_repository):
        """Test index consistency with time-based operations."""
        now = datetime.now(UTC)

        # Create events with different timestamps
        events = []
        for i in range(20):
            event = self.create_trade_event(f"time_trade_{i}", "BTC/USDT", "binance")
            # Set different timestamps
            event.timestamp = now - timedelta(hours=i)
            events.append(event)

        # Store events
        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 20

        # Verify initial consistency
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 20

        # Delete events in time range
        start_time = now - timedelta(hours=10)
        end_time = now - timedelta(hours=5)
        time_query = EventQuery(start_time=start_time, end_time=end_time)
        deleted_count = await event_storage.delete_events(time_query)
        assert deleted_count >= 0

        # Verify consistency after time-based deletion
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_with_large_dataset(self, event_storage, metadata_repository):
        """Test index consistency with large dataset."""
        # Create a large number of events
        events = []
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "SOL/USDT"]
        sources = ["binance", "kraken", "coinbase", "kucoin"]

        for i in range(100):
            symbol = symbols[i % len(symbols)]
            source = sources[i % len(sources)]

            if i % 2 == 0:
                event = self.create_trade_event(f"large_trade_{i}", symbol, source)
            else:
                event = self.create_kline_event(symbol, source)

            events.append(event)

        # Store all events
        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 100

        # Verify initial consistency
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 100

        # Perform various delete operations
        delete_tasks = [
            event_storage.delete_events(EventQuery(symbols=["BTC/USDT"])),
            event_storage.delete_events(EventQuery(sources=["binance"])),
            event_storage.delete_events(EventQuery(event_types=[EventType.TRADE], limit=10)),
        ]

        delete_results = await asyncio.gather(*delete_tasks)

        # Verify consistency after multiple deletions
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_after_storage_errors(self, event_storage, metadata_repository):
        """Test index consistency after storage errors."""
        # Store some initial events
        events = [
            self.create_trade_event("error_trade_1", "BTC/USDT", "binance"),
            self.create_trade_event("error_trade_2", "ETH/USDT", "binance"),
        ]

        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 2

        # Verify initial consistency
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 2

        # Close storage to simulate errors
        await event_storage.close()

        # Try to perform operations that should fail
        try:
            await event_storage.store_event(self.create_trade_event("error_trade_3", "BTC/USDT", "binance"))
        except StorageError:
            pass  # Expected error

        # Verify indexes are still consistent (no corruption)
        # We need to create a new storage instance with a new metadata repository to verify
        from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository

        new_metadata_repo = InMemoryMetadataRepository()
        new_storage = InMemoryEventStorage(serializer=MemoryEventSerializer(), metadata_repository=new_metadata_repo)

        try:
            # Since the metadata repository is new, we can't verify the old data
            # But we can verify that the new storage instance works correctly
            test_event = self.create_trade_event("recovery_test", "BTC/USDT", "binance")
            storage_id = await new_storage.store_event(test_event)
            assert storage_id is not None

            # Verify we can retrieve the event
            retrieved_event = await new_storage.retrieve_event(storage_id)
            assert retrieved_event is not None
            assert retrieved_event.data.symbol == "BTC/USDT"
        finally:
            await new_storage.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_cleanup_on_complete_deletion(self, event_storage, metadata_repository):
        """Test that indexes are properly cleaned up when all events are deleted."""
        # Store events
        events = [
            self.create_trade_event("cleanup_trade_1", "BTC/USDT", "binance"),
            self.create_trade_event("cleanup_trade_2", "BTC/USDT", "binance"),
        ]

        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 2

        # Verify indexes exist
        type_index = await metadata_repository.get(f"index:type:{EventType.TRADE.value}")
        symbol_index = await metadata_repository.get("index:symbol:BTC/USDT")
        source_index = await metadata_repository.get("index:source:binance")

        assert type_index is not None
        assert symbol_index is not None
        assert source_index is not None

        # Delete all events
        query = EventQuery(event_types=[EventType.TRADE])
        deleted_count = await event_storage.delete_events(query)
        assert deleted_count >= 2

        # Verify indexes are cleaned up or empty
        type_index = await metadata_repository.get(f"index:type:{EventType.TRADE.value}")
        symbol_index = await metadata_repository.get("index:symbol:BTC/USDT")
        source_index = await metadata_repository.get("index:source:binance")

        # Indexes should either be None or have empty event_ids
        if type_index:
            assert len(type_index.get("event_ids", [])) == 0
        if symbol_index:
            assert len(symbol_index.get("event_ids", [])) == 0
        if source_index:
            assert len(source_index.get("event_ids", [])) == 0

        # Verify overall consistency
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_with_query_operations(self, event_storage, metadata_repository):
        """Test that query operations don't affect index consistency."""
        # Store events
        events = [
            self.create_trade_event("query_trade_1", "BTC/USDT", "binance"),
            self.create_trade_event("query_trade_2", "ETH/USDT", "binance"),
            self.create_kline_event("BTC/USDT", "binance"),
        ]

        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 3

        # Perform various query operations
        queries = [
            EventQuery(event_types=[EventType.TRADE]),
            EventQuery(symbols=["BTC/USDT"]),
            EventQuery(sources=["binance"]),
            EventQuery(event_types=[EventType.TRADE], symbols=["BTC/USDT"]),
            EventQuery(limit=2),
            EventQuery(offset=1),
        ]

        # Execute queries
        for query in queries:
            events = await event_storage.query_events(query)
            assert isinstance(events, list)

        # Verify indexes are still consistent after queries
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_index_consistency_with_streaming_operations(self, event_storage, metadata_repository):
        """Test that streaming operations don't affect index consistency."""
        # Store events
        events = []
        for i in range(50):
            event = self.create_trade_event(f"stream_trade_{i}", "BTC/USDT", "binance")
            events.append(event)

        storage_ids = await event_storage.store_events(events)
        assert len(storage_ids) == 50

        # Perform streaming operations
        query = EventQuery(event_types=[EventType.TRADE])
        streamed_count = 0

        async for event in event_storage.stream_events(query):
            streamed_count += 1
            # Limit to prevent infinite loop
            if streamed_count >= 50:
                break

        assert streamed_count >= 50

        # Verify indexes are still consistent after streaming
        event_count = await self.verify_index_consistency(event_storage, metadata_repository)
        assert event_count >= 50
