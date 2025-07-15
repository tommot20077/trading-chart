# ABOUTME: Unit tests for InMemoryEventStorage implementation
# ABOUTME: Tests event storage functionality using TDD approach with mocked MetadataRepository

import pytest
import pytest_asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.interfaces.storage.metadata_repository import AbstractMetadataRepository
from core.models.data.event import BaseEvent
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.event.event_storage_stats import EventStorageStats
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval, TradeSide
from core.exceptions.base import StorageError


class TestInMemoryEventStorageUnit:
    """Unit tests for InMemoryEventStorage using mocked dependencies."""

    @pytest.fixture
    def mock_serializer(self):
        """Create a mock event serializer."""
        serializer = AsyncMock(spec=AbstractEventSerializer)
        serializer.serialize.return_value = b'{"test": "data"}'
        serializer.deserialize.return_value = Mock(spec=BaseEvent)
        return serializer

    @pytest.fixture
    def mock_metadata_repo(self):
        """Create a mock metadata repository."""
        repo = AsyncMock(spec=AbstractMetadataRepository)
        repo.set.return_value = None
        repo.get.return_value = None
        repo.exists.return_value = False
        repo.delete.return_value = False
        repo.list_keys.return_value = []
        repo.close.return_value = None
        return repo

    @pytest.fixture
    def sample_trade(self):
        """Create a sample Trade object for testing."""
        return Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.50"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )

    @pytest.fixture
    def sample_trade_event(self, sample_trade):
        """Create a sample TradeEvent for testing."""
        return TradeEvent(source="test_exchange", symbol="BTC/USDT", data=sample_trade, priority=EventPriority.NORMAL)

    @pytest.fixture
    def sample_kline(self):
        """Create a sample Kline object for testing."""
        now = datetime.now(UTC)
        open_time = now.replace(second=0, microsecond=0)
        close_time = open_time.replace(minute=open_time.minute + 1)
        return Kline(
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

    @pytest.fixture
    def sample_kline_event(self, sample_kline):
        """Create a sample KlineEvent for testing."""
        return KlineEvent(source="test_exchange", symbol="BTC/USDT", data=sample_kline, priority=EventPriority.NORMAL)

    @pytest.fixture
    def event_storage(self, mock_serializer, mock_metadata_repo):
        """Create InMemoryEventStorage instance with mocked dependencies."""
        # This will fail initially since we haven't implemented the class yet
        # This is the TDD approach - write the test first
        from core.implementations.memory.storage.event_storage import InMemoryEventStorage

        return InMemoryEventStorage(serializer=mock_serializer, metadata_repository=mock_metadata_repo)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, mock_serializer, mock_metadata_repo):
        """Test InMemoryEventStorage initialization."""
        # This test will fail initially - that's expected in TDD
        from core.implementations.memory.storage.event_storage import InMemoryEventStorage

        storage = InMemoryEventStorage(serializer=mock_serializer, metadata_repository=mock_metadata_repo)

        assert storage.serializer == mock_serializer
        assert storage._metadata_repo == mock_metadata_repo
        assert not storage._closed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_event_success(self, event_storage, sample_trade_event, mock_serializer, mock_metadata_repo):
        """Test successful event storage."""
        # Setup mocks
        mock_serializer.serialize.return_value = b'{"serialized": "event"}'
        mock_metadata_repo.set.return_value = None

        # Execute
        storage_id = await event_storage.store_event(sample_trade_event)

        # Verify
        assert isinstance(storage_id, str)
        assert len(storage_id) > 0

        # Verify serializer was called
        mock_serializer.serialize.assert_called_once_with(sample_trade_event)

        # Verify metadata repository was called to store the event
        mock_metadata_repo.set.assert_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_events_batch_success(
        self, event_storage, sample_trade_event, sample_kline_event, mock_serializer, mock_metadata_repo
    ):
        """Test successful batch event storage."""
        events = [sample_trade_event, sample_kline_event]

        # Setup mocks
        mock_serializer.serialize.return_value = b'{"serialized": "event"}'
        mock_metadata_repo.set.return_value = None

        # Execute
        storage_ids = await event_storage.store_events(events)

        # Verify
        assert len(storage_ids) == 2
        assert all(isinstance(sid, str) for sid in storage_ids)
        assert all(len(sid) > 0 for sid in storage_ids)

        # Verify serializer was called for each event
        assert mock_serializer.serialize.call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_event_success(self, event_storage, sample_trade_event, mock_serializer, mock_metadata_repo):
        """Test successful event retrieval."""
        storage_id = "test-storage-id"
        serialized_data = b'{"serialized": "event"}'

        # Setup mocks
        mock_metadata_repo.get.return_value = {"data": serialized_data.decode()}
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        retrieved_event = await event_storage.retrieve_event(storage_id)

        # Verify
        assert retrieved_event == sample_trade_event
        mock_metadata_repo.get.assert_called_once()
        mock_serializer.deserialize.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_event_not_found(self, event_storage, mock_metadata_repo):
        """Test event retrieval when event doesn't exist."""
        storage_id = "non-existent-id"

        # Setup mocks
        mock_metadata_repo.get.return_value = None

        # Execute
        retrieved_event = await event_storage.retrieve_event(storage_id)

        # Verify
        assert retrieved_event is None
        mock_metadata_repo.get.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_events_by_type(self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event):
        """Test querying events by event type."""
        query = EventQuery(event_types=[EventType.TRADE])

        # Setup mocks - simulate finding events in type index
        mock_metadata_repo.get.side_effect = [
            {"event_ids": ["id1", "id2"]},  # type index
            {"data": b'{"event": "data1"}'},  # event data
            {"data": b'{"event": "data2"}'},  # event data
        ]
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        events = await event_storage.query_events(query)

        # Verify
        assert len(events) == 2
        assert all(event == sample_trade_event for event in events)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_events_by_symbol(self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event):
        """Test querying events by symbol."""
        query = EventQuery(symbols=["BTC/USDT"])

        # Setup mocks
        mock_metadata_repo.get.side_effect = [
            {"event_ids": ["id1"]},  # symbol index
            {"data": b'{"event": "data1"}'},  # event data
        ]
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        events = await event_storage.query_events(query)

        # Verify
        assert len(events) == 1
        assert events[0] == sample_trade_event

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_events_by_time_range(
        self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event
    ):
        """Test querying events by time range."""
        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)
        query = EventQuery(start_time=start_time, end_time=end_time)

        # Setup mocks - simulate time-based filtering with time shard indexes
        def mock_get_side_effect(key):
            # Mock time shard indexes
            if key.startswith("index:time:"):
                return {"event_ids": ["id1", "id2"]}
            # Mock event data
            elif key == "event:id1":
                return {"data": b'{"event": "data1"}', "timestamp": start_time.isoformat()}
            elif key == "event:id2":
                return {"data": b'{"event": "data2"}', "timestamp": end_time.isoformat()}
            return None

        mock_metadata_repo.get.side_effect = mock_get_side_effect
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        events = await event_storage.query_events(query)

        # Verify
        assert len(events) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_events(self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event):
        """Test streaming events."""
        query = EventQuery(event_types=[EventType.TRADE])

        # Setup mocks
        mock_metadata_repo.get.side_effect = [
            {"event_ids": ["id1", "id2"]},  # type index
            {"data": b'{"event": "data1"}'},  # event data
            {"data": b'{"event": "data2"}'},  # event data
        ]
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        events = []
        async for event in event_storage.stream_events(query):
            events.append(event)

        # Verify
        assert len(events) == 2
        assert all(event == sample_trade_event for event in events)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_event_success(self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event):
        """Test successful event deletion."""
        storage_id = "test-storage-id"

        # Setup mocks - need to mock retrieve_event first
        mock_metadata_repo.get.return_value = {"data": b'{"event": "data"}'}
        mock_serializer.deserialize.return_value = sample_trade_event
        mock_metadata_repo.delete.return_value = True

        # Execute
        result = await event_storage.delete_event(storage_id)

        # Verify
        assert result is True
        mock_metadata_repo.delete.assert_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_event_not_found(self, event_storage, mock_metadata_repo):
        """Test event deletion when event doesn't exist."""
        storage_id = "non-existent-id"

        # Setup mocks
        mock_metadata_repo.delete.return_value = False

        # Execute
        result = await event_storage.delete_event(storage_id)

        # Verify
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_events_by_query(self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event):
        """Test deleting events by query."""
        query = EventQuery(event_types=[EventType.TRADE])

        # Setup mocks - simulate finding and deleting events
        # We need to mock all the get calls for indexes and events
        def mock_get_side_effect(key):
            if key == "index:type:trade":
                return {"event_ids": ["id1", "id2"]}
            elif key.startswith("event:"):
                return {"data": b'{"event": "data"}'}
            elif key.startswith("index:"):
                return {"event_ids": ["id1", "id2"]}  # For index cleanup
            return None

        mock_metadata_repo.get.side_effect = mock_get_side_effect
        mock_serializer.deserialize.return_value = sample_trade_event
        mock_metadata_repo.delete.return_value = True
        mock_metadata_repo.set.return_value = None  # For index updates

        # Execute
        deleted_count = await event_storage.delete_events(query)

        # Verify
        assert deleted_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_stats(self, event_storage, mock_metadata_repo):
        """Test getting storage statistics."""
        # Setup mocks - only return actual event keys, not index keys
        mock_metadata_repo.list_keys.return_value = ["event:id1", "event:id2"]
        mock_metadata_repo.get.side_effect = [
            {"event_type": "trade", "timestamp": datetime.now(UTC).isoformat(), "data": "some_data"},  # first event
            {
                "event_type": "kline",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": "some_other_data",
            },  # second event
        ]

        # Execute
        stats = await event_storage.get_stats()

        # Verify
        assert isinstance(stats, EventStorageStats)
        assert stats.total_events == 2
        assert isinstance(stats.events_by_type, dict)
        assert "trade" in stats.events_by_type
        assert "kline" in stats.events_by_type

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, event_storage, mock_metadata_repo):
        """Test health check when storage is healthy."""
        # Setup mocks - health check does set/get/delete operations
        mock_metadata_repo.set.return_value = None
        mock_metadata_repo.get.return_value = {"test": "data"}
        mock_metadata_repo.delete.return_value = True

        # Execute
        is_healthy = await event_storage.health_check()

        # Verify
        assert is_healthy is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, event_storage, mock_metadata_repo):
        """Test health check when storage is unhealthy."""
        # Setup mocks
        mock_metadata_repo.exists.side_effect = Exception("Connection failed")

        # Execute
        is_healthy = await event_storage.health_check()

        # Verify
        assert is_healthy is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close(self, event_storage, mock_metadata_repo):
        """Test closing the event storage."""
        # Execute
        await event_storage.close()

        # Verify
        assert event_storage._closed is True
        mock_metadata_repo.close.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_operations_after_close_raise_error(self, event_storage, sample_trade_event):
        """Test that operations after close raise appropriate errors."""
        # Close the storage
        await event_storage.close()

        # Verify operations raise errors
        with pytest.raises(StorageError) as exc_info:
            await event_storage.store_event(sample_trade_event)
        assert "closed" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_serialization_error_handling(
        self, event_storage, sample_trade_event, mock_serializer, mock_metadata_repo
    ):
        """Test handling of serialization errors."""
        # Setup mocks to simulate serialization failure
        mock_serializer.serialize.side_effect = Exception("Serialization failed")

        # Execute and verify error handling
        with pytest.raises(StorageError) as exc_info:
            await event_storage.store_event(sample_trade_event)
        assert "serialization" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metadata_repository_error_handling(
        self, event_storage, sample_trade_event, mock_serializer, mock_metadata_repo
    ):
        """Test handling of metadata repository errors."""
        # Setup mocks
        mock_serializer.serialize.return_value = b'{"test": "data"}'
        mock_metadata_repo.set.side_effect = Exception("Storage failed")

        # Execute and verify error handling
        with pytest.raises(StorageError) as exc_info:
            await event_storage.store_event(sample_trade_event)
        assert "storage" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_limit_and_offset(
        self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event
    ):
        """Test querying with limit and offset parameters."""
        query = EventQuery(limit=10, offset=5)

        # Setup mocks
        mock_metadata_repo.list_keys.return_value = [f"event:id{i}" for i in range(20)]
        mock_metadata_repo.get.return_value = {"data": b'{"event": "data"}'}
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        events = await event_storage.query_events(query)

        # Verify pagination is applied
        assert len(events) <= 10  # Should respect limit

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_ordering(self, event_storage, mock_metadata_repo, mock_serializer, sample_trade_event):
        """Test querying with ordering parameters."""
        query = EventQuery(order_by="timestamp", order_desc=True)

        # Setup mocks
        mock_metadata_repo.list_keys.return_value = ["event:id1", "event:id2"]
        mock_metadata_repo.get.side_effect = [
            {"data": b'{"event": "data1"}', "timestamp": "2023-01-01T00:00:00Z"},
            {"data": b'{"event": "data2"}', "timestamp": "2023-01-02T00:00:00Z"},
        ]
        mock_serializer.deserialize.return_value = sample_trade_event

        # Execute
        events = await event_storage.query_events(query)

        # Verify events are returned (ordering logic will be tested in integration tests)
        assert len(events) == 2


class TestInMemoryEventStorageIntegration:
    """Integration tests for InMemoryEventStorage with real MetadataRepository."""

    @pytest_asyncio.fixture
    async def real_metadata_repo(self):
        """Create a real InMemoryMetadataRepository for integration testing."""
        from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository

        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest.fixture
    def real_serializer(self):
        """Create a real MemoryEventSerializer for integration testing."""
        from core.implementations.memory.event.event_serializer import MemoryEventSerializer

        return MemoryEventSerializer()

    @pytest_asyncio.fixture
    async def integrated_event_storage(self, real_serializer, real_metadata_repo):
        """Create InMemoryEventStorage with real dependencies."""
        from core.implementations.memory.storage.event_storage import InMemoryEventStorage

        storage = InMemoryEventStorage(serializer=real_serializer, metadata_repository=real_metadata_repo)
        yield storage
        await storage.close()

    @pytest.fixture
    def sample_trade(self):
        """Create a sample Trade object for testing."""
        from decimal import Decimal
        from core.models.data.trade import Trade
        from core.models.data.enum import TradeSide

        return Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.50"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )

    @pytest.fixture
    def sample_trade_event(self, sample_trade):
        """Create a sample TradeEvent for testing."""
        from core.models.event.trade_event import TradeEvent
        from core.models.event.event_priority import EventPriority

        return TradeEvent(source="test_exchange", symbol="BTC/USDT", data=sample_trade, priority=EventPriority.NORMAL)

    @pytest.fixture
    def sample_kline(self):
        """Create a sample Kline object for testing."""
        from decimal import Decimal
        from core.models.data.kline import Kline
        from core.models.data.enum import KlineInterval

        now = datetime.now(UTC)
        open_time = now.replace(second=0, microsecond=0)
        close_time = open_time.replace(minute=open_time.minute + 1)
        return Kline(
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

    @pytest.fixture
    def sample_kline_event(self, sample_kline):
        """Create a sample KlineEvent for testing."""
        from core.models.event.Kline_event import KlineEvent
        from core.models.event.event_priority import EventPriority

        return KlineEvent(source="test_exchange", symbol="BTC/USDT", data=sample_kline, priority=EventPriority.NORMAL)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_event_lifecycle(self, integrated_event_storage, sample_trade_event):
        """Test complete event lifecycle: store, retrieve, query, delete."""
        # Store event
        storage_id = await integrated_event_storage.store_event(sample_trade_event)
        assert storage_id is not None

        # Retrieve event
        retrieved_event = await integrated_event_storage.retrieve_event(storage_id)
        assert retrieved_event is not None
        assert retrieved_event.event_id == sample_trade_event.event_id

        # Query events
        query = EventQuery(event_types=[EventType.TRADE])
        events = await integrated_event_storage.query_events(query)
        assert len(events) >= 1

        # Delete event
        deleted = await integrated_event_storage.delete_event(storage_id)
        assert deleted is True

        # Verify deletion
        retrieved_after_delete = await integrated_event_storage.retrieve_event(storage_id)
        assert retrieved_after_delete is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_operations(self, integrated_event_storage, sample_trade_event, sample_kline_event):
        """Test batch storage and retrieval operations."""
        events = [sample_trade_event, sample_kline_event]

        # Store batch
        storage_ids = await integrated_event_storage.store_events(events)
        assert len(storage_ids) == 2

        # Retrieve all
        retrieved_events = []
        for storage_id in storage_ids:
            event = await integrated_event_storage.retrieve_event(storage_id)
            if event:
                retrieved_events.append(event)

        assert len(retrieved_events) == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_queries(self, integrated_event_storage, sample_trade_event, sample_kline_event):
        """Test complex query scenarios."""
        # Store different types of events
        await integrated_event_storage.store_event(sample_trade_event)
        await integrated_event_storage.store_event(sample_kline_event)

        # Query by type
        trade_query = EventQuery(event_types=[EventType.TRADE])
        trade_events = await integrated_event_storage.query_events(trade_query)
        assert len(trade_events) >= 1

        # Query by symbol
        symbol_query = EventQuery(symbols=["BTC/USDT"])
        symbol_events = await integrated_event_storage.query_events(symbol_query)
        assert len(symbol_events) >= 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_large_dataset(self, integrated_event_storage, sample_trade_event):
        """Test streaming functionality with larger dataset."""
        # Store multiple events
        events = []
        for i in range(50):
            event = TradeEvent(
                source=f"exchange_{i % 3}",
                symbol="BTC/USDT",
                data=sample_trade_event.data,
                priority=EventPriority.NORMAL,
            )
            events.append(event)

        await integrated_event_storage.store_events(events)

        # Stream events
        query = EventQuery(event_types=[EventType.TRADE])
        streamed_count = 0
        async for event in integrated_event_storage.stream_events(query):
            streamed_count += 1
            assert isinstance(event, TradeEvent)

        assert streamed_count >= 50

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_statistics_accuracy(self, integrated_event_storage, sample_trade_event, sample_kline_event):
        """Test that statistics accurately reflect stored data."""
        # Store events
        await integrated_event_storage.store_event(sample_trade_event)
        await integrated_event_storage.store_event(sample_kline_event)

        # Get statistics
        stats = await integrated_event_storage.get_stats()

        # Verify statistics
        assert stats.total_events >= 2
        assert EventType.TRADE.value in stats.events_by_type
        assert EventType.KLINE.value in stats.events_by_type
        assert stats.events_by_type[EventType.TRADE.value] >= 1
        assert stats.events_by_type[EventType.KLINE.value] >= 1
