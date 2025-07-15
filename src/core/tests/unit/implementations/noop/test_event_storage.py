# ABOUTME: Unit tests for NoOpEventStorage implementation
# ABOUTME: Tests no-operation functionality and performance characteristics

import pytest
from datetime import datetime, UTC
from unittest.mock import Mock

from core.implementations.noop.event_storage import NoOpEventStorage
from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.models.data.event import BaseEvent
from core.models.event.event_query import EventQuery
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_storage_stats import EventStorageStats


class TestNoOpEventStorage:
    """Test suite for NoOpEventStorage implementation."""

    @pytest.fixture
    def mock_serializer(self):
        """Create a mock event serializer."""
        return Mock(spec=AbstractEventSerializer)

    @pytest.fixture
    def noop_storage(self, mock_serializer):
        """Create a NoOpEventStorage instance."""
        return NoOpEventStorage(mock_serializer)

    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return BaseEvent(
            event_type=EventType.TRADE,
            source="test_exchange",
            symbol="BTCUSDT",
            data={"price": 50000, "volume": 1.5},
            priority=EventPriority.NORMAL,
            timestamp=datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_initialization(self, mock_serializer):
        """Test NoOpEventStorage initialization."""
        storage = NoOpEventStorage(mock_serializer)

        assert storage.serializer == mock_serializer
        assert not storage._closed
        assert storage._event_count == 0

    @pytest.mark.asyncio
    async def test_store_event_returns_fake_id(self, noop_storage, sample_event):
        """Test that store_event returns a fake storage ID."""
        storage_id = await noop_storage.store_event(sample_event)

        assert isinstance(storage_id, str)
        assert storage_id.startswith("noop-")
        assert len(storage_id) > 5
        assert noop_storage._event_count == 1

    @pytest.mark.asyncio
    async def test_store_events_returns_fake_ids(self, noop_storage, sample_event):
        """Test that store_events returns fake storage IDs."""
        events = [sample_event, sample_event, sample_event]

        storage_ids = await noop_storage.store_events(events)

        assert len(storage_ids) == 3
        assert all(isinstance(sid, str) for sid in storage_ids)
        assert all(sid.startswith("noop-") for sid in storage_ids)
        assert noop_storage._event_count == 3

    @pytest.mark.asyncio
    async def test_retrieve_event_returns_none(self, noop_storage):
        """Test that retrieve_event always returns None."""
        result = await noop_storage.retrieve_event("any-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_events_returns_empty_list(self, noop_storage):
        """Test that query_events always returns empty list."""
        query = EventQuery(event_types=[EventType.TRADE])

        result = await noop_storage.query_events(query)

        assert result == []

    @pytest.mark.asyncio
    async def test_stream_events_yields_nothing(self, noop_storage):
        """Test that stream_events yields nothing."""
        query = EventQuery(event_types=[EventType.TRADE])

        events = []
        async for event in noop_storage.stream_events(query):
            events.append(event)

        assert events == []

    @pytest.mark.asyncio
    async def test_delete_event_returns_true(self, noop_storage):
        """Test that delete_event always returns True."""
        result = await noop_storage.delete_event("any-id")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_events_returns_zero(self, noop_storage):
        """Test that delete_events always returns 0."""
        query = EventQuery(event_types=[EventType.TRADE])

        result = await noop_storage.delete_events(query)

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_stats_returns_fake_stats(self, noop_storage, sample_event):
        """Test that get_stats returns fake statistics."""
        # Store some events to increment the counter
        await noop_storage.store_event(sample_event)
        await noop_storage.store_event(sample_event)

        stats = await noop_storage.get_stats()

        assert isinstance(stats, EventStorageStats)
        assert stats.total_events == 2
        assert stats.events_by_type == {}
        assert stats.storage_size_bytes == 0
        assert stats.oldest_event_time is None
        assert stats.newest_event_time is None
        assert stats.avg_event_size_bytes == 0.0

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, noop_storage):
        """Test that health_check always returns True when not closed."""
        result = await noop_storage.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_closed(self, noop_storage):
        """Test that health_check returns False when closed."""
        await noop_storage.close()

        result = await noop_storage.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_close_sets_closed_flag(self, noop_storage):
        """Test that close sets the closed flag."""
        await noop_storage.close()

        assert noop_storage._closed is True

    @pytest.mark.asyncio
    async def test_operations_after_close_raise_error(self, noop_storage, sample_event):
        """Test that operations after close raise RuntimeError."""
        await noop_storage.close()

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.store_event(sample_event)

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.store_events([sample_event])

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.retrieve_event("any-id")

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.query_events(EventQuery())

        with pytest.raises(RuntimeError, match="closed"):
            async for _ in noop_storage.stream_events(EventQuery()):
                pass

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.delete_event("any-id")

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.delete_events(EventQuery())

        with pytest.raises(RuntimeError, match="closed"):
            await noop_storage.get_stats()

    @pytest.mark.asyncio
    async def test_performance_characteristics(self, noop_storage, sample_event):
        """Test that NoOp operations are fast and don't consume resources."""
        import time

        # Test batch storage performance
        events = [sample_event] * 1000

        start_time = time.time()
        storage_ids = await noop_storage.store_events(events)
        end_time = time.time()

        # Should be very fast
        assert end_time - start_time < 0.1  # Less than 100ms
        assert len(storage_ids) == 1000
        assert noop_storage._event_count == 1000

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, noop_storage, sample_event):
        """Test concurrent operations work correctly."""
        import asyncio

        # Run multiple operations concurrently
        tasks = [noop_storage.store_event(sample_event) for _ in range(100)]

        results = await asyncio.gather(*tasks)

        assert len(results) == 100
        assert all(isinstance(r, str) for r in results)
        assert noop_storage._event_count == 100

    @pytest.mark.asyncio
    async def test_serializer_not_used(self, noop_storage, sample_event):
        """Test that the serializer is not actually used."""
        mock_serializer = noop_storage.serializer

        # Perform operations
        await noop_storage.store_event(sample_event)
        await noop_storage.retrieve_event("any-id")

        # Verify serializer methods were not called
        assert not mock_serializer.serialize.called
        assert not mock_serializer.deserialize.called

    @pytest.mark.asyncio
    async def test_empty_batch_operations(self, noop_storage):
        """Test operations with empty inputs."""
        # Empty batch storage
        result = await noop_storage.store_events([])
        assert result == []
        assert noop_storage._event_count == 0

    @pytest.mark.asyncio
    async def test_complex_query_handling(self, noop_storage):
        """Test that complex queries are handled correctly."""
        from datetime import timedelta

        # Create a complex query
        query = EventQuery(
            event_types=[EventType.TRADE, EventType.KLINE],
            symbols=["BTCUSDT", "ETHUSDT"],
            sources=["binance", "kraken"],
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC),
            limit=100,
            offset=50,
            order_by="timestamp",
            order_desc=True,
            metadata_filters={"exchange": "binance"},
        )

        # Should handle any query without errors
        result = await noop_storage.query_events(query)
        assert result == []

        # Should handle streaming too
        events = []
        async for event in noop_storage.stream_events(query):
            events.append(event)
        assert events == []

    @pytest.mark.asyncio
    async def test_stats_consistency(self, noop_storage, sample_event):
        """Test that stats remain consistent across operations."""
        # Initial stats
        stats1 = await noop_storage.get_stats()
        assert stats1.total_events == 0

        # Store some events
        await noop_storage.store_event(sample_event)
        await noop_storage.store_events([sample_event, sample_event])

        # Stats should reflect the count
        stats2 = await noop_storage.get_stats()
        assert stats2.total_events == 3

        # Delete operations don't affect the count (since nothing is actually stored)
        await noop_storage.delete_event("any-id")
        await noop_storage.delete_events(EventQuery())

        stats3 = await noop_storage.get_stats()
        assert stats3.total_events == 3  # Count unchanged
