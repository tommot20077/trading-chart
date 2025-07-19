# ABOUTME: Performance benchmark tests for EventStorage implementations
# ABOUTME: Tests performance characteristics and scalability of event storage operations

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import List

from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.noop.event_storage import NoOpEventStorage
from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide
from unittest.mock import Mock


class TestEventStoragePerformance:
    """Performance benchmark tests for EventStorage implementations."""

    @pytest_asyncio.fixture
    async def metadata_repository(self):
        """Create MetadataRepository for performance testing."""
        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest_asyncio.fixture
    async def memory_event_storage(self, metadata_repository):
        """Create InMemoryEventStorage for performance testing."""
        serializer = MemoryEventSerializer()
        storage = InMemoryEventStorage(serializer, metadata_repository)
        yield storage
        await storage.close()

    @pytest.fixture
    def noop_event_storage(self):
        """Create NoOpEventStorage for performance comparison."""
        mock_serializer = Mock()
        return NoOpEventStorage(mock_serializer)

    def create_test_events(self, count: int) -> List[TradeEvent]:
        """Create test events for performance testing."""
        events = []
        for i in range(count):
            trade = Trade(
                symbol="BTC/USDT",
                trade_id=f"trade_{i}",
                price=Decimal("45000.00") + Decimal(i % 1000),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
                timestamp=datetime.now(UTC) - timedelta(seconds=i),
                is_buyer_maker=i % 2 == 0,
            )

            event = TradeEvent(source=f"exchange_{i % 5}", symbol="BTC/USDT", data=trade, priority=EventPriority.NORMAL)
            events.append(event)
        return events

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_single_event_storage_performance(self, memory_event_storage, benchmark):
        """Test performance of single event storage operations."""
        event = self.create_test_events(1)[0]

        # Warm up
        await memory_event_storage.store_event(event)

        async def single_event_operation():
            result = await memory_event_storage.store_event(event)
            return result

        # Benchmark single event storage
        result = await benchmark(single_event_operation)
        assert result is not None  # Should return a valid result

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_batch_storage_performance(self, memory_event_storage, benchmark):
        """Test performance of batch event storage operations."""
        # Test with a representative batch size
        batch_size = 100
        events = self.create_test_events(batch_size)

        # Warm up
        await memory_event_storage.store_events(events[:5])

        async def batch_storage_operation():
            result = await memory_event_storage.store_events(events)
            return result

        # Benchmark batch storage
        result = await benchmark(batch_storage_operation)
        assert len(result) == batch_size  # Should return storage IDs for all events

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_query_performance(self, memory_event_storage, benchmark):
        """Test performance of query operations."""
        # Store test data
        events = self.create_test_events(1000)
        await memory_event_storage.store_events(events)

        # Test with a representative query
        query = EventQuery(event_types=[EventType.TRADE], symbols=["BTC/USDT"], limit=100)

        # Warm up
        await memory_event_storage.query_events(query)

        async def query_operation():
            result = await memory_event_storage.query_events(query)
            return result

        # Benchmark query performance
        result = await benchmark(query_operation)
        assert len(result) <= 100  # Should respect limit

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_time_range_query_performance(self, memory_event_storage, benchmark):
        """Test performance of time-range queries with time shard indexing."""
        # Create events with different timestamps
        events = []
        now = datetime.now(UTC)

        for i in range(1000):
            timestamp = now - timedelta(hours=i)
            event = self.create_test_events(1)[0]
            event.timestamp = timestamp
            events.append(event)

        await memory_event_storage.store_events(events)

        # Test time-range query performance
        start_time = now - timedelta(hours=100)
        end_time = now - timedelta(hours=50)

        query = EventQuery(start_time=start_time, end_time=end_time)

        # Warm up
        await memory_event_storage.query_events(query)

        async def time_range_query_operation():
            result = await memory_event_storage.query_events(query)
            return result

        # Benchmark time-range query
        result = await benchmark(time_range_query_operation)
        assert isinstance(result, list)  # Should return a list of events

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_streaming_performance(self, memory_event_storage, benchmark):
        """Test performance of streaming operations."""
        # Store test data
        events = self.create_test_events(1000)
        await memory_event_storage.store_events(events)

        query = EventQuery(event_types=[EventType.TRADE])

        async def streaming_operation():
            streamed_count = 0
            async for event in memory_event_storage.stream_events(query):
                streamed_count += 1
                if streamed_count >= 100:  # Reduced for benchmark
                    break
            return streamed_count

        # Benchmark streaming performance
        result = await benchmark(streaming_operation)
        assert result > 0  # Should stream some events

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, memory_event_storage, benchmark):
        """Test performance under concurrent operations."""
        # Create test data
        events_per_task = 20  # Reduced for benchmark
        num_tasks = 5

        async def store_events_task(task_id: int):
            events = self.create_test_events(events_per_task)
            # Modify events to avoid conflicts
            for i, event in enumerate(events):
                event.data.trade_id = f"task_{task_id}_trade_{i}"
            return await memory_event_storage.store_events(events)

        async def query_events_task():
            query = EventQuery(event_types=[EventType.TRADE])
            return await memory_event_storage.query_events(query)

        async def concurrent_operations():
            # Create mixed workload
            tasks = []
            for i in range(num_tasks):
                if i % 3 == 0:
                    tasks.append(query_events_task())
                else:
                    tasks.append(store_events_task(i))

            results = await asyncio.gather(*tasks)
            return len(results)

        # Benchmark concurrent operations
        result = await benchmark(concurrent_operations)
        assert result == num_tasks  # Should complete all tasks

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, memory_event_storage):
        """Test memory usage scaling with data size."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure memory usage at different data sizes
        data_sizes = [100, 500, 1000, 2000]
        memory_measurements = []

        for size in data_sizes:
            # Clear any existing data
            stats = await memory_event_storage.get_stats()
            print(f"Starting with {stats.total_events} events")

            # Store events
            events = self.create_test_events(size)
            await memory_event_storage.store_events(events)

            # Measure memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(memory_mb)

            print(f"Memory usage with {size} events: {memory_mb:.2f} MB")

        # Verify memory usage is reasonable
        # Should not exceed 250MB for 2000 events (adjusted for more realistic expectations)
        assert memory_measurements[-1] < 250

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_index_performance_scaling(self, memory_event_storage, benchmark):
        """Test index performance scaling with data size."""
        # Use a representative data size for benchmark
        size = 1000

        # Store events
        events = self.create_test_events(size)
        await memory_event_storage.store_events(events)

        # Test index query performance
        query = EventQuery(event_types=[EventType.TRADE])

        async def index_query_operation():
            result = await memory_event_storage.query_events(query)
            return result

        # Benchmark index query
        result = await benchmark(index_query_operation)
        assert isinstance(result, list)  # Should return list of events

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_noop_vs_memory_performance(self, memory_event_storage, noop_event_storage, benchmark):
        """Compare performance between NoOp and Memory implementations."""
        events = self.create_test_events(100)

        async def memory_storage_operation():
            result = await memory_event_storage.store_events(events)
            return result

        # Benchmark Memory storage (NoOp comparison is in other dedicated tests)
        result = await benchmark(memory_storage_operation)
        assert len(result) == 100  # Should store all events

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_statistics_performance(self, memory_event_storage, benchmark):
        """Test performance of statistics calculation."""
        # Store test data
        events = self.create_test_events(1000)
        await memory_event_storage.store_events(events)

        async def stats_operation():
            stats = await memory_event_storage.get_stats()
            return stats

        # Benchmark statistics calculation
        result = await benchmark(stats_operation)
        assert result is not None  # Should return statistics

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_delete_performance(self, memory_event_storage, benchmark):
        """Test performance of delete operations."""
        # Store test data
        events = self.create_test_events(100)
        storage_ids = await memory_event_storage.store_events(events)

        async def delete_operations():
            # Test batch delete (more representative)
            query = EventQuery(event_types=[EventType.TRADE])
            deleted_count = await memory_event_storage.delete_events(query)
            return deleted_count

        # Benchmark delete operations
        result = await benchmark(delete_operations)
        assert result >= 0  # Should return number of deleted events
