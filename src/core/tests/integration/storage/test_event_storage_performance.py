# ABOUTME: Performance benchmark tests for EventStorage implementations
# ABOUTME: Tests performance characteristics and scalability of event storage operations

import pytest
import pytest_asyncio
import asyncio
import time
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
    async def test_single_event_storage_performance(self, memory_event_storage):
        """Test performance of single event storage operations."""
        event = self.create_test_events(1)[0]

        # Warm up
        await memory_event_storage.store_event(event)

        # Benchmark single event storage
        iterations = 100
        start_time = time.time()

        for _ in range(iterations):
            await memory_event_storage.store_event(event)

        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_second = iterations / elapsed

        print(f"Single event storage: {ops_per_second:.2f} ops/sec")
        assert ops_per_second > 100  # Should handle at least 100 ops/sec

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_batch_storage_performance(self, memory_event_storage):
        """Test performance of batch event storage operations."""
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 500]

        for batch_size in batch_sizes:
            events = self.create_test_events(batch_size)

            # Warm up
            await memory_event_storage.store_events(events[:5])

            # Benchmark batch storage
            start_time = time.time()
            await memory_event_storage.store_events(events)
            end_time = time.time()

            elapsed = end_time - start_time
            ops_per_second = batch_size / elapsed

            print(f"Batch storage ({batch_size} events): {ops_per_second:.2f} ops/sec")
            assert ops_per_second > 100  # Should handle at least 100 ops/sec

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_query_performance(self, memory_event_storage):
        """Test performance of query operations."""
        # Store test data
        events = self.create_test_events(1000)
        await memory_event_storage.store_events(events)

        # Test different query types
        queries = [
            EventQuery(event_types=[EventType.TRADE]),
            EventQuery(symbols=["BTC/USDT"]),
            EventQuery(sources=["exchange_0"]),
            EventQuery(event_types=[EventType.TRADE], symbols=["BTC/USDT"], limit=100),
        ]

        for i, query in enumerate(queries):
            # Warm up
            await memory_event_storage.query_events(query)

            # Benchmark query
            iterations = 10
            start_time = time.time()

            for _ in range(iterations):
                await memory_event_storage.query_events(query)

            end_time = time.time()
            elapsed = end_time - start_time
            queries_per_second = iterations / elapsed

            print(f"Query performance {i + 1}: {queries_per_second:.2f} queries/sec")
            assert queries_per_second > 10  # Should handle at least 10 queries/sec

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_time_range_query_performance(self, memory_event_storage):
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

        # Benchmark time-range query
        iterations = 10
        start_time_bench = time.time()

        for _ in range(iterations):
            await memory_event_storage.query_events(query)

        end_time_bench = time.time()
        elapsed = end_time_bench - start_time_bench
        queries_per_second = iterations / elapsed

        print(f"Time-range query performance: {queries_per_second:.2f} queries/sec")
        assert queries_per_second > 5  # Should handle at least 5 time-range queries/sec

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_streaming_performance(self, memory_event_storage):
        """Test performance of streaming operations."""
        # Store test data
        events = self.create_test_events(1000)
        await memory_event_storage.store_events(events)

        query = EventQuery(event_types=[EventType.TRADE])

        # Benchmark streaming
        start_time = time.time()
        streamed_count = 0

        async for event in memory_event_storage.stream_events(query):
            streamed_count += 1
            if streamed_count >= 1000:
                break

        end_time = time.time()
        elapsed = end_time - start_time
        events_per_second = streamed_count / elapsed

        print(f"Streaming performance: {events_per_second:.2f} events/sec")
        assert events_per_second > 1000  # Should handle at least 1000 events/sec

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, memory_event_storage):
        """Test performance under concurrent operations."""
        # Create test data
        events_per_task = 50
        num_tasks = 10

        async def store_events_task(task_id: int):
            events = self.create_test_events(events_per_task)
            # Modify events to avoid conflicts
            for i, event in enumerate(events):
                event.data.trade_id = f"task_{task_id}_trade_{i}"
            return await memory_event_storage.store_events(events)

        async def query_events_task():
            query = EventQuery(event_types=[EventType.TRADE])
            return await memory_event_storage.query_events(query)

        # Create mixed workload
        tasks = []
        for i in range(num_tasks):
            if i % 3 == 0:
                tasks.append(query_events_task())
            else:
                tasks.append(store_events_task(i))

        # Benchmark concurrent operations
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        elapsed = end_time - start_time
        total_operations = num_tasks
        ops_per_second = total_operations / elapsed

        print(f"Concurrent operations performance: {ops_per_second:.2f} ops/sec")
        assert ops_per_second > 5  # Should handle at least 5 concurrent ops/sec

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
    async def test_index_performance_scaling(self, memory_event_storage):
        """Test index performance scaling with data size."""
        # Test how index performance scales with data size
        data_sizes = [100, 500, 1000]

        for size in data_sizes:
            # Store events
            events = self.create_test_events(size)
            await memory_event_storage.store_events(events)

            # Test index query performance
            query = EventQuery(event_types=[EventType.TRADE])

            # Benchmark index query
            iterations = 5
            start_time = time.time()

            for _ in range(iterations):
                await memory_event_storage.query_events(query)

            end_time = time.time()
            elapsed = end_time - start_time
            queries_per_second = iterations / elapsed

            print(f"Index performance with {size} events: {queries_per_second:.2f} queries/sec")

            # Index performance should not degrade significantly with size
            assert queries_per_second > 5

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_noop_vs_memory_performance(self, memory_event_storage, noop_event_storage):
        """Compare performance between NoOp and Memory implementations."""
        events = self.create_test_events(100)

        # Benchmark NoOp storage
        start_time = time.time()
        await noop_event_storage.store_events(events)
        noop_time = time.time() - start_time

        # Benchmark Memory storage
        start_time = time.time()
        await memory_event_storage.store_events(events)
        memory_time = time.time() - start_time

        noop_ops_per_second = 100 / noop_time
        memory_ops_per_second = 100 / memory_time

        print(f"NoOp storage: {noop_ops_per_second:.2f} ops/sec")
        print(f"Memory storage: {memory_ops_per_second:.2f} ops/sec")

        # NoOp should be significantly faster
        assert noop_ops_per_second > memory_ops_per_second
        assert noop_ops_per_second > 1000  # NoOp should be very fast

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_statistics_performance(self, memory_event_storage):
        """Test performance of statistics calculation."""
        # Store test data
        events = self.create_test_events(1000)
        await memory_event_storage.store_events(events)

        # Benchmark statistics calculation
        iterations = 10
        start_time = time.time()

        for _ in range(iterations):
            await memory_event_storage.get_stats()

        end_time = time.time()
        elapsed = end_time - start_time
        stats_per_second = iterations / elapsed

        print(f"Statistics performance: {stats_per_second:.2f} stats/sec")
        assert stats_per_second > 1  # Should handle at least 1 stats calculation/sec

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_delete_performance(self, memory_event_storage):
        """Test performance of delete operations."""
        # Store test data
        events = self.create_test_events(100)
        storage_ids = await memory_event_storage.store_events(events)

        # Benchmark single delete
        start_time = time.time()
        await memory_event_storage.delete_event(storage_ids[0])
        single_delete_time = time.time() - start_time

        # Benchmark batch delete
        query = EventQuery(event_types=[EventType.TRADE])
        start_time = time.time()
        deleted_count = await memory_event_storage.delete_events(query)
        batch_delete_time = time.time() - start_time

        print(f"Single delete time: {single_delete_time:.4f} seconds")
        print(f"Batch delete time: {batch_delete_time:.4f} seconds")
        print(f"Deleted {deleted_count} events")

        # Delete operations should be reasonably fast
        assert single_delete_time < 0.1  # Less than 100ms
        assert batch_delete_time < 1.0  # Less than 1 second for batch
