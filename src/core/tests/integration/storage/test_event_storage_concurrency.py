# ABOUTME: Concurrency tests for EventStorage implementations
# ABOUTME: Tests thread safety and data consistency under concurrent operations

import pytest
import pytest_asyncio
import asyncio
import time_machine
from datetime import datetime, UTC
from decimal import Decimal
from typing import List

from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide
from core.exceptions.base import StorageError


class TestEventStorageConcurrency:
    """Concurrency tests for EventStorage implementations."""

    @pytest_asyncio.fixture
    async def metadata_repository(self):
        """Create MetadataRepository for concurrency testing."""
        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest_asyncio.fixture
    async def event_storage(self, metadata_repository):
        """Create EventStorage for concurrency testing."""
        serializer = MemoryEventSerializer()
        storage = InMemoryEventStorage(serializer, metadata_repository)
        yield storage
        await storage.close()

    def create_test_event(self, trade_id: str, symbol: str = "BTC/USDT", source: str = "binance") -> TradeEvent:
        """Create a test event with unique trade_id."""
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

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_single_event_storage(self, event_storage):
        """Test concurrent single event storage operations."""
        num_tasks = 20
        events_per_task = 10

        async def store_events_task(task_id: int):
            results = []
            for i in range(events_per_task):
                event = self.create_test_event(f"task_{task_id}_event_{i}")
                storage_id = await event_storage.store_event(event)
                results.append(storage_id)
            return results

        # Run concurrent tasks
        tasks = [store_events_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify all events were stored
        all_storage_ids = []
        for task_results in results:
            all_storage_ids.extend(task_results)

        # All storage IDs should be unique
        assert len(all_storage_ids) == num_tasks * events_per_task
        assert len(set(all_storage_ids)) == len(all_storage_ids)

        # Verify we can retrieve all events
        retrieved_count = 0
        for storage_id in all_storage_ids:
            event = await event_storage.retrieve_event(storage_id)
            if event:
                retrieved_count += 1

        assert retrieved_count == len(all_storage_ids)

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_batch_storage(self, event_storage):
        """Test concurrent batch storage operations."""
        num_tasks = 10
        events_per_batch = 20

        async def batch_store_task(task_id: int):
            events = []
            for i in range(events_per_batch):
                event = self.create_test_event(f"batch_{task_id}_event_{i}")
                events.append(event)
            return await event_storage.store_events(events)

        # Run concurrent batch operations
        tasks = [batch_store_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify all batches were stored
        all_storage_ids = []
        for batch_results in results:
            all_storage_ids.extend(batch_results)

        expected_count = num_tasks * events_per_batch
        assert len(all_storage_ids) == expected_count
        assert len(set(all_storage_ids)) == len(all_storage_ids)

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, event_storage):
        """Test concurrent read and write operations."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Pre-populate some data
            initial_events = [self.create_test_event(f"initial_event_{i}") for i in range(50)]
            await event_storage.store_events(initial_events)

            num_writers = 5
            num_readers = 10
            events_per_writer = 10

        async def writer_task(writer_id: int):
            events = []
            for i in range(events_per_writer):
                event = self.create_test_event(f"writer_{writer_id}_event_{i}")
                events.append(event)
            return await event_storage.store_events(events)

        async def reader_task(reader_id: int):
            results = []
            for _ in range(10):  # Each reader performs 10 read operations
                query = EventQuery(event_types=[EventType.TRADE])
                events = await event_storage.query_events(query)
                results.append(len(events))
                traveller.shift(0.001)  # Small delay to allow interleaving
            return results

        # Run concurrent read/write operations
        writer_tasks = [writer_task(i) for i in range(num_writers)]
        reader_tasks = [reader_task(i) for i in range(num_readers)]

        all_tasks = writer_tasks + reader_tasks
        results = await asyncio.gather(*all_tasks)

        # Verify writers completed successfully
        writer_results = results[:num_writers]
        for writer_result in writer_results:
            assert len(writer_result) == events_per_writer

        # Verify readers got consistent results
        reader_results = results[num_writers:]
        for reader_result in reader_results:
            assert len(reader_result) == 10
            assert all(count >= 50 for count in reader_result)  # At least initial events

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_query_operations(self, event_storage):
        """Test concurrent query operations."""
        # Store test data with different attributes
        events = []
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        sources = ["binance", "kraken", "coinbase"]

        for i in range(100):
            symbol = symbols[i % len(symbols)]
            source = sources[i % len(sources)]
            event = self.create_test_event(f"query_test_{i}", symbol, source)
            events.append(event)

        await event_storage.store_events(events)

        # Define different query types
        queries = [
            EventQuery(event_types=[EventType.TRADE]),
            EventQuery(symbols=["BTC/USDT"]),
            EventQuery(sources=["binance"]),
            EventQuery(symbols=["ETH/USDT"], sources=["kraken"]),
            EventQuery(event_types=[EventType.TRADE], limit=10),
        ]

        async def query_task(query_id: int):
            query = queries[query_id % len(queries)]
            results = []
            for _ in range(10):
                events = await event_storage.query_events(query)
                results.append(len(events))
            return results

        # Run concurrent queries
        num_tasks = 20
        tasks = [query_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify all queries completed successfully
        for result in results:
            assert len(result) == 10
            assert all(count >= 0 for count in result)

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_streaming(self, event_storage):
        """Test concurrent streaming operations."""
        # Store test data
        events = [self.create_test_event(f"stream_test_{i}") for i in range(200)]
        await event_storage.store_events(events)

        async def streaming_task(task_id: int):
            query = EventQuery(event_types=[EventType.TRADE])
            streamed_events = []
            async for event in event_storage.stream_events(query):
                streamed_events.append(event)
                # Limit to prevent infinite streaming
                if len(streamed_events) >= 50:
                    break
            return len(streamed_events)

        # Run concurrent streaming operations
        num_tasks = 10
        tasks = [streaming_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify all streaming operations completed successfully
        for result in results:
            assert result >= 50

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_delete_operations(self, event_storage):
        """Test concurrent delete operations."""
        # Store test data
        events = [self.create_test_event(f"delete_test_{i}") for i in range(100)]
        storage_ids = await event_storage.store_events(events)

        # Split storage IDs for different delete operations
        single_delete_ids = storage_ids[:50]
        batch_delete_remaining = storage_ids[50:]

        async def single_delete_task(storage_ids: List[str]):
            deleted_count = 0
            for storage_id in storage_ids:
                if await event_storage.delete_event(storage_id):
                    deleted_count += 1
            return deleted_count

        async def batch_delete_task():
            query = EventQuery(event_types=[EventType.TRADE])
            return await event_storage.delete_events(query)

        # Run concurrent delete operations
        single_delete_tasks = [
            single_delete_task(single_delete_ids[i : i + 10]) for i in range(0, len(single_delete_ids), 10)
        ]
        batch_delete_tasks = [batch_delete_task()]

        all_tasks = single_delete_tasks + batch_delete_tasks
        results = await asyncio.gather(*all_tasks)

        # Verify delete operations completed
        single_delete_results = results[:-1]
        batch_delete_results = results[-1:]

        total_single_deleted = sum(single_delete_results)
        total_batch_deleted = sum(batch_delete_results)

        assert total_single_deleted >= 0
        assert total_batch_deleted >= 0

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_index_consistency_under_concurrency(self, event_storage):
        """Test index consistency under concurrent operations."""
        num_tasks = 10
        events_per_task = 20

        async def store_task(task_id: int):
            events = []
            for i in range(events_per_task):
                # Use consistent symbol for index testing
                event = self.create_test_event(f"index_test_{task_id}_{i}", "BTC/USDT")
                events.append(event)
            return await event_storage.store_events(events)

        # Store events concurrently
        tasks = [store_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify index consistency
        query = EventQuery(symbols=["BTC/USDT"])
        events = await event_storage.query_events(query)

        expected_count = num_tasks * events_per_task
        assert len(events) >= expected_count

        # Verify all events have correct symbol
        for event in events:
            assert event.symbol == "BTC/USDT"

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_statistics_consistency_under_concurrency(self, event_storage):
        """Test statistics consistency under concurrent operations."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Store initial data
            initial_events = [self.create_test_event(f"stats_test_{i}") for i in range(50)]
            await event_storage.store_events(initial_events)

            async def stats_task():
                results = []
                for _ in range(10):
                    stats = await event_storage.get_stats()
                    results.append(stats.total_events)
                    traveller.shift(0.001)
                return results

            async def store_task():
                events = [self.create_test_event(f"concurrent_stats_{i}") for i in range(10)]
                return await event_storage.store_events(events)

            # Run concurrent stats and store operations
            stats_tasks = [stats_task() for _ in range(5)]
            store_tasks = [store_task() for _ in range(3)]

            all_tasks = stats_tasks + store_tasks
            results = await asyncio.gather(*all_tasks)

            # Verify statistics are consistent
            stats_results = results[:5]
            for stats_result in stats_results:
                assert len(stats_result) == 10
                assert all(count >= 50 for count in stats_result)

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, event_storage):
        """Test concurrent health check operations."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:

            async def health_check_task():
                results = []
                for _ in range(20):
                    health = await event_storage.health_check()
                    results.append(health)
                    traveller.shift(0.001)
                return results

            async def store_task():
                event = self.create_test_event("health_check_test")
                return await event_storage.store_event(event)

            # Run concurrent health checks and operations
            health_tasks = [health_check_task() for _ in range(5)]
            store_tasks = [store_task() for _ in range(10)]

            all_tasks = health_tasks + store_tasks
            results = await asyncio.gather(*all_tasks)

            # Verify health checks are consistent
            health_results = results[:5]
            for health_result in health_results:
                assert len(health_result) == 20
                assert all(health is True for health in health_result)

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_error_handling_under_concurrency(self, event_storage):
        """Test error handling under concurrent operations."""
        # Store some test data
        events = [self.create_test_event(f"error_test_{i}") for i in range(20)]
        storage_ids = await event_storage.store_events(events)

        # Close storage to trigger errors
        await event_storage.close()

        async def error_task():
            try:
                event = self.create_test_event("error_event")
                await event_storage.store_event(event)
                return "success"
            except StorageError:
                return "error"
            except Exception as e:
                return f"unexpected_error: {type(e).__name__}"

        # Run concurrent operations that should fail
        num_tasks = 10
        tasks = [error_task() for _ in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify all operations failed appropriately
        for result in results:
            assert result == "error"

    @pytest.mark.concurrency
    @pytest.mark.asyncio
    async def test_resource_cleanup_under_concurrency(self, event_storage):
        """Test resource cleanup under concurrent operations."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Store events and then close while operations might be running
            events = [self.create_test_event(f"cleanup_test_{i}") for i in range(50)]
            await event_storage.store_events(events)

            async def query_task():
                try:
                    query = EventQuery(event_types=[EventType.TRADE])
                    return await event_storage.query_events(query)
                except Exception:
                    return []

            async def close_task():
                traveller.shift(0.01)  # Small delay
                await event_storage.close()

            # Run queries concurrently with close
            query_tasks = [query_task() for _ in range(5)]
            close_task_list = [close_task()]

            all_tasks = query_tasks + close_task_list
            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            # Verify cleanup completed without hanging
            assert len(results) == 6  # 5 queries + 1 close

            # Verify subsequent operations fail
            with pytest.raises(StorageError):
                await event_storage.store_event(self.create_test_event("after_close"))
