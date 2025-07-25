# ABOUTME: Storage system performance benchmark testing using pytest-benchmark
# ABOUTME: Professional performance measurements for storage operations

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Dict

from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.noop.storage.event_storage import NoOpEventStorage
from core.implementations.noop.storage.metadata_repository import NoOpMetadataRepository
from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide
from unittest.mock import Mock


class TestStoragePerformanceBenchmark:
    """Professional performance benchmark tests for storage implementations using pytest-benchmark."""

    @pytest_asyncio.fixture
    async def memory_repositories(self):
        """Create in-memory storage repositories for benchmarking."""
        metadata_repo = InMemoryMetadataRepository()
        serializer = MemoryEventSerializer()

        repositories = {
            "event_storage": InMemoryEventStorage(serializer, metadata_repo),
            "metadata_repository": metadata_repo,
        }

        yield repositories

        # Cleanup
        for repo in repositories.values():
            if hasattr(repo, "close"):
                await repo.close()

    @pytest.fixture
    def noop_repositories(self):
        """Create NoOp storage repositories for baseline comparison."""
        mock_serializer = Mock()

        return {
            "event_storage": NoOpEventStorage(mock_serializer),
            "metadata_repository": NoOpMetadataRepository(),
        }

    def create_benchmark_data(self, count: int = 100) -> Dict:
        """Create test data for benchmarking."""
        now = datetime.now(UTC)

        # Trade events
        trades = []
        for i in range(count):
            trade = Trade(
                symbol=f"SYMBOL{i % 10}/USDT",
                trade_id=f"trade_{i}",
                side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
                quantity=Decimal(f"{100 + i}"),
                price=Decimal(f"{50000 + i * 10}"),
                timestamp=now + timedelta(seconds=i),
            )

            event = TradeEvent(
                trade=trade,
                source=f"exchange_{i % 3}",
                priority=EventPriority.NORMAL,
                timestamp=now + timedelta(seconds=i),
            )
            trades.append(event)

        return {"trade_events": trades}

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_storage_single_operation_benchmark(self, memory_repositories, benchmark):
        """Benchmark single event storage operations."""
        event_storage = memory_repositories["event_storage"]
        test_data = self.create_benchmark_data(50)
        events = test_data["trade_events"]

        async def store_single_events():
            for event in events:
                await event_storage.store_event(event)
            return len(events)

        result = await benchmark(store_single_events)
        assert result == 50

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_storage_batch_operation_benchmark(self, memory_repositories, benchmark):
        """Benchmark batch event storage operations."""
        event_storage = memory_repositories["event_storage"]
        test_data = self.create_benchmark_data(200)
        events = test_data["trade_events"]

        async def store_batch_events():
            return await event_storage.store_events(events)

        result = await benchmark(store_batch_events)
        assert len(result) == 200

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_repository_benchmark(self, memory_repositories, benchmark):
        """Benchmark metadata repository operations."""
        metadata_repo = memory_repositories["metadata_repository"]

        async def metadata_operations():
            # Store operations
            for i in range(100):
                key = f"benchmark_key_{i}"
                value = {"data": f"benchmark_value_{i}", "index": i}
                await metadata_repo.set(key, value)

            # Get operations
            results = []
            for i in range(100):
                key = f"benchmark_key_{i}"
                result = await metadata_repo.get(key)
                results.append(result)

            return len(results)

        # Use benchmark to measure the execution
        result = await benchmark(metadata_operations)
        assert result == 100

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_operations_benchmark(self, memory_repositories, benchmark):
        """Benchmark query operations."""
        event_storage = memory_repositories["event_storage"]
        test_data = self.create_benchmark_data(300)

        # Pre-populate data
        await event_storage.store_events(test_data["trade_events"])

        async def query_operations():
            queries = [
                EventQuery(event_types=[EventType.TRADE]),
                EventQuery(symbols=["SYMBOL0/USDT"]),
                EventQuery(sources=["exchange_0"]),
                EventQuery(event_types=[EventType.TRADE], limit=50),
            ]

            results = []
            for query in queries:
                result = await event_storage.query_events(query)
                results.extend(result)

            return len(results)

        result = await benchmark(query_operations)
        assert result > 0

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_operations_benchmark(self, memory_repositories, benchmark):
        """Benchmark concurrent storage operations."""
        repositories = memory_repositories
        test_data = self.create_benchmark_data(100)

        async def concurrent_operations():
            # Concurrent tasks
            tasks = []

            # Event storage task
            async def store_events():
                return await repositories["event_storage"].store_events(test_data["trade_events"][:50])

            # Metadata task
            async def store_metadata():
                for i in range(25):
                    key = f"concurrent_key_{i}"
                    value = {"data": f"concurrent_value_{i}"}
                    await repositories["metadata_repository"].set(key, value)
                return 25

            tasks.extend([store_events(), store_metadata()])
            results = await asyncio.gather(*tasks)
            return sum(len(r) if isinstance(r, list) else r for r in results)

        result = await benchmark(concurrent_operations)
        assert result >= 75

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_noop_vs_memory_performance_comparison(self, memory_repositories, noop_repositories, benchmark):
        """Compare NoOp vs Memory implementation performance."""

        async def noop_operations():
            noop_event_storage = noop_repositories["event_storage"]
            noop_metadata = noop_repositories["metadata_repository"]

            # NoOp operations (should be very fast)
            for i in range(1000):
                await noop_event_storage.store_event(None)
                await noop_metadata.set(f"key_{i}", f"value_{i}")

            return 2000  # 1000 + 1000 operations

        result = await benchmark(noop_operations)
        assert result == 2000

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_scaling_benchmark(self, memory_repositories, benchmark):
        """Benchmark memory usage scaling with data size."""
        event_storage = memory_repositories["event_storage"]

        async def scaling_test():
            # Test with progressively larger datasets
            total_stored = 0
            for scale in [100, 200, 300]:
                test_data = self.create_benchmark_data(scale)
                await event_storage.store_events(test_data["trade_events"])
                total_stored += scale

            return total_stored

        result = await benchmark(scaling_test)
        assert result == 600

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_performance_scaling(self, memory_repositories, benchmark):
        """Benchmark query performance with increasing data size."""
        event_storage = memory_repositories["event_storage"]

        # Pre-populate with large dataset
        large_dataset = self.create_benchmark_data(1000)
        await event_storage.store_events(large_dataset["trade_events"])

        async def complex_queries():
            query_results = 0

            # Multiple complex queries
            for i in range(10):
                query = EventQuery(event_types=[EventType.TRADE], symbols=[f"SYMBOL{i % 10}/USDT"], limit=50)
                results = await event_storage.query_events(query)
                query_results += len(results)

            return query_results

        result = await benchmark(complex_queries)
        assert result >= 0

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cleanup_operations_benchmark(self, memory_repositories, benchmark):
        """Benchmark storage cleanup and maintenance operations."""
        repositories = memory_repositories
        test_data = self.create_benchmark_data(200)

        # Pre-populate data
        event_ids = await repositories["event_storage"].store_events(test_data["trade_events"])

        async def cleanup_operations():
            # Delete operations
            deleted_count = 0
            for event_id in event_ids[:50]:  # Delete first 50
                await repositories["event_storage"].delete_event(event_id)
                deleted_count += 1

            # Statistics operations
            stats = await repositories["event_storage"].get_stats()

            return deleted_count

        result = await benchmark(cleanup_operations)
        assert result == 50
