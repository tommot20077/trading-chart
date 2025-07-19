# ABOUTME: Storage system stability verification with internal cycle testing
# ABOUTME: Memory leak detection, resource monitoring, and error handling verification

import pytest
import pytest_asyncio
import asyncio
import psutil
import gc
import os
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Dict
from dataclasses import dataclass

from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.noop.event_storage import NoOpEventStorage
from core.implementations.noop.storage.metadata_repository import NoOpMetadataRepository
from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide
from unittest.mock import Mock


@dataclass
class StabilityMetrics:
    """Stability metrics data structure."""

    operation_name: str
    success_count: int
    error_count: int
    success_rate: float
    memory_growth_mb: float
    max_memory_mb: float
    duration_seconds: float


class ResourceMonitor:
    """Monitor system resources during testing."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.peak_memory = None

    def start_monitoring(self):
        """Start resource monitoring."""
        gc.collect()  # Clean up before monitoring
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def get_memory_growth(self) -> float:
        """Get memory growth in MB."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        return current_memory - self.start_memory if self.start_memory else 0


class TestStorageStabilityVerification:
    """Storage system stability verification with internal testing cycles."""

    # Stability thresholds
    MEMORY_LEAK_THRESHOLD = 100.0  # MB
    SUCCESS_RATE_THRESHOLD = 0.95  # 95%
    MAX_MEMORY_THRESHOLD = 300.0  # MB

    @pytest_asyncio.fixture
    async def memory_repositories(self):
        """Create in-memory storage repositories for stability testing."""
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
        """Create NoOp storage repositories for baseline testing."""
        mock_serializer = Mock()

        return {
            "event_storage": NoOpEventStorage(mock_serializer),
            "metadata_repository": NoOpMetadataRepository(),
        }

    @pytest.fixture
    def resource_monitor(self):
        """Resource monitoring fixture."""
        monitor = ResourceMonitor()
        yield monitor

    def create_test_data(self, count: int = 50) -> Dict:
        """Create test data for stability testing."""
        now = datetime.now(UTC)

        trades = []
        for i in range(count):
            trade = Trade(
                symbol=f"SYMBOL{i % 5}/USDT",
                trade_id=f"trade_{i}",
                side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
                quantity=Decimal(f"{100 + i}"),
                price=Decimal(f"{50000 + i * 10}"),
                timestamp=now + timedelta(seconds=i),
            )

            event = TradeEvent(
                trade=trade,
                source=f"exchange_{i % 2}",
                priority=EventPriority.NORMAL,
                timestamp=now + timedelta(seconds=i),
            )
            trades.append(event)

        return {"trade_events": trades}

    def _validate_stability_metrics(self, metrics: StabilityMetrics):
        """Validate stability metrics against thresholds."""
        assert metrics.success_rate >= self.SUCCESS_RATE_THRESHOLD, (
            f"Success rate {metrics.success_rate:.2%} below threshold {self.SUCCESS_RATE_THRESHOLD:.2%}"
        )

        assert metrics.memory_growth_mb < self.MEMORY_LEAK_THRESHOLD, (
            f"Memory growth {metrics.memory_growth_mb:.2f}MB exceeds threshold {self.MEMORY_LEAK_THRESHOLD}MB"
        )

        assert metrics.max_memory_mb < self.MAX_MEMORY_THRESHOLD, (
            f"Peak memory {metrics.max_memory_mb:.2f}MB exceeds threshold {self.MAX_MEMORY_THRESHOLD}MB"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_storage_stability_with_cycles(self, memory_repositories, resource_monitor):
        """Test event storage stability with internal cycles."""
        event_storage = memory_repositories["event_storage"]
        monitor = resource_monitor
        monitor.start_monitoring()

        success_count = 0
        error_count = 0
        start_time = asyncio.get_event_loop().time()

        print("🔄 Running event storage stability test (10 cycles)...")

        for cycle in range(10):
            try:
                test_data = self.create_test_data(25)

                # Store events
                event_ids = await event_storage.store_events(test_data["trade_events"])
                assert len(event_ids) == 25

                # Query events
                query = EventQuery(event_types=[EventType.TRADE])
                results = await event_storage.query_events(query)
                assert len(results) >= 25

                # Update peak memory
                monitor.update_peak_memory()

                success_count += 1
                print(f"  ✅ Cycle {cycle + 1}/10 completed")

            except Exception as e:
                error_count += 1
                print(f"  ❌ Cycle {cycle + 1}/10 failed: {e}")

        # Calculate metrics
        duration = asyncio.get_event_loop().time() - start_time
        success_rate = success_count / 10
        memory_growth = monitor.get_memory_growth()

        metrics = StabilityMetrics(
            operation_name="event_storage_cycles",
            success_count=success_count,
            error_count=error_count,
            success_rate=success_rate,
            memory_growth_mb=memory_growth,
            max_memory_mb=monitor.peak_memory,
            duration_seconds=duration,
        )

        print(
            f"📊 Stability Results: {success_rate:.1%} success rate, "
            f"{memory_growth:.2f}MB growth, {duration:.2f}s duration"
        )

        self._validate_stability_metrics(metrics)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_repository_stability_cycles(self, memory_repositories, resource_monitor):
        """Test metadata repository stability with internal cycles."""
        metadata_repo = memory_repositories["metadata_repository"]
        monitor = resource_monitor
        monitor.start_monitoring()

        success_count = 0
        error_count = 0
        start_time = asyncio.get_event_loop().time()

        print("🔄 Running metadata repository stability test (10 cycles)...")

        for cycle in range(10):
            try:
                # Store and retrieve metadata
                for i in range(20):
                    key = f"stability_key_{cycle}_{i}"
                    value = {"data": f"stability_value_{cycle}_{i}", "cycle": cycle}

                    await metadata_repo.set(key, value)
                    retrieved = await metadata_repo.get(key)
                    assert retrieved == value

                monitor.update_peak_memory()
                success_count += 1
                print(f"  ✅ Cycle {cycle + 1}/10 completed")

            except Exception as e:
                error_count += 1
                print(f"  ❌ Cycle {cycle + 1}/10 failed: {e}")

        # Calculate metrics
        duration = asyncio.get_event_loop().time() - start_time
        success_rate = success_count / 10
        memory_growth = monitor.get_memory_growth()

        metrics = StabilityMetrics(
            operation_name="metadata_cycles",
            success_count=success_count,
            error_count=error_count,
            success_rate=success_rate,
            memory_growth_mb=memory_growth,
            max_memory_mb=monitor.peak_memory,
            duration_seconds=duration,
        )

        print(
            f"📊 Stability Results: {success_rate:.1%} success rate, "
            f"{memory_growth:.2f}MB growth, {duration:.2f}s duration"
        )

        self._validate_stability_metrics(metrics)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_combined_operations_stability(self, memory_repositories, resource_monitor):
        """Test combined storage operations stability."""
        repositories = memory_repositories
        monitor = resource_monitor
        monitor.start_monitoring()

        success_count = 0
        error_count = 0
        start_time = asyncio.get_event_loop().time()

        print("🔄 Running combined operations stability test (10 cycles)...")

        for cycle in range(10):
            try:
                test_data = self.create_test_data(15)

                # Combined operations
                event_ids = await repositories["event_storage"].store_events(test_data["trade_events"])

                for i, event_id in enumerate(event_ids[:5]):
                    key = f"event_meta_{cycle}_{i}"
                    value = {"event_id": event_id, "processed": True}
                    await repositories["metadata_repository"].set(key, value)

                # Verify operations
                query = EventQuery(event_types=[EventType.TRADE])
                events = await repositories["event_storage"].query_events(query)
                assert len(events) >= 15

                monitor.update_peak_memory()
                success_count += 1
                print(f"  ✅ Cycle {cycle + 1}/10 completed")

            except Exception as e:
                error_count += 1
                print(f"  ❌ Cycle {cycle + 1}/10 failed: {e}")

        # Calculate metrics
        duration = asyncio.get_event_loop().time() - start_time
        success_rate = success_count / 10
        memory_growth = monitor.get_memory_growth()

        metrics = StabilityMetrics(
            operation_name="combined_operations",
            success_count=success_count,
            error_count=error_count,
            success_rate=success_rate,
            memory_growth_mb=memory_growth,
            max_memory_mb=monitor.peak_memory,
            duration_seconds=duration,
        )

        print(
            f"📊 Stability Results: {success_rate:.1%} success rate, "
            f"{memory_growth:.2f}MB growth, {duration:.2f}s duration"
        )

        self._validate_stability_metrics(metrics)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, memory_repositories, resource_monitor):
        """Test for memory leaks during repeated operations."""
        event_storage = memory_repositories["event_storage"]
        monitor = resource_monitor
        monitor.start_monitoring()

        print("🔍 Running memory leak detection test...")

        # Run many operations to detect leaks
        for batch in range(20):
            test_data = self.create_test_data(25)
            await event_storage.store_events(test_data["trade_events"])

            # Force garbage collection every few batches
            if batch % 5 == 0:
                gc.collect()
                monitor.update_peak_memory()
                current_growth = monitor.get_memory_growth()
                print(f"  📈 Batch {batch + 1}/20: {current_growth:.2f}MB growth")

        final_growth = monitor.get_memory_growth()

        # Memory leak validation
        assert final_growth < self.MEMORY_LEAK_THRESHOLD, f"Potential memory leak detected: {final_growth:.2f}MB growth"

        print(f"✅ Memory leak check passed: {final_growth:.2f}MB total growth")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_stability_verification(self, memory_repositories, resource_monitor):
        """Test stability under concurrent access patterns."""
        repositories = memory_repositories
        monitor = resource_monitor
        monitor.start_monitoring()

        success_count = 0
        error_count = 0

        print("🔄 Running concurrent stability test (5 cycles)...")

        for cycle in range(5):
            try:
                test_data = self.create_test_data(20)

                # Concurrent tasks
                async def store_events_task():
                    return await repositories["event_storage"].store_events(test_data["trade_events"][:10])

                async def metadata_task():
                    for i in range(10):
                        key = f"concurrent_{cycle}_{i}"
                        value = {"cycle": cycle, "index": i}
                        await repositories["metadata_repository"].set(key, value)
                    return 10

                # Run tasks concurrently
                results = await asyncio.gather(store_events_task(), metadata_task())

                # Verify results
                assert len(results[0]) == 10  # Event IDs
                assert results[1] == 10  # Metadata operations

                monitor.update_peak_memory()
                success_count += 1
                print(f"  ✅ Concurrent cycle {cycle + 1}/5 completed")

            except Exception as e:
                error_count += 1
                print(f"  ❌ Concurrent cycle {cycle + 1}/5 failed: {e}")

        success_rate = success_count / 5
        assert success_rate >= self.SUCCESS_RATE_THRESHOLD, (
            f"Concurrent stability failed: {success_rate:.1%} success rate"
        )

        print(f"✅ Concurrent stability: {success_rate:.1%} success rate")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_stability(self, memory_repositories, resource_monitor):
        """Test stability of error handling and recovery."""
        repositories = memory_repositories
        monitor = resource_monitor
        monitor.start_monitoring()

        recovery_count = 0

        print("🔄 Running error handling stability test...")

        for cycle in range(5):
            try:
                # Intentionally cause some errors and test recovery
                test_data = self.create_test_data(10)

                # Normal operations
                await repositories["event_storage"].store_events(test_data["trade_events"])

                # Try to access non-existent data (should handle gracefully)
                query = EventQuery(symbols=["NON_EXISTENT_SYMBOL"])
                results = await repositories["event_storage"].query_events(query)
                assert isinstance(results, list)  # Should return empty list, not error

                # Try to get non-existent metadata
                result = await repositories["metadata_repository"].get("non_existent_key")
                assert result is None  # Should return None, not error

                recovery_count += 1
                print(f"  ✅ Error handling cycle {cycle + 1}/5 completed")

            except Exception as e:
                print(f"  ⚠️ Unexpected error in cycle {cycle + 1}/5: {e}")

        recovery_rate = recovery_count / 5
        assert recovery_rate >= 0.8, f"Error handling stability failed: {recovery_rate:.1%} recovery rate"

        print(f"✅ Error handling stability: {recovery_rate:.1%} recovery rate")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_noop_implementations_stability(self, noop_repositories, resource_monitor):
        """Test stability of NoOp implementations."""
        repositories = noop_repositories
        monitor = resource_monitor
        monitor.start_monitoring()

        print("🔄 Running NoOp implementations stability test...")

        # NoOp operations should always succeed and be very fast
        for cycle in range(100):  # Many operations since NoOp is fast
            # These should never fail
            await repositories["event_storage"].store_event(None)
            await repositories["metadata_repository"].set(f"key_{cycle}", f"value_{cycle}")

            # Query operations
            await repositories["event_storage"].query_events(None)
            await repositories["metadata_repository"].get(f"key_{cycle}")

        memory_growth = monitor.get_memory_growth()

        # NoOp should have minimal memory impact
        assert memory_growth < 10.0, f"NoOp memory growth too high: {memory_growth:.2f}MB"

        print(f"✅ NoOp stability: {memory_growth:.2f}MB growth for 400 operations")
