# ABOUTME: Integration tests for complete storage workflow from DataProvider to repositories
# ABOUTME: Tests end-to-end data flow through DataProvider → EventStorage → KlineRepository → MetadataRepository

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, Any

from core.implementations.noop.data.provider import NoOpDataProvider
from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.kline_repository import InMemoryKlineRepository
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.models.data.enum import KlineInterval
from core.models.event.Kline_event import KlineEvent
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.event.event_type import EventType


class TestStorageWorkflowIntegration:
    """
    Integration tests for complete storage workflow.

    Tests the complete data flow from:
    DataProvider → EventStorage → KlineRepository → MetadataRepository

    Validates:
    - End-to-end data pipeline integrity
    - Data consistency across all layers
    - Performance of complete workflow
    - Error propagation through the pipeline
    - Concurrent workflow execution
    - Resource cleanup coordination
    """

    @pytest_asyncio.fixture
    async def data_provider(self):
        """Create and configure NoOpDataProvider."""
        provider = NoOpDataProvider()
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    async def metadata_repository(self):
        """Create a clean InMemoryMetadataRepository."""
        repo = InMemoryMetadataRepository()
        yield repo
        await repo.close()

    @pytest_asyncio.fixture
    async def kline_repository(self):
        """Create a clean InMemoryKlineRepository."""
        repo = InMemoryKlineRepository()
        yield repo
        await repo.close()

    @pytest.fixture
    def event_serializer(self):
        """Create MemoryEventSerializer."""
        return MemoryEventSerializer()

    @pytest_asyncio.fixture
    async def event_storage(self, event_serializer, metadata_repository):
        """Create InMemoryEventStorage with real dependencies."""
        storage = InMemoryEventStorage(serializer=event_serializer, metadata_repository=metadata_repository)
        yield storage
        await storage.close()

    @pytest.fixture
    def workflow_coordinator(self):
        """Create a simple workflow coordinator for testing."""

        class WorkflowCoordinator:
            def __init__(self, data_provider, event_storage, kline_repository, metadata_repository):
                self.data_provider = data_provider
                self.event_storage = event_storage
                self.kline_repository = kline_repository
                self.metadata_repository = metadata_repository

            async def process_kline_data(
                self, symbol: str, interval: KlineInterval, start_time: datetime, end_time: datetime
            ) -> Dict[str, Any]:
                """Process kline data through the complete workflow."""
                # Step 1: Fetch data from provider
                klines = await self.data_provider.get_klines(symbol, interval, start_time, end_time, limit=100)

                # Step 2: Convert to events and store in EventStorage
                events = []
                for kline in klines:
                    event = KlineEvent(
                        source=self.data_provider.name, symbol=symbol, data=kline, priority=EventPriority.NORMAL
                    )
                    events.append(event)

                event_ids = await self.event_storage.store_events(events)

                # Step 3: Store klines in KlineRepository
                saved_count = await self.kline_repository.save_batch(klines)

                # Step 4: Update metadata
                if klines:
                    await self.metadata_repository.set_last_sync_time(
                        symbol, f"klines_{interval.value}", klines[-1].close_time
                    )

                    # Store workflow status
                    workflow_status = {
                        "symbol": symbol,
                        "interval": interval.value,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "klines_fetched": len(klines),
                        "events_stored": len(event_ids),
                        "klines_saved": saved_count,
                        "completed_at": datetime.now(UTC).isoformat(),
                    }

                    await self.metadata_repository.set(f"workflow:{symbol}:{interval.value}", workflow_status)

                return {
                    "klines_fetched": len(klines),
                    "events_stored": len(event_ids),
                    "klines_saved": saved_count,
                    "event_ids": event_ids,
                }

        return WorkflowCoordinator

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test complete workflow execution from data provider to final storage."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1
        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        # Execute complete workflow
        result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)

        # Verify workflow completion
        assert result["klines_fetched"] > 0
        assert result["events_stored"] == result["klines_fetched"]
        assert result["klines_saved"] == result["klines_fetched"]
        assert len(result["event_ids"]) == result["klines_fetched"]

        # Verify data exists in EventStorage
        query = EventQuery(event_types=[EventType.KLINE], symbols=[symbol])
        events = await event_storage.query_events(query)
        assert len(events) >= result["events_stored"]

        # Verify data exists in KlineRepository
        stored_klines = await kline_repository.query(symbol, interval, start_time, end_time)
        assert len(stored_klines) == result["klines_saved"]

        # Verify metadata is updated
        sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
        assert sync_time is not None

        workflow_status = await metadata_repository.get(f"workflow:{symbol}:{interval.value}")
        assert workflow_status is not None
        assert workflow_status["klines_fetched"] == result["klines_fetched"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_symbol_workflow_execution(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test workflow execution with multiple symbols."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        interval = KlineInterval.MINUTE_5
        start_time = datetime.now(UTC) - timedelta(hours=2)
        end_time = datetime.now(UTC)

        results = {}

        # Execute workflow for each symbol
        for symbol in symbols:
            result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)
            results[symbol] = result

        # Verify all workflows completed successfully
        for symbol, result in results.items():
            assert result["klines_fetched"] > 0
            assert result["events_stored"] == result["klines_fetched"]
            assert result["klines_saved"] == result["klines_fetched"]

            # Verify symbol-specific data
            stored_klines = await kline_repository.query(symbol, interval, start_time, end_time)
            assert len(stored_klines) == result["klines_saved"]

            sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
            assert sync_time is not None

        # Verify no cross-contamination between symbols
        for symbol in symbols:
            symbol_klines = await kline_repository.query(symbol, interval, start_time, end_time)
            for kline in symbol_klines:
                assert kline.symbol == symbol

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_different_intervals(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test workflow with different time intervals."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "ETH/USDT"
        intervals = [KlineInterval.MINUTE_1, KlineInterval.MINUTE_5, KlineInterval.HOUR_1]
        start_time = datetime.now(UTC) - timedelta(hours=3)
        end_time = datetime.now(UTC)

        results = {}

        # Execute workflow for each interval
        for interval in intervals:
            result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)
            results[interval] = result

        # Verify all intervals processed successfully
        for interval, result in results.items():
            assert result["klines_fetched"] > 0

            # Verify interval-specific storage
            stored_klines = await kline_repository.query(symbol, interval, start_time, end_time)
            assert len(stored_klines) == result["klines_saved"]

            # Verify all stored klines have correct interval
            for kline in stored_klines:
                assert kline.interval == interval

            # Verify interval-specific metadata
            sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
            assert sync_time is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test concurrent workflow execution for multiple symbols and intervals."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        # Define multiple workflow tasks
        workflow_tasks = [
            ("BTC/USDT", KlineInterval.MINUTE_1),
            ("ETH/USDT", KlineInterval.MINUTE_1),
            ("BTC/USDT", KlineInterval.MINUTE_5),
            ("ETH/USDT", KlineInterval.MINUTE_5),
        ]

        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        async def execute_workflow_task(symbol: str, interval: KlineInterval):
            """Execute a single workflow task."""
            result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)
            return (symbol, interval, result)

        # Execute all workflows concurrently
        tasks = [execute_workflow_task(symbol, interval) for symbol, interval in workflow_tasks]
        results = await asyncio.gather(*tasks)

        # Verify all concurrent workflows completed successfully
        for symbol, interval, result in results:
            assert result["klines_fetched"] > 0
            assert result["events_stored"] == result["klines_fetched"]
            assert result["klines_saved"] == result["klines_fetched"]

            # Verify data isolation
            stored_klines = await kline_repository.query(symbol, interval, start_time, end_time)
            assert len(stored_klines) == result["klines_saved"]

            for kline in stored_klines:
                assert kline.symbol == symbol
                assert kline.interval == interval

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_error_handling_and_recovery(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test workflow error handling and recovery mechanisms."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "TEST/ERROR"
        interval = KlineInterval.MINUTE_1
        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        # First, execute a successful workflow
        normal_symbol = "BTC/USDT"
        normal_result = await coordinator.process_kline_data(normal_symbol, interval, start_time, end_time)
        assert normal_result["klines_fetched"] > 0

        # Test workflow with potential errors (using NoOp provider which shouldn't fail)
        # But we can test partial failure scenarios
        try:
            # This should succeed with NoOp provider
            error_result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)

            # Even if it succeeds, verify error handling doesn't corrupt normal data
            normal_klines = await kline_repository.query(normal_symbol, interval, start_time, end_time)
            assert len(normal_klines) == normal_result["klines_saved"]

        except Exception:
            # If an error occurs, verify it doesn't affect other data
            normal_klines = await kline_repository.query(normal_symbol, interval, start_time, end_time)
            assert len(normal_klines) == normal_result["klines_saved"]

            # Verify error data is not partially stored
            error_klines = await kline_repository.query(symbol, interval, start_time, end_time)
            assert len(error_klines) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_performance_monitoring(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test workflow performance monitoring and metrics collection."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1
        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        # Measure workflow execution time
        workflow_start = datetime.now(UTC)
        result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)
        workflow_end = datetime.now(UTC)

        execution_time = (workflow_end - workflow_start).total_seconds()

        # Store performance metrics
        performance_metrics = {
            "symbol": symbol,
            "interval": interval.value,
            "execution_time_seconds": execution_time,
            "klines_per_second": result["klines_fetched"] / execution_time if execution_time > 0 else 0,
            "total_klines": result["klines_fetched"],
            "workflow_start": workflow_start.isoformat(),
            "workflow_end": workflow_end.isoformat(),
        }

        await metadata_repository.set(f"performance:{symbol}:{interval.value}", performance_metrics)

        # Verify performance data
        stored_metrics = await metadata_repository.get(f"performance:{symbol}:{interval.value}")
        assert stored_metrics is not None
        assert stored_metrics["total_klines"] == result["klines_fetched"]
        assert stored_metrics["execution_time_seconds"] > 0

        # Verify reasonable performance (should be very fast with NoOp provider)
        assert execution_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_data_consistency_validation(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test data consistency validation across the complete workflow."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "ETH/USDT"
        interval = KlineInterval.MINUTE_5
        start_time = datetime.now(UTC) - timedelta(hours=2)
        end_time = datetime.now(UTC)

        # Execute workflow
        result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)

        # Validate data consistency across all storage layers

        # 1. Verify EventStorage data
        query = EventQuery(event_types=[EventType.KLINE], symbols=[symbol])
        stored_events = await event_storage.query_events(query)
        kline_events = [e for e in stored_events if isinstance(e, KlineEvent)]

        # 2. Verify KlineRepository data
        stored_klines = await kline_repository.query(symbol, interval, start_time, end_time)

        # 3. Cross-validate data consistency
        assert len(kline_events) >= result["events_stored"]
        assert len(stored_klines) == result["klines_saved"]

        # Verify each kline event corresponds to a stored kline
        event_klines = [event.data for event in kline_events if event.symbol == symbol]

        # Sort both by timestamp for comparison
        event_klines.sort(key=lambda k: k.open_time)
        stored_klines.sort(key=lambda k: k.open_time)

        # Compare key attributes
        for event_kline, stored_kline in zip(event_klines[: len(stored_klines)], stored_klines):
            assert event_kline.symbol == stored_kline.symbol
            assert event_kline.interval == stored_kline.interval
            assert event_kline.open_time == stored_kline.open_time
            assert event_kline.close_price == stored_kline.close_price

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_resource_cleanup(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test proper resource cleanup after workflow execution."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1
        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        # Execute workflow
        result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)
        assert result["klines_fetched"] > 0

        # Get initial resource usage
        initial_kline_memory = await kline_repository.get_memory_usage()
        initial_metadata_memory = await metadata_repository.get_memory_usage()
        initial_event_stats = await event_storage.get_stats()

        # Verify resources are being used
        assert initial_kline_memory["total_klines"] > 0
        assert initial_metadata_memory["total_keys"] > 0
        assert initial_event_stats.total_events > 0

        # Perform cleanup operations
        await kline_repository.clear_all()
        await metadata_repository.clear_all()

        # Verify cleanup effectiveness
        final_kline_memory = await kline_repository.get_memory_usage()
        final_metadata_memory = await metadata_repository.get_memory_usage()

        assert final_kline_memory["total_klines"] == 0
        assert final_metadata_memory["total_keys"] == 0

        # Verify repositories are still operational after cleanup
        assert not final_kline_memory["is_closed"]
        assert not final_metadata_memory["is_closed"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_large_dataset(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test workflow performance with larger datasets."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "BTC/USDT"
        interval = KlineInterval.MINUTE_1
        start_time = datetime.now(UTC) - timedelta(days=1)  # Request 1 day of data
        end_time = datetime.now(UTC)

        # Execute workflow with larger dataset
        result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)

        # Verify workflow handles larger datasets
        assert result["klines_fetched"] > 0
        assert result["events_stored"] == result["klines_fetched"]
        assert result["klines_saved"] == result["klines_fetched"]

        # Verify data integrity with larger dataset
        stored_klines = await kline_repository.query(symbol, interval, start_time, end_time)
        assert len(stored_klines) == result["klines_saved"]

        # Verify memory usage is reasonable
        memory_usage = await kline_repository.get_memory_usage()
        assert memory_usage["total_klines"] == result["klines_saved"]

        # Verify metadata is properly maintained
        sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
        assert sync_time is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_streaming_integration(
        self, data_provider, event_storage, kline_repository, metadata_repository, workflow_coordinator
    ):
        """Test workflow integration with streaming data."""
        coordinator = workflow_coordinator(data_provider, event_storage, kline_repository, metadata_repository)

        symbol = "ETH/USDT"
        interval = KlineInterval.MINUTE_1
        start_time = datetime.now(UTC) - timedelta(hours=1)
        end_time = datetime.now(UTC)

        # Execute initial workflow
        initial_result = await coordinator.process_kline_data(symbol, interval, start_time, end_time)

        # Simulate streaming updates by processing additional data
        stream_start = end_time
        stream_end = end_time + timedelta(minutes=10)

        stream_result = await coordinator.process_kline_data(symbol, interval, stream_start, stream_end)

        # Verify streaming integration
        total_klines = await kline_repository.count(symbol, interval)
        expected_total = initial_result["klines_saved"] + stream_result["klines_saved"]
        assert total_klines == expected_total

        # Verify latest sync time is updated
        final_sync_time = await metadata_repository.get_last_sync_time(symbol, f"klines_{interval.value}")
        assert final_sync_time is not None

        # Verify event storage contains all data
        query = EventQuery(event_types=[EventType.KLINE], symbols=[symbol])
        all_events = await event_storage.query_events(query)
        kline_events = [e for e in all_events if isinstance(e, KlineEvent) and e.symbol == symbol]
        assert len(kline_events) >= expected_total
