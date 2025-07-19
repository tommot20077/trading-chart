# ABOUTME: Event system quality assurance tests for Phase 6 (E6.1, E6.2, E6.3)
# ABOUTME: Tests coverage analysis, performance benchmarks, and stability verification

import asyncio

import pytest
import pytest_asyncio
import time_machine

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.event.event_type import EventType
from core.models.event.Kline_event import KlineEvent
from core.models.network.enum import ConnectionStatus
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval
from datetime import datetime, UTC
from decimal import Decimal


@pytest.mark.integration
class TestEventSystemQualityAssurance:
    """Quality assurance tests for the event system (Phase 6)."""

    @pytest_asyncio.fixture
    async def event_system(self):
        """Create complete event system for quality testing."""
        event_bus = InMemoryEventBus()
        serializer = MemoryEventSerializer()
        metadata_repo = InMemoryMetadataRepository()
        event_storage = InMemoryEventStorage(serializer=serializer, metadata_repository=metadata_repo)

        return {
            "event_bus": event_bus,
            "event_storage": event_storage,
            "serializer": serializer,
            "metadata_repo": metadata_repo,
        }

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_e61_event_system_coverage_analysis(self, event_system):
        """E6.1: Event system test coverage analysis."""
        event_bus = event_system["event_bus"]
        event_storage = event_system["event_storage"]

        # Test coverage of all event types
        tested_event_types = set()

        # Test CONNECTION events
        connection_event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED,
            connection_id="coverage_test_conn",
            priority=EventPriority.HIGH,
            source="coverage_test",
        )

        event_bus.subscribe(EventType.CONNECTION, lambda e: tested_event_types.add(e.event_type))
        await event_bus.publish(connection_event)
        await asyncio.sleep(0.05)

        # Test ERROR events
        error_event = ErrorEvent(
            error="Coverage test error",
            error_code="COVERAGE_ERR",
            priority=EventPriority.NORMAL,
            source="coverage_test",
        )

        event_bus.subscribe(EventType.ERROR, lambda e: tested_event_types.add(e.event_type))
        await event_bus.publish(error_event)
        await asyncio.sleep(0.05)

        # Test KLINE events
        kline_data = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2022, 1, 1, 12, 0, 0, tzinfo=UTC),
            close_time=datetime(2022, 1, 1, 12, 1, 0, tzinfo=UTC),
            open_price=Decimal("50000.0"),
            high_price=Decimal("50100.0"),
            low_price=Decimal("49900.0"),
            close_price=Decimal("50050.0"),
            volume=Decimal("100.0"),
            quote_volume=Decimal("5002500.0"),
            trades_count=10
        )
        kline_event = KlineEvent(
            data=kline_data,
            priority=EventPriority.NORMAL,
            source="coverage_test",
        )

        event_bus.subscribe(EventType.KLINE, lambda e: tested_event_types.add(e.event_type))
        await event_bus.publish(kline_event)
        await asyncio.sleep(0.05)

        # Verify coverage
        assert "connection" in tested_event_types
        assert "error" in tested_event_types
        assert "kline" in tested_event_types

        # Test storage coverage
        async def store_handler(event):
            await event_storage.store_event(event)

        event_bus.subscribe(EventType.CONNECTION, store_handler)
        event_bus.subscribe(EventType.ERROR, store_handler)
        event_bus.subscribe(EventType.KLINE, store_handler)

        # Publish more events for storage coverage
        for i in range(3):
            status = ConnectionStatus.CONNECTED if i % 2 == 0 else ConnectionStatus.DISCONNECTED
            test_event = ConnectionEvent(
                status=status, connection_id=f"coverage_conn_{i}", priority=EventPriority.LOW, source="coverage_test"
            )
            await event_bus.publish(test_event)

        await asyncio.sleep(0.1)

        stored_events = await event_storage.get_events(EventQuery())
        assert len(stored_events) >= 3

        # Calculate coverage metrics
        total_event_types = len(EventType)
        covered_event_types = len(tested_event_types)
        coverage_percentage = (covered_event_types / total_event_types) * 100

        print(f"Event Type Coverage: {coverage_percentage:.1f}% ({covered_event_types}/{total_event_types})")

        # Coverage should be reasonable for tested types
        assert coverage_percentage >= 30.0  # At least 30% coverage with tested types
