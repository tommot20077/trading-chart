from core.models.network.enum import ConnectionStatus
# ABOUTME: Integration tests for EventBus + EventStorage cross-component functionality (E5.1)
# ABOUTME: Tests event publishing with persistent storage integration

import asyncio
import pytest
import pytest_asyncio
import time_machine
from datetime import datetime, timezone, timedelta

from core.interfaces.event.event_bus import AbstractEventBus
from core.interfaces.event.event_storage import AbstractEventStorage
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.event.event_type import EventType
from core.models.data.enum import TradeSide, AssetClass
from core.models.data.trade import Trade
from core.models.data.kline import Kline


@pytest.mark.integration
class TestEventBusStorageIntegration:
    """Integration tests for EventBus + EventStorage cross-component functionality."""

    @pytest_asyncio.fixture
    async def event_bus(self) -> AbstractEventBus:
        """Create event bus instance."""
        bus = InMemoryEventBus()
        yield bus
        await bus.close()

    @pytest_asyncio.fixture
    async def event_storage(self) -> AbstractEventStorage:
        """Create event storage instance."""
        from core.implementations.memory.event.event_serializer import MemoryEventSerializer
        from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository

        serializer = MemoryEventSerializer()
        metadata_repo = InMemoryMetadataRepository()
        storage = InMemoryEventStorage(serializer=serializer, metadata_repository=metadata_repo)
        yield storage
        await storage.close()

    @pytest_asyncio.fixture
    async def integrated_system(self, event_bus: AbstractEventBus, event_storage: AbstractEventStorage):
        """Create integrated event bus + storage system."""

        # Set up event bus to automatically store events
        async def store_event_handler(event):
            await event_storage.store_event(event)

        # Subscribe storage handler to all event types
        from core.models.event.event_type import EventType

        for event_type in EventType:
            event_bus.subscribe(event_type, store_event_handler)

        return {"event_bus": event_bus, "event_storage": event_storage, "store_handler": store_event_handler}

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_publish_and_storage_integration(self, integrated_system):
        """Test that events published to EventBus are automatically stored in EventStorage."""
        event_bus = integrated_system["event_bus"]
        event_storage = integrated_system["event_storage"]

        # Create test events
        current_time = datetime.now(timezone.utc)

        trade_event = TradeEvent(
            data=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=current_time,
                received_at=current_time,
                side=TradeSide.BUY,
                trade_id="trade_123",
                asset_class=AssetClass.DIGITAL,
            ),
            source="test",
            priority=EventPriority.HIGH,
        )

        from datetime import timedelta

        open_time = current_time
        close_time = current_time + timedelta(minutes=1)

        kline_event = KlineEvent(
            data=Kline(
                symbol="ETHUSDT",
                exchange="binance",
                open_price=3000.0,
                high_price=3100.0,
                low_price=2950.0,
                close_price=3050.0,
                volume=100.0,
                open_time=open_time,
                close_time=close_time,
                start_time=open_time,
                end_time=close_time,
                interval="1m",
                quote_volume=1000.0,
                trades_count=10,
                asset_class=AssetClass.DIGITAL,
            ),
            source="test",
            priority=EventPriority.NORMAL,
        )

        # Publish events
        await event_bus.publish(trade_event)
        await event_bus.publish(kline_event)

        # Allow async processing
        await asyncio.sleep(0.1)

        # Verify events are stored
        stored_events = await event_storage.query_events(EventQuery())
        assert len(stored_events) == 2

        # Verify event types and content
        event_types = {event.event_type for event in stored_events}
        assert "trade" in event_types
        assert "kline" in event_types

        # Verify specific event data
        trade_events = [e for e in stored_events if e.event_type == "trade"]
        assert len(trade_events) == 1
        assert trade_events[0].data.symbol == "BTCUSDT"
        assert trade_events[0].data.price == 50000.0

        kline_events = [e for e in stored_events if e.event_type == "kline"]
        assert len(kline_events) == 1
        assert kline_events[0].data.symbol == "ETHUSDT"
        assert kline_events[0].data.close_price == 3050.0

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_storage_query_integration(self, integrated_system):
        """Test querying stored events through EventStorage after EventBus publishing."""
        event_bus = integrated_system["event_bus"]
        event_storage = integrated_system["event_storage"]

        # Publish multiple events with different priorities
        events = []
        for i in range(5):
            event = TradeEvent(
                data=Trade(
                    symbol="BTCUSDT",
                    exchange="binance",
                    price=50000.0 + i * 100,
                    quantity=1.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.BUY,
                    trade_id=f"trade_{i}",
                    asset_class=AssetClass.DIGITAL,
                ),
                source="test",
                priority=EventPriority.HIGH if i % 2 == 0 else EventPriority.LOW,
            )
            events.append(event)
            await event_bus.publish(event)

        await asyncio.sleep(0.1)

        # Query by event type
        trade_query = EventQuery(event_types=[EventType.TRADE])
        trade_events = await event_storage.query_events(trade_query)
        assert len(trade_events) == 5

        # Query by priority - filter manually since priority is not supported in query
        all_events = await event_storage.query_events(EventQuery())
        high_priority_events = [e for e in all_events if e.priority == EventPriority.HIGH]
        assert len(high_priority_events) == 3  # Events 0, 2, 4

        # Query with limit
        limited_query = EventQuery(limit=3)
        limited_events = await event_storage.query_events(limited_query)
        assert len(limited_events) == 3

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_storage_persistence_across_bus_operations(self, integrated_system):
        """Test that stored events persist across multiple EventBus operations."""
        event_bus = integrated_system["event_bus"]
        event_storage = integrated_system["event_storage"]

        # Phase 1: Publish initial events
        initial_event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED, connection_id="conn_1", priority=EventPriority.NORMAL, source="test"
        )
        await event_bus.publish(initial_event)
        await asyncio.sleep(0.05)

        # Verify initial storage
        events_after_phase1 = await event_storage.query_events(EventQuery())
        assert len(events_after_phase1) == 1

        # Phase 2: Add more events
        for i in range(3):
            error_event = ErrorEvent(
                error=f"Test error {i}", error_code=f"ERR_{i}", priority=EventPriority.HIGH, source="test"
            )
            await event_bus.publish(error_event)

        await asyncio.sleep(0.1)

        # Verify cumulative storage
        events_after_phase2 = await event_storage.query_events(EventQuery())
        assert len(events_after_phase2) == 4

        # Verify event types
        event_types = {event.event_type for event in events_after_phase2}
        assert "connection" in event_types
        assert "error" in event_types

        # Phase 3: Subscribe new handler and publish more events
        processed_events = []

        async def tracking_handler(event):
            processed_events.append(event)

        event_bus.subscribe(EventType.TRADE, tracking_handler)

        trade_event = TradeEvent(
            data=Trade(
                symbol="ETHUSDT",
                exchange="binance",
                price=3000.0,
                quantity=2.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.SELL,
                trade_id="trade_final",
                asset_class=AssetClass.DIGITAL,
            ),
            source="test",
            priority=EventPriority.NORMAL,
        )
        await event_bus.publish(trade_event)
        await asyncio.sleep(0.1)

        # Verify final storage state
        final_events = await event_storage.query_events(EventQuery())
        assert len(final_events) == 5

        # Verify new handler received the event
        assert len(processed_events) == 1
        assert processed_events[0].event_type == "trade"

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_concurrent_event_bus_storage_operations(self, integrated_system):
        """Test concurrent EventBus publishing and EventStorage operations."""
        event_bus = integrated_system["event_bus"]
        event_storage = integrated_system["event_storage"]

        # Concurrent publishing tasks
        async def publish_trade_events():
            for i in range(10):
                event = TradeEvent(
                    data=Trade(
                        symbol="BTCUSDT",
                        exchange="binance",
                        price=50000.0 + i,
                        quantity=1.0,
                        timestamp=datetime.now(timezone.utc),
                        received_at=datetime.now(timezone.utc),
                        side=TradeSide.BUY,
                        trade_id=f"concurrent_trade_{i}",
                        asset_class=AssetClass.DIGITAL,
                    ),
                    source="test",
                    priority=EventPriority.NORMAL,
                )
                await event_bus.publish(event)
                await asyncio.sleep(0.01)

        async def publish_kline_events():
            for i in range(8):
                current_time = datetime.now(timezone.utc)
                open_time = current_time
                close_time = current_time + timedelta(minutes=1)

                event = KlineEvent(
                    data=Kline(
                        symbol="ETHUSDT",
                        exchange="binance",
                        open_price=3000.0 + i,
                        high_price=3100.0 + i,
                        low_price=2950.0 + i,
                        close_price=3050.0 + i,
                        volume=100.0,
                        open_time=open_time,
                        close_time=close_time,
                        start_time=open_time,
                        end_time=close_time,
                        interval="1m",
                        quote_volume=1000.0,
                        trades_count=10,
                        asset_class=AssetClass.DIGITAL,
                    ),
                    source="test",
                    priority=EventPriority.LOW,
                )
                await event_bus.publish(event)
                await asyncio.sleep(0.015)

        # Run concurrent publishing
        await asyncio.gather(publish_trade_events(), publish_kline_events())

        # Allow processing to complete
        await asyncio.sleep(0.2)

        # Verify all events are stored
        all_events = await event_storage.query_events(EventQuery())
        assert len(all_events) == 18  # 10 trades + 8 klines

        # Verify event distribution
        trade_events = await event_storage.query_events(EventQuery(event_types=[EventType.TRADE]))
        kline_events = await event_storage.query_events(EventQuery(event_types=[EventType.KLINE]))

        assert len(trade_events) == 10
        assert len(kline_events) == 8

        # Verify no data corruption
        trade_ids = {event.data.trade_id for event in trade_events}
        expected_trade_ids = {f"concurrent_trade_{i}" for i in range(10)}
        assert trade_ids == expected_trade_ids

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_bus_storage_error_handling(self, integrated_system):
        """Test error handling in EventBus + EventStorage integration."""
        event_bus = integrated_system["event_bus"]
        event_storage = integrated_system["event_storage"]

        # Create a handler that might fail
        failed_events = []

        async def failing_handler(event):
            if "fail" in getattr(event.data, "trade_id", ""):
                failed_events.append(event)
                raise ValueError("Simulated handler failure")

        event_bus.subscribe(EventType.TRADE, failing_handler)

        # Publish events - some will cause handler failures
        success_event = TradeEvent(
            data=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="success_trade",
                asset_class=AssetClass.DIGITAL,
            ),
            source="test",
            priority=EventPriority.NORMAL,
        )

        fail_event = TradeEvent(
            data=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=51000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="fail_trade",
                asset_class=AssetClass.DIGITAL,
            ),
            source="test",
            priority=EventPriority.NORMAL,
        )

        # Publish both events
        await event_bus.publish(success_event)
        await event_bus.publish(fail_event)
        await asyncio.sleep(0.1)

        # Verify that storage still works despite handler failures
        stored_events = await event_storage.query_events(EventQuery(event_types=[EventType.TRADE]))
        assert len(stored_events) == 2

        # Verify both events are stored regardless of handler failures
        trade_ids = {event.data.trade_id for event in stored_events}
        assert "success_trade" in trade_ids
        assert "fail_trade" in trade_ids

        # Verify the failing handler was called
        assert len(failed_events) == 1
        assert failed_events[0].data.trade_id == "fail_trade"

    @pytest.mark.asyncio
    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_storage_statistics_integration(self, integrated_system):
        """Test EventStorage statistics integration with EventBus operations."""
        event_bus = integrated_system["event_bus"]
        event_storage = integrated_system["event_storage"]

        # Publish various events
        events_to_publish = [
            TradeEvent(
                data=Trade(
                    symbol="BTCUSDT",
                    exchange="binance",
                    price=50000.0,
                    quantity=1.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.BUY,
                    trade_id="stats_trade_1",
                    asset_class=AssetClass.DIGITAL,
                ),
                source="test",
                priority=EventPriority.HIGH,
            ),
            TradeEvent(
                data=Trade(
                    symbol="ETHUSDT",
                    exchange="binance",
                    price=3000.0,
                    quantity=2.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.SELL,
                    trade_id="stats_trade_2",
                    asset_class=AssetClass.DIGITAL,
                ),
                source="test",
                priority=EventPriority.NORMAL,
            ),
            KlineEvent(
                data=Kline(
                    symbol="BTCUSDT",
                    exchange="binance",
                    open_price=50000.0,
                    high_price=51000.0,
                    low_price=49500.0,
                    close_price=50500.0,
                    volume=100.0,
                    open_time=datetime.now(timezone.utc),
                    close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                    interval="1m",
                    quote_volume=1000.0,
                    trades_count=10,
                    asset_class=AssetClass.DIGITAL,
                ),
                source="test",
                priority=EventPriority.LOW,
            ),
        ]

        # Publish all events
        for event in events_to_publish:
            await event_bus.publish(event)

        await asyncio.sleep(0.1)

        # Get storage statistics
        stats = await event_storage.get_statistics()

        # Verify statistics
        assert stats.total_events == 3
        assert stats.events_by_type["trade"] == 2
        assert stats.events_by_type["kline"] == 1
        # EventStorageStats doesn't have events_by_priority, so we verify what it has
        assert stats.storage_size_bytes > 0
        assert stats.avg_event_size_bytes > 0
