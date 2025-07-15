from core.models.network.enum import ConnectionStatus
import pytest_asyncio
# ABOUTME: End-to-end integration tests for complete event flow (E5.3)
# ABOUTME: Tests complete event flow: publish  serialize  store  deserialize  process

import asyncio
import pytest
import time_machine
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.memory.event.event_middleware_bus import EventMiddlewareBus
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.memory.storage.event_storage import InMemoryEventStorage

# from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_priority import EventPriority
from core.models.event.event_query import EventQuery
from core.models.data.enum import TradeSide, AssetClass, KlineInterval
from core.models.data.trade import Trade
from core.models.data.kline import Kline


@pytest.mark.integration
class TestEventFlowEndToEndIntegration:
    """End-to-end integration tests for complete event flow."""

    @pytest_asyncio.fixture
    async def event_system(self):
        """Create complete event system with all components."""
        # Create components
        event_bus = InMemoryEventBus()
        middleware_pipeline = None  # InMemoryMiddlewarePipeline()
        middleware_bus = EventMiddlewareBus(base_bus=event_bus)
        event_serializer = MemoryEventSerializer()
        from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository

        metadata_repository = InMemoryMetadataRepository()
        event_storage = InMemoryEventStorage(serializer=event_serializer, metadata_repository=metadata_repository)

        # Set up event flow: bus  serializer  storage
        async def serialize_and_store_handler(event):
            # Serialize event
            serialized_data = event_serializer.serialize(event)

            # Store serialized event
            await event_storage.store_event(event)

            # Store serialization metadata for verification
            event._serialized_data = serialized_data

        # Subscribe storage handler to all events (only to event_bus, not middleware_bus to avoid duplication)
        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            event_bus.subscribe(event_type, serialize_and_store_handler)

        return {
            "event_bus": event_bus,
            "middleware_bus": middleware_bus,
            "event_serializer": event_serializer,
            "event_storage": event_storage,
            "middleware_pipeline": middleware_pipeline,
        }

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_complete_event_flow_through_event_bus(self, event_system):
        """Test complete event flow: EventBus  serialize  store  deserialize  process."""
        event_bus = event_system["event_bus"]
        event_serializer = event_system["event_serializer"]
        event_storage = event_system["event_storage"]

        # Step 1: Publish event through EventBus
        original_event = TradeEvent(
            source="integration_test",
            trade=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="e2e_trade_1",
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.HIGH,
        )

        await event_bus.publish(original_event)
        await asyncio.sleep(0.1)

        # Step 2: Verify event was stored
        stored_events = await event_storage.query_events(EventQuery())
        assert len(stored_events) == 1
        stored_event = stored_events[0]

        # Step 3: Verify serialization occurred
        assert hasattr(original_event, "_serialized_data")
        serialized_data = original_event._serialized_data
        assert serialized_data is not None
        assert isinstance(serialized_data, (str, bytes, dict))

        # Step 4: Deserialize and verify data integrity
        deserialized_event = event_serializer.deserialize(serialized_data)

        # Verify event integrity through the complete flow
        assert deserialized_event.event_type == original_event.event_type
        assert deserialized_event.priority == original_event.priority
        assert deserialized_event.data.trade_id == "e2e_trade_1"
        assert deserialized_event.data.symbol == "BTCUSDT"
        assert deserialized_event.data.price == 50000.0

        # Step 5: Verify stored event matches original
        assert stored_event.event_type == "trade"
        assert stored_event.data.trade_id == "e2e_trade_1"

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_complete_event_flow_through_middleware_bus(self, event_system):
        """Test complete event flow through EventMiddlewareBus."""
        middleware_bus = event_system["middleware_bus"]
        event_serializer = event_system["event_serializer"]
        event_storage = event_system["event_storage"]

        # Publish event through MiddlewareBus
        original_event = KlineEvent(
            source="integration_test",
            kline=Kline(
                symbol="ETHUSDT",
                exchange="binance",
                open_price=3000.0,
                high_price=3100.0,
                low_price=2950.0,
                close_price=3050.0,
                volume=100.0,
                quote_volume=1000.0,
                trades_count=10,
                interval=KlineInterval.MINUTE_1,
                open_time=datetime.now(timezone.utc),
                close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.NORMAL,
        )

        await middleware_bus.publish(original_event)
        await asyncio.sleep(0.1)

        # Verify complete flow
        stored_events = await event_storage.query_events(EventQuery(event_types=[EventType.KLINE]))
        assert len(stored_events) == 1

        # Verify serialization and deserialization
        assert hasattr(original_event, "_serialized_data")
        deserialized_event = event_serializer.deserialize(original_event._serialized_data)

        assert deserialized_event.event_type == "kline"
        assert deserialized_event.data.symbol == "ETHUSDT"
        assert deserialized_event.data.close_price == 3050.0

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_multiple_event_types_end_to_end_flow(self, event_system):
        """Test end-to-end flow with multiple event types."""
        event_bus = event_system["event_bus"]
        event_serializer = event_system["event_serializer"]
        event_storage = event_system["event_storage"]

        # Create various event types
        events_to_publish = [
            TradeEvent(
                source="integration_test",
                trade=Trade(
                    symbol="BTCUSDT",
                    exchange="binance",
                    price=50000.0,
                    quantity=1.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.BUY,
                    trade_id="multi_trade",
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.HIGH,
            ),
            KlineEvent(
                source="integration_test",
                kline=Kline(
                    symbol="ETHUSDT",
                    exchange="binance",
                    open_price=3000.0,
                    high_price=3100.0,
                    low_price=2950.0,
                    close_price=3050.0,
                    volume=100.0,
                    quote_volume=1000.0,
                    trades_count=10,
                    interval=KlineInterval.MINUTE_1,
                    open_time=datetime.now(timezone.utc),
                    close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.NORMAL,
            ),
            ConnectionEvent(status=ConnectionStatus.CONNECTED, connection_id="multi_conn", priority=EventPriority.LOW),
            ErrorEvent(
                source="integration_test",
                error="Multi-type test error",
                error_code="MULTI_ERR",
                priority=EventPriority.HIGH,
            ),
        ]

        # Publish all events
        for event in events_to_publish:
            await event_bus.publish(event)

        await asyncio.sleep(0.2)

        # Verify all events went through complete flow
        stored_events = await event_storage.query_events(EventQuery())
        assert len(stored_events) == 4

        # Verify each event type
        event_types = {event.event_type for event in stored_events}
        assert event_types == {"trade", "kline", "connection", "error"}

        # Verify serialization for each event
        for original_event in events_to_publish:
            assert hasattr(original_event, "_serialized_data")
            deserialized_event = event_serializer.deserialize(original_event._serialized_data)
            assert deserialized_event.event_type == original_event.event_type
            assert deserialized_event.priority == original_event.priority

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_concurrent_end_to_end_event_processing(self, event_system):
        """Test concurrent end-to-end event processing."""
        event_bus = event_system["event_bus"]
        middleware_bus = event_system["middleware_bus"]
        event_storage = event_system["event_storage"]

        # Concurrent publishing through different buses
        async def publish_through_event_bus():
            for i in range(5):
                event = TradeEvent(
                    source="integration_test",
                    trade=Trade(
                        symbol="BTCUSDT",
                        exchange="binance",
                        price=50000.0 + i * 100,
                        quantity=1.0,
                        timestamp=datetime.now(timezone.utc),
                        received_at=datetime.now(timezone.utc),
                        side=TradeSide.BUY,
                        trade_id=f"concurrent_eb_trade_{i}",
                    ),
                    priority=EventPriority.NORMAL,
                )
                await event_bus.publish(event)
                await asyncio.sleep(0.01)

        async def publish_through_middleware_bus():
            for i in range(3):
                event = KlineEvent(
                    source="integration_test",
                    kline=Kline(
                        symbol="ETHUSDT",
                        exchange="binance",
                        open_price=3000.0 + i * 50,
                        high_price=3100.0 + i * 50,
                        low_price=2950.0 + i * 50,
                        close_price=3050.0 + i * 50,
                        volume=100.0,
                        quote_volume=1000.0,
                        trades_count=10,
                        interval=KlineInterval.MINUTE_1,
                        open_time=datetime.now(timezone.utc),
                        close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                        asset_class=AssetClass.DIGITAL,
                    ),
                    priority=EventPriority.LOW,
                )
                await middleware_bus.publish(event)
                await asyncio.sleep(0.015)

        # Run concurrent publishing
        await asyncio.gather(publish_through_event_bus(), publish_through_middleware_bus())

        await asyncio.sleep(0.3)

        # Verify all events completed the flow
        all_events = await event_storage.query_events(EventQuery())
        assert len(all_events) == 8  # 5 trades + 3 klines

        # Verify event distribution
        trade_events = await event_storage.query_events(EventQuery(event_types=[EventType.TRADE]))
        kline_events = await event_storage.query_events(EventQuery(event_types=[EventType.KLINE]))

        assert len(trade_events) == 5
        assert len(kline_events) == 3

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_flow_with_processing_handlers(self, event_system):
        """Test end-to-end flow with additional processing handlers."""
        event_bus = event_system["event_bus"]
        event_storage = event_system["event_storage"]

        # Track processing stages
        processing_log = []

        async def preprocessing_handler(event):
            processing_log.append(f"preprocess_{event.event_type}")

        async def postprocessing_handler(event):
            processing_log.append(f"postprocess_{event.event_type}")

        # Subscribe processing handlers
        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            event_bus.subscribe(event_type, preprocessing_handler)
            event_bus.subscribe(event_type, postprocessing_handler)

        # Publish test event
        test_event = TradeEvent(
            source="integration_test",
            trade=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="processing_trade",
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.HIGH,
        )

        await event_bus.publish(test_event)
        await asyncio.sleep(0.1)

        # Verify processing occurred
        assert len(processing_log) >= 2
        assert "preprocess_trade" in processing_log
        assert "postprocess_trade" in processing_log

        # Verify storage still works
        stored_events = await event_storage.query_events(EventQuery())
        assert len(stored_events) == 1
        assert stored_events[0].data.trade_id == "processing_trade"

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_flow_error_recovery(self, event_system):
        """Test end-to-end flow error recovery."""
        event_bus = event_system["event_bus"]
        event_storage = event_system["event_storage"]

        # Add a handler that fails for specific events
        failed_events = []

        async def failing_handler(event):
            trade_id = getattr(event.data, "trade_id", "")
            if "fail" in trade_id:
                failed_events.append(event)
                raise ValueError("Simulated processing failure")

        event_bus.subscribe(EventType.TRADE, failing_handler)

        # Publish mix of successful and failing events
        events = [
            TradeEvent(
                source="integration_test",
                trade=Trade(
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
                priority=EventPriority.NORMAL,
            ),
            TradeEvent(
                source="integration_test",
                trade=Trade(
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
                priority=EventPriority.NORMAL,
            ),
            TradeEvent(
                source="integration_test",
                trade=Trade(
                    symbol="ETHUSDT",
                    exchange="binance",
                    price=3000.0,
                    quantity=2.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.SELL,
                    trade_id="another_success",
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.LOW,
            ),
        ]

        # Publish all events
        for event in events:
            await event_bus.publish(event)

        await asyncio.sleep(0.2)

        # Verify storage works despite handler failures
        stored_events = await event_storage.query_events(EventQuery(event_types=[EventType.TRADE]))
        assert len(stored_events) == 3

        # Verify failed events were still processed by failing handler
        assert len(failed_events) == 1
        assert failed_events[0].data.trade_id == "fail_trade"

        # Verify all events have serialization data
        for original_event in events:
            assert hasattr(original_event, "_serialized_data")

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_flow_performance_characteristics(self, event_system):
        """Test performance characteristics of end-to-end event flow."""
        event_bus = event_system["event_bus"]
        event_storage = event_system["event_storage"]

        # Measure processing time for batch of events
        start_time = asyncio.get_event_loop().time()

        # Publish batch of events
        batch_size = 20
        for i in range(batch_size):
            event = TradeEvent(
                source="integration_test",
                trade=Trade(
                    symbol="BTCUSDT",
                    exchange="binance",
                    price=50000.0 + i,
                    quantity=1.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.BUY,
                    trade_id=f"perf_trade_{i}",
                ),
                priority=EventPriority.NORMAL,
            )
            await event_bus.publish(event)

        # Wait for all processing to complete
        await asyncio.sleep(0.5)

        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        # Verify all events were processed
        stored_events = await event_storage.query_events(EventQuery())
        assert len(stored_events) == batch_size

        # Performance assertions (reasonable thresholds)
        assert processing_time < 2.0  # Should complete within 2 seconds
        avg_time_per_event = processing_time / batch_size
        assert avg_time_per_event < 0.1  # Less than 100ms per event on average

        # Verify data integrity wasn't compromised for performance
        trade_ids = {event.data.trade_id for event in stored_events}
        expected_ids = {f"perf_trade_{i}" for i in range(batch_size)}
        assert trade_ids == expected_ids

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_event_flow_data_consistency_verification(self, event_system):
        """Test data consistency through complete event flow."""
        event_bus = event_system["event_bus"]
        event_serializer = event_system["event_serializer"]
        event_storage = event_system["event_storage"]

        # Create event with complex data
        original_event = KlineEvent(
            source="integration_test",
            kline=Kline(
                symbol="BTCUSDT",
                exchange="binance",
                open_price=50000.123456,  # High precision
                high_price=50500.987654,
                low_price=49500.111111,
                close_price=50250.555555,
                volume=123.456789,
                quote_volume=1000.0,
                trades_count=10,
                interval=KlineInterval.MINUTE_5,
                open_time=datetime.now(timezone.utc),
                close_time=datetime.now(timezone.utc) + timedelta(minutes=5),
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.HIGH,
        )

        # Publish and process
        await event_bus.publish(original_event)
        await asyncio.sleep(0.1)

        # Verify storage
        stored_events = await event_storage.query_events(EventQuery())
        assert len(stored_events) == 1
        stored_event = stored_events[0]

        # Verify serialization/deserialization preserves precision
        serialized_data = original_event._serialized_data
        deserialized_event = event_serializer.deserialize(serialized_data)

        # Check precision preservation
        assert abs(deserialized_event.data.open_price - Decimal("50000.123456")) < Decimal("0.000001")
        assert abs(deserialized_event.data.high_price - Decimal("50500.987654")) < Decimal("0.000001")
        assert abs(deserialized_event.data.low_price - Decimal("49500.111111")) < Decimal("0.000001")
        assert abs(deserialized_event.data.close_price - Decimal("50250.555555")) < Decimal("0.000001")
        assert abs(deserialized_event.data.volume - Decimal("123.456789")) < Decimal("0.000001")

        # Verify stored data matches original
        assert stored_event.data.symbol == "BTCUSDT"
        assert stored_event.data.interval == "5m"
        assert stored_event.priority == EventPriority.HIGH
