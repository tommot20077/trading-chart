from core.models.network.enum import ConnectionStatus
import pytest_asyncio
# ABOUTME: Integration tests for EventMiddlewareBus basic functionality (E5.2)
# ABOUTME: Tests middleware event bus basic event processing without complex middleware

import asyncio
import pytest
import time_machine
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from core.implementations.memory.event.event_middleware_bus import EventMiddlewareBus

# from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_priority import EventPriority
from core.models.data.enum import TradeSide, AssetClass, KlineInterval
from core.models.data.trade import Trade
from core.models.data.kline import Kline


@pytest.mark.integration
class TestEventMiddlewareBusIntegration:
    """Integration tests for EventMiddlewareBus basic functionality."""

    @pytest_asyncio.fixture
    async def middleware_pipeline(self):
        """Create middleware pipeline instance."""
        # Use None for now since pipeline import has issues
        return None

    @pytest_asyncio.fixture
    async def middleware_bus(self, middleware_pipeline):
        """Create EventMiddlewareBus instance."""
        from core.implementations.memory.event.event_bus import InMemoryEventBus

        base_bus = InMemoryEventBus()
        return EventMiddlewareBus(base_bus=base_bus)

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_basic_event_publishing_through_middleware_bus(self, middleware_bus):
        """Test basic event publishing through EventMiddlewareBus."""
        received_events = []

        async def event_handler(event):
            received_events.append(event)

        # Subscribe to events
        middleware_bus.subscribe(EventType.TRADE, event_handler)

        # Create and publish event
        trade_event = TradeEvent(
            source="binance",
            symbol="BTCUSDT",
            data=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="middleware_trade_1",
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.HIGH,
        )

        await middleware_bus.publish(trade_event)
        await asyncio.sleep(0.1)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0].event_type == "trade"
        assert received_events[0].data.trade_id == "middleware_trade_1"

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_multiple_event_types_through_middleware_bus(self, middleware_bus):
        """Test handling multiple event types through EventMiddlewareBus."""
        trade_events = []
        kline_events = []
        connection_events = []

        async def trade_handler(event):
            trade_events.append(event)

        async def kline_handler(event):
            kline_events.append(event)

        async def connection_handler(event):
            connection_events.append(event)

        # Subscribe to different event types
        middleware_bus.subscribe(EventType.TRADE, trade_handler)
        middleware_bus.subscribe(EventType.KLINE, kline_handler)
        middleware_bus.subscribe(EventType.CONNECTION, connection_handler)

        # Publish different types of events
        trade_event = TradeEvent(
            source="binance",
            symbol="BTCUSDT",
            data=Trade(
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
            priority=EventPriority.NORMAL,
        )

        kline_event = KlineEvent(
            source="binance",
            symbol="ETHUSDT",
            data=Kline(
                symbol="ETHUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=datetime.now(timezone.utc),
                close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                open_price=Decimal("3000.0"),
                high_price=Decimal("3100.0"),
                low_price=Decimal("2950.0"),
                close_price=Decimal("3050.0"),
                volume=Decimal("100.0"),
                quote_volume=Decimal("1000.0"),
                trades_count=10,
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.LOW,
        )

        connection_event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED,
            source="binance",
            symbol=None,
            data={"connection_id": "conn_multi"},
            priority=EventPriority.HIGH,
        )

        # Publish all events
        await middleware_bus.publish(trade_event)
        await middleware_bus.publish(kline_event)
        await middleware_bus.publish(connection_event)
        await asyncio.sleep(0.1)

        # Verify each handler received correct events
        assert len(trade_events) == 1
        assert len(kline_events) == 1
        assert len(connection_events) == 1

        assert trade_events[0].data.trade_id == "multi_trade"
        assert kline_events[0].data.symbol == "ETHUSDT"
        assert connection_events[0].data["connection_id"] == "conn_multi"

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_subscription_management(self, middleware_bus):
        """Test subscription management in EventMiddlewareBus."""
        received_events = []

        @pytest.mark.asyncio
        async def test_handler(event):
            received_events.append(event)

        # Subscribe to trade events
        subscription_id = middleware_bus.subscribe(EventType.TRADE, test_handler)

        # Publish event - should be received
        trade_event1 = TradeEvent(
            source="binance",
            symbol="BTCUSDT",
            data=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="sub_trade_1",
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.NORMAL,
        )

        await middleware_bus.publish(trade_event1)
        await asyncio.sleep(0.05)

        assert len(received_events) == 1

        # Unsubscribe
        middleware_bus.unsubscribe(subscription_id)

        # Publish another event - should not be received
        trade_event2 = TradeEvent(
            source="binance",
            symbol="ETHUSDT",
            data=Trade(
                symbol="ETHUSDT",
                exchange="binance",
                price=3000.0,
                quantity=2.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.SELL,
                trade_id="sub_trade_2",
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.NORMAL,
        )

        await middleware_bus.publish(trade_event2)
        await asyncio.sleep(0.05)

        # Should still be 1 (no new events received)
        assert len(received_events) == 1

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_wildcard_subscription(self, middleware_bus):
        """Test wildcard subscription in EventMiddlewareBus."""
        all_events = []

        async def wildcard_handler(event):
            all_events.append(event)

        # Subscribe to all events with wildcard
        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            middleware_bus.subscribe(event_type, wildcard_handler)

        # Publish different types of events
        events_to_publish = [
            TradeEvent(
                source="binance",
                symbol="BTCUSDT",
                data=Trade(
                    symbol="BTCUSDT",
                    exchange="binance",
                    price=50000.0,
                    quantity=1.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.BUY,
                    trade_id="wildcard_trade",
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.HIGH,
            ),
            KlineEvent(
                source="binance",
                symbol="ETHUSDT",
                data=Kline(
                    symbol="ETHUSDT",
                    interval=KlineInterval.MINUTE_1,
                    open_time=datetime.now(timezone.utc),
                    close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                    open_price=Decimal("3000.0"),
                    high_price=Decimal("3100.0"),
                    low_price=Decimal("2950.0"),
                    close_price=Decimal("3050.0"),
                    volume=Decimal("100.0"),
                    quote_volume=Decimal("1000.0"),
                    trades_count=10,
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.NORMAL,
            ),
            ErrorEvent(
                error="Wildcard test error",
                error_code="WILDCARD_ERR",
                source="binance",
                symbol="BTC/USDT",
                data={"context": "wildcard_test"},
                priority=EventPriority.LOW,
            ),
        ]

        # Publish all events
        for event in events_to_publish:
            await middleware_bus.publish(event)

        await asyncio.sleep(0.1)

        # Verify all events were received
        assert len(all_events) == 3

        event_types = {event.event_type for event in all_events}
        assert event_types == {"trade", "kline", "error"}

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_concurrent_operations(self, middleware_bus):
        """Test concurrent operations in EventMiddlewareBus."""
        trade_events = []
        kline_events = []

        async def trade_handler(event):
            trade_events.append(event)
            await asyncio.sleep(0.01)  # Simulate processing time

        async def kline_handler(event):
            kline_events.append(event)
            await asyncio.sleep(0.01)  # Simulate processing time

        # Subscribe handlers
        middleware_bus.subscribe(EventType.TRADE, trade_handler)
        middleware_bus.subscribe(EventType.KLINE, kline_handler)

        # Concurrent publishing tasks
        async def publish_trades():
            for i in range(5):
                event = TradeEvent(
                    source="binance",
                    symbol="BTCUSDT",
                    data=Trade(
                        symbol="BTCUSDT",
                        exchange="binance",
                        price=50000.0 + i * 100,
                        quantity=1.0,
                        timestamp=datetime.now(timezone.utc),
                        received_at=datetime.now(timezone.utc),
                        side=TradeSide.BUY,
                        trade_id=f"concurrent_trade_{i}",
                    ),
                    priority=EventPriority.NORMAL,
                )
                await middleware_bus.publish(event)
                await asyncio.sleep(0.005)

        async def publish_klines():
            for i in range(3):
                event = KlineEvent(
                    source="binance",
                    symbol="ETHUSDT",
                    data=Kline(
                        symbol="ETHUSDT",
                        interval=KlineInterval.MINUTE_1,
                        open_time=datetime.now(timezone.utc),
                        close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
                        open_price=Decimal(f"{3000.0 + i * 50}"),
                        high_price=Decimal(f"{3100.0 + i * 50}"),
                        low_price=Decimal(f"{2950.0 + i * 50}"),
                        close_price=Decimal(f"{3050.0 + i * 50}"),
                        volume=Decimal("100.0"),
                        quote_volume=Decimal("1000.0"),
                        trades_count=10,
                        asset_class=AssetClass.DIGITAL,
                    ),
                    priority=EventPriority.LOW,
                )
                await middleware_bus.publish(event)
                await asyncio.sleep(0.008)

        # Run concurrent publishing
        await asyncio.gather(publish_trades(), publish_klines())
        await asyncio.sleep(0.2)  # Allow all processing to complete

        # Verify all events were processed
        assert len(trade_events) == 5
        assert len(kline_events) == 3

        # Verify event order and content
        trade_ids = [event.data.trade_id for event in trade_events]
        expected_trade_ids = [f"concurrent_trade_{i}" for i in range(5)]
        assert trade_ids == expected_trade_ids

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_wait_for_functionality(self, middleware_bus):
        """Test wait_for functionality in EventMiddlewareBus."""
        # Start waiting for a specific event
        wait_task = asyncio.create_task(middleware_bus.wait_for(EventType.TRADE, timeout=1.0))

        # Give wait_for a moment to set up
        await asyncio.sleep(0.05)

        # Publish the expected event
        trade_event = TradeEvent(
            source="binance",
            symbol="BTCUSDT",
            data=Trade(
                symbol="BTCUSDT",
                exchange="binance",
                price=50000.0,
                quantity=1.0,
                timestamp=datetime.now(timezone.utc),
                received_at=datetime.now(timezone.utc),
                side=TradeSide.BUY,
                trade_id="wait_for_trade",
                asset_class=AssetClass.DIGITAL,
            ),
            priority=EventPriority.HIGH,
        )

        await middleware_bus.publish(trade_event)

        # Wait for the event to be received
        received_event = await wait_task

        # Verify the correct event was received
        assert received_event is not None
        assert received_event.event_type == "trade"
        assert received_event.data.trade_id == "wait_for_trade"

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_wait_for_timeout(self, middleware_bus):
        """Test wait_for timeout functionality in EventMiddlewareBus."""
        # Wait for an event that won't come
        with pytest.raises(asyncio.TimeoutError):
            await middleware_bus.wait_for(EventType.ERROR, timeout=0.1)

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_error_handling(self, middleware_bus):
        """Test error handling in EventMiddlewareBus."""
        successful_events = []
        failed_events = []

        async def failing_handler(event):
            if "fail" in event.data.trade_id:
                failed_events.append(event)
                raise ValueError("Simulated handler failure")
            else:
                successful_events.append(event)

        middleware_bus.subscribe(EventType.TRADE, failing_handler)

        # Publish events - some will cause failures
        success_event = TradeEvent(
            source="binance",
            symbol="BTCUSDT",
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
            priority=EventPriority.NORMAL,
        )

        fail_event = TradeEvent(
            source="binance",
            symbol="BTCUSDT",
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
            priority=EventPriority.NORMAL,
        )

        # Publish both events
        await middleware_bus.publish(success_event)
        await middleware_bus.publish(fail_event)
        await asyncio.sleep(0.1)

        # Verify that successful events are processed and failures are handled
        assert len(successful_events) == 1
        assert len(failed_events) == 1
        assert successful_events[0].data.trade_id == "success_trade"
        assert failed_events[0].data.trade_id == "fail_trade"

    @time_machine.travel("2025-01-17T10:00:00Z")
    @pytest.mark.asyncio
    async def test_middleware_bus_priority_handling(self, middleware_bus):
        """Test priority handling in EventMiddlewareBus."""
        received_events = []

        async def priority_handler(event):
            # Extract identifier based on event type
            if hasattr(event.data, "trade_id"):
                identifier = event.data.trade_id
            elif hasattr(event.data, "error_code"):
                identifier = event.data.error_code
            else:
                identifier = str(event.data)
            received_events.append((event.priority, identifier))

        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            middleware_bus.subscribe(event_type, priority_handler)

        # Publish events with different priorities
        events = [
            TradeEvent(
                source="binance",
                symbol="BTCUSDT",
                data=Trade(
                    symbol="BTCUSDT",
                    exchange="binance",
                    price=50000.0,
                    quantity=1.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.BUY,
                    trade_id="low_priority",
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.LOW,
            ),
            ErrorEvent(
                error="High priority error",
                error_code="HIGH_PRIORITY",
                source="binance",
                symbol="BTC/USDT",
                data={"context": "priority_test"},
                priority=EventPriority.HIGH,
            ),
            TradeEvent(
                source="binance",
                symbol="ETHUSDT",
                data=Trade(
                    symbol="ETHUSDT",
                    exchange="binance",
                    price=3000.0,
                    quantity=2.0,
                    timestamp=datetime.now(timezone.utc),
                    received_at=datetime.now(timezone.utc),
                    side=TradeSide.SELL,
                    trade_id="medium_priority",
                    asset_class=AssetClass.DIGITAL,
                ),
                priority=EventPriority.NORMAL,
            ),
        ]

        # Publish all events
        for event in events:
            await middleware_bus.publish(event)

        await asyncio.sleep(0.1)

        # Verify all events were received
        assert len(received_events) == 3

        # Verify priorities are preserved
        priorities = [priority for priority, _ in received_events]
        assert EventPriority.LOW in priorities
        assert EventPriority.NORMAL in priorities
        assert EventPriority.HIGH in priorities
