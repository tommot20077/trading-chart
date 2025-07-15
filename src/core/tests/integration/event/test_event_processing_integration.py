# ABOUTME: Integration tests for basic event processing flow (E2.1)
# ABOUTME: Tests event publish → handler trigger → state update complete flow

import pytest
import asyncio
from typing import List, Dict, Any

from core.interfaces.event.event_bus import AbstractEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


@pytest.mark.integration
class TestEventProcessingIntegration:
    """
    Integration tests for basic event processing flow.

    Tests the complete flow: event publish → handler trigger → state update
    This covers task E2.1 from the integration test plan.
    """

    @pytest.mark.asyncio
    async def test_basic_event_publish_and_handler_trigger(
        self, event_bus: AbstractEventBus, sample_trade_event: BaseEvent, sync_event_handlers: Dict[str, Any]
    ):
        """Test basic event publishing triggers registered handlers."""
        # Arrange: Subscribe handler to trade events
        handler = sync_event_handlers["trade_handler"]
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Act: Publish trade event
        await event_bus.publish(sample_trade_event)

        # Allow event processing to complete
        await asyncio.sleep(0.1)

        # Assert: Handler was called with the event
        assert sync_event_handlers["trade_handler_calls"] == 1
        assert sync_event_handlers["trade_handler_last_event"] == sample_trade_event

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_async_event_handler_execution(
        self, event_bus: AbstractEventBus, sample_kline_event: BaseEvent, async_event_handlers: Dict[str, Any]
    ):
        """Test asynchronous event handlers are properly executed."""
        # Arrange: Subscribe async handler to kline events
        handler = async_event_handlers["async_kline_handler"]
        subscription_id = event_bus.subscribe(EventType.KLINE, handler)

        # Act: Publish kline event
        await event_bus.publish(sample_kline_event)

        # Allow async handler to complete
        await asyncio.sleep(0.1)

        # Assert: Async handler was called
        assert async_event_handlers["async_kline_handler_calls"] == 1
        assert async_event_handlers["async_kline_handler_last_event"] == sample_kline_event

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_multiple_handlers_for_same_event_type(
        self,
        event_bus: AbstractEventBus,
        sample_trade_event: BaseEvent,
        sync_event_handlers: Dict[str, Any],
        async_event_handlers: Dict[str, Any],
    ):
        """Test multiple handlers can be registered for the same event type."""
        # Arrange: Subscribe both sync and async handlers to trade events
        sync_handler = sync_event_handlers["trade_handler"]
        async_handler = async_event_handlers["async_trade_handler"]

        sync_sub_id = event_bus.subscribe(EventType.TRADE, sync_handler)
        async_sub_id = event_bus.subscribe(EventType.TRADE, async_handler)

        # Act: Publish trade event
        await event_bus.publish(sample_trade_event)

        # Allow all handlers to complete
        await asyncio.sleep(0.1)

        # Assert: Both handlers were called
        assert sync_event_handlers["trade_handler_calls"] == 1
        assert async_event_handlers["async_trade_handler_calls"] == 1

        # Cleanup
        event_bus.unsubscribe(sync_sub_id)
        event_bus.unsubscribe(async_sub_id)

    @pytest.mark.asyncio
    async def test_event_filtering_by_symbol(
        self, event_bus: AbstractEventBus, event_factory, sync_event_handlers: Dict[str, Any]
    ):
        """Test event filtering by symbol works correctly."""
        # Arrange: Subscribe handler with symbol filter
        handler = sync_event_handlers["trade_handler"]
        subscription_id = event_bus.subscribe(EventType.TRADE, handler, filter_symbol="BTC/USDT")

        # Act: Publish events with different symbols
        btc_event = event_factory(EventType.TRADE, symbol="BTC/USDT")
        eth_event = event_factory(EventType.TRADE, symbol="ETH/USDT")

        await event_bus.publish(btc_event)
        await event_bus.publish(eth_event)

        # Allow processing to complete
        await asyncio.sleep(0.1)

        # Assert: Only BTC event triggered the handler
        assert sync_event_handlers["trade_handler_calls"] == 1
        assert sync_event_handlers["trade_handler_last_event"].symbol == "BTC/USDT"

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_event_bus_state_updates(
        self, event_bus: AbstractEventBus, mixed_events: List[BaseEvent], sync_event_handlers: Dict[str, Any]
    ):
        """Test event bus state is properly updated during processing."""
        # Arrange: Subscribe universal handler to all event types
        handler = sync_event_handlers["universal_handler"]
        subscriptions = []

        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            sub_id = event_bus.subscribe(event_type, handler)
            subscriptions.append(sub_id)

        # Act: Publish multiple events
        for event in mixed_events:
            await event_bus.publish(event)

        # Allow all processing to complete
        await asyncio.sleep(0.2)

        # Assert: Event bus state reflects all publications
        assert sync_event_handlers["universal_handler_calls"] == len(mixed_events)
        assert event_bus.get_subscription_count() == len(subscriptions)

        # Cleanup
        for sub_id in subscriptions:
            event_bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_error_handling_in_event_processing(self, event_bus: AbstractEventBus, sample_trade_event: BaseEvent):
        """Test error handling when event handlers fail."""

        # Arrange: Create a handler that raises an exception
        def failing_handler(event: BaseEvent):
            raise ValueError("Handler intentionally failed")

        subscription_id = event_bus.subscribe(EventType.TRADE, failing_handler)

        # Act & Assert: Publishing should not raise exception
        # (Error handling should be internal to the event bus)
        try:
            await event_bus.publish(sample_trade_event)
            await asyncio.sleep(0.1)
        except ValueError:
            pytest.fail("Event bus should handle handler exceptions internally")

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_concurrent_event_publishing(
        self, event_bus: AbstractEventBus, event_factory, sync_event_handlers: Dict[str, Any]
    ):
        """Test concurrent event publishing works correctly."""
        # Arrange: Subscribe handler
        handler = sync_event_handlers["trade_handler"]
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Act: Publish multiple events concurrently
        events = [event_factory(EventType.TRADE) for _ in range(10)]

        # Publish all events concurrently
        await asyncio.gather(*[event_bus.publish(event) for event in events])

        # Allow all processing to complete
        await asyncio.sleep(0.2)

        # Assert: All events were processed
        assert sync_event_handlers["trade_handler_calls"] == len(events)

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_subscription_management_during_processing(
        self, event_bus: AbstractEventBus, sample_trade_event: BaseEvent, sync_event_handlers: Dict[str, Any]
    ):
        """Test subscription management while events are being processed."""
        # Arrange: Subscribe handler
        handler = sync_event_handlers["trade_handler"]
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Act: Start publishing events and unsubscribe during processing
        publish_task = asyncio.create_task(event_bus.publish(sample_trade_event))

        # Unsubscribe while event might still be processing
        unsubscribed_count = event_bus.unsubscribe(subscription_id)

        # Wait for publish to complete
        await publish_task
        await asyncio.sleep(0.1)

        # Assert: Unsubscription was successful
        assert unsubscribed_count == 1
        assert event_bus.get_subscription_count(EventType.TRADE) == 0

    @pytest.mark.asyncio
    async def test_event_bus_performance_metrics(
        self, event_bus: AbstractEventBus, event_factory, performance_monitor, event_test_config: Dict[str, Any]
    ):
        """Test event processing performance meets requirements."""
        # Arrange: Subscribe a simple handler
        call_count = 0

        def counting_handler(event: BaseEvent):
            nonlocal call_count
            call_count += 1

        subscription_id = event_bus.subscribe(EventType.TRADE, counting_handler)

        # Act: Measure event publishing performance
        events = [event_factory(EventType.TRADE) for _ in range(100)]

        performance_monitor.start_timer("batch_publish")

        for event in events:
            await event_bus.publish(event)

        # Wait for all processing to complete
        await asyncio.sleep(0.5)

        duration = performance_monitor.end_timer("batch_publish")

        # Assert: Performance meets requirements
        avg_latency_ms = (duration * 1000) / len(events)
        threshold_ms = event_test_config["performance_thresholds"]["publish_latency_ms"]

        assert avg_latency_ms < threshold_ms, (
            f"Average latency {avg_latency_ms:.2f}ms exceeds threshold {threshold_ms}ms"
        )
        assert call_count == len(events), "Not all events were processed"

        # Cleanup
        event_bus.unsubscribe(subscription_id)
