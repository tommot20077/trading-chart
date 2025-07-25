# ABOUTME: Complete integration tests for InMemoryEventBus functionality (E2.3)
# ABOUTME: Tests publish, subscribe, unsubscribe, wait_for and all integrated scenarios

import pytest
import asyncio
from typing import Dict, Any

from core.interfaces.event.event_bus import AbstractEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


@pytest.mark.integration
class TestEventBusCompleteIntegration:
    """
    Complete integration tests for InMemoryEventBus functionality.

    Tests publish, subscribe, unsubscribe, wait_for and all integrated scenarios.
    This covers task E2.3 from the integration test plan.
    """

    @pytest.mark.asyncio
    async def test_complete_publish_subscribe_lifecycle(
        self, event_bus: AbstractEventBus, sample_trade_event: BaseEvent, sync_event_handlers: Dict[str, Any]
    ):
        """Test complete publish-subscribe lifecycle."""
        # Arrange: Subscribe handler
        handler = sync_event_handlers["trade_handler"]
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Verify subscription was created
        assert event_bus.get_subscription_count(EventType.TRADE) == 1
        assert event_bus.get_subscription_count() == 1

        # Act: Publish event
        await event_bus.publish(sample_trade_event)
        await asyncio.sleep(0.1)

        # Assert: Handler was called
        assert sync_event_handlers["trade_handler_calls"] == 1

        # Act: Unsubscribe
        unsubscribed_count = event_bus.unsubscribe(subscription_id)

        # Assert: Subscription was removed
        assert unsubscribed_count == 1
        assert event_bus.get_subscription_count(EventType.TRADE) == 0
        assert event_bus.get_subscription_count() == 0

        # Act: Publish again after unsubscribe
        await event_bus.publish(sample_trade_event)
        await asyncio.sleep(0.1)

        # Assert: Handler was not called again
        assert sync_event_handlers["trade_handler_calls"] == 1

    @pytest.mark.asyncio
    async def test_wait_for_functionality(self, event_bus: AbstractEventBus, event_factory):
        """Test wait_for functionality with various scenarios."""
        # Test 1: Basic wait_for
        event_to_wait_for = event_factory(EventType.TRADE)

        # Start waiting in background
        wait_task = asyncio.create_task(event_bus.wait_for(EventType.TRADE, timeout=2.0))

        # Give wait_for time to set up
        await asyncio.sleep(0.1)

        # Publish the event
        await event_bus.publish(event_to_wait_for)

        # Wait for the result
        received_event = await wait_task

        # Assert: Correct event was received
        assert received_event is not None
        assert received_event.event_type == EventType.TRADE
        assert received_event.event_id == event_to_wait_for.event_id

    @pytest.mark.asyncio
    async def test_wait_for_with_filter(self, event_bus: AbstractEventBus, event_factory):
        """Test wait_for with custom filter function."""
        # Arrange: Create events with different symbols
        btc_event = event_factory(EventType.TRADE, symbol="BTC/USDT")
        eth_event = event_factory(EventType.TRADE, symbol="ETH/USDT")

        # Define filter for BTC events only
        def btc_filter(event: BaseEvent) -> bool:
            return event.symbol == "BTC/USDT"

        # Start waiting for BTC event
        wait_task = asyncio.create_task(event_bus.wait_for(EventType.TRADE, timeout=2.0, filter_func=btc_filter))

        await asyncio.sleep(0.1)

        # Publish ETH event first (should be ignored)
        await event_bus.publish(eth_event)
        await asyncio.sleep(0.1)

        # Publish BTC event (should be received)
        await event_bus.publish(btc_event)

        # Wait for result
        received_event = await wait_task

        # Assert: BTC event was received, not ETH
        assert received_event is not None
        assert received_event.symbol == "BTC/USDT"
        assert received_event.event_id == btc_event.event_id

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self, event_bus: AbstractEventBus):
        """Test wait_for timeout functionality."""
        # Act: Wait for event that will never come
        start_time = asyncio.get_event_loop().time()

        with pytest.raises(asyncio.TimeoutError):
            await event_bus.wait_for(EventType.TRADE, timeout=0.5)

        end_time = asyncio.get_event_loop().time()

        # Assert: Timeout occurred and duration is approximately correct
        assert 0.4 <= (end_time - start_time) <= 0.7  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_multiple_concurrent_wait_for(self, event_bus: AbstractEventBus, event_factory):
        """Test multiple concurrent wait_for operations."""
        # Arrange: Create different events
        trade_event = event_factory(EventType.TRADE)
        kline_event = event_factory(EventType.KLINE)

        # Start multiple wait_for operations
        trade_wait_task = asyncio.create_task(event_bus.wait_for(EventType.TRADE, timeout=2.0))
        kline_wait_task = asyncio.create_task(event_bus.wait_for(EventType.KLINE, timeout=2.0))

        await asyncio.sleep(0.1)

        # Publish events
        await event_bus.publish(kline_event)
        await event_bus.publish(trade_event)

        # Wait for results
        trade_result, kline_result = await asyncio.gather(trade_wait_task, kline_wait_task)

        # Assert: Each wait_for received the correct event
        assert trade_result.event_type == EventType.TRADE
        assert kline_result.event_type == EventType.KLINE

    @pytest.mark.asyncio
    async def test_unsubscribe_all_functionality(
        self, event_bus: AbstractEventBus, sync_event_handlers: Dict[str, Any]
    ):
        """Test unsubscribe_all functionality."""
        # Arrange: Subscribe multiple handlers to same event type
        handler1 = sync_event_handlers["trade_handler"]
        handler2 = sync_event_handlers["universal_handler"]

        sub1 = event_bus.subscribe(EventType.TRADE, handler1)
        sub2 = event_bus.subscribe(EventType.TRADE, handler2)
        sub3 = event_bus.subscribe(EventType.KLINE, handler1)

        # Verify subscriptions
        assert event_bus.get_subscription_count(EventType.TRADE) == 2
        assert event_bus.get_subscription_count(EventType.KLINE) == 1
        assert event_bus.get_subscription_count() == 3

        # Act: Unsubscribe all TRADE handlers
        unsubscribed_count = event_bus.unsubscribe_all(EventType.TRADE)

        # Assert: Only TRADE subscriptions were removed
        assert unsubscribed_count == 2
        assert event_bus.get_subscription_count(EventType.TRADE) == 0
        assert event_bus.get_subscription_count(EventType.KLINE) == 1
        assert event_bus.get_subscription_count() == 1

        # Act: Unsubscribe all remaining handlers
        total_unsubscribed = event_bus.unsubscribe_all()

        # Assert: All subscriptions were removed
        assert total_unsubscribed == 1
        assert event_bus.get_subscription_count() == 0

    @pytest.mark.asyncio
    async def test_event_bus_state_management(self, event_bus: AbstractEventBus, sync_event_handlers: Dict[str, Any]):
        """Test event bus state management and properties."""
        # Test initial state
        assert not event_bus.is_closed
        assert event_bus.get_subscription_count() == 0

        # Add subscriptions
        handler = sync_event_handlers["trade_handler"]
        sub1 = event_bus.subscribe(EventType.TRADE, handler)
        sub2 = event_bus.subscribe(EventType.KLINE, handler)

        # Test state with subscriptions
        assert event_bus.get_subscription_count() == 2
        assert event_bus.get_subscription_count(EventType.TRADE) == 1
        assert event_bus.get_subscription_count(EventType.KLINE) == 1

        # Test detailed subscription info (if available)
        if hasattr(event_bus, "get_subscriptions_for_type"):
            trade_subs = event_bus.get_subscriptions_for_type(EventType.TRADE)
            assert len(trade_subs) == 1
            assert trade_subs[0]["event_type"] == EventType.TRADE.value

        # Clean up
        event_bus.unsubscribe(sub1)
        event_bus.unsubscribe(sub2)

    @pytest.mark.asyncio
    async def test_event_bus_error_resilience(self, event_bus: AbstractEventBus, sample_trade_event: BaseEvent):
        """Test event bus resilience to handler errors."""
        # Arrange: Create handlers with different error behaviors
        successful_calls = []

        def successful_handler(event: BaseEvent):
            successful_calls.append(event.event_id)

        def failing_handler(event: BaseEvent):
            raise RuntimeError("Handler failed intentionally")

        def another_successful_handler(event: BaseEvent):
            successful_calls.append(f"second_{event.event_id}")

        # Subscribe all handlers
        sub1 = event_bus.subscribe(EventType.TRADE, successful_handler)
        sub2 = event_bus.subscribe(EventType.TRADE, failing_handler)
        sub3 = event_bus.subscribe(EventType.TRADE, another_successful_handler)

        # Act: Publish event
        await event_bus.publish(sample_trade_event)
        await asyncio.sleep(0.1)

        # Assert: Successful handlers were called despite failing handler
        assert len(successful_calls) == 2
        assert sample_trade_event.event_id in successful_calls
        assert f"second_{sample_trade_event.event_id}" in successful_calls

        # Clean up
        event_bus.unsubscribe(sub1)
        event_bus.unsubscribe(sub2)
        event_bus.unsubscribe(sub3)

    @pytest.mark.asyncio
    async def test_mixed_sync_async_handlers(
        self,
        event_bus: AbstractEventBus,
        sample_trade_event: BaseEvent,
        sync_event_handlers: Dict[str, Any],
        async_event_handlers: Dict[str, Any],
    ):
        """Test mixing synchronous and asynchronous handlers."""
        # Arrange: Subscribe both sync and async handlers
        sync_handler = sync_event_handlers["trade_handler"]
        async_handler = async_event_handlers["async_trade_handler"]

        sync_sub = event_bus.subscribe(EventType.TRADE, sync_handler)
        async_sub = event_bus.subscribe(EventType.TRADE, async_handler)

        # Act: Publish event
        await event_bus.publish(sample_trade_event)
        await asyncio.sleep(0.2)  # Allow async handler to complete

        # Assert: Both handlers were called
        assert sync_event_handlers["trade_handler_calls"] == 1
        assert async_event_handlers["async_trade_handler_calls"] == 1

        # Clean up
        event_bus.unsubscribe(sync_sub)
        event_bus.unsubscribe(async_sub)

    @pytest.mark.asyncio
    async def test_event_bus_performance_under_load(
        self, event_bus: AbstractEventBus, event_factory, performance_monitor, event_test_config: Dict[str, Any]
    ):
        """Test event bus performance under high load."""
        # Arrange: Create multiple handlers
        call_counts = {"handler1": 0, "handler2": 0, "handler3": 0}

        def create_counting_handler(name: str):
            def handler(event: BaseEvent):
                call_counts[name] += 1

            return handler

        # Subscribe multiple handlers
        subscriptions = []
        for name in call_counts.keys():
            handler = create_counting_handler(name)
            sub_id = event_bus.subscribe(EventType.TRADE, handler)
            subscriptions.append(sub_id)

        # Create many events
        num_events = 1000
        events = [event_factory(EventType.TRADE) for _ in range(num_events)]

        # Act: Publish all events and measure performance
        performance_monitor.start_timer("high_load_publish")

        for event in events:
            await event_bus.publish(event)

        # Wait for all events to be processed using precise synchronization
        flush_success = await event_bus.flush_queue(timeout=10.0)
        if not flush_success:
            pytest.fail(f"Event queue flush timed out after 10 seconds. Queue size: {event_bus.get_queue_size()}")

        duration = performance_monitor.end_timer("high_load_publish")

        # Assert: All events were processed by all handlers
        for name, count in call_counts.items():
            assert count == num_events, f"Handler {name} processed {count}/{num_events} events"

        # Assert: Performance is acceptable
        avg_latency_ms = (duration * 1000) / num_events
        threshold_ms = event_test_config["performance_thresholds"]["publish_latency_ms"]

        assert avg_latency_ms < threshold_ms, (
            f"Average latency {avg_latency_ms:.2f}ms exceeds threshold {threshold_ms}ms"
        )

        # Clean up
        for sub_id in subscriptions:
            event_bus.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_event_bus_close_functionality(self, event_bus: AbstractEventBus, sample_trade_event: BaseEvent):
        """Test event bus close functionality and state after closing."""
        # Verify initial state
        assert not event_bus.is_closed

        # Add a subscription
        def dummy_handler(event: BaseEvent):
            pass

        sub_id = event_bus.subscribe(EventType.TRADE, dummy_handler)

        # Publish an event to verify normal operation
        await event_bus.publish(sample_trade_event)

        # Close the event bus
        await event_bus.close()

        # Assert: Event bus is closed
        assert event_bus.is_closed

        # Note: Behavior after close depends on implementation
        # Some implementations might raise errors, others might ignore operations
        # This test documents the expected behavior
