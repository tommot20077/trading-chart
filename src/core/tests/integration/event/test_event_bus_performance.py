# ABOUTME: Performance and benchmark tests for InMemoryEventBus
# ABOUTME: Tests throughput, latency, and system behavior under load

import asyncio
import pytest
import pytest_asyncio
import time
from datetime import datetime, UTC

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class TestEventBusPerformance:
    """Performance and benchmark tests for InMemoryEventBus."""

    @pytest_asyncio.fixture
    async def high_performance_bus(self):
        """Create a high-performance event bus for testing."""
        bus = InMemoryEventBus(max_queue_size=10000, handler_timeout=5.0, max_concurrent_handlers=50)
        yield bus
        await bus.close()

    @pytest.fixture
    def event_factory(self):
        """Factory function for creating test events."""

        def _create_event(event_type=EventType.TRADE, priority=EventPriority.NORMAL, data=None):
            return BaseEvent(
                event_type=event_type,
                source="perf_test",
                symbol="BTCUSDT",
                data=data or {"price": 50000, "volume": 1.0},
                priority=priority,
            )

        return _create_event

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_high_throughput_publishing(self, high_performance_bus, event_factory):
        """Test high-throughput event publishing."""
        processed_events = []

        def counting_handler(event):
            processed_events.append(event.event_id)

        # Subscribe handler
        subscription_id = high_performance_bus.subscribe(EventType.TRADE, counting_handler)

        # Measure publishing performance
        num_events = 1000
        events = [event_factory() for _ in range(num_events)]

        start_time = time.time()

        # Publish all events
        for event in events:
            await high_performance_bus.publish(event)

        publish_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Verify all events were processed
        assert len(processed_events) == num_events

        # Calculate throughput
        throughput = num_events / publish_time
        print(f"Publishing throughput: {throughput:.2f} events/second")

        # Verify reasonable performance (should be > 1000 events/second)
        assert throughput > 1000, f"Publishing throughput too low: {throughput:.2f} events/second"

        # Verify statistics
        stats = high_performance_bus.get_statistics()
        assert stats["published_count"] == num_events
        assert stats["processed_count"] == num_events
        assert stats["error_count"] == 0
        assert stats["dropped_count"] == 0

        # Cleanup
        high_performance_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_publishing_performance(self, high_performance_bus, event_factory):
        """Test concurrent event publishing performance."""
        processed_events = []
        process_lock = asyncio.Lock()

        async def thread_safe_handler(event):
            async with process_lock:
                processed_events.append(event.event_id)

        # Subscribe handler
        subscription_id = high_performance_bus.subscribe(EventType.TRADE, thread_safe_handler)

        # Create events for concurrent publishing
        num_events = 500
        events = [event_factory() for _ in range(num_events)]

        # Measure concurrent publishing performance
        start_time = time.time()

        # Publish all events concurrently
        await asyncio.gather(*[high_performance_bus.publish(event) for event in events])

        publish_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Verify all events were processed
        assert len(processed_events) == num_events

        # Calculate concurrent throughput
        concurrent_throughput = num_events / publish_time
        print(f"Concurrent publishing throughput: {concurrent_throughput:.2f} events/second")

        # Verify reasonable concurrent performance
        assert concurrent_throughput > 500, f"Concurrent throughput too low: {concurrent_throughput:.2f} events/second"

        # Cleanup
        high_performance_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_latency_measurement(self, high_performance_bus, event_factory):
        """Test event processing latency."""
        latencies = []

        def latency_handler(event):
            # Calculate latency from event timestamp to processing time
            process_time = datetime.now(UTC)
            latency = (process_time - event.timestamp).total_seconds() * 1000  # ms
            latencies.append(latency)

        # Subscribe handler
        subscription_id = high_performance_bus.subscribe(EventType.TRADE, latency_handler)

        # Publish events with measured latency
        num_events = 100
        for _ in range(num_events):
            event = event_factory()
            await high_performance_bus.publish(event)
            await asyncio.sleep(0.01)  # Small delay between events

        # Wait for processing
        await asyncio.sleep(1.0)

        # Verify all events were processed
        assert len(latencies) == num_events

        # Calculate latency statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(f"Latency stats - Avg: {avg_latency:.2f}ms, Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms")

        # Verify reasonable latency (should be < 100ms average)
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
        assert max_latency < 500, f"Max latency too high: {max_latency:.2f}ms"

        # Cleanup
        high_performance_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_priority_queue_performance(self, high_performance_bus, event_factory):
        """Test priority queue performance with mixed priorities."""
        processing_order = []

        def priority_handler(event):
            processing_order.append(event.priority.value)

        # Subscribe handler
        subscription_id = high_performance_bus.subscribe(EventType.TRADE, priority_handler)

        # Create events with different priorities
        num_events = 1000
        priorities = [EventPriority.HIGH, EventPriority.NORMAL, EventPriority.LOW, EventPriority.CRITICAL]
        events = [event_factory(priority=priorities[i % len(priorities)]) for i in range(num_events)]

        # Measure priority queue performance
        start_time = time.time()

        # Publish all events
        for event in events:
            await high_performance_bus.publish(event)

        publish_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(3.0)

        # Verify all events were processed
        assert len(processing_order) == num_events

        # Calculate throughput
        throughput = num_events / publish_time
        print(f"Priority queue throughput: {throughput:.2f} events/second")

        # Verify priority ordering is maintained
        # Count events by priority
        priority_counts = {}
        for priority in processing_order:
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Verify all priorities are represented
        assert EventPriority.CRITICAL.value in priority_counts
        assert EventPriority.HIGH.value in priority_counts
        assert EventPriority.NORMAL.value in priority_counts
        assert EventPriority.LOW.value in priority_counts

        # Verify reasonable performance
        assert throughput > 500, f"Priority queue throughput too low: {throughput:.2f} events/second"

        # Cleanup
        high_performance_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, high_performance_bus, event_factory):
        """Test memory usage behavior under high load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        processed_events = []

        def memory_handler(event):
            processed_events.append(event.event_id)

        # Subscribe handler
        subscription_id = high_performance_bus.subscribe(EventType.TRADE, memory_handler)

        # Create and publish many events
        num_events = 2000
        events = [event_factory() for _ in range(num_events)]

        for event in events:
            await high_performance_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(3.0)

        # Measure memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"Memory usage - Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB"
        )

        # Verify all events were processed
        assert len(processed_events) == num_events

        # Verify reasonable memory usage (should not increase by more than 100MB)
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.2f}MB"

        # Cleanup
        high_performance_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_queue_overflow_behavior(self, event_factory):
        """Test behavior when queue overflows."""
        # Create bus with small queue
        small_bus = InMemoryEventBus(max_queue_size=50, handler_timeout=1.0)

        try:
            # Slow handler to cause queue buildup
            async def slow_handler(event):
                await asyncio.sleep(0.1)

            # Subscribe handler
            subscription_id = small_bus.subscribe(EventType.TRADE, slow_handler)

            # Publish events rapidly to overflow queue
            num_events = 100
            published_count = 0
            dropped_count = 0

            for i in range(num_events):
                try:
                    await small_bus.publish(event_factory())
                    published_count += 1
                except RuntimeError:
                    dropped_count += 1

            # Wait for processing
            await asyncio.sleep(2.0)

            # Verify overflow behavior
            assert published_count > 0
            assert dropped_count > 0
            assert published_count + dropped_count == num_events

            # Verify statistics
            stats = small_bus.get_statistics()
            assert stats["dropped_count"] == dropped_count
            assert stats["published_count"] == published_count

            print(f"Queue overflow test - Published: {published_count}, Dropped: {dropped_count}")

            # Cleanup
            small_bus.unsubscribe(subscription_id)

        finally:
            await small_bus.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_event_types_performance(self, high_performance_bus, event_factory):
        """Test performance with multiple event types."""
        results = {EventType.TRADE: [], EventType.KLINE: [], EventType.CONNECTION: [], EventType.ERROR: []}

        def trade_handler(event):
            results[EventType.TRADE].append(event.event_id)

        def kline_handler(event):
            results[EventType.KLINE].append(event.event_id)

        def connection_handler(event):
            results[EventType.CONNECTION].append(event.event_id)

        def error_handler(event):
            results[EventType.ERROR].append(event.event_id)

        # Subscribe handlers
        subscriptions = [
            high_performance_bus.subscribe(EventType.TRADE, trade_handler),
            high_performance_bus.subscribe(EventType.KLINE, kline_handler),
            high_performance_bus.subscribe(EventType.CONNECTION, connection_handler),
            high_performance_bus.subscribe(EventType.ERROR, error_handler),
        ]

        # Create mixed events
        num_events_per_type = 250
        event_types = [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]
        events = []

        for event_type in event_types:
            for _ in range(num_events_per_type):
                events.append(event_factory(event_type=event_type))

        # Measure performance
        start_time = time.time()

        # Publish all events
        for event in events:
            await high_performance_bus.publish(event)

        publish_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Verify all events were processed
        total_processed = sum(len(events) for events in results.values())
        assert total_processed == num_events_per_type * len(event_types)

        # Verify each event type was processed correctly
        for event_type in event_types:
            assert len(results[event_type]) == num_events_per_type

        # Calculate throughput
        throughput = total_processed / publish_time
        print(f"Multiple event types throughput: {throughput:.2f} events/second")

        # Verify reasonable performance
        assert throughput > 500, f"Multi-type throughput too low: {throughput:.2f} events/second"

        # Cleanup
        for sub_id in subscriptions:
            high_performance_bus.unsubscribe(sub_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_handler_timeout_performance(self, high_performance_bus, event_factory):
        """Test performance impact of handler timeouts."""
        completed_handlers = []
        timed_out_handlers = []

        async def timeout_handler(event):
            await asyncio.sleep(6.0)  # Will timeout (longer than 5 second timeout)
            completed_handlers.append(event)

        def quick_handler(event):
            completed_handlers.append(event)

        # Subscribe handlers
        timeout_sub = high_performance_bus.subscribe(EventType.TRADE, timeout_handler)
        quick_sub = high_performance_bus.subscribe(EventType.KLINE, quick_handler)

        # Publish events
        num_events = 50

        start_time = time.time()

        # Publish events to both handlers
        for i in range(num_events):
            await high_performance_bus.publish(event_factory(event_type=EventType.TRADE))
            await high_performance_bus.publish(event_factory(event_type=EventType.KLINE))

        publish_time = time.time() - start_time

        # Wait for processing (including timeouts)
        await asyncio.sleep(6.0)

        # Verify performance
        total_events = num_events * 2
        throughput = total_events / publish_time

        print(f"Timeout test throughput: {throughput:.2f} events/second")

        # Verify statistics
        stats = high_performance_bus.get_statistics()
        assert stats["timeout_count"] == num_events  # Trade events timed out
        assert stats["processed_count"] == total_events

        # Quick handlers should complete
        assert len(completed_handlers) == num_events  # Only KLINE events completed

        # Cleanup
        high_performance_bus.unsubscribe(timeout_sub)
        high_performance_bus.unsubscribe(quick_sub)
