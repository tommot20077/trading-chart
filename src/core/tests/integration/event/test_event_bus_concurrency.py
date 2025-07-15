# ABOUTME: Concurrency and load testing for InMemoryEventBus
# ABOUTME: Tests system behavior under high concurrency and stress conditions

import asyncio
import pytest
import pytest_asyncio
import time
import random

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class TestEventBusConcurrency:
    """Concurrency and load tests for InMemoryEventBus."""

    @pytest_asyncio.fixture
    async def concurrent_bus(self):
        """Create a high-concurrency event bus for testing."""
        bus = InMemoryEventBus(max_queue_size=5000, handler_timeout=2.0, max_concurrent_handlers=100)
        yield bus
        await bus.close()

    @pytest.fixture
    def event_factory(self):
        """Factory function for creating test events."""

        def _create_event(event_type=EventType.TRADE, priority=EventPriority.NORMAL, data=None):
            return BaseEvent(
                event_type=event_type,
                source="concurrency_test",
                symbol=f"SYMBOL{random.randint(1, 10)}",
                data=data or {"price": random.uniform(1000, 5000), "volume": random.uniform(0.1, 10.0)},
                priority=priority,
            )

        return _create_event

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_publishers_single_consumer(self, concurrent_bus, event_factory):
        """Test multiple concurrent publishers with single consumer."""
        processed_events = []
        process_lock = asyncio.Lock()

        async def consumer_handler(event):
            async with process_lock:
                processed_events.append(event.event_id)

        # Subscribe consumer
        subscription_id = concurrent_bus.subscribe(EventType.TRADE, consumer_handler)

        # Create multiple publisher tasks
        num_publishers = 10
        events_per_publisher = 100

        async def publisher_task(publisher_id):
            for i in range(events_per_publisher):
                event = event_factory()
                await concurrent_bus.publish(event)
                await asyncio.sleep(0.001)  # Small delay to simulate realistic publishing

        # Run all publishers concurrently
        start_time = time.time()
        publisher_tasks = [publisher_task(i) for i in range(num_publishers)]
        await asyncio.gather(*publisher_tasks)
        publish_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Verify all events were processed
        total_events = num_publishers * events_per_publisher
        assert len(processed_events) == total_events

        # Calculate throughput
        throughput = total_events / publish_time
        print(f"Concurrent publishers throughput: {throughput:.2f} events/second")

        # Verify reasonable performance
        assert throughput > 1000, f"Concurrent publishers throughput too low: {throughput:.2f} events/second"

        # Verify no duplicate processing
        assert len(set(processed_events)) == total_events

        # Cleanup
        concurrent_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_single_publisher_concurrent_consumers(self, concurrent_bus, event_factory):
        """Test single publisher with multiple concurrent consumers."""
        consumer_results = {}

        async def create_consumer(consumer_id):
            results = []

            async def consumer_handler(event):
                results.append(event.event_id)
                await asyncio.sleep(0.001)  # Simulate processing time

            subscription_id = concurrent_bus.subscribe(EventType.TRADE, consumer_handler)
            consumer_results[consumer_id] = (subscription_id, results)

        # Create multiple consumers
        num_consumers = 20
        for i in range(num_consumers):
            await create_consumer(i)

        # Publish events
        num_events = 500
        events = [event_factory() for _ in range(num_events)]

        start_time = time.time()
        for event in events:
            await concurrent_bus.publish(event)
        publish_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(3.0)

        # Verify all consumers processed all events
        for consumer_id, (subscription_id, results) in consumer_results.items():
            assert len(results) == num_events, (
                f"Consumer {consumer_id} processed {len(results)} events, expected {num_events}"
            )

        # Calculate throughput
        throughput = num_events / publish_time
        print(f"Concurrent consumers throughput: {throughput:.2f} events/second")

        # Verify performance
        assert throughput > 500, f"Concurrent consumers throughput too low: {throughput:.2f} events/second"

        # Cleanup
        for consumer_id, (subscription_id, results) in consumer_results.items():
            concurrent_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_publishers_and_consumers(self, concurrent_bus, event_factory):
        """Test concurrent publishers and consumers."""
        consumer_results = {}
        publisher_counts = {}

        # Create consumers
        num_consumers = 10
        for i in range(num_consumers):
            results = []

            async def consumer_handler(event, consumer_id=i):
                results.append(event.event_id)
                await asyncio.sleep(0.001)

            subscription_id = concurrent_bus.subscribe(EventType.TRADE, consumer_handler)
            consumer_results[i] = (subscription_id, results)

        # Create publishers
        num_publishers = 5
        events_per_publisher = 100

        async def publisher_task(publisher_id):
            count = 0
            for i in range(events_per_publisher):
                event = event_factory()
                await concurrent_bus.publish(event)
                count += 1
                await asyncio.sleep(0.001)
            publisher_counts[publisher_id] = count

        # Run publishers and consumers concurrently
        start_time = time.time()
        publisher_tasks = [publisher_task(i) for i in range(num_publishers)]
        await asyncio.gather(*publisher_tasks)
        execution_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(3.0)

        # Verify results
        total_published = sum(publisher_counts.values())
        total_consumed = sum(len(results) for _, results in consumer_results.values())

        expected_total = num_publishers * events_per_publisher * num_consumers
        assert total_consumed == expected_total

        # Calculate throughput
        throughput = total_published / execution_time
        print(f"Concurrent pub/sub throughput: {throughput:.2f} events/second")

        # Verify performance
        assert throughput > 500, f"Concurrent pub/sub throughput too low: {throughput:.2f} events/second"

        # Cleanup
        for consumer_id, (subscription_id, results) in consumer_results.items():
            concurrent_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_priority_under_concurrency(self, concurrent_bus, event_factory):
        """Test priority handling under high concurrency."""
        processing_order = []
        process_lock = asyncio.Lock()

        async def priority_handler(event):
            async with process_lock:
                processing_order.append(event.priority.value)

        # Subscribe handler
        subscription_id = concurrent_bus.subscribe(EventType.TRADE, priority_handler)

        # Create concurrent publishing tasks with different priorities
        priorities = [EventPriority.CRITICAL, EventPriority.HIGH, EventPriority.NORMAL, EventPriority.LOW]

        async def priority_publisher(priority, count):
            for _ in range(count):
                event = event_factory(priority=priority)
                await concurrent_bus.publish(event)
                await asyncio.sleep(0.001)

        # Run concurrent publishers
        events_per_priority = 100
        start_time = time.time()

        publisher_tasks = [priority_publisher(priority, events_per_priority) for priority in priorities]
        await asyncio.gather(*publisher_tasks)

        execution_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(3.0)

        # Verify all events were processed
        total_events = len(priorities) * events_per_priority
        assert len(processing_order) == total_events

        # Verify priority distribution
        priority_counts = {}
        for priority_value in processing_order:
            priority_counts[priority_value] = priority_counts.get(priority_value, 0) + 1

        # Each priority should have the expected count
        for priority in priorities:
            assert priority_counts[priority.value] == events_per_priority

        # Verify that higher priorities were processed first (generally)
        # Due to concurrency, we can't guarantee perfect ordering, but we can check trends
        critical_positions = [i for i, p in enumerate(processing_order) if p == EventPriority.CRITICAL.value]
        low_positions = [i for i, p in enumerate(processing_order) if p == EventPriority.LOW.value]

        # Most critical events should be processed before most low priority events
        avg_critical_pos = sum(critical_positions) / len(critical_positions)
        avg_low_pos = sum(low_positions) / len(low_positions)

        assert avg_critical_pos < avg_low_pos, (
            "Critical events should generally be processed before low priority events"
        )

        # Calculate throughput
        throughput = total_events / execution_time
        print(f"Priority concurrency throughput: {throughput:.2f} events/second")

        # Cleanup
        concurrent_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dynamic_subscription_management(self, concurrent_bus, event_factory):
        """Test dynamic subscription and unsubscription under load."""
        active_subscriptions = {}
        processed_counts = {}

        def create_handler(handler_id):
            def handler(event):
                if handler_id not in processed_counts:
                    processed_counts[handler_id] = 0
                processed_counts[handler_id] += 1

            return handler

        # Start with some initial subscriptions
        for i in range(10):
            handler = create_handler(i)
            sub_id = concurrent_bus.subscribe(EventType.TRADE, handler)
            active_subscriptions[i] = sub_id

        # Start publishing events
        async def publisher_task():
            for _ in range(500):
                event = event_factory()
                await concurrent_bus.publish(event)
                await asyncio.sleep(0.002)

        # Start dynamic subscription management
        async def subscription_manager():
            for i in range(100):
                await asyncio.sleep(0.01)

                # Randomly add or remove subscriptions
                if random.random() < 0.3 and len(active_subscriptions) < 20:
                    # Add new subscription
                    new_id = max(active_subscriptions.keys()) + 1 if active_subscriptions else 0
                    handler = create_handler(new_id)
                    sub_id = concurrent_bus.subscribe(EventType.TRADE, handler)
                    active_subscriptions[new_id] = sub_id

                elif random.random() < 0.2 and len(active_subscriptions) > 5:
                    # Remove random subscription
                    handler_id = random.choice(list(active_subscriptions.keys()))
                    sub_id = active_subscriptions.pop(handler_id)
                    concurrent_bus.unsubscribe(sub_id)

        # Run tasks concurrently
        start_time = time.time()
        await asyncio.gather(publisher_task(), subscription_manager())
        execution_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Verify system stability
        assert len(active_subscriptions) >= 5  # At least some subscriptions remain
        assert len(processed_counts) >= 5  # At least some processing occurred

        # Verify statistics
        stats = concurrent_bus.get_statistics()
        assert stats["error_count"] == 0  # No errors during dynamic management
        assert stats["published_count"] == 500

        print(f"Dynamic subscription test completed in {execution_time:.2f}s")
        print(f"Final active subscriptions: {len(active_subscriptions)}")
        print(f"Handlers that processed events: {len(processed_counts)}")

        # Cleanup
        for handler_id, sub_id in active_subscriptions.items():
            concurrent_bus.unsubscribe(sub_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_under_concurrency(self, concurrent_bus, event_factory):
        """Test error handling and isolation under high concurrency."""
        successful_processes = []
        failed_processes = []

        def failing_handler(event):
            failed_processes.append(event.event_id)
            raise ValueError("Intentional handler failure")

        def successful_handler(event):
            successful_processes.append(event.event_id)

        # Subscribe both handlers
        fail_sub = concurrent_bus.subscribe(EventType.TRADE, failing_handler)
        success_sub = concurrent_bus.subscribe(EventType.TRADE, successful_handler)

        # Publish events concurrently
        num_events = 200

        async def concurrent_publisher():
            tasks = []
            for _ in range(num_events):
                event = event_factory()
                tasks.append(concurrent_bus.publish(event))
                if len(tasks) >= 50:  # Batch publish
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)

        start_time = time.time()
        await concurrent_publisher()
        execution_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(3.0)

        # Verify error isolation
        assert len(successful_processes) == num_events  # All events processed by successful handler
        assert len(failed_processes) == num_events  # All events processed by failing handler

        # Verify statistics
        stats = concurrent_bus.get_statistics()
        assert stats["error_count"] == num_events  # All failing handlers generated errors
        assert stats["processed_count"] == num_events  # Processing continued despite errors

        # Calculate throughput
        throughput = num_events / execution_time
        print(f"Error handling concurrency throughput: {throughput:.2f} events/second")

        # Verify performance wasn't severely impacted by errors
        assert throughput > 200, f"Error handling throughput too low: {throughput:.2f} events/second"

        # Cleanup
        concurrent_bus.unsubscribe(fail_sub)
        concurrent_bus.unsubscribe(success_sub)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stress_test_sustained_load(self, concurrent_bus, event_factory):
        """Test system behavior under sustained high load."""
        processed_events = []
        error_count = 0

        async def stress_handler(event):
            nonlocal error_count
            try:
                processed_events.append(event.event_id)
                # Simulate variable processing time
                await asyncio.sleep(random.uniform(0.001, 0.005))
            except Exception:
                error_count += 1

        # Subscribe handler
        subscription_id = concurrent_bus.subscribe(EventType.TRADE, stress_handler)

        # Sustained load test
        duration = 10  # seconds
        target_rate = 100  # events per second

        async def sustained_publisher():
            start_time = time.time()
            published_count = 0

            while time.time() - start_time < duration:
                event = event_factory()
                await concurrent_bus.publish(event)
                published_count += 1

                # Maintain target rate
                elapsed = time.time() - start_time
                expected_count = int(elapsed * target_rate)
                if published_count > expected_count:
                    await asyncio.sleep(0.001)

            return published_count

        # Run sustained load
        start_time = time.time()
        published_count = await sustained_publisher()
        execution_time = time.time() - start_time

        # Wait for processing to complete
        await asyncio.sleep(5.0)

        # Verify system stability
        assert error_count == 0, f"Errors occurred during stress test: {error_count}"
        assert len(processed_events) == published_count, (
            f"Processed {len(processed_events)} events, published {published_count}"
        )

        # Verify statistics
        stats = concurrent_bus.get_statistics()
        assert stats["error_count"] == 0
        assert stats["timeout_count"] == 0
        assert stats["dropped_count"] == 0

        # Calculate actual throughput
        actual_throughput = published_count / execution_time
        print(f"Stress test - Published: {published_count} events in {execution_time:.2f}s")
        print(f"Actual throughput: {actual_throughput:.2f} events/second")
        print(f"Target throughput: {target_rate} events/second")

        # Verify throughput was close to target
        assert actual_throughput >= target_rate * 0.9, f"Throughput {actual_throughput:.2f} below target {target_rate}"

        # Cleanup
        concurrent_bus.unsubscribe(subscription_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_graceful_shutdown_under_load(self, event_factory):
        """Test graceful shutdown behavior under high load."""
        # Create dedicated bus for shutdown test
        shutdown_bus = InMemoryEventBus(max_queue_size=1000, handler_timeout=1.0)

        processed_events = []

        async def shutdown_handler(event):
            processed_events.append(event.event_id)
            await asyncio.sleep(0.01)  # Simulate processing time

        # Subscribe handler
        subscription_id = shutdown_bus.subscribe(EventType.TRADE, shutdown_handler)

        # Start publishing events
        async def publisher_task():
            for i in range(500):
                event = event_factory()
                try:
                    await shutdown_bus.publish(event)
                    await asyncio.sleep(0.001)
                except RuntimeError:
                    # Expected when bus is closing
                    break

        # Start publisher and let it run briefly
        publisher_task_handle = asyncio.create_task(publisher_task())
        await asyncio.sleep(0.5)  # Let some events accumulate

        # Initiate shutdown
        start_shutdown = time.time()
        shutdown_task = asyncio.create_task(shutdown_bus.close())

        # Wait for shutdown to complete
        await shutdown_task
        shutdown_time = time.time() - start_shutdown

        # Cancel publisher if still running
        if not publisher_task_handle.done():
            publisher_task_handle.cancel()
            try:
                await publisher_task_handle
            except asyncio.CancelledError:
                pass

        # Verify graceful shutdown
        assert shutdown_bus.is_closed
        assert shutdown_time < 10.0, f"Shutdown took too long: {shutdown_time:.2f}s"

        # Verify some events were processed
        assert len(processed_events) > 0, "No events were processed before shutdown"

        print(f"Graceful shutdown completed in {shutdown_time:.2f}s")
        print(f"Processed {len(processed_events)} events during shutdown")

        # Verify final statistics
        stats = shutdown_bus.get_statistics()
        print(f"Final stats - Published: {stats['published_count']}, Processed: {stats['processed_count']}")

        # No additional cleanup needed as bus is closed
