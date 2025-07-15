# ABOUTME: Unit tests for concurrent event processing in InMemoryEventBus
# ABOUTME: Tests timeout handling, error isolation, and concurrent handler execution

import asyncio
import pytest
import pytest_asyncio

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class TestConcurrentProcessing:
    """Tests for concurrent event processing functionality."""

    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create a fresh event bus for each test."""
        bus = InMemoryEventBus(max_queue_size=100, handler_timeout=0.5, max_concurrent_handlers=10)
        yield bus
        await bus.close()

    @pytest.fixture
    def event_factory(self):
        """Factory function for creating test events."""

        def _create_event(event_type=EventType.TRADE, priority=EventPriority.NORMAL, data=None):
            return BaseEvent(
                event_type=event_type,
                source="test_source",
                symbol="BTCUSDT",
                data=data or {"test": "data"},
                priority=priority,
            )

        return _create_event

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_handler_execution(self, event_bus, event_factory):
        """Test that multiple handlers can execute concurrently."""
        start_times = []
        end_times = []

        async def slow_handler(event):
            start_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.2)
            end_times.append(asyncio.get_event_loop().time())

        # Subscribe multiple handlers
        subscriptions = []
        for i in range(3):
            sub_id = event_bus.subscribe(EventType.TRADE, slow_handler)
            subscriptions.append(sub_id)

        # Publish event
        await event_bus.publish(event_factory())

        # Wait for all handlers to complete
        await asyncio.sleep(0.5)

        # Verify concurrent execution
        assert len(start_times) == 3
        assert len(end_times) == 3

        # All handlers should start around the same time (within 0.1 seconds)
        max_start_diff = max(start_times) - min(start_times)
        assert max_start_diff < 0.1

        # Cleanup
        for sub_id in subscriptions:
            event_bus.unsubscribe(sub_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handler_timeout_protection(self, event_bus, event_factory):
        """Test that handlers are protected by timeout."""
        completed_handlers = []

        async def timeout_handler(event):
            await asyncio.sleep(1.0)  # Longer than timeout
            completed_handlers.append(event)

        def quick_handler(event):
            completed_handlers.append(event)

        # Subscribe handlers
        timeout_sub = event_bus.subscribe(EventType.TRADE, timeout_handler)
        quick_sub = event_bus.subscribe(EventType.TRADE, quick_handler)

        # Publish event
        await event_bus.publish(event_factory())

        # Wait for processing
        await asyncio.sleep(0.8)

        # Only quick handler should complete
        assert len(completed_handlers) == 1

        # Verify timeout statistics
        stats = event_bus.get_statistics()
        assert stats["timeout_count"] == 1
        assert stats["processed_count"] == 1

        # Cleanup
        event_bus.unsubscribe(timeout_sub)
        event_bus.unsubscribe(quick_sub)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_isolation(self, event_bus, event_factory):
        """Test that handler errors don't affect other handlers."""
        successful_handlers = []

        def failing_handler(event):
            raise ValueError("Handler error")

        def successful_handler(event):
            successful_handlers.append(event)

        # Subscribe handlers
        fail_sub = event_bus.subscribe(EventType.TRADE, failing_handler)
        success_sub = event_bus.subscribe(EventType.TRADE, successful_handler)

        # Publish event
        await event_bus.publish(event_factory())

        # Wait for processing
        await asyncio.sleep(0.2)

        # Successful handler should still execute
        assert len(successful_handlers) == 1

        # Verify error statistics
        stats = event_bus.get_statistics()
        assert stats["error_count"] == 1
        assert stats["processed_count"] == 1

        # Cleanup
        event_bus.unsubscribe(fail_sub)
        event_bus.unsubscribe(success_sub)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, event_bus, event_factory):
        """Test that semaphore limits concurrent handler execution."""
        # Create event bus with limited concurrency
        limited_bus = InMemoryEventBus(max_concurrent_handlers=2, handler_timeout=1.0)

        try:
            active_handlers = []
            max_concurrent = 0

            async def tracking_handler(event):
                active_handlers.append(event)
                nonlocal max_concurrent
                max_concurrent = max(max_concurrent, len(active_handlers))
                await asyncio.sleep(0.2)
                active_handlers.remove(event)

            # Subscribe multiple handlers
            subscriptions = []
            for i in range(5):
                sub_id = limited_bus.subscribe(EventType.TRADE, tracking_handler)
                subscriptions.append(sub_id)

            # Publish event
            await limited_bus.publish(event_factory())

            # Wait for processing
            await asyncio.sleep(0.5)

            # Should not exceed max_concurrent_handlers
            assert max_concurrent <= 2

            # Cleanup
            for sub_id in subscriptions:
                limited_bus.unsubscribe(sub_id)

        finally:
            await limited_bus.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handler_thread_execution(self, event_bus, event_factory):
        """Test that synchronous handlers run in thread pool."""
        import threading

        handler_thread_ids = []
        main_thread_id = threading.get_ident()

        def sync_handler(event):
            handler_thread_ids.append(threading.get_ident())

        # Subscribe synchronous handler
        sub_id = event_bus.subscribe(EventType.TRADE, sync_handler)

        # Publish event
        await event_bus.publish(event_factory())

        # Wait for processing
        await asyncio.sleep(0.2)

        # Handler should run in different thread
        assert len(handler_thread_ids) == 1
        assert handler_thread_ids[0] != main_thread_id

        # Cleanup
        event_bus.unsubscribe(sub_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mixed_sync_async_handlers(self, event_bus, event_factory):
        """Test mixed synchronous and asynchronous handlers."""
        sync_results = []
        async_results = []

        def sync_handler(event):
            sync_results.append(event.event_id)

        async def async_handler(event):
            async_results.append(event.event_id)

        # Subscribe both types
        sync_sub = event_bus.subscribe(EventType.TRADE, sync_handler)
        async_sub = event_bus.subscribe(EventType.TRADE, async_handler)

        # Publish event
        event = event_factory()
        await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Both handlers should execute
        assert len(sync_results) == 1
        assert len(async_results) == 1
        assert sync_results[0] == event.event_id
        assert async_results[0] == event.event_id

        # Cleanup
        event_bus.unsubscribe(sync_sub)
        event_bus.unsubscribe(async_sub)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_high_volume_concurrent_publishing(self, event_bus, event_factory):
        """Test high volume concurrent event publishing."""
        processed_events = []
        process_lock = asyncio.Lock()

        async def counting_handler(event):
            async with process_lock:
                processed_events.append(event.event_id)

        # Subscribe handler
        sub_id = event_bus.subscribe(EventType.TRADE, counting_handler)

        # Publish many events concurrently
        events = [event_factory() for _ in range(50)]
        publish_tasks = [event_bus.publish(event) for event in events]
        await asyncio.gather(*publish_tasks)

        # Wait for processing
        await asyncio.sleep(1.0)

        # All events should be processed
        assert len(processed_events) == 50

        # Verify all event IDs are present
        expected_ids = {event.event_id for event in events}
        actual_ids = set(processed_events)
        assert actual_ids == expected_ids

        # Cleanup
        event_bus.unsubscribe(sub_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pause_resume_processing(self, event_bus, event_factory):
        """Test pausing and resuming event processing."""
        processed_events = []

        def handler(event):
            processed_events.append(event)

        # Subscribe handler
        sub_id = event_bus.subscribe(EventType.TRADE, handler)

        # Publish events
        events = [event_factory() for _ in range(3)]
        for event in events:
            await event_bus.publish(event)

        # Wait for initial processing
        await asyncio.sleep(0.2)
        initial_count = len(processed_events)

        # Pause processing
        await event_bus.pause_processing()

        # Publish more events
        more_events = [event_factory() for _ in range(2)]
        for event in more_events:
            await event_bus.publish(event)

        # Wait and verify no additional processing
        await asyncio.sleep(0.2)
        assert len(processed_events) == initial_count

        # Resume processing
        await event_bus.resume_processing()

        # Wait for remaining events to process
        await asyncio.sleep(0.2)

        # All events should now be processed
        assert len(processed_events) == 5

        # Cleanup
        event_bus.unsubscribe(sub_id)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_configuration(self, event_bus, event_factory):
        """Test handler timeout configuration."""
        # Test getting timeout
        assert event_bus.get_handler_timeout() == 0.5

        # Test setting timeout
        event_bus.set_handler_timeout(2.0)
        assert event_bus.get_handler_timeout() == 2.0

        # Test invalid timeout
        with pytest.raises(ValueError, match="Timeout must be positive"):
            event_bus.set_handler_timeout(-1.0)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            event_bus.set_handler_timeout(0.0)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_pending_events(self, event_bus, event_factory):
        """Test graceful shutdown with pending events."""
        # Create a custom event bus with longer timeout for this test
        test_bus = InMemoryEventBus(handler_timeout=2.0, max_concurrent_handlers=10)

        try:
            processed_events = []
            processing_started = []

            async def slow_handler(event):
                # Mark that processing started
                processing_started.append(event)
                # Simulate processing time (should be less than timeout)
                await asyncio.sleep(0.1)
                processed_events.append(event)

            # Subscribe handler
            sub_id = test_bus.subscribe(EventType.TRADE, slow_handler)

            # Publish events
            events = [event_factory() for _ in range(5)]
            for event in events:
                await test_bus.publish(event)

            # Allow brief time for async tasks to start
            await asyncio.sleep(0.05)

            # Start shutdown (should wait for processing)
            await test_bus.close()

            # Some events should have started processing
            assert len(processing_started) >= 1  # At least one event started processing

            # All started events should be processed during graceful shutdown
            assert len(processed_events) == len(processing_started)

            # Bus should be closed
            assert test_bus.is_closed

        finally:
            if not test_bus.is_closed:
                await test_bus.close()
