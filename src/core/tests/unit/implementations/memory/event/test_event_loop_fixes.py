# ABOUTME: Tests for event loop runtime error fixes and resource cleanup
# ABOUTME: Validates GeneratorExit handling and proper coroutine lifecycle management

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


class TestEventLoopFixes:
    """Test cases for event loop runtime error fixes."""

    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create a fresh event bus for each test."""
        bus = InMemoryEventBus(max_queue_size=100, handler_timeout=1.0)
        yield bus
        await bus.close()

    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return BaseEvent(
            event_type=EventType.TRADE, source="test_source", symbol="BTCUSDT", data={"price": 50000, "quantity": 1.0}
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager_support(self):
        """Test that event bus can be used as an async context manager."""
        events_received = []

        async with InMemoryEventBus() as bus:
            # Subscribe to events
            def handler(event):
                events_received.append(event)

            bus.subscribe(EventType.TRADE, handler)

            # Publish an event
            event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})
            await bus.publish(event)

            # Give some time for processing
            await asyncio.sleep(0.1)

        # After context exit, bus should be closed
        assert bus.is_closed
        assert len(events_received) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_active_tasks(self, event_bus):
        """Test that event bus shuts down gracefully even with active tasks."""
        slow_handler_called = []

        async def slow_handler(event):
            """A handler that takes some time to complete."""
            slow_handler_called.append(event)
            await asyncio.sleep(0.5)  # Simulate slow processing

        # Subscribe the slow handler
        event_bus.subscribe(EventType.TRADE, slow_handler)

        # Publish an event
        event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})
        await event_bus.publish(event)

        # Give a moment for the task to start
        await asyncio.sleep(0.1)

        # Close should handle active tasks gracefully
        await event_bus.close()

        assert event_bus.is_closed
        # Handler should have been called
        assert len(slow_handler_called) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_with_no_running_event_loop_handling(self, event_bus):
        """Test that close() handles cases where event loop might be closed."""
        # Subscribe a handler
        handler_called = []

        def handler(event):
            handler_called.append(event)

        event_bus.subscribe(EventType.TRADE, handler)

        # Publish an event
        event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})
        await event_bus.publish(event)

        # Normal close should work fine
        await event_bus.close()
        assert event_bus.is_closed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_close_calls_are_safe(self, event_bus):
        """Test that calling close() multiple times is safe."""
        # First close
        await event_bus.close()
        assert event_bus.is_closed

        # Second close should not raise any errors
        await event_bus.close()
        assert event_bus.is_closed

        # Third close should also be safe
        await event_bus.close()
        assert event_bus.is_closed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_processing_task_handles_queue_timeout(self, event_bus):
        """Test that processing task handles queue timeout gracefully."""
        # Start processing by publishing an event
        event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})
        await event_bus.publish(event)

        # Wait a bit to ensure processing starts
        await asyncio.sleep(0.2)

        # Close should work even with the timeout mechanism
        await event_bus.close()
        assert event_bus.is_closed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_in_processing_loop_is_handled(self, event_bus):
        """Test that errors in processing loop are handled gracefully."""
        error_handler_called = []

        def error_handler(event):
            error_handler_called.append(event)
            raise RuntimeError("Simulated handler error")

        # Subscribe the error handler
        event_bus.subscribe(EventType.TRADE, error_handler)

        # Publish an event
        event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})
        await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Check statistics show the error was counted
        stats = event_bus.get_statistics()
        assert stats["error_count"] > 0

        # Close should still work
        await event_bus.close()
        assert event_bus.is_closed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_processing_task_with_closed_loop(self):
        """Test that _start_processing_task handles closed event loop gracefully."""
        bus = InMemoryEventBus()

        # Mock a closed event loop scenario
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_closed.return_value = True
            mock_get_loop.return_value = mock_loop

            # This should not raise an error
            bus._start_processing_task()
            assert bus._processing_task is None

        await bus.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_publish_after_close_raises_error(self, event_bus):
        """Test that publishing after close raises appropriate error."""
        await event_bus.close()

        event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})

        with pytest.raises(RuntimeError, match="Event bus is closed"):
            await event_bus.publish(event)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_after_close_raises_error(self, event_bus):
        """Test that subscribing after close raises appropriate error."""
        await event_bus.close()

        def handler(event):
            pass

        with pytest.raises(RuntimeError, match="Event bus is closed"):
            event_bus.subscribe(EventType.TRADE, handler)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_for_after_close_raises_error(self, event_bus):
        """Test that wait_for after close raises appropriate error."""
        await event_bus.close()

        with pytest.raises(RuntimeError, match="Event bus is closed"):
            await event_bus.wait_for(EventType.TRADE, timeout=1.0)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resource_cleanup_order(self, event_bus):
        """Test that resources are cleaned up in the correct order."""
        handler_called = []

        async def async_handler(event):
            handler_called.append(event)
            await asyncio.sleep(0.1)

        # Subscribe handler
        event_bus.subscribe(EventType.TRADE, async_handler)

        # Publish event
        event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={})
        await event_bus.publish(event)

        # Start a wait_for operation
        wait_task = asyncio.create_task(event_bus.wait_for(EventType.KLINE, timeout=5.0))

        # Give some time for operations to start
        await asyncio.sleep(0.05)

        # Close should cancel waiting operations and clean up properly
        close_task = asyncio.create_task(event_bus.close())

        # Wait for close to complete
        await close_task

        # Cancel the wait task if it's still running
        if not wait_task.done():
            wait_task.cancel()
            try:
                await wait_task
            except asyncio.CancelledError:
                pass

        assert event_bus.is_closed
        assert len(handler_called) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pause_and_resume_processing(self, event_bus):
        """Test that pause and resume work correctly."""
        handler_called = []

        def handler(event):
            handler_called.append(event)

        event_bus.subscribe(EventType.TRADE, handler)

        # Publish event before pausing
        event1 = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={"id": 1})
        await event_bus.publish(event1)
        await asyncio.sleep(0.1)

        # Pause processing
        await event_bus.pause_processing()

        # Publish event while paused - should still queue
        event2 = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTCUSDT", data={"id": 2})
        await event_bus.publish(event2)
        await asyncio.sleep(0.1)

        # Resume processing
        await event_bus.resume_processing()
        await asyncio.sleep(0.1)

        # Both events should be processed
        assert len(handler_called) == 2

        await event_bus.close()
