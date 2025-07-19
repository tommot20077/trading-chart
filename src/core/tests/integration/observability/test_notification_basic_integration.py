# ABOUTME: Basic integration tests for NotificationHandler and EventBus collaboration
# ABOUTME: Simplified tests to verify core notification integration functionality

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any
from datetime import datetime, UTC

# Basic notification integration tests - event bus and notification handler collaboration

from core.implementations.memory.observability.notification_handler import (
    InMemoryNotificationHandler,
)
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class SimpleAlertEvent(BaseEvent[Dict[str, Any]]):
    """Simple alert event for testing notification integration."""

    def __init__(self, event_id: str, severity: str, rule_name: str, message: str):
        priority = EventPriority.HIGH if severity == "critical" else EventPriority.NORMAL
        super().__init__(
            event_id=event_id,
            event_type=EventType.ALERT,
            timestamp=datetime.now(UTC),
            source="test",
            symbol="TESTUSDT",
            data={"severity": severity, "rule_name": rule_name, "message": message},
            priority=priority,
        )


@pytest_asyncio.fixture
async def event_bus():
    """Create event bus for testing."""
    bus = InMemoryEventBus(max_queue_size=100, handler_timeout=5.0)
    try:
        yield bus
    finally:
        await bus.close()


@pytest_asyncio.fixture
async def notification_handler():
    """Create notification handler for testing."""
    handler = InMemoryNotificationHandler(
        max_queue_size=100,
        max_history_size=500,
        history_retention_hours=1.0,
        cleanup_interval_seconds=60.0,
        max_retry_attempts=2,
        simulate_failure_rate=0.0,
        processing_delay_seconds=0.0,
    )
    try:
        yield handler
    finally:
        await handler.close()


class TestBasicNotificationIntegration:
    """Test basic notification handler integration with event bus."""

    @pytest.mark.asyncio
    async def test_simple_event_to_notification(self, event_bus, notification_handler):
        """Test simple event to notification flow."""
        notifications_sent = []

        async def simple_handler(event: BaseEvent):
            """Simple handler that sends notifications."""
            alert_data = {
                "id": f"simple-{event.event_id}",
                "rule_name": event.data["rule_name"],
                "severity": event.data["severity"],
                "message": event.data["message"],
                "metadata": {},
            }

            success, message = await notification_handler.send_notification(alert_data)
            notifications_sent.append(success)

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, simple_handler)

        # Test: Send a simple event
        event = SimpleAlertEvent(
            event_id="simple-001", severity="warning", rule_name="test_rule", message="Test message"
        )

        await event_bus.publish(event)

        # Wait for processing with timeout
        timeout_count = 0
        while len(notifications_sent) == 0 and timeout_count < 50:
            await asyncio.sleep(0.1)
            timeout_count += 1

        # Verify
        assert len(notifications_sent) == 1
        assert notifications_sent[0] is True

        # Check statistics
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 1
        assert stats["total_failed"] == 0

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_multiple_events_to_notifications(self, event_bus, notification_handler):
        """Test multiple events to notifications."""
        notifications_received = []

        async def multi_handler(event: BaseEvent):
            """Handler for multiple events."""
            alert_data = {
                "id": f"multi-{event.event_id}",
                "rule_name": event.data["rule_name"],
                "severity": event.data["severity"],
                "message": event.data["message"],
                "metadata": {},
            }

            success, _ = await notification_handler.send_notification(alert_data)
            notifications_received.append((event.event_id, success))

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, multi_handler)

        # Test: Send multiple events
        events = [
            SimpleAlertEvent("multi-001", "critical", "rule_1", "Message 1"),
            SimpleAlertEvent("multi-002", "warning", "rule_2", "Message 2"),
            SimpleAlertEvent("multi-003", "info", "rule_3", "Message 3"),
        ]

        for event in events:
            await event_bus.publish(event)

        # Wait for processing
        timeout_count = 0
        while len(notifications_received) < 3 and timeout_count < 100:
            await asyncio.sleep(0.1)
            timeout_count += 1

        # Verify
        assert len(notifications_received) == 3
        for event_id, success in notifications_received:
            assert success is True

        # Check statistics
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 3
        assert stats["total_failed"] == 0

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_notification_failure_handling(self, event_bus):
        """Test handling of notification failures."""
        # Create handler with failures
        failing_handler = InMemoryNotificationHandler(
            max_queue_size=100,
            simulate_failure_rate=1.0,  # Always fail
        )

        try:
            failure_results = []

            async def failure_handler(event: BaseEvent):
                """Handler that expects failures."""
                alert_data = {
                    "id": f"fail-{event.event_id}",
                    "rule_name": event.data["rule_name"],
                    "severity": event.data["severity"],
                    "message": event.data["message"],
                    "metadata": {},
                }

                success, message = await failing_handler.send_notification(alert_data)
                failure_results.append((event.event_id, success, message))

            # Subscribe to alerts
            subscription_id = event_bus.subscribe(EventType.ALERT, failure_handler)

            # Test: Send event that will fail
            event = SimpleAlertEvent("fail-001", "critical", "fail_rule", "Fail message")
            await event_bus.publish(event)

            # Wait for processing
            timeout_count = 0
            while len(failure_results) == 0 and timeout_count < 50:
                await asyncio.sleep(0.1)
                timeout_count += 1

            # Verify
            assert len(failure_results) == 1
            event_id, success, message = failure_results[0]
            assert success is False
            assert "failure" in message.lower() or "error" in message.lower()

            # Check statistics
            stats = await failing_handler.get_notification_statistics()
            assert stats["total_sent"] == 0
            assert stats["total_failed"] == 1

            # Cleanup
            event_bus.unsubscribe(subscription_id)

        finally:
            await failing_handler.close()

    @pytest.mark.asyncio
    async def test_notification_history_retrieval(self, event_bus, notification_handler):
        """Test notification history retrieval."""

        async def history_handler(event: BaseEvent):
            """Handler for history testing."""
            alert_data = {
                "id": f"hist-{event.event_id}",
                "rule_name": event.data["rule_name"],
                "severity": event.data["severity"],
                "message": event.data["message"],
                "metadata": {},
            }
            await notification_handler.send_notification(alert_data)

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, history_handler)

        # Test: Send events for history
        events = [
            SimpleAlertEvent("hist-001", "critical", "hist_rule_1", "History message 1"),
            SimpleAlertEvent("hist-002", "warning", "hist_rule_2", "History message 2"),
        ]

        for event in events:
            await event_bus.publish(event)

        # Wait for processing
        timeout_count = 0
        stats = await notification_handler.get_notification_statistics()
        while stats["total_sent"] < 2 and timeout_count < 50:
            await asyncio.sleep(0.1)
            stats = await notification_handler.get_notification_statistics()
            timeout_count += 1

        # Test history retrieval
        history = await notification_handler.get_notification_history(limit=10)
        assert len(history) == 2

        # Verify history record structure
        record = history[0]
        assert "id" in record
        assert "alert_data" in record
        assert "status" in record
        assert record["status"] == "sent"

        # Test filtering by rule
        rule1_history = await notification_handler.get_notification_history(limit=10, rule_name="hist_rule_1")
        assert len(rule1_history) == 1
        assert rule1_history[0]["alert_data"]["rule_name"] == "hist_rule_1"

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_event_bus_notification_statistics(self, event_bus, notification_handler):
        """Test that both event bus and notification handler maintain statistics."""
        processed_events = []

        async def stats_handler(event: BaseEvent):
            """Handler for statistics testing."""
            alert_data = {
                "id": f"stats-{event.event_id}",
                "rule_name": event.data["rule_name"],
                "severity": event.data["severity"],
                "message": event.data["message"],
                "metadata": {},
            }

            success, _ = await notification_handler.send_notification(alert_data)
            processed_events.append(event.event_id)

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, stats_handler)

        # Test: Send events
        num_events = 5
        for i in range(num_events):
            event = SimpleAlertEvent(f"stats-{i:03d}", "info", "stats_rule", f"Stats message {i}")
            await event_bus.publish(event)

        # Wait for processing
        timeout_count = 0
        while len(processed_events) < num_events and timeout_count < 100:
            await asyncio.sleep(0.1)
            timeout_count += 1

        # Verify event bus statistics
        bus_stats = event_bus.get_statistics()
        assert bus_stats["published_count"] == num_events
        assert bus_stats["processed_count"] == num_events
        assert bus_stats["error_count"] == 0

        # Verify notification handler statistics
        handler_stats = await notification_handler.get_notification_statistics()
        assert handler_stats["total_sent"] == num_events
        assert handler_stats["total_failed"] == 0
        assert handler_stats["history_size"] == num_events

        # Cleanup
        event_bus.unsubscribe(subscription_id)
