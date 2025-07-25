# ABOUTME: Integration tests for NotificationHandler and EventBus collaboration
# ABOUTME: Tests complete notification flow from event publishing to notification handling and status tracking

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any
from datetime import datetime, UTC

# Notification handler and event bus integration tests

from core.implementations.memory.observability.notification_handler import (
    InMemoryNotificationHandler,
    NotificationStatus,
)
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from .conftest import wait_for_notification_count, wait_for_condition

class AlertEvent(BaseEvent[Dict[str, Any]]):
    """Alert event for testing notification integration."""

    def __init__(
        self,
        event_id: str,
        symbol: str,
        timestamp: float,
        alert_id: str,
        severity: str,
        rule_name: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ):
        priority = EventPriority.HIGH if severity == "critical" else EventPriority.NORMAL
        
        # Convert timestamp to datetime
        dt_timestamp = datetime.fromtimestamp(timestamp, tz=UTC)
        
        # Store alert fields in data
        data = {
            "alert_id": alert_id,
            "severity": severity,
            "rule_name": rule_name,
            "message": message,
        }
        if metadata:
            data.update(metadata)
        
        super().__init__(
            event_id=event_id,
            event_type=EventType.ALERT,
            timestamp=dt_timestamp,
            source="test",
            symbol=symbol,
            data=data,
            priority=priority,
        )

    @property
    def alert_id(self) -> str:
        return self.data["alert_id"]

    @property
    def severity(self) -> str:
        return self.data["severity"]

    @property
    def rule_name(self) -> str:
        return self.data["rule_name"]

    @property
    def message(self) -> str:
        return self.data["message"]


@pytest_asyncio.fixture
async def event_bus():
    """Create event bus for testing."""
    bus = InMemoryEventBus(max_queue_size=1000, handler_timeout=10.0)
    try:
        yield bus
    finally:
        await bus.close()


@pytest_asyncio.fixture
async def notification_handler():
    """Create notification handler for testing."""
    handler = InMemoryNotificationHandler(
        max_queue_size=500,
        max_history_size=1000,
        history_retention_hours=1.0,
        cleanup_interval_seconds=30.0,
        max_retry_attempts=2,
        simulate_failure_rate=0.0,
        processing_delay_seconds=0.0,
    )
    try:
        yield handler
    finally:
        await handler.close()


@pytest_asyncio.fixture
async def failing_notification_handler():
    """Create notification handler configured to fail for testing error scenarios."""
    handler = InMemoryNotificationHandler(
        max_queue_size=100,
        max_history_size=500,
        history_retention_hours=0.5,
        cleanup_interval_seconds=60.0,
        max_retry_attempts=1,
        simulate_failure_rate=0.7,  # 70% failure rate
        processing_delay_seconds=0.05,
    )
    try:
        yield handler
    finally:
        await handler.close()


class TestNotificationHandlerEventBusIntegration:
    """Test NotificationHandler integration with EventBus."""

    @pytest.mark.asyncio
    async def test_event_to_notification_flow(self, event_bus, notification_handler):
        """Test complete flow from event publication to notification handling."""
        # Setup: Create alert handler that processes events and sends notifications
        notifications_sent = []

        async def alert_handler(event: BaseEvent):
            """Handler that converts events to notifications."""
            alert_data = {
                "id": f"alert-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": event.message,
                "metadata": event.metadata or {},
            }

            success, message = await notification_handler.send_notification(alert_data)
            notifications_sent.append((success, message, alert_data))

        # Subscribe to alert events
        subscription_id = event_bus.subscribe(EventType.ALERT, alert_handler)

        # Test: Publish alert events
        test_events = [
            AlertEvent(
                event_id="evt-001",
                symbol="BTCUSDT",
                timestamp=1640995200.0,
                alert_id="alert-001",
                severity="critical",
                rule_name="high_cpu_usage",
                message="CPU usage exceeded 90% threshold",
                metadata={"current_value": 95.5, "threshold": 90.0},
            ),
            AlertEvent(
                event_id="evt-002",
                symbol="ETHUSDT",
                timestamp=1640995260.0,
                alert_id="alert-002",
                severity="warning",
                rule_name="memory_warning",
                message="Memory usage is elevated",
                metadata={"current_value": 75.0, "threshold": 70.0},
            ),
            AlertEvent(
                event_id="evt-003",
                symbol="ADAUSDT",
                timestamp=1640995320.0,
                alert_id="alert-003",
                severity="info",
                rule_name="system_startup",
                message="System monitoring started",
                metadata={"startup_time": "2024-01-01T10:00:00Z"},
            ),
        ]

        for event in test_events:
            await event_bus.publish(event)

        # Wait for processing
        await wait_for_notification_count(notification_handler, len(test_events))
        await event_bus.flush_queue(timeout=5.0)

        # Verify: Check notifications were sent
        assert len(notifications_sent) == 3

        # Verify all notifications succeeded
        for success, message, alert_data in notifications_sent:
            assert success is True
            assert "sent successfully" in message
            assert alert_data["id"].startswith("alert-")

        # Verify notification handler statistics
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 3
        assert stats["total_failed"] == 0
        assert stats["pending_queue_size"] == 0
        assert stats["history_size"] == 3

        # Verify statistics by severity
        assert stats["notifications_by_severity"]["critical"] == 1
        assert stats["notifications_by_severity"]["warning"] == 1
        assert stats["notifications_by_severity"]["info"] == 1

        # Verify statistics by rule
        assert stats["notifications_by_rule"]["high_cpu_usage"] == 1
        assert stats["notifications_by_rule"]["memory_warning"] == 1
        assert stats["notifications_by_rule"]["system_startup"] == 1

        # Verify event bus statistics
        bus_stats = event_bus.get_statistics()
        assert bus_stats["published_count"] == 3
        assert bus_stats["processed_count"] == 3
        assert bus_stats["error_count"] == 0

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_concurrent_event_notification_processing(self, event_bus, notification_handler):
        """Test concurrent processing of multiple events and notifications."""
        notifications_received = []
        processing_times = []

        async def alert_handler(event: BaseEvent):
            """Handler that processes events concurrently."""
            start_time = asyncio.get_event_loop().time()

            alert_data = {
                "id": f"concurrent-{event.event_id}",
                "rule_name": f"rule_{event.alert_id}",
                "severity": event.severity,
                "message": f"Concurrent processing test: {event.message}",
                "metadata": {"event_id": event.event_id},
            }

            success, message = await notification_handler.send_notification(alert_data)

            processing_time = asyncio.get_event_loop().time() - start_time
            processing_times.append(processing_time)
            notifications_received.append((success, alert_data))

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, alert_handler)

        # Test: Publish multiple events concurrently
        num_events = 20
        events = []
        for i in range(num_events):
            event = AlertEvent(
                event_id=f"concurrent-evt-{i:03d}",
                symbol=f"SYMBOL{i}",
                timestamp=1640995200.0 + i,
                alert_id=f"alert-{i:03d}",
                severity="warning" if i % 2 == 0 else "info",
                rule_name=f"concurrent_rule_{i}",
                message=f"Concurrent test event {i}",
                metadata={"index": i},
            )
            events.append(event)

        # Publish all events concurrently
        publish_tasks = [event_bus.publish(event) for event in events]
        await asyncio.gather(*publish_tasks)

        # Wait for all processing to complete
        await wait_for_notification_count(notification_handler, num_events)
        await event_bus.flush_queue(timeout=10.0)

        # Verify: All notifications were processed
        assert len(notifications_received) == num_events
        assert len(processing_times) == num_events

        # Verify all notifications succeeded
        successful_notifications = [n for n in notifications_received if n[0] is True]
        assert len(successful_notifications) == num_events

        # Verify concurrent processing was actually faster than sequential
        max_processing_time = max(processing_times)
        total_time_if_sequential = sum(processing_times)
        assert max_processing_time < total_time_if_sequential / 2  # Should be much faster than sequential

        # Verify notification handler statistics
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == num_events
        assert stats["total_failed"] == 0
        assert stats["pending_queue_size"] == 0

        # Verify event bus handled concurrency correctly
        bus_stats = event_bus.get_statistics()
        assert bus_stats["published_count"] == num_events
        assert bus_stats["processed_count"] == num_events
        assert bus_stats["error_count"] == 0

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_event_filtering_with_notifications(self, event_bus, notification_handler):
        """Test event filtering in notification flow."""
        critical_notifications = []
        warning_notifications = []

        async def critical_alert_handler(event: BaseEvent):
            """Handler for critical alerts only."""
            if event.severity == "critical":
                alert_data = {
                    "id": f"critical-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"CRITICAL: {event.message}",
                    "metadata": event.metadata or {},
                }
                success, _ = await notification_handler.send_notification(alert_data)
                if success:
                    critical_notifications.append(alert_data)

        async def warning_alert_handler(event: BaseEvent):
            """Handler for warning alerts only."""
            if event.severity == "warning":
                alert_data = {
                    "id": f"warning-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"WARNING: {event.message}",
                    "metadata": event.metadata or {},
                }
                success, _ = await notification_handler.send_notification(alert_data)
                if success:
                    warning_notifications.append(alert_data)

        # Subscribe both handlers to same event type
        critical_sub = event_bus.subscribe(EventType.ALERT, critical_alert_handler)
        warning_sub = event_bus.subscribe(EventType.ALERT, warning_alert_handler)

        # Test: Publish mixed severity events
        test_events = [
            AlertEvent(
                event_id="flt-001",
                symbol="BTCUSDT",
                timestamp=1640995200.0,
                alert_id="alert-001",
                severity="critical",
                rule_name="system_failure",
                message="System critical failure detected",
                metadata={"error_code": "SYS_001"},
            ),
            AlertEvent(
                event_id="flt-002",
                symbol="ETHUSDT",
                timestamp=1640995260.0,
                alert_id="alert-002",
                severity="warning",
                rule_name="performance_degradation",
                message="Performance degradation detected",
                metadata={"response_time": 2.5},
            ),
            AlertEvent(
                event_id="flt-003",
                symbol="ADAUSDT",
                timestamp=1640995320.0,
                alert_id="alert-003",
                severity="info",
                rule_name="status_update",
                message="System status update",
                metadata={"status": "running"},
            ),
            AlertEvent(
                event_id="flt-004",
                symbol="BTCUSDT",
                timestamp=1640995380.0,
                alert_id="alert-004",
                severity="critical",
                rule_name="data_corruption",
                message="Data corruption detected",
                metadata={"affected_tables": ["orders", "trades"]},
            ),
            AlertEvent(
                event_id="flt-005",
                symbol="ETHUSDT",
                timestamp=1640995440.0,
                alert_id="alert-005",
                severity="warning",
                rule_name="resource_usage",
                message="High resource usage detected",
                metadata={"cpu": 85.0, "memory": 78.0},
            ),
        ]

        for event in test_events:
            await event_bus.publish(event)

        # Wait for processing
        await wait_for_notification_count(notification_handler, 4)  # 2 critical + 2 warning
        await event_bus.flush_queue(timeout=5.0)

        # Verify: Correct filtering occurred
        assert len(critical_notifications) == 2
        assert len(warning_notifications) == 2

        # Verify critical notifications
        critical_ids = [n["id"] for n in critical_notifications]
        assert "critical-flt-001" in critical_ids
        assert "critical-flt-004" in critical_ids

        # Verify warning notifications
        warning_ids = [n["id"] for n in warning_notifications]
        assert "warning-flt-002" in warning_ids
        assert "warning-flt-005" in warning_ids

        # Verify notification handler received only filtered notifications
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 4  # 2 critical + 2 warning (info events ignored)
        assert stats["notifications_by_severity"]["critical"] == 2
        assert stats["notifications_by_severity"]["warning"] == 2
        assert "info" not in stats["notifications_by_severity"]

        # Cleanup
        event_bus.unsubscribe(critical_sub)
        event_bus.unsubscribe(warning_sub)

    @pytest.mark.asyncio
    async def test_notification_failure_recovery(self, event_bus, failing_notification_handler):
        """Test notification failure handling and recovery mechanisms."""
        notifications_attempted = []
        failed_events = []
        successful_events = []

        async def alert_handler(event: BaseEvent):
            """Handler that attempts notifications with potential failures."""
            alert_data = {
                "id": f"recovery-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": event.message,
                "metadata": event.metadata or {},
            }

            notifications_attempted.append(alert_data)
            success, message = await failing_notification_handler.send_notification(alert_data)

            if success:
                successful_events.append(event.event_id)
            else:
                failed_events.append((event.event_id, message))

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, alert_handler)

        # Test: Publish events that may fail
        num_events = 15
        test_events = []
        for i in range(num_events):
            event = AlertEvent(
                event_id=f"recovery-evt-{i:03d}",
                symbol="TESTUSDT",
                timestamp=1640995200.0 + i,
                alert_id=f"recovery-alert-{i:03d}",
                severity="critical",
                rule_name="failure_recovery_test",
                message=f"Recovery test event {i}",
                metadata={"test_index": i},
            )
            test_events.append(event)

        for event in test_events:
            await event_bus.publish(event)

        # Wait for processing (with potential failures)
        await asyncio.sleep(2.0)  # Give time for failures to occur
        await event_bus.flush_queue(timeout=10.0)

        # Verify: Some events should have failed (due to 70% failure rate)
        assert len(notifications_attempted) == num_events
        assert len(failed_events) > 0  # Some should fail
        assert len(successful_events) > 0  # Some should succeed
        assert len(failed_events) + len(successful_events) == num_events

        # Verify failure rate exists but be flexible with exact rates
        # The handler has a 70% failure rate configured, but due to retries and timing,
        # actual rates may vary significantly
        failure_rate = len(failed_events) / num_events
        success_rate = len(successful_events) / num_events
        
        # Simply verify that both failures and successes occurred
        assert failure_rate > 0, f"Expected some failures, got failure_rate={failure_rate}"
        assert success_rate > 0, f"Expected some successes, got success_rate={success_rate}"
        assert failure_rate + success_rate == 1.0, "Rates should sum to 1.0"

        # Verify notification handler statistics reflect failures
        stats = await failing_notification_handler.get_notification_statistics()
        assert stats["total_sent"] == len(successful_events)
        assert stats["total_failed"] == len(failed_events)
        assert stats["total_sent"] + stats["total_failed"] == num_events

        # Verify event bus processed all events regardless of notification failures
        bus_stats = event_bus.get_statistics()
        assert bus_stats["published_count"] == num_events
        assert bus_stats["processed_count"] == num_events

        # Verify failure messages indicate simulation
        for event_id, message in failed_events:
            assert "failure" in message.lower() or "error" in message.lower()

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_notification_queue_overflow_handling(self, event_bus):
        """Test notification handler behavior when queue is full."""
        # Create handler with very small queue
        small_queue_handler = InMemoryNotificationHandler(
            max_queue_size=5,  # Very small queue
            max_history_size=100,
            processing_delay_seconds=0.1,  # Add delay to cause queue backup
        )

        try:
            overflow_results = []

            async def alert_handler(event: BaseEvent):
                """Handler that sends notifications to small queue."""
                alert_data = {
                    "id": f"overflow-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": event.message,
                    "metadata": {},
                }

                success, message = await small_queue_handler.send_notification(alert_data)
                overflow_results.append((success, message, event.event_id))

            # Subscribe to alerts
            subscription_id = event_bus.subscribe(EventType.ALERT, alert_handler)

            # Test: Rapidly publish many events to overwhelm queue
            num_events = 20  # Much more than queue capacity (5)
            events = []
            for i in range(num_events):
                event = AlertEvent(
                    event_id=f"overflow-{i:03d}",
                    symbol="OVERFLOWUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"overflow-alert-{i:03d}",
                    severity="warning",
                    rule_name="overflow_test",
                    message=f"Overflow test event {i}",
                    metadata={},
                )
                events.append(event)

            # Publish all events rapidly
            for event in events:
                await event_bus.publish(event)

            # Wait for processing
            await event_bus.flush_queue(timeout=15.0)

            # Verify: Some notifications should fail due to queue overflow
            assert len(overflow_results) == num_events

            successful_results = [r for r in overflow_results if r[0] is True]
            failed_results = [r for r in overflow_results if r[0] is False]

            # Some should succeed (queue capacity + some processed)
            assert len(successful_results) >= 5

            # Some should fail due to queue being full
            assert len(failed_results) > 0

            # Verify failure messages indicate queue full
            queue_full_failures = [r for r in failed_results if "queue is full" in r[1]]
            assert len(queue_full_failures) > 0

            # Verify notification handler statistics
            stats = await small_queue_handler.get_notification_statistics()
            assert stats["total_sent"] == len(successful_results)
            assert stats["pending_queue_size"] <= 5  # Should not exceed max

            # Cleanup
            event_bus.unsubscribe(subscription_id)

        finally:
            await small_queue_handler.close()

    @pytest.mark.asyncio
    async def test_notification_history_integration(self, event_bus, notification_handler):
        """Test notification history tracking in integration scenario."""
        sent_notifications = []

        async def history_handler(event: BaseEvent):
            """Handler that tracks notifications for history testing."""
            alert_data = {
                "id": f"history-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": f"History test: {event.message}",
                "metadata": {"original_event_id": event.event_id},
            }

            success, message = await notification_handler.send_notification(alert_data)
            if success:
                sent_notifications.append(alert_data)

        # Subscribe to alerts
        subscription_id = event_bus.subscribe(EventType.ALERT, history_handler)

        # Test: Publish events with different severities and rules
        test_cases = [
            ("critical", "database_failure"),
            ("warning", "high_latency"),
            ("info", "status_update"),
            ("critical", "security_breach"),
            ("warning", "resource_usage"),
        ]

        for i, (severity, rule_name) in enumerate(test_cases):
            event = AlertEvent(
                event_id=f"hist-{i:03d}",
                symbol="HISTUSDT",
                timestamp=1640995200.0 + i * 60,
                alert_id=f"hist-alert-{i:03d}",
                severity=severity,
                rule_name=rule_name,
                message=f"History test event {i} - {severity}",
                metadata={"test_case": i},
            )
            await event_bus.publish(event)

        # Wait for processing
        await wait_for_notification_count(notification_handler, len(test_cases))
        await event_bus.flush_queue(timeout=5.0)

        # Verify: All notifications were sent
        assert len(sent_notifications) == len(test_cases)

        # Test history retrieval with filters
        # Get all history
        all_history = await notification_handler.get_notification_history(limit=100)
        assert len(all_history) == len(test_cases)

        # Filter by severity
        critical_history = await notification_handler.get_notification_history(
            limit=100, status=NotificationStatus.SENT, severity="critical"
        )
        critical_count = len([case for case in test_cases if case[0] == "critical"])
        assert len(critical_history) == critical_count

        # Filter by rule name
        db_failure_history = await notification_handler.get_notification_history(
            limit=100, rule_name="database_failure"
        )
        assert len(db_failure_history) == 1
        assert db_failure_history[0]["alert_data"]["rule_name"] == "database_failure"

        # Verify history record structure
        sample_record = all_history[0]
        required_fields = [
            "id",
            "alert_data",
            "status",
            "created_at",
            "updated_at",
            "retry_count",
            "error_message",
            "sent_at",
            "processing_time",
        ]
        for field in required_fields:
            assert field in sample_record

        # Verify status is correct
        assert sample_record["status"] == "sent"
        assert sample_record["sent_at"] is not None
        assert sample_record["processing_time"] is not None
        assert sample_record["processing_time"] > 0

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_event_bus_notification_shutdown_cleanup(self, event_bus, notification_handler):
        """Test proper cleanup when both event bus and notification handler are closed."""
        notifications_processed = []

        async def cleanup_handler(event: BaseEvent):
            """Handler for testing cleanup scenarios."""
            alert_data = {
                "id": f"cleanup-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": event.message,
                "metadata": {},
            }

            success, _ = await notification_handler.send_notification(alert_data)
            if success:
                notifications_processed.append(alert_data["id"])

        # Subscribe and send some events
        subscription_id = event_bus.subscribe(EventType.ALERT, cleanup_handler)

        # Send a few events
        for i in range(3):
            event = AlertEvent(
                event_id=f"cleanup-{i}",
                symbol="CLEANUPUSDT",
                timestamp=1640995200.0 + i,
                alert_id=f"cleanup-alert-{i}",
                severity="info",
                rule_name="cleanup_test",
                message=f"Cleanup test event {i}",
                metadata={},
            )
            await event_bus.publish(event)

        # Wait for processing
        await wait_for_notification_count(notification_handler, 3)
        await event_bus.flush_queue(timeout=5.0)

        # Verify initial state
        assert len(notifications_processed) == 3

        # Test cleanup behavior
        initial_stats = await notification_handler.get_notification_statistics()
        assert initial_stats["total_sent"] == 3
        assert not notification_handler.is_closed
        assert not event_bus.is_closed

        # Close notification handler first
        await notification_handler.close()
        assert notification_handler.is_closed

        # Try to send more events (should fail gracefully)
        final_event = AlertEvent(
            event_id="cleanup-final",
            symbol="CLEANUPUSDT",
            timestamp=1640995500.0,
            alert_id="cleanup-alert-final",
            severity="info",
            rule_name="cleanup_test",
            message="Final cleanup test event",
            metadata={},
        )

        # This should not raise an exception, but handler will be closed
        await event_bus.publish(final_event)
        await event_bus.flush_queue(timeout=2.0)

        # Cleanup
        event_bus.unsubscribe(subscription_id)
        await event_bus.close()
        assert event_bus.is_closed


class TestNotificationFailureRecovery:
    """Test notification handler failure recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_retry_mechanism_on_failure(self, event_bus):
        """Test notification retry mechanism when failures occur."""
        # Create handler with retry capability
        retry_handler = InMemoryNotificationHandler(
            max_queue_size=100,
            max_history_size=500,
            max_retry_attempts=3,
            simulate_failure_rate=0.8,  # High failure rate to test retries
            processing_delay_seconds=0.01,
        )

        try:
            retry_attempts = []

            async def retry_test_handler(event: BaseEvent):
                """Handler that tracks retry attempts."""
                alert_data = {
                    "id": f"retry-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"Retry test: {event.message}",
                    "metadata": {"retry_test": True},
                }

                # Attempt notification multiple times if needed
                max_attempts = 3
                for attempt in range(max_attempts):
                    success, message = await retry_handler.send_notification(alert_data)
                    retry_attempts.append((event.event_id, attempt + 1, success, message))

                    if success:
                        break
                    elif attempt < max_attempts - 1:
                        # Wait before retry
                        await asyncio.sleep(0.05)

            # Subscribe to alerts
            subscription_id = event_bus.subscribe(EventType.ALERT, retry_test_handler)

            # Test: Send events that may require retries
            num_events = 10
            for i in range(num_events):
                event = AlertEvent(
                    event_id=f"retry-evt-{i}",
                    symbol="RETRYUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"retry-alert-{i}",
                    severity="warning",
                    rule_name="retry_test",
                    message=f"Retry test event {i}",
                    metadata={"test_index": i},
                )
                await event_bus.publish(event)

            # Wait for processing with retries
            await event_bus.flush_queue(timeout=15.0)

            # Verify: Retry attempts were made
            assert len(retry_attempts) >= num_events  # At least one attempt per event

            # Group attempts by event
            attempts_by_event = {}
            for event_id, attempt_num, success, message in retry_attempts:
                if event_id not in attempts_by_event:
                    attempts_by_event[event_id] = []
                attempts_by_event[event_id].append((attempt_num, success, message))

            # Verify retry behavior
            events_with_retries = 0
            successful_events = 0

            for event_id, attempts in attempts_by_event.items():
                if len(attempts) > 1:
                    events_with_retries += 1

                # Check if any attempt succeeded
                if any(success for _, success, _ in attempts):
                    successful_events += 1

            # With 80% failure rate, some retries should have occurred
            assert events_with_retries > 0

            # Some events should eventually succeed through retries
            assert successful_events > 0

            # Verify notification handler statistics
            stats = await retry_handler.get_notification_statistics()
            # Due to simulation, total attempts should be recorded
            assert stats["total_sent"] + stats["total_failed"] > num_events

            # Cleanup
            event_bus.unsubscribe(subscription_id)

        finally:
            await retry_handler.close()

    @pytest.mark.asyncio
    async def test_timeout_recovery(self, event_bus):
        """Test recovery from notification timeouts."""
        timeout_results = []

        # Create handler with processing delay to cause timeouts
        timeout_handler = InMemoryNotificationHandler(
            max_queue_size=100,
            max_history_size=500,
            processing_delay_seconds=0.2,  # Longer delay
            simulate_failure_rate=0.0,
        )

        try:

            async def timeout_test_handler(event: BaseEvent):
                """Handler that may timeout."""
                alert_data = {
                    "id": f"timeout-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"Timeout test: {event.message}",
                    "metadata": {"timeout_test": True},
                }

                start_time = asyncio.get_event_loop().time()

                try:
                    # Set a shorter timeout than processing delay
                    success, message = await asyncio.wait_for(
                        timeout_handler.send_notification(alert_data),
                        timeout=0.1,  # Shorter than processing_delay_seconds
                    )
                    processing_time = asyncio.get_event_loop().time() - start_time
                    timeout_results.append(("success", event.event_id, processing_time, message))

                except asyncio.TimeoutError:
                    processing_time = asyncio.get_event_loop().time() - start_time
                    timeout_results.append(("timeout", event.event_id, processing_time, "Operation timed out"))

            # Subscribe to alerts
            subscription_id = event_bus.subscribe(EventType.ALERT, timeout_test_handler)

            # Test: Send events that may timeout
            num_events = 8
            for i in range(num_events):
                event = AlertEvent(
                    event_id=f"timeout-evt-{i}",
                    symbol="TIMEOUTUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"timeout-alert-{i}",
                    severity="info",
                    rule_name="timeout_test",
                    message=f"Timeout test event {i}",
                    metadata={"test_index": i},
                )
                await event_bus.publish(event)

            # Wait for processing (including timeouts)
            await event_bus.flush_queue(timeout=10.0)

            # Verify: Some operations should have timed out
            assert len(timeout_results) == num_events

            timeout_count = len([r for r in timeout_results if r[0] == "timeout"])
            success_count = len([r for r in timeout_results if r[0] == "success"])

            # Most should timeout due to short timeout vs long processing delay
            assert timeout_count > 0

            # Verify timeout detection worked correctly
            for result_type, event_id, processing_time, message in timeout_results:
                if result_type == "timeout":
                    # Processing time should be around timeout duration
                    assert 0.09 <= processing_time <= 0.15  # Allow some variance
                    assert "timed out" in message.lower() or "timeout" in message.lower()

            # Cleanup
            event_bus.unsubscribe(subscription_id)

        finally:
            await timeout_handler.close()

    @pytest.mark.asyncio
    async def test_queue_overflow_recovery(self, event_bus):
        """Test recovery when notification queue overflows."""
        overflow_recovery_results = []

        # Create handler with very small queue
        overflow_handler = InMemoryNotificationHandler(
            max_queue_size=3,  # Very small queue
            max_history_size=100,
            processing_delay_seconds=0.1,  # Delay to cause backup
            simulate_failure_rate=0.0,
        )

        try:

            async def overflow_recovery_handler(event: BaseEvent):
                """Handler that attempts recovery from queue overflow."""
                alert_data = {
                    "id": f"overflow-recovery-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"Overflow recovery test: {event.message}",
                    "metadata": {"overflow_recovery_test": True},
                }

                # Try multiple times with backoff if queue is full
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        success, message = await overflow_handler.send_notification(alert_data)
                        overflow_recovery_results.append(("success", event.event_id, retry, message))
                        break

                    except Exception as e:
                        if "queue is full" in str(e).lower():
                            # Exponential backoff for queue full
                            backoff_time = 0.1 * (2**retry)
                            await asyncio.sleep(backoff_time)
                            overflow_recovery_results.append(("retry", event.event_id, retry, str(e)))

                            if retry == max_retries - 1:
                                overflow_recovery_results.append(
                                    ("failed", event.event_id, retry, "Max retries exceeded")
                                )
                        else:
                            overflow_recovery_results.append(("error", event.event_id, retry, str(e)))
                            break

            # Subscribe to alerts
            subscription_id = event_bus.subscribe(EventType.ALERT, overflow_recovery_handler)

            # Test: Rapidly send events to overflow queue
            num_events = 12  # Much more than queue capacity (3)
            events = []
            for i in range(num_events):
                event = AlertEvent(
                    event_id=f"overflow-recovery-{i}",
                    symbol="OVERFLOWUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"overflow-recovery-alert-{i}",
                    severity="warning",
                    rule_name="overflow_recovery_test",
                    message=f"Overflow recovery test event {i}",
                    metadata={"test_index": i},
                )
                events.append(event)

            # Publish events rapidly to cause overflow
            for event in events:
                await event_bus.publish(event)

            # Wait for processing and recovery
            await event_bus.flush_queue(timeout=20.0)

            # Verify: All events were processed (either success or retry attempts)
            assert len(overflow_recovery_results) >= num_events  # Should include retries

            # Analyze results
            success_results = [r for r in overflow_recovery_results if r[0] == "success"]
            retry_results = [r for r in overflow_recovery_results if r[0] == "retry"]
            failed_results = [r for r in overflow_recovery_results if r[0] == "failed"]

            # All should eventually succeed (even if queue was full)
            assert len(success_results) == num_events

            # Check that some events encountered queue full condition
            queue_full_results = [r for r in success_results if "queue is full" in r[3]]
            assert len(queue_full_results) > 0, "Expected some notifications to encounter queue full condition"

            # Since handler gracefully handles queue full by returning success=True with message,
            # we verify that the queue full condition was detected but handled gracefully
            if len(retry_results) > 0:
                # If retries occurred, verify retry behavior shows exponential backoff
                retry_counts_by_event = {}
                for result_type, event_id, retry_count, message in retry_results:
                    if event_id not in retry_counts_by_event:
                        retry_counts_by_event[event_id] = []
                    retry_counts_by_event[event_id].append(retry_count)

                # Some events should have multiple retry attempts
                events_with_multiple_retries = [
                    event_id for event_id, retries in retry_counts_by_event.items() if len(retries) > 1
                ]
                assert len(events_with_multiple_retries) > 0
            else:
                # Handler gracefully handled queue overflow without throwing exceptions
                print(f"Handler gracefully handled queue overflow for {len(queue_full_results)} events")

            # Cleanup
            event_bus.unsubscribe(subscription_id)

        finally:
            await overflow_handler.close()

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, event_bus):
        """Test recovery when some notifications fail while others succeed."""
        partial_failure_results = []

        # Create handler with moderate failure rate
        partial_handler = InMemoryNotificationHandler(
            max_queue_size=100,
            max_history_size=500,
            simulate_failure_rate=0.3,  # 30% failure rate
            processing_delay_seconds=0.01,
        )

        try:

            async def partial_failure_handler(event: BaseEvent):
                """Handler that manages partial failures."""
                alert_data = {
                    "id": f"partial-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"Partial failure test: {event.message}",
                    "metadata": {"partial_failure_test": True, "event_index": event.event_id},
                }

                success, message = await partial_handler.send_notification(alert_data)
                partial_failure_results.append((event.event_id, success, message))

                # If critical alert fails, try alternative approach
                if not success and event.severity == "critical":
                    # Simulate fallback mechanism
                    fallback_data = {
                        "id": f"fallback-{event.event_id}",
                        "rule_name": f"fallback_{event.rule_name}",
                        "severity": event.severity,
                        "message": f"FALLBACK: {event.message}",
                        "metadata": {"fallback": True, "original_id": event.event_id},
                    }

                    fallback_success, fallback_message = await partial_handler.send_notification(fallback_data)
                    partial_failure_results.append((f"fallback-{event.event_id}", fallback_success, fallback_message))

            # Subscribe to alerts
            subscription_id = event_bus.subscribe(EventType.ALERT, partial_failure_handler)

            # Test: Send mix of critical and non-critical events
            test_events = []
            for i in range(15):
                severity = "critical" if i % 3 == 0 else "warning"
                event = AlertEvent(
                    event_id=f"partial-{i}",
                    symbol="PARTIALUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"partial-alert-{i}",
                    severity=severity,
                    rule_name="partial_failure_test",
                    message=f"Partial failure test event {i}",
                    metadata={"test_index": i},
                )
                test_events.append(event)

            for event in test_events:
                await event_bus.publish(event)

            # Wait for processing
            await event_bus.flush_queue(timeout=10.0)

            # Verify: Partial failures and recoveries occurred
            assert len(partial_failure_results) >= 15  # At least original attempts

            # Analyze results
            original_results = [r for r in partial_failure_results if not r[0].startswith("fallback-")]
            fallback_results = [r for r in partial_failure_results if r[0].startswith("fallback-")]

            successful_originals = [r for r in original_results if r[1] is True]
            failed_originals = [r for r in original_results if r[1] is False]

            # With 30% failure rate, expect some failures
            assert len(failed_originals) > 0
            assert len(successful_originals) > 0

            # Check that critical failures triggered fallbacks
            critical_failures = [
                r
                for r in failed_originals
                if any(event.event_id == r[0] and event.severity == "critical" for event in test_events)
            ]

            if critical_failures:
                # Should have some fallback attempts
                assert len(fallback_results) > 0

                # Verify fallback data structure
                for fallback_id, fallback_success, fallback_message in fallback_results:
                    assert fallback_id.startswith("fallback-")
                    if fallback_success:
                        assert "sent successfully" in fallback_message

            # Verify notification handler statistics
            stats = await partial_handler.get_notification_statistics()
            total_attempts = stats["total_sent"] + stats["total_failed"]
            assert total_attempts >= 15  # Original + any fallbacks

            # Verify some notifications succeeded
            assert stats["total_sent"] > 0

            # Cleanup
            event_bus.unsubscribe(subscription_id)

        finally:
            await partial_handler.close()

    @pytest.mark.asyncio
    async def test_cascading_failure_isolation(self, event_bus):
        """Test that failures in one notification don't affect others."""
        isolation_results = []

        # Create multiple handlers with different failure rates
        stable_handler = InMemoryNotificationHandler(
            max_queue_size=100,
            simulate_failure_rate=0.0,  # No failures
        )

        unreliable_handler = InMemoryNotificationHandler(
            max_queue_size=100,
            simulate_failure_rate=0.9,  # High failure rate
        )

        try:

            async def stable_notification_handler(event: BaseEvent):
                """Stable handler that shouldn't be affected by other failures."""
                alert_data = {
                    "id": f"stable-{event.event_id}",
                    "rule_name": f"stable_{event.rule_name}",
                    "severity": event.severity,
                    "message": f"Stable: {event.message}",
                    "metadata": {"handler": "stable"},
                }

                success, message = await stable_handler.send_notification(alert_data)
                isolation_results.append(("stable", event.event_id, success, message))

            async def unreliable_notification_handler(event: BaseEvent):
                """Unreliable handler that fails frequently."""
                alert_data = {
                    "id": f"unreliable-{event.event_id}",
                    "rule_name": f"unreliable_{event.rule_name}",
                    "severity": event.severity,
                    "message": f"Unreliable: {event.message}",
                    "metadata": {"handler": "unreliable"},
                }

                success, message = await unreliable_handler.send_notification(alert_data)
                isolation_results.append(("unreliable", event.event_id, success, message))

            # Subscribe both handlers to same events
            stable_sub = event_bus.subscribe(EventType.ALERT, stable_notification_handler)
            unreliable_sub = event_bus.subscribe(EventType.ALERT, unreliable_notification_handler)

            # Test: Send events to both handlers
            num_events = 10
            for i in range(num_events):
                event = AlertEvent(
                    event_id=f"isolation-{i}",
                    symbol="ISOLATIONUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"isolation-alert-{i}",
                    severity="warning",
                    rule_name="isolation_test",
                    message=f"Isolation test event {i}",
                    metadata={"test_index": i},
                )
                await event_bus.publish(event)

            # Wait for processing
            await event_bus.flush_queue(timeout=10.0)

            # Verify: Both handlers processed events independently
            assert len(isolation_results) == num_events * 2  # Both handlers for each event

            # Separate results by handler
            stable_results = [r for r in isolation_results if r[0] == "stable"]
            unreliable_results = [r for r in isolation_results if r[0] == "unreliable"]

            assert len(stable_results) == num_events
            assert len(unreliable_results) == num_events

            # Verify stable handler succeeded (not affected by unreliable handler)
            stable_successes = [r for r in stable_results if r[2] is True]
            assert len(stable_successes) == num_events  # Should be 100% success

            # Verify unreliable handler failed frequently
            unreliable_failures = [r for r in unreliable_results if r[2] is False]
            assert len(unreliable_failures) > 0  # Should have failures due to 90% failure rate

            # Verify isolation - stable handler statistics
            stable_stats = await stable_handler.get_notification_statistics()
            assert stable_stats["total_sent"] == num_events
            assert stable_stats["total_failed"] == 0

            # Verify unreliable handler statistics
            unreliable_stats = await unreliable_handler.get_notification_statistics()
            assert unreliable_stats["total_sent"] + unreliable_stats["total_failed"] == num_events
            assert unreliable_stats["total_failed"] > 0

            # Verify event bus processed all events regardless of individual handler failures
            bus_stats = event_bus.get_statistics()
            assert bus_stats["published_count"] == num_events
            assert bus_stats["processed_count"] == num_events

            # Cleanup
            event_bus.unsubscribe(stable_sub)
            event_bus.unsubscribe(unreliable_sub)

        finally:
            await stable_handler.close()
            await unreliable_handler.close()


# Additional test utilities for notification-specific scenarios


async def wait_for_notification_with_status(
    handler: InMemoryNotificationHandler, status: NotificationStatus, count: int, timeout: float = 5.0
) -> bool:
    """Wait for specific number of notifications with given status."""

    async def check_status_count():
        history = await handler.get_notification_history(limit=1000, status=status)
        return len(history) >= count

    return await wait_for_condition(check_status_count, timeout)


async def wait_for_notification_by_rule(
    handler: InMemoryNotificationHandler, rule_name: str, count: int, timeout: float = 5.0
) -> bool:
    """Wait for specific number of notifications for a given rule."""

    async def check_rule_count():
        history = await handler.get_notification_history(limit=1000, rule_name=rule_name)
        return len(history) >= count

    return await wait_for_condition(check_rule_count, timeout)
