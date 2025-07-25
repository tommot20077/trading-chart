# ABOUTME: Integration tests for notification handler statistics and history tracking
# ABOUTME: Tests comprehensive notification history management, statistics collection, and automated cleanup functionality

import asyncio
import pytest
import pytest_asyncio
import time
import time_machine
from typing import Dict, Any
from datetime import datetime, UTC, timezone

# Notification statistics and history integration tests

from core.implementations.memory.observability.notification_handler import (
    InMemoryNotificationHandler,
    NotificationStatus,
)
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from .conftest import wait_for_notification_count, wait_for_condition

class StatsTestEvent(BaseEvent[Dict[str, Any]]):
    """Test event for statistics and history testing."""

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
        priority_map = {"critical": EventPriority.HIGH, "warning": EventPriority.NORMAL, "info": EventPriority.LOW}
        priority = priority_map.get(severity, EventPriority.NORMAL)
        
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
async def stats_event_bus():
    """Create event bus for statistics testing."""
    bus = InMemoryEventBus(max_queue_size=2000, handler_timeout=15.0)
    try:
        yield bus
    finally:
        await bus.close()


@pytest_asyncio.fixture
async def stats_notification_handler():
    """Create notification handler optimized for statistics testing."""
    handler = InMemoryNotificationHandler(
        max_queue_size=1000,
        max_history_size=2000,
        history_retention_hours=2.0,
        cleanup_interval_seconds=10.0,  # Faster cleanup for testing
        max_retry_attempts=2,
        simulate_failure_rate=0.0,
        processing_delay_seconds=0.0,
    )
    try:
        yield handler
    finally:
        await handler.close()


@pytest_asyncio.fixture
async def cleanup_notification_handler():
    """Create notification handler for cleanup testing with shorter retention."""
    handler = InMemoryNotificationHandler(
        max_queue_size=500,
        max_history_size=100,  # Small history for cleanup testing
        history_retention_hours=0.01,  # Very short retention (36 seconds)
        cleanup_interval_seconds=5.0,  # Fast cleanup
        max_retry_attempts=1,
        simulate_failure_rate=0.0,
        processing_delay_seconds=0.0,
    )
    try:
        yield handler
    finally:
        await handler.close()


class TestNotificationHistoryTracking:
    """Test notification history tracking and management."""

    @pytest.mark.asyncio
    async def test_comprehensive_history_tracking(self, stats_event_bus, stats_notification_handler):
        """Test comprehensive notification history tracking across different scenarios."""
        history_data = []

        async def history_tracking_handler(event: BaseEvent):
            """Handler that tracks all notification attempts."""
            alert_data = {
                "id": f"history-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": f"History tracking: {event.message}",
                "metadata": {"original_event_id": event.event_id, "tracking_test": True, "timestamp": event.timestamp},
            }

            success, message = await stats_notification_handler.send_notification(alert_data)
            history_data.append((event.event_id, success, message, time.time()))

        # Subscribe to alerts
        subscription_id = stats_event_bus.subscribe(EventType.ALERT, history_tracking_handler)

        # Test: Create comprehensive test data with various scenarios
        test_scenarios = [
            # Scenario 1: Different severities
            ("critical", "database_failure", "Database connection lost"),
            ("warning", "high_latency", "API response time elevated"),
            ("info", "status_update", "System health check completed"),
            # Scenario 2: Different rule types
            ("critical", "security_breach", "Unauthorized access detected"),
            ("warning", "resource_usage", "CPU usage above threshold"),
            ("info", "maintenance", "Scheduled maintenance started"),
            # Scenario 3: Repeated rules (different events)
            ("critical", "database_failure", "Database timeout occurred"),
            ("warning", "high_latency", "Network latency spike detected"),
            ("info", "status_update", "Monitoring system restarted"),
            # Scenario 4: Edge cases
            ("critical", "system_failure", "Complete system failure"),
            ("warning", "memory_leak", "Memory usage growing steadily"),
            ("info", "backup_complete", "Daily backup completed successfully"),
        ]

        # Create and publish events
        for i, (severity, rule_name, message) in enumerate(test_scenarios):
            event = StatsTestEvent(
                event_id=f"history-evt-{i:03d}",
                symbol="HISTORYUSDT",
                timestamp=1640995200.0 + i * 60,  # Space events 1 minute apart
                alert_id=f"history-alert-{i:03d}",
                severity=severity,
                rule_name=rule_name,
                message=message,
                metadata={"scenario_index": i, "test_type": "history_tracking"},
            )
            await stats_event_bus.publish(event)

        # Wait for all processing
        await wait_for_notification_count(stats_notification_handler, len(test_scenarios))
        await stats_event_bus.flush_queue(timeout=10.0)

        # Verify: All events were processed
        assert len(history_data) == len(test_scenarios)

        # Test history retrieval with different filters
        # 1. Get all history
        all_history = await stats_notification_handler.get_notification_history(limit=100)
        assert len(all_history) == len(test_scenarios)

        # 2. Filter by severity
        critical_history = await stats_notification_handler.get_notification_history(limit=100, severity="critical")
        expected_critical = len([s for s in test_scenarios if s[0] == "critical"])
        assert len(critical_history) == expected_critical

        warning_history = await stats_notification_handler.get_notification_history(limit=100, severity="warning")
        expected_warning = len([s for s in test_scenarios if s[0] == "warning"])
        assert len(warning_history) == expected_warning

        info_history = await stats_notification_handler.get_notification_history(limit=100, severity="info")
        expected_info = len([s for s in test_scenarios if s[0] == "info"])
        assert len(info_history) == expected_info

        # 3. Filter by rule name
        db_failure_history = await stats_notification_handler.get_notification_history(
            limit=100, rule_name="database_failure"
        )
        expected_db_failures = len([s for s in test_scenarios if s[1] == "database_failure"])
        assert len(db_failure_history) == expected_db_failures

        # 4. Filter by status
        sent_history = await stats_notification_handler.get_notification_history(
            limit=100, status=NotificationStatus.SENT
        )
        assert len(sent_history) == len(test_scenarios)  # All should be sent successfully

        # 5. Test history record completeness
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
            assert field in sample_record, f"Missing field: {field}"

        # Verify record data integrity
        assert sample_record["status"] == "sent"
        assert sample_record["sent_at"] is not None
        assert sample_record["processing_time"] is not None
        assert sample_record["processing_time"] >= 0
        assert sample_record["retry_count"] == 0  # No retries for successful sends
        assert sample_record["error_message"] is None  # No errors for successful sends

        # 6. Test history ordering (newest first)
        timestamps = [record["created_at"] for record in all_history]
        assert timestamps == sorted(timestamps, reverse=True), "History should be ordered newest first"

        # Cleanup
        stats_event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_history_with_failures(self, stats_event_bus):
        """Test history tracking when notifications include failures."""
        # Create handler with some failure rate
        failing_handler = InMemoryNotificationHandler(
            max_queue_size=500,
            max_history_size=1000,
            simulate_failure_rate=0.4,  # 40% failure rate
            processing_delay_seconds=0.01,
        )

        try:
            failure_results = []

            async def failure_tracking_handler(event: BaseEvent):
                """Handler that tracks both successes and failures."""
                alert_data = {
                    "id": f"failure-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"Failure test: {event.message}",
                    "metadata": {"failure_test": True},
                }

                success, message = await failing_handler.send_notification(alert_data)
                failure_results.append((event.event_id, success, message))

            # Subscribe to alerts
            subscription_id = stats_event_bus.subscribe(EventType.ALERT, failure_tracking_handler)

            # Test: Send multiple events to get mix of successes and failures
            num_events = 20
            for i in range(num_events):
                event = StatsTestEvent(
                    event_id=f"failure-{i:03d}",
                    symbol="FAILUREUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"failure-alert-{i:03d}",
                    severity="warning",
                    rule_name="failure_test",
                    message=f"Failure test event {i}",
                    metadata={"test_index": i},
                )
                await stats_event_bus.publish(event)

            # Wait for processing
            await stats_event_bus.flush_queue(timeout=15.0)

            # Verify: Mix of successes and failures
            assert len(failure_results) == num_events
            successes = [r for r in failure_results if r[1] is True]
            failures = [r for r in failure_results if r[1] is False]

            # Should have both successes and failures (with 40% failure rate)
            assert len(failures) > 0
            assert len(successes) > 0

            # Test history for both statuses
            sent_history = await failing_handler.get_notification_history(limit=100, status=NotificationStatus.SENT)
            failed_history = await failing_handler.get_notification_history(limit=100, status=NotificationStatus.FAILED)

            assert len(sent_history) == len(successes)
            assert len(failed_history) == len(failures)

            # Verify failed records have error information
            for failed_record in failed_history:
                assert failed_record["status"] == "failed"
                assert failed_record["error_message"] is not None
                assert failed_record["sent_at"] is None
                assert (
                    "failure" in failed_record["error_message"].lower()
                    or "error" in failed_record["error_message"].lower()
                )

            # Verify successful records are complete
            for sent_record in sent_history:
                assert sent_record["status"] == "sent"
                assert sent_record["sent_at"] is not None
                assert sent_record["processing_time"] is not None
                assert sent_record["error_message"] is None

            # Cleanup
            stats_event_bus.unsubscribe(subscription_id)

        finally:
            await failing_handler.close()

    @pytest.mark.asyncio
    async def test_history_pagination_and_limits(self, stats_event_bus, stats_notification_handler):
        """Test history retrieval with pagination and limits."""

        async def pagination_handler(event: BaseEvent):
            """Handler for pagination testing."""
            alert_data = {
                "id": f"page-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": f"Pagination test: {event.message}",
                "metadata": {"pagination_test": True},
            }
            await stats_notification_handler.send_notification(alert_data)

        # Subscribe to alerts
        subscription_id = stats_event_bus.subscribe(EventType.ALERT, pagination_handler)

        # Test: Create many events to test pagination
        num_events = 50
        for i in range(num_events):
            event = StatsTestEvent(
                event_id=f"page-{i:03d}",
                symbol="PAGEUSDT",
                timestamp=1640995200.0 + i,
                alert_id=f"page-alert-{i:03d}",
                severity="info",
                rule_name="pagination_test",
                message=f"Pagination test event {i}",
                metadata={"page_index": i},
            )
            await stats_event_bus.publish(event)

        # Wait for processing
        await wait_for_notification_count(stats_notification_handler, num_events)
        await stats_event_bus.flush_queue(timeout=10.0)

        # Test different limit values
        # 1. Small limit
        small_page = await stats_notification_handler.get_notification_history(limit=5)
        assert len(small_page) == 5

        # 2. Medium limit
        medium_page = await stats_notification_handler.get_notification_history(limit=20)
        assert len(medium_page) == 20

        # 3. Large limit (more than available)
        large_page = await stats_notification_handler.get_notification_history(limit=100)
        assert len(large_page) == num_events  # Should return all available

        # 4. Test ordering consistency across different limits
        first_5 = await stats_notification_handler.get_notification_history(limit=5)
        first_10 = await stats_notification_handler.get_notification_history(limit=10)

        # First 5 of the 10-item result should match the 5-item result
        assert [r["id"] for r in first_5] == [r["id"] for r in first_10[:5]]

        # 5. Test that pagination maintains chronological order
        all_records = await stats_notification_handler.get_notification_history(limit=num_events)
        timestamps = [r["created_at"] for r in all_records]
        assert timestamps == sorted(timestamps, reverse=True), "Records should be in descending chronological order"

        # Cleanup
        stats_event_bus.unsubscribe(subscription_id)


class TestNotificationStatisticsCollection:
    """Test notification statistics collection and aggregation."""

    @pytest.mark.asyncio
    async def test_comprehensive_statistics_collection(self, stats_event_bus, stats_notification_handler):
        """Test comprehensive statistics collection across multiple dimensions."""

        async def stats_collection_handler(event: BaseEvent):
            """Handler for statistics collection testing."""
            alert_data = {
                "id": f"stats-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": f"Stats test: {event.message}",
                "metadata": {"stats_test": True},
            }
            await stats_notification_handler.send_notification(alert_data)

        # Subscribe to alerts
        subscription_id = stats_event_bus.subscribe(EventType.ALERT, stats_collection_handler)

        # Test: Create events with diverse characteristics for statistics
        test_data = [
            # Different severities
            ("critical", "system_failure", 5),  # 5 critical system_failure
            ("warning", "performance_issue", 8),  # 8 warning performance_issue
            ("info", "status_update", 12),  # 12 info status_update
            ("critical", "security_alert", 3),  # 3 critical security_alert
            ("warning", "resource_warning", 6),  # 6 warning resource_warning
            ("info", "maintenance_notice", 4),  # 4 info maintenance_notice
        ]

        event_counter = 0
        expected_totals = {"critical": 0, "warning": 0, "info": 0}
        expected_rules = {}

        for severity, rule_name, count in test_data:
            expected_totals[severity] += count
            expected_rules[rule_name] = count

            for i in range(count):
                event = StatsTestEvent(
                    event_id=f"stats-{event_counter:03d}",
                    symbol="STATSUSDT",
                    timestamp=1640995200.0 + event_counter,
                    alert_id=f"stats-alert-{event_counter:03d}",
                    severity=severity,
                    rule_name=rule_name,
                    message=f"Stats test event {event_counter} - {severity} {rule_name}",
                    metadata={"test_index": event_counter},
                )
                await stats_event_bus.publish(event)
                event_counter += 1

        total_events = sum(count for _, _, count in test_data)

        # Wait for processing
        await wait_for_notification_count(stats_notification_handler, total_events)
        await stats_event_bus.flush_queue(timeout=15.0)

        # Verify: Comprehensive statistics
        stats = await stats_notification_handler.get_notification_statistics()

        # 1. Overall totals
        assert stats["total_sent"] == total_events
        assert stats["total_failed"] == 0  # No failures expected
        assert stats["total_expired"] == 0
        assert stats["pending_queue_size"] == 0
        assert stats["history_size"] == total_events

        # 2. Processing time statistics
        assert stats["average_processing_time"] >= 0
        assert isinstance(stats["average_processing_time"], (int, float))

        # 3. Statistics by severity
        severity_stats = stats["notifications_by_severity"]
        for severity, expected_count in expected_totals.items():
            assert severity_stats[severity] == expected_count, (
                f"Severity {severity}: expected {expected_count}, got {severity_stats.get(severity, 0)}"
            )

        # 4. Statistics by rule
        rule_stats = stats["notifications_by_rule"]
        for rule_name, expected_count in expected_rules.items():
            assert rule_stats[rule_name] == expected_count, (
                f"Rule {rule_name}: expected {expected_count}, got {rule_stats.get(rule_name, 0)}"
            )

        # 5. System status
        assert stats["is_closed"] is False
        assert stats["last_cleanup_time"] >= 0

        # Cleanup
        stats_event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_statistics_with_mixed_results(self, stats_event_bus):
        """Test statistics collection with mixed success/failure results."""
        # Create handler with failures for testing
        mixed_handler = InMemoryNotificationHandler(
            max_queue_size=500,
            max_history_size=1000,
            simulate_failure_rate=0.25,  # 25% failure rate
            processing_delay_seconds=0.01,
        )

        try:

            async def mixed_results_handler(event: BaseEvent):
                """Handler for mixed results testing."""
                alert_data = {
                    "id": f"mixed-{event.event_id}",
                    "rule_name": event.rule_name,
                    "severity": event.severity,
                    "message": f"Mixed results test: {event.message}",
                    "metadata": {"mixed_test": True},
                }
                await mixed_handler.send_notification(alert_data)

            # Subscribe to alerts
            subscription_id = stats_event_bus.subscribe(EventType.ALERT, mixed_results_handler)

            # Test: Send events with different characteristics
            severities = ["critical", "warning", "info"]
            rules = ["rule_A", "rule_B", "rule_C"]
            num_events = 30

            for i in range(num_events):
                severity = severities[i % len(severities)]
                rule_name = rules[i % len(rules)]

                event = StatsTestEvent(
                    event_id=f"mixed-{i:03d}",
                    symbol="MIXEDUSDT",
                    timestamp=1640995200.0 + i,
                    alert_id=f"mixed-alert-{i:03d}",
                    severity=severity,
                    rule_name=rule_name,
                    message=f"Mixed test event {i}",
                    metadata={"test_index": i},
                )
                await stats_event_bus.publish(event)

            # Wait for processing
            await stats_event_bus.flush_queue(timeout=15.0)

            # Verify: Statistics reflect mixed results
            stats = await mixed_handler.get_notification_statistics()

            # Should have both successes and failures
            assert stats["total_sent"] > 0
            assert stats["total_failed"] > 0
            assert stats["total_sent"] + stats["total_failed"] == num_events

            # Verify failure rate is approximately as expected (25% ± variance)
            failure_rate = stats["total_failed"] / num_events
            assert 0.1 <= failure_rate <= 0.4, f"Failure rate {failure_rate} outside expected range"

            # Verify statistics by severity include both success and failure cases
            severity_stats = stats["notifications_by_severity"]
            total_by_severity = sum(severity_stats.values())
            assert total_by_severity == num_events

            # Verify statistics by rule
            rule_stats = stats["notifications_by_rule"]
            total_by_rule = sum(rule_stats.values())
            assert total_by_rule == num_events

            # Each rule should have approximately equal distribution
            expected_per_rule = num_events // len(rules)
            for rule_name in rules:
                actual_count = rule_stats.get(rule_name, 0)
                # Allow some variance due to randomness
                assert expected_per_rule - 3 <= actual_count <= expected_per_rule + 3

            # Cleanup
            stats_event_bus.unsubscribe(subscription_id)

        finally:
            await mixed_handler.close()

    @pytest.mark.asyncio
    async def test_real_time_statistics_updates(self, stats_event_bus, stats_notification_handler):
        """Test that statistics are updated in real-time as notifications are processed."""
        statistics_snapshots = []

        async def real_time_handler(event: BaseEvent):
            """Handler that captures statistics at different points."""
            # Capture statistics before sending notification
            pre_stats = await stats_notification_handler.get_notification_statistics()

            alert_data = {
                "id": f"realtime-{event.event_id}",
                "rule_name": event.rule_name,
                "severity": event.severity,
                "message": f"Real-time test: {event.message}",
                "metadata": {"realtime_test": True},
            }

            success, _ = await stats_notification_handler.send_notification(alert_data)

            # Capture statistics after sending notification
            post_stats = await stats_notification_handler.get_notification_statistics()

            statistics_snapshots.append(
                {"event_id": event.event_id, "success": success, "pre_stats": pre_stats, "post_stats": post_stats}
            )

        # Subscribe to alerts
        subscription_id = stats_event_bus.subscribe(EventType.ALERT, real_time_handler)

        # Test: Send events one by one to observe incremental updates
        num_events = 10
        for i in range(num_events):
            event = StatsTestEvent(
                event_id=f"realtime-{i:03d}",
                symbol="REALTIMEUSDT",
                timestamp=1640995200.0 + i,
                alert_id=f"realtime-alert-{i:03d}",
                severity="info",
                rule_name="realtime_test",
                message=f"Real-time test event {i}",
                metadata={"test_index": i},
            )
            await stats_event_bus.publish(event)

            # Wait for this specific event to be processed
            await asyncio.sleep(0.1)  # Small delay to ensure processing

        # Wait for all processing to complete
        await stats_event_bus.flush_queue(timeout=10.0)

        # Verify: Statistics updated incrementally
        assert len(statistics_snapshots) == num_events

        for i, snapshot in enumerate(statistics_snapshots):
            pre_stats = snapshot["pre_stats"]
            post_stats = snapshot["post_stats"]
            success = snapshot["success"]

            if success:
                # Should see increment in total_sent
                assert post_stats["total_sent"] == pre_stats["total_sent"] + 1
                assert post_stats["total_failed"] == pre_stats["total_failed"]
            else:
                # Should see increment in total_failed (though not expected with 0% failure rate)
                assert post_stats["total_failed"] == pre_stats["total_failed"] + 1
                assert post_stats["total_sent"] == pre_stats["total_sent"]

            # History size should increase
            assert post_stats["history_size"] == pre_stats["history_size"] + 1

            # Severity and rule counts should update
            assert (
                post_stats["notifications_by_severity"]["info"]
                == pre_stats["notifications_by_severity"].get("info", 0) + 1
            )
            assert (
                post_stats["notifications_by_rule"]["realtime_test"]
                == pre_stats["notifications_by_rule"].get("realtime_test", 0) + 1
            )

        # Verify final state
        final_stats = await stats_notification_handler.get_notification_statistics()
        assert final_stats["total_sent"] == num_events
        assert final_stats["history_size"] == num_events
        assert final_stats["notifications_by_severity"]["info"] == num_events
        assert final_stats["notifications_by_rule"]["realtime_test"] == num_events

        # Cleanup
        stats_event_bus.unsubscribe(subscription_id)


class TestNotificationAutomaticCleanup:
    """Test automatic cleanup functionality for notification history."""

    @pytest.mark.asyncio
    async def test_automatic_history_cleanup(self, stats_event_bus, cleanup_notification_handler):
        """Test automatic cleanup of old notification history."""
        
        # Create notifications directly in time-machine context to ensure old timestamps
        with time_machine.travel(datetime(2024, 1, 1, tzinfo=timezone.utc), tick=False):
            # Test: Create notifications that will exceed retention time
            initial_events = 20
            for i in range(initial_events):
                alert_data = {
                    "id": f"cleanup-{i:03d}",
                    "rule_name": "cleanup_test",
                    "severity": "info",
                    "message": f"Cleanup test event {i}",
                    "metadata": {"test_index": i, "batch": "initial"},
                }
                success, _ = await cleanup_notification_handler.send_notification(alert_data)
                assert success, f"Failed to send notification {i}"

        # Verify initial state (outside time machine to use real time for verification)
        initial_stats = await cleanup_notification_handler.get_notification_statistics()
        assert initial_stats["total_sent"] == initial_events
        assert initial_stats["history_size"] == initial_events

        # Now advance time and trigger cleanup in time-machine context
        with time_machine.travel(datetime(2024, 1, 1, 0, 0, 45, tzinfo=timezone.utc), tick=False):
            # Manually trigger cleanup since time-machine doesn't affect background tasks
            await cleanup_notification_handler._cleanup_old_notifications()

        # Verify: Old history should be cleaned up (outside time machine for verification)
        post_cleanup_stats = await cleanup_notification_handler.get_notification_statistics()
        
        # History size should be smaller due to cleanup (old events removed)
        assert post_cleanup_stats["history_size"] < initial_events, f"Expected history size < {initial_events}, got {post_cleanup_stats['history_size']}"

        # Total sent should remain the same (cleanup doesn't affect this counter)
        assert post_cleanup_stats["total_sent"] == initial_events

    @pytest.mark.asyncio
    async def test_history_size_limit_cleanup(self, stats_event_bus, cleanup_notification_handler):
        """Test cleanup when history size limit is exceeded."""
        
        # Test: Send more events than max_history_size (100) directly
        num_events = 150  # Exceeds the 100 limit
        for i in range(num_events):
            alert_data = {
                "id": f"sizelimit-{i:03d}",
                "rule_name": "size_limit_test",
                "severity": "info",
                "message": f"Size limit test event {i}",
                "metadata": {"test_index": i},
            }
            success, _ = await cleanup_notification_handler.send_notification(alert_data)
            assert success, f"Failed to send notification {i}"

            # Add small delays periodically to allow cleanup
            if i % 20 == 0:
                await asyncio.sleep(0.01)

        # Manually trigger cleanup to enforce size limit
        await cleanup_notification_handler._cleanup_old_notifications()

        # Verify: History size should be limited
        stats = await cleanup_notification_handler.get_notification_statistics()

        # All events should have been processed
        assert stats["total_sent"] == num_events

        # History size should not exceed the limit
        assert stats["history_size"] <= 100  # Max history size

        # Get current history
        current_history = await cleanup_notification_handler.get_notification_history(limit=200)
        assert len(current_history) <= 100

        # History should contain the most recent events
        # Check that the highest numbered events are present
        present_indices = []
        for record in current_history:
            if "sizelimit-" in record["id"]:
                # Extract index from ID like "sizelimit-142"
                try:
                    index = int(record["id"].split("-")[1])
                    present_indices.append(index)
                except (IndexError, ValueError):
                    pass

        if present_indices:
            present_indices.sort()
            # Should contain recent events (higher indices)
            max_present = max(present_indices)
            min_present = min(present_indices)

            # Most recent events should be present
            assert max_present >= num_events - 50  # Within last 50 events

            # Verify chronological consistency
            assert max_present > min_present

    @pytest.mark.asyncio
    async def test_cleanup_preserves_statistics(self, stats_event_bus, cleanup_notification_handler):
        """Test that cleanup preserves overall statistics while cleaning history."""

        # Phase 1: Initial events with old timestamps
        with time_machine.travel(datetime(2024, 1, 1, tzinfo=timezone.utc), tick=False):
            phase1_events = 30
            for i in range(phase1_events):
                severity = "critical" if i % 3 == 0 else "warning"
                alert_data = {
                    "id": f"preserve-p1-{i:03d}",
                    "rule_name": "preservation_test_phase1",
                    "severity": severity,
                    "message": f"Phase 1 preservation test event {i}",
                    "metadata": {"test_index": i, "phase": 1},
                }
                success, _ = await cleanup_notification_handler.send_notification(alert_data)
                assert success, f"Failed to send notification {i}"

        # Capture phase 1 statistics
        phase1_stats = await cleanup_notification_handler.get_notification_statistics()

        # Advance time to trigger cleanup (retention period + cleanup interval)
        with time_machine.travel(datetime(2024, 1, 1, 0, 0, 45, tzinfo=timezone.utc), tick=False):
            # Manually trigger cleanup since time-machine doesn't affect background tasks
            await cleanup_notification_handler._cleanup_old_notifications()

        # Phase 2: More events after cleanup (with current timestamps)
        phase2_events = 25
        for i in range(phase2_events):
            severity = "info" if i % 2 == 0 else "warning"
            alert_data = {
                "id": f"preserve-p2-{i:03d}",
                "rule_name": "preservation_test_phase2",
                "severity": severity,
                "message": f"Phase 2 preservation test event {i}",
                "metadata": {"test_index": i, "phase": 2},
            }
            success, _ = await cleanup_notification_handler.send_notification(alert_data)
            assert success, f"Failed to send notification {i}"

        # Capture final statistics
        final_stats = await cleanup_notification_handler.get_notification_statistics()

        # Verify: Overall statistics should be preserved despite cleanup
        # Total counts should include all events
        assert final_stats["total_sent"] == phase1_events + phase2_events

        # History size may be smaller due to cleanup
        assert final_stats["history_size"] <= phase1_events + phase2_events

        # Severity statistics should include all events from both phases
        expected_critical = len([i for i in range(phase1_events) if i % 3 == 0])  # Phase 1 critical
        expected_warning = (
            len([i for i in range(phase1_events) if i % 3 != 0])  # Phase 1 warning
            + len([i for i in range(phase2_events) if i % 2 != 0])
        )  # Phase 2 warning
        expected_info = len([i for i in range(phase2_events) if i % 2 == 0])  # Phase 2 info

        severity_stats = final_stats["notifications_by_severity"]
        assert severity_stats.get("critical", 0) == expected_critical
        assert severity_stats.get("warning", 0) == expected_warning
        assert severity_stats.get("info", 0) == expected_info

        # Rule statistics should include both phases
        rule_stats = final_stats["notifications_by_rule"]
        assert rule_stats.get("preservation_test_phase1", 0) == phase1_events
        assert rule_stats.get("preservation_test_phase2", 0) == phase2_events

        # Cleanup timestamp should be updated
        assert final_stats["last_cleanup_time"] > phase1_stats["last_cleanup_time"]


# Additional test utilities for statistics and cleanup testing


async def wait_for_cleanup_cycle(handler: InMemoryNotificationHandler, timeout: float = 10.0) -> bool:
    """Wait for at least one cleanup cycle to complete."""
    initial_stats = await handler.get_notification_statistics()
    initial_cleanup_time = initial_stats["last_cleanup_time"]

    async def check_cleanup():
        stats = await handler.get_notification_statistics()
        return stats["last_cleanup_time"] > initial_cleanup_time

    return await wait_for_condition(check_cleanup, timeout)


async def wait_for_history_size_reduction(
    handler: InMemoryNotificationHandler, initial_size: int, timeout: float = 10.0
) -> bool:
    """Wait for history size to be reduced from initial size."""

    async def check_size_reduction():
        stats = await handler.get_notification_statistics()
        return stats["history_size"] < initial_size

    return await wait_for_condition(check_size_reduction, timeout)
