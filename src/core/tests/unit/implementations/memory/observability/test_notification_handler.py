# ABOUTME: Unit tests for InMemoryNotificationHandler implementation
# ABOUTME: Tests notification queue management, history tracking, and statistics collection

import asyncio
import pytest
import pytest_asyncio
import time
import time_machine

from core.implementations.memory.observability.notification_handler import (
    InMemoryNotificationHandler,
    NotificationStatus,
)


class TestInMemoryNotificationHandler:
    """Test the InMemoryNotificationHandler class."""

    @pytest_asyncio.fixture
    async def notification_handler(self):
        """Create a notification handler for testing."""
        handler = InMemoryNotificationHandler(
            max_queue_size=100,
            max_history_size=500,
            history_retention_hours=1.0,
            cleanup_interval_seconds=60.0,
            max_retry_attempts=3,
            simulate_failure_rate=0.0,
            processing_delay_seconds=0.0,
        )
        yield handler
        # Cleanup
        await handler.close()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test notification handler initialization."""
        handler = InMemoryNotificationHandler(
            max_queue_size=1000,
            max_history_size=5000,
            history_retention_hours=24.0,
            cleanup_interval_seconds=300.0,
            max_retry_attempts=5,
            simulate_failure_rate=0.1,
            processing_delay_seconds=0.5,
        )

        assert handler.max_queue_size == 1000
        assert handler.max_history_size == 5000
        assert handler.history_retention_seconds == 24.0 * 3600
        assert handler.cleanup_interval_seconds == 300.0
        assert handler.max_retry_attempts == 5
        assert handler.simulate_failure_rate == 0.1
        assert handler.processing_delay_seconds == 0.5
        assert handler.is_closed is False

        await handler.close()

    @pytest.mark.asyncio
    async def test_send_notification_success(self, notification_handler):
        """Test successful notification sending."""
        alert_data = {
            "id": "alert-001",
            "rule_name": "high_cpu_usage",
            "severity": "warning",
            "message": "CPU usage is above 80%",
            "labels": {"service": "web"},
            "annotations": {"runbook": "https://example.com/runbook"},
            "fired_at": "2024-01-01T12:00:00Z",
            "trace_id": "trace-123",
        }

        success, message = await notification_handler.send_notification(alert_data)

        assert success is True
        assert "Notification sent successfully" in message
        assert "ID:" in message

        # Check statistics
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 1
        assert stats["total_failed"] == 0
        assert stats["notifications_by_severity"]["warning"] == 1
        assert stats["notifications_by_rule"]["high_cpu_usage"] == 1

    @pytest.mark.asyncio
    async def test_send_notification_invalid_data(self, notification_handler):
        """Test sending notification with invalid data."""
        # Test with non-dict data
        success, message = await notification_handler.send_notification("invalid")
        assert success is False
        assert "alert_data must be a dictionary" in message

        # Test with missing required fields
        incomplete_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            # Missing severity and message
        }

        success, message = await notification_handler.send_notification(incomplete_data)
        assert success is False
        assert "Missing required fields" in message

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_send_notification_queue_full(self, notification_handler):
        """Test notification sending when queue is full."""
        with time_machine.travel("2024-01-01 12:00:00", tick=True) as traveller:
            # Set a very small queue size for testing
            handler = InMemoryNotificationHandler(
                max_queue_size=2,
                processing_delay_seconds=10.0,  # Use long delay to keep queue full
            )

            alert_data = {
                "id": "alert-001",
                "rule_name": "test_rule",
                "severity": "warning",
                "message": "Test message",
            }

            # Start filling the queue without waiting for completion
            tasks = []
            for i in range(2):
                alert_data_copy = alert_data.copy()
                alert_data_copy["id"] = f"alert-{i:03d}"
                task = asyncio.create_task(handler.send_notification(alert_data_copy))
                tasks.append(task)

            # Allow async operations to start but not complete
            await asyncio.sleep(0.01)  # Very short delay to let tasks start

            # Next notification should fail due to queue being full
            alert_data["id"] = "alert-overflow"
            success, message = await handler.send_notification(alert_data)
            assert success is False
            assert "Notification queue is full" in message

            # Cancel the pending tasks and cleanup
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await handler.close()

    @pytest.mark.asyncio
    async def test_send_notification_with_failure_simulation(self):
        """Test notification sending with failure simulation."""
        handler = InMemoryNotificationHandler(
            simulate_failure_rate=1.0  # Always fail
        )

        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "error",
            "message": "Test message",
        }

        success, message = await handler.send_notification(alert_data)
        assert success is False
        assert "Simulated failure for testing" in message

        # Check statistics
        stats = await handler.get_notification_statistics()
        assert stats["total_sent"] == 0
        assert stats["total_failed"] == 1

        await handler.close()

    @pytest.mark.asyncio
    async def test_send_notification_with_processing_delay(self):
        """Test notification sending with processing delay."""
        handler = InMemoryNotificationHandler(processing_delay_seconds=0.1)

        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "info",
            "message": "Test message",
        }

        start_time = time.time()
        success, message = await handler.send_notification(alert_data)
        end_time = time.time()

        assert success is True
        assert (end_time - start_time) >= 0.1  # Should have taken at least 0.1 seconds

        await handler.close()

    @pytest.mark.asyncio
    async def test_get_notification_history(self, notification_handler):
        """Test getting notification history."""
        # Send several notifications
        for i in range(5):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": f"rule_{i % 2}",
                "severity": "warning" if i % 2 == 0 else "error",
                "message": f"Test message {i}",
            }
            await notification_handler.send_notification(alert_data)

        # Get all history
        history = await notification_handler.get_notification_history()
        assert len(history) == 5

        # Check sorting (newest first)
        timestamps = [record["created_at"] for record in history]
        assert timestamps == sorted(timestamps, reverse=True)

        # Test filtering by status
        sent_history = await notification_handler.get_notification_history(status=NotificationStatus.SENT)
        assert len(sent_history) == 5
        assert all(record["status"] == "sent" for record in sent_history)

        # Test filtering by rule name
        rule_0_history = await notification_handler.get_notification_history(rule_name="rule_0")
        assert len(rule_0_history) == 3  # Indices 0, 2, 4

        # Test filtering by severity
        warning_history = await notification_handler.get_notification_history(severity="warning")
        assert len(warning_history) == 3  # Indices 0, 2, 4

        # Test limit
        limited_history = await notification_handler.get_notification_history(limit=3)
        assert len(limited_history) == 3

    @pytest.mark.asyncio
    async def test_get_notification_statistics(self, notification_handler):
        """Test getting comprehensive notification statistics."""
        # Send notifications with different severities and rules
        notifications = [
            {"rule_name": "rule_a", "severity": "error"},
            {"rule_name": "rule_a", "severity": "warning"},
            {"rule_name": "rule_b", "severity": "error"},
            {"rule_name": "rule_b", "severity": "info"},
        ]

        for i, notif in enumerate(notifications):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": notif["rule_name"],
                "severity": notif["severity"],
                "message": f"Test message {i}",
            }
            await notification_handler.send_notification(alert_data)

        stats = await notification_handler.get_notification_statistics()

        assert stats["total_sent"] == 4
        assert stats["total_failed"] == 0
        assert stats["notifications_by_severity"]["error"] == 2
        assert stats["notifications_by_severity"]["warning"] == 1
        assert stats["notifications_by_severity"]["info"] == 1
        assert stats["notifications_by_rule"]["rule_a"] == 2
        assert stats["notifications_by_rule"]["rule_b"] == 2
        assert stats["pending_queue_size"] == 0
        assert stats["history_size"] == 4
        assert stats["is_closed"] is False

    @pytest.mark.asyncio
    async def test_get_pending_notifications(self, notification_handler):
        """Test getting pending notifications."""
        # For this test, we need to simulate pending notifications
        # We'll use a handler with processing delay and check during processing
        handler = InMemoryNotificationHandler(
            processing_delay_seconds=0.5  # Delay for processing
        )

        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Test message",
        }

        # Start notification sending but don't wait for completion
        task = asyncio.create_task(handler.send_notification(alert_data))

        # Allow the notification to be queued but not processed yet
        await asyncio.sleep(0.01)  # Small delay to allow queueing

        # Check pending notifications
        pending = await handler.get_pending_notifications()
        assert len(pending) == 1
        assert pending[0]["alert_data"]["id"] == "alert-001"

        # Wait for task to complete and cleanup
        await task

        # After completion, pending should be empty
        pending = await handler.get_pending_notifications()
        assert len(pending) == 0

        await handler.close()

    @pytest.mark.asyncio
    async def test_clear_history(self, notification_handler):
        """Test clearing notification history."""
        # Send some notifications
        for i in range(3):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": "test_rule",
                "severity": "info",
                "message": f"Test message {i}",
            }
            await notification_handler.send_notification(alert_data)

        # Verify history exists
        history = await notification_handler.get_notification_history()
        assert len(history) == 3

        # Clear history
        cleared_count = await notification_handler.clear_history()
        assert cleared_count == 3

        # Verify history is empty
        history = await notification_handler.get_notification_history()
        assert len(history) == 0

        stats = await notification_handler.get_notification_statistics()
        assert stats["history_size"] == 0

    @pytest.mark.asyncio
    async def test_simulate_processing_failure(self, notification_handler):
        """Test simulating processing failure for specific notifications."""
        # Send a notification that will be processed normally
        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Test message",
        }
        await notification_handler.send_notification(alert_data)

        # Get the notification ID from history
        history = await notification_handler.get_notification_history()
        notification_id = history[0]["id"]

        # Since the notification was already processed, simulate_processing_failure should return False
        result = await notification_handler.simulate_processing_failure(notification_id)
        assert result is False  # Already processed

        # For testing pending notifications, we need a different approach
        # This would require a more complex setup with delayed processing

    @pytest.mark.asyncio
    async def test_concurrent_notifications(self, notification_handler):
        """Test handling concurrent notifications."""

        async def send_notification(i):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": f"rule_{i % 3}",
                "severity": "warning" if i % 2 == 0 else "error",
                "message": f"Test message {i}",
            }
            return await notification_handler.send_notification(alert_data)

        # Send 10 notifications concurrently
        tasks = [send_notification(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(success for success, _ in results)

        # Check statistics
        stats = await notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 10
        assert stats["total_failed"] == 0
        assert stats["history_size"] == 10

    @time_machine.travel("2024-01-01 12:00:00", tick=False)
    @pytest.mark.asyncio
    async def test_cleanup_old_notifications(self):
        """Test cleanup of old notifications."""
        handler = InMemoryNotificationHandler(
            history_retention_hours=1.0,  # 1 hour retention
            cleanup_interval_seconds=60.0,
        )

        # Send notification at initial time
        alert_data = {
            "id": "alert-old",
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Old notification",
        }
        await handler.send_notification(alert_data)

        # Move forward 2 hours (beyond retention period)
        with time_machine.travel("2024-01-01 14:00:00", tick=False):
            # Send another notification
            alert_data["id"] = "alert-new"
            alert_data["message"] = "New notification"
            await handler.send_notification(alert_data)

            # Manually trigger cleanup
            await handler._cleanup_old_notifications()

            # Check history - only new notification should remain
            history = await handler.get_notification_history()
            assert len(history) == 1
            assert history[0]["alert_data"]["id"] == "alert-new"

        await handler.close()

    @pytest.mark.asyncio
    async def test_cleanup_history_size_limit(self):
        """Test cleanup when history exceeds size limit."""
        handler = InMemoryNotificationHandler(
            max_history_size=3,  # Very small limit
            history_retention_hours=24.0,  # Long retention
        )

        # Send 5 notifications
        for i in range(5):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": "test_rule",
                "severity": "info",
                "message": f"Test message {i}",
            }
            await handler.send_notification(alert_data)

        # Manually trigger cleanup
        await handler._cleanup_old_notifications()

        # Should only have 3 most recent notifications
        history = await handler.get_notification_history()
        assert len(history) == 3

        # Should be the most recent ones (2, 3, 4)
        ids = [record["alert_data"]["id"] for record in history]
        assert "alert-002" in ids
        assert "alert-003" in ids
        assert "alert-004" in ids

        await handler.close()

    @pytest.mark.asyncio
    async def test_handler_close_cleanup(self, notification_handler):
        """Test proper cleanup when closing handler."""
        # Send some notifications
        for i in range(3):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": "test_rule",
                "severity": "info",
                "message": f"Test message {i}",
            }
            await notification_handler.send_notification(alert_data)

        # Verify data exists
        stats = await notification_handler.get_notification_statistics()
        assert stats["history_size"] == 3

        # Close handler
        await notification_handler.close()

        # Verify handler is closed
        assert notification_handler.is_closed is True

        # Verify cleanup completed
        assert len(notification_handler._notification_history) == 0
        assert len(notification_handler._pending_queue) == 0

    @pytest.mark.asyncio
    async def test_handler_operations_after_close(self, notification_handler):
        """Test that operations fail after handler is closed."""
        # Close the handler
        await notification_handler.close()

        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Test message",
        }

        # Operations should fail after close
        with pytest.raises(RuntimeError, match="Notification handler is closed"):
            await notification_handler.send_notification(alert_data)

        with pytest.raises(RuntimeError, match="Notification handler is closed"):
            await notification_handler.get_notification_history()

        with pytest.raises(RuntimeError, match="Notification handler is closed"):
            await notification_handler.clear_history()

    @pytest.mark.asyncio
    async def test_error_handling_in_cleanup_loop(self):
        """Test error handling in cleanup loop."""
        handler = InMemoryNotificationHandler(
            cleanup_interval_seconds=0.01  # Very short for testing
        )

        # Mock the cleanup method to raise an exception
        original_cleanup = handler._cleanup_old_notifications

        call_count = 0

        async def mock_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated cleanup error")
            return await original_cleanup()

        handler._cleanup_old_notifications = mock_cleanup

        # Allow cleanup operations to complete
        # The cleanup loop starts with sleep, so we need to wait for that
        await asyncio.sleep(0.05)  # Wait longer than cleanup_interval_seconds

        # Should have attempted cleanup despite error
        assert call_count >= 2

        await handler.close()


@pytest.mark.asyncio
async def test_notification_handler_integration():
    """Integration test simulating real-world notification handling."""
    handler = InMemoryNotificationHandler(
        max_queue_size=100,
        max_history_size=1000,
        simulate_failure_rate=0.1,  # 10% failure rate
    )

    # Simulate various alert types
    alert_types = [
        {"rule_name": "high_cpu", "severity": "warning"},
        {"rule_name": "high_memory", "severity": "error"},
        {"rule_name": "disk_full", "severity": "critical"},
        {"rule_name": "service_down", "severity": "critical"},
    ]

    sent_count = 0
    failed_count = 0

    # Send many notifications
    for i in range(50):
        alert_type = alert_types[i % len(alert_types)]
        alert_data = {
            "id": f"alert-{i:03d}",
            "rule_name": alert_type["rule_name"],
            "severity": alert_type["severity"],
            "message": f"Alert {i}: {alert_type['rule_name']}",
            "labels": {"service": f"service_{i % 5}"},
            "fired_at": f"2024-01-01T12:{i:02d}:00Z",
        }

        success, _ = await handler.send_notification(alert_data)
        if success:
            sent_count += 1
        else:
            failed_count += 1

    # Verify statistics
    stats = await handler.get_notification_statistics()
    assert stats["total_sent"] == sent_count
    assert stats["total_failed"] == failed_count
    assert stats["total_sent"] + stats["total_failed"] == 50

    # Verify we have notifications for all severities
    assert "warning" in stats["notifications_by_severity"]
    assert "error" in stats["notifications_by_severity"]
    assert "critical" in stats["notifications_by_severity"]

    # Verify we have notifications for all rules
    for alert_type in alert_types:
        assert alert_type["rule_name"] in stats["notifications_by_rule"]

    # Test history filtering
    critical_history = await handler.get_notification_history(severity="critical")
    assert len(critical_history) > 0
    assert all(record["alert_data"]["severity"] == "critical" for record in critical_history)

    await handler.close()
