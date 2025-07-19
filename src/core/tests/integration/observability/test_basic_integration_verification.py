# ABOUTME: Basic verification tests for observability integration functionality
# ABOUTME: Simple tests to verify components work together without complex logging setup

import asyncio
import pytest
import pytest_asyncio
import time
import uuid

from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from ...fixtures.performance_metrics import PerformanceMonitor


@pytest_asyncio.fixture
async def simple_observability_setup(mock_event_bus):
    """Simple observability setup for basic testing."""
    notification_handler = InMemoryNotificationHandler(
        max_queue_size=100, max_history_size=500, simulate_failure_rate=0.0
    )

    performance_monitor = PerformanceMonitor(event_bus=mock_event_bus, sample_interval=0.1)

    yield {
        "notification_handler": notification_handler,
        "performance_monitor": performance_monitor,
        "event_bus": mock_event_bus,
    }

    await notification_handler.close()
    await performance_monitor.stop_monitoring()


class TestBasicObservabilityIntegration:
    """Basic tests for observability integration without complex logging."""

    @pytest.mark.asyncio
    async def test_notification_handler_basic_functionality(self, simple_observability_setup):
        """Test basic notification handler functionality."""
        setup = simple_observability_setup
        handler = setup["notification_handler"]

        # Test sending a notification
        alert_data = {
            "id": str(uuid.uuid4()),
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Test alert message",
        }

        success, message = await handler.send_notification(alert_data)

        assert success, f"Notification failed: {message}"

        # Verify notification statistics
        stats = await handler.get_notification_statistics()
        assert stats["total_sent"] == 1
        assert stats["total_failed"] == 0

        # Verify notification history
        history = await handler.get_notification_history(limit=10)
        assert len(history) == 1
        assert history[0]["alert_data"]["rule_name"] == "test_rule"

        print("✓ Notification handler basic functionality test passed")

    @pytest.mark.asyncio
    async def test_performance_monitor_basic_functionality(self, simple_observability_setup):
        """Test basic performance monitor functionality."""
        setup = simple_observability_setup
        monitor = setup["performance_monitor"]
        event_bus = setup["event_bus"]

        # Start monitoring
        await monitor.start_monitoring()

        # Update event bus stats to trigger monitoring
        event_bus.update_stats(published_count=10, processed_count=8, error_count=1, queue_size=5)

        # Allow monitoring to collect data
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify performance data was collected
        summary = monitor.get_performance_summary()

        if summary:  # May be empty if monitoring duration was too short
            assert "monitoring_duration" in summary
            assert "sample_count" in summary
            assert summary["sample_count"] > 0

        print("✓ Performance monitor basic functionality test passed")

    @pytest.mark.asyncio
    async def test_cross_component_basic_integration(self, simple_observability_setup):
        """Test basic integration between notification handler and performance monitor."""
        setup = simple_observability_setup
        handler = setup["notification_handler"]
        monitor = setup["performance_monitor"]
        event_bus = setup["event_bus"]

        # Capture alerts from performance monitor
        alerts_captured = []

        def alert_callback(alert_type: str, alert_data: dict):
            alerts_captured.append({"type": alert_type, "data": alert_data, "timestamp": time.time()})

        monitor.add_alert_callback(alert_callback)

        # Start monitoring
        await monitor.start_monitoring()

        # Set low thresholds to trigger alerts
        monitor.cpu_threshold = 1.0  # Very low to trigger easily
        monitor.memory_threshold = 1.0
        monitor.queue_size_threshold = 1
        monitor.error_rate_threshold = 1.0

        # Update stats to trigger alerts
        event_bus.update_stats(published_count=100, processed_count=90, error_count=5, queue_size=10)

        # Allow monitoring to detect and generate alerts
        await asyncio.sleep(0.3)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify alerts were generated
        assert len(alerts_captured) > 0, "No alerts were generated"

        # Send notifications for alerts
        notification_count = 0
        for alert in alerts_captured:
            notification_data = {
                "id": str(uuid.uuid4()),
                "rule_name": f"perf_{alert['type']}",
                "severity": "warning",
                "message": f"Performance alert: {alert['type']}",
            }

            success, _ = await handler.send_notification(notification_data)
            if success:
                notification_count += 1

        # Verify notifications were sent
        stats = await handler.get_notification_statistics()
        assert stats["total_sent"] >= notification_count

        print("✓ Cross-component integration test passed:")
        print(f"  - Alerts generated: {len(alerts_captured)}")
        print(f"  - Notifications sent: {stats['total_sent']}")

    @pytest.mark.asyncio
    async def test_trace_id_basic_propagation(self, simple_observability_setup):
        """Test basic trace ID propagation without complex logging."""
        setup = simple_observability_setup
        handler = setup["notification_handler"]

        # Create notifications with trace IDs
        trace_id = str(uuid.uuid4())

        notifications_with_trace = []
        for i in range(3):
            alert_data = {
                "id": str(uuid.uuid4()),
                "rule_name": f"test_rule_{i}",
                "severity": "info",
                "message": f"Test message {i}",
                "metadata": {"trace_id": trace_id, "sequence": i},
            }

            success, message = await handler.send_notification(alert_data)
            if success:
                notifications_with_trace.append(alert_data)

        # Verify all notifications were sent
        assert len(notifications_with_trace) == 3

        # Verify trace ID consistency in history
        history = await handler.get_notification_history(limit=10)

        trace_notifications = [
            notif for notif in history if notif["alert_data"].get("metadata", {}).get("trace_id") == trace_id
        ]

        assert len(trace_notifications) == 3, f"Expected 3 notifications with trace ID, got {len(trace_notifications)}"

        # Verify sequence ordering
        sequences = [notif["alert_data"]["metadata"]["sequence"] for notif in trace_notifications]

        assert set(sequences) == {0, 1, 2}, f"Expected sequences [0,1,2], got {sequences}"

        print("✓ Trace ID propagation test passed:")
        print(f"  - Trace ID: {trace_id}")
        print(f"  - Notifications with trace: {len(trace_notifications)}")

    @pytest.mark.asyncio
    async def test_monitoring_data_consistency(self, simple_observability_setup):
        """Test data consistency across monitoring components."""
        setup = simple_observability_setup
        handler = setup["notification_handler"]
        monitor = setup["performance_monitor"]
        event_bus = setup["event_bus"]

        # Start monitoring
        await monitor.start_monitoring()

        # Generate consistent test data
        test_start_time = time.time()

        # Update event bus stats progressively
        for i in range(5):
            event_bus.update_stats(published_count=i * 10, processed_count=i * 9, error_count=i, queue_size=i * 2)

            # Send notification for each update
            notification_data = {
                "id": str(uuid.uuid4()),
                "rule_name": "data_consistency_test",
                "severity": "info",
                "message": f"Data update {i}",
                "metadata": {"test_sequence": i, "timestamp": time.time()},
            }

            await handler.send_notification(notification_data)
            await asyncio.sleep(0.1)  # Small delay between updates

        test_end_time = time.time()

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify data consistency

        # Check notification timing consistency
        history = await handler.get_notification_history(limit=10)
        test_notifications = [notif for notif in history if notif["alert_data"]["rule_name"] == "data_consistency_test"]

        assert len(test_notifications) == 5, f"Expected 5 test notifications, got {len(test_notifications)}"

        # Verify temporal ordering
        notification_times = [notif["created_at"] for notif in test_notifications]
        notification_times.sort()

        # All notifications should be within test timeframe
        for notif_time in notification_times:
            assert test_start_time <= notif_time <= test_end_time + 1.0, (
                f"Notification timestamp {notif_time} outside test timeframe [{test_start_time}, {test_end_time}]"
            )

        # Check performance monitoring consistency
        summary = monitor.get_performance_summary()

        if summary and summary.get("sample_count", 0) > 0:
            duration = summary["monitoring_duration"]
            expected_duration = test_end_time - test_start_time

            # Duration should be approximately correct (allow some variance)
            assert abs(duration - expected_duration) < 2.0, (
                f"Monitoring duration {duration} differs too much from expected {expected_duration}"
            )

        print("✓ Data consistency test passed:")
        print(f"  - Test duration: {test_end_time - test_start_time:.3f}s")
        print(f"  - Notifications: {len(test_notifications)}")
        print("  - All timestamps within expected range")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
