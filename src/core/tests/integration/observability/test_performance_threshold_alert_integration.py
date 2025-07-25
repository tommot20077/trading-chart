# ABOUTME: Integration tests for performance threshold alerting system including various alert types and notification workflows
# ABOUTME: Validates threshold detection, alert generation, notification delivery, and alert system integration

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, Any
from unittest.mock import patch
import time
from dataclasses import dataclass

# Previously skipped due to timeout issues - now fixed

from ...fixtures.performance_metrics import PerformanceMonitor
from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from .conftest import wait_for_alerts, wait_for_notification_count


@dataclass
class AlertTestScenario:
    """Test scenario for alert generation."""

    name: str
    alert_type: str
    trigger_condition: Dict[str, Any]
    expected_threshold_breach: bool
    description: str


class TestPerformanceThresholdAlerts:
    """Integration tests for performance threshold alerting system."""

    @pytest_asyncio.fixture
    async def alert_monitor(self, mock_event_bus):
        """Create PerformanceMonitor configured for alert testing."""
        monitor = PerformanceMonitor(
            event_bus=mock_event_bus,
            sample_interval=0.05,  # Fast sampling for tests
        )

        # Configure test thresholds to be more sensitive for testing
        monitor.cpu_threshold = 70.0
        monitor.memory_threshold = 75.0
        monitor.queue_size_threshold = 50
        monitor.error_rate_threshold = 5.0

        yield monitor

        # Cleanup
        await monitor.stop_monitoring()
        monitor.clear_metrics()

    @pytest_asyncio.fixture
    async def notification_handler_for_alerts(self):
        """Create notification handler for alert integration testing."""
        handler = InMemoryNotificationHandler(
            max_queue_size=500,
            max_history_size=1000,
            history_retention_hours=1.0,
            cleanup_interval_seconds=30.0,
            max_retry_attempts=3,
            simulate_failure_rate=0.0,
            processing_delay_seconds=0.0,
        )

        yield handler

        # Cleanup
        await handler.close()

    @pytest.fixture
    def alert_scenarios(self):
        """Provide various alert test scenarios."""
        return [
            AlertTestScenario(
                name="high_cpu_alert",
                alert_type="high_cpu",
                trigger_condition={"cpu_percent": 85.0},
                expected_threshold_breach=True,
                description="CPU usage exceeds 70% threshold",
            ),
            AlertTestScenario(
                name="high_memory_alert",
                alert_type="high_memory",
                trigger_condition={"memory_percent": 90.0},
                expected_threshold_breach=True,
                description="Memory usage exceeds 75% threshold",
            ),
            AlertTestScenario(
                name="high_queue_size_alert",
                alert_type="high_queue_size",
                trigger_condition={"queue_size": 100},
                expected_threshold_breach=True,
                description="Queue size exceeds 50 events threshold",
            ),
            AlertTestScenario(
                name="high_error_rate_alert",
                alert_type="high_error_rate",
                trigger_condition={"processed_count": 100, "error_count": 10},
                expected_threshold_breach=True,
                description="Error rate exceeds 5% threshold",
            ),
            AlertTestScenario(
                name="normal_operation",
                alert_type="none",
                trigger_condition={"cpu_percent": 30.0, "memory_percent": 40.0, "queue_size": 10},
                expected_threshold_breach=False,
                description="Normal operation within all thresholds",
            ),
        ]

    @pytest.mark.asyncio
    async def test_cpu_threshold_alert_generation(self, alert_monitor, alert_collector):
        """Test CPU threshold alert generation and data accuracy."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Mock high CPU usage
        with patch.object(monitor.process, "cpu_percent", return_value=85.0):
            await monitor.start_monitoring()

            # Wait for threshold detection
            await wait_for_alerts(alert_collector, 1, timeout=2.0)

            await monitor.stop_monitoring()

        # Verify CPU alert was generated
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        assert len(cpu_alerts) > 0

        # Verify alert data
        alert = cpu_alerts[0]
        assert alert["type"] == "high_cpu"
        assert alert["data"]["current"] == 85.0
        assert alert["data"]["threshold"] == 70.0
        assert alert["data"]["current"] > alert["data"]["threshold"]

    @pytest.mark.asyncio
    async def test_memory_threshold_alert_generation(self, alert_monitor, alert_collector):
        """Test memory threshold alert generation and data accuracy."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Mock high memory usage
        with patch.object(monitor.process, "memory_percent", return_value=90.0):
            await monitor.start_monitoring()

            # Wait for threshold detection
            await wait_for_alerts(alert_collector, 1, timeout=2.0)

            await monitor.stop_monitoring()

        # Verify memory alert was generated
        memory_alerts = alert_collector.get_by_type("high_memory")
        assert len(memory_alerts) > 0

        # Verify alert data
        alert = memory_alerts[0]
        assert alert["type"] == "high_memory"
        assert alert["data"]["current"] == 90.0
        assert alert["data"]["threshold"] == 75.0
        assert alert["data"]["current"] > alert["data"]["threshold"]

    @pytest.mark.asyncio
    async def test_queue_size_threshold_alert(self, alert_monitor, alert_collector, mock_event_bus):
        """Test queue size threshold alert generation."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Set high queue size
        mock_event_bus.update_stats(queue_size=100)

        await monitor.start_monitoring()

        # Wait for threshold detection
        await wait_for_alerts(alert_collector, 1, timeout=2.0)

        await monitor.stop_monitoring()

        # Verify queue size alert was generated
        queue_alerts = alert_collector.get_by_type("high_queue_size")
        assert len(queue_alerts) > 0

        # Verify alert data
        alert = queue_alerts[0]
        assert alert["type"] == "high_queue_size"
        assert alert["data"]["current"] == 100
        assert alert["data"]["threshold"] == 50
        assert alert["data"]["current"] > alert["data"]["threshold"]

    @pytest.mark.asyncio
    async def test_error_rate_threshold_alert(self, alert_monitor, alert_collector, mock_event_bus):
        """Test error rate threshold alert generation."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Set high error rate (10% error rate)
        mock_event_bus.update_stats(processed_count=100, error_count=10)

        await monitor.start_monitoring()

        # Wait for threshold detection
        await wait_for_alerts(alert_collector, 1, timeout=2.0)

        await monitor.stop_monitoring()

        # Verify error rate alert was generated
        error_alerts = alert_collector.get_by_type("high_error_rate")
        assert len(error_alerts) > 0

        # Verify alert data
        alert = error_alerts[0]
        assert alert["type"] == "high_error_rate"
        assert alert["data"]["current"] == 10.0  # 10% error rate
        assert alert["data"]["threshold"] == 5.0
        assert alert["data"]["current"] > alert["data"]["threshold"]

    @pytest.mark.asyncio
    async def test_multiple_threshold_breaches(self, alert_monitor, alert_collector, mock_event_bus):
        """Test multiple simultaneous threshold breaches."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Configure multiple threshold breaches
        mock_event_bus.update_stats(
            queue_size=75,  # Above threshold (50)
            processed_count=100,
            error_count=8,  # 8% error rate, above threshold (5%)
        )

        # Mock high CPU and memory
        with (
            patch.object(monitor.process, "cpu_percent", return_value=80.0),
            patch.object(monitor.process, "memory_percent", return_value=85.0),
        ):
            await monitor.start_monitoring()

            # Wait for multiple alerts
            await wait_for_alerts(alert_collector, 4, timeout=3.0)

            await monitor.stop_monitoring()

        # Verify all alert types were generated
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        memory_alerts = alert_collector.get_by_type("high_memory")
        queue_alerts = alert_collector.get_by_type("high_queue_size")
        error_alerts = alert_collector.get_by_type("high_error_rate")

        assert len(cpu_alerts) > 0
        assert len(memory_alerts) > 0
        assert len(queue_alerts) > 0
        assert len(error_alerts) > 0

        # Verify total alert count
        total_alerts = len(cpu_alerts) + len(memory_alerts) + len(queue_alerts) + len(error_alerts)
        assert total_alerts >= 4

    @pytest.mark.asyncio
    async def test_alert_notification_integration(self, alert_monitor, notification_handler_for_alerts, mock_event_bus):
        """Test integration between alerts and notification system."""
        monitor = alert_monitor
        handler = notification_handler_for_alerts
        notifications_sent = []

        # Create alert-to-notification bridge with tracking
        async def alert_to_notification(alert_type: str, alert_data: Dict[str, Any]):
            notification = {
                "id": f"alert_{alert_type}_{int(time.time())}",
                "rule_name": f"threshold_{alert_type}",
                "severity": "critical" if alert_type in ["high_cpu", "high_memory"] else "warning",
                "message": f"{alert_type.replace('_', ' ').title()}: {alert_data['current']} > {alert_data['threshold']}",
                "metadata": alert_data,
            }
            success, message = await handler.send_notification(notification)
            notifications_sent.append((alert_type, success, message))

        # Connect alert system to notification handler
        monitor.add_alert_callback(alert_to_notification)

        # Trigger multiple alert conditions to ensure we get alerts
        mock_event_bus.update_stats(queue_size=80)  # Above threshold (50)

        with patch.object(monitor.process, "cpu_percent", return_value=85.0):  # Above threshold (70)
            await monitor.start_monitoring()

            # Wait longer for alerts and notifications to be generated
            await asyncio.sleep(1.0)

            await monitor.stop_monitoring()

        # Wait for notification processing
        await wait_for_notification_count(handler, 2, timeout=5.0)

        # Verify notifications were attempted
        assert len(notifications_sent) >= 2, f"Expected at least 2 notifications sent, got {len(notifications_sent)}"

        # Verify notifications were sent successfully
        successful_notifications = [n for n in notifications_sent if n[1] is True]
        assert len(successful_notifications) >= 2, f"Expected at least 2 successful notifications, got {len(successful_notifications)}"

        # Verify notification statistics
        stats = await handler.get_notification_statistics()
        assert stats["total_sent"] >= 2, f"Expected at least 2 total_sent, got {stats['total_sent']}"

        # Verify notification history
        history = await handler.get_notification_history()
        assert len(history) >= 2, f"Expected at least 2 notifications in history, got {len(history)}"

        # Verify notification content
        for notification_record in history:
            alert_data = notification_record["alert_data"]
            assert "rule_name" in alert_data
            assert "severity" in alert_data
            assert "message" in alert_data
            assert "metadata" in alert_data

    @pytest.mark.asyncio
    async def test_alert_callback_error_handling(self, alert_monitor, mock_event_bus):
        """Test error handling in alert callback functions."""
        monitor = alert_monitor

        # Create callback that raises exceptions
        def failing_callback(alert_type: str, alert_data: Dict[str, Any]):
            raise Exception(f"Callback error for {alert_type}")

        # Create successful callback for comparison
        successful_calls = []

        def successful_callback(alert_type: str, alert_data: Dict[str, Any]):
            successful_calls.append((alert_type, alert_data))

        # Add both callbacks
        monitor.add_alert_callback(failing_callback)
        monitor.add_alert_callback(successful_callback)

        # Trigger alert
        mock_event_bus.update_stats(queue_size=100)

        await monitor.start_monitoring()
        await asyncio.sleep(0.3)
        await monitor.stop_monitoring()

        # Verify successful callback still worked despite failing callback
        assert len(successful_calls) > 0
        alert_type, alert_data = successful_calls[0]
        assert alert_type == "high_queue_size"
        assert alert_data["current"] == 100

    @pytest.mark.asyncio
    async def test_threshold_configuration_and_updates(self, alert_monitor, alert_collector, mock_event_bus):
        """Test dynamic threshold configuration and updates."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Set initial conservative thresholds
        monitor.cpu_threshold = 90.0
        monitor.memory_threshold = 90.0
        monitor.queue_size_threshold = 200

        # Test condition that shouldn't trigger alerts initially
        mock_event_bus.update_stats(queue_size=100)

        with patch.object(monitor.process, "cpu_percent", return_value=85.0):
            await monitor.start_monitoring()
            await asyncio.sleep(0.2)

            # Should have no alerts with high thresholds
            initial_count = alert_collector.count()

            # Update thresholds to be more sensitive
            monitor.cpu_threshold = 80.0
            monitor.queue_size_threshold = 50

            # Wait for new threshold checks
            await asyncio.sleep(0.3)

            await monitor.stop_monitoring()

        # Should now have alerts with updated thresholds
        final_count = alert_collector.count()
        assert final_count > initial_count

        # Verify specific alerts were generated
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        queue_alerts = alert_collector.get_by_type("high_queue_size")

        assert len(cpu_alerts) > 0
        assert len(queue_alerts) > 0

    @pytest.mark.asyncio
    async def test_alert_frequency_and_deduplication(self, alert_monitor, alert_collector, mock_event_bus):
        """Test alert frequency behavior and potential deduplication."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Set conditions that continuously breach thresholds
        mock_event_bus.update_stats(queue_size=100)

        with patch.object(monitor.process, "cpu_percent", return_value=85.0):
            await monitor.start_monitoring()

            # Run for longer period to see alert frequency
            await asyncio.sleep(1.0)

            await monitor.stop_monitoring()

        # Analyze alert frequency
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        queue_alerts = alert_collector.get_by_type("high_queue_size")

        # Should have multiple alerts due to continuous monitoring
        assert len(cpu_alerts) > 1
        assert len(queue_alerts) > 1

        # Verify alerts have different timestamps
        cpu_timestamps = [alert["timestamp"] for alert in cpu_alerts]
        assert len(set(cpu_timestamps)) > 1  # Should have different timestamps

    @pytest.mark.asyncio
    async def test_alert_data_accuracy_and_consistency(self, alert_monitor, alert_collector, mock_event_bus):
        """Test accuracy and consistency of alert data."""
        monitor = alert_monitor
        monitor.add_alert_callback(alert_collector)

        # Set precise test conditions
        test_cpu = 82.5
        test_memory = 78.3
        test_queue_size = 75
        test_error_rate = 7.5  # 7.5% error rate

        mock_event_bus.update_stats(
            queue_size=test_queue_size,
            processed_count=200,
            error_count=15,  # 15/200 = 7.5%
        )

        with (
            patch.object(monitor.process, "cpu_percent", return_value=test_cpu),
            patch.object(monitor.process, "memory_percent", return_value=test_memory),
        ):
            await monitor.start_monitoring()
            await asyncio.sleep(0.3)
            await monitor.stop_monitoring()

        # Verify alert data accuracy
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        memory_alerts = alert_collector.get_by_type("high_memory")
        queue_alerts = alert_collector.get_by_type("high_queue_size")
        error_alerts = alert_collector.get_by_type("high_error_rate")

        # Check CPU alert accuracy
        if len(cpu_alerts) > 0:
            cpu_alert = cpu_alerts[0]
            assert cpu_alert["data"]["current"] == test_cpu
            assert cpu_alert["data"]["threshold"] == monitor.cpu_threshold

        # Check memory alert accuracy
        if len(memory_alerts) > 0:
            memory_alert = memory_alerts[0]
            assert memory_alert["data"]["current"] == test_memory
            assert memory_alert["data"]["threshold"] == monitor.memory_threshold

        # Check queue alert accuracy
        if len(queue_alerts) > 0:
            queue_alert = queue_alerts[0]
            assert queue_alert["data"]["current"] == test_queue_size
            assert queue_alert["data"]["threshold"] == monitor.queue_size_threshold

        # Check error rate alert accuracy
        if len(error_alerts) > 0:
            error_alert = error_alerts[0]
            assert error_alert["data"]["current"] == test_error_rate
            assert error_alert["data"]["threshold"] == monitor.error_rate_threshold

    @pytest.mark.asyncio
    async def test_alert_system_performance_impact(self, alert_monitor, mock_event_bus):
        """Test performance impact of alert system on monitoring."""
        monitor = alert_monitor

        # Add multiple alert callbacks
        callback_calls = []

        def counting_callback(alert_type: str, alert_data: Dict[str, Any]):
            callback_calls.append((alert_type, alert_data, time.time()))

        # Add multiple callbacks to test performance
        for i in range(10):
            monitor.add_alert_callback(counting_callback)

        # Trigger multiple alert conditions
        mock_event_bus.update_stats(queue_size=100, processed_count=100, error_count=10)

        with (
            patch.object(monitor.process, "cpu_percent", return_value=85.0),
            patch.object(monitor.process, "memory_percent", return_value=85.0),
        ):
            start_time = time.time()

            await monitor.start_monitoring()
            await asyncio.sleep(0.5)
            await monitor.stop_monitoring()

            end_time = time.time()

        # Verify monitoring completed in reasonable time
        monitoring_duration = end_time - start_time
        assert monitoring_duration < 2.0  # Should complete quickly

        # Verify callbacks were called
        assert len(callback_calls) > 0

        # Verify multiple callbacks were triggered per alert
        # (10 callbacks × multiple alert types × multiple monitoring cycles)
        assert len(callback_calls) >= 40  # Conservative estimate


@pytest.mark.integration
class TestThresholdAlertWorkflow:
    """Comprehensive integration tests for threshold alert workflow."""

    @pytest_asyncio.fixture
    async def alert_monitor(self, mock_event_bus):
        """Create PerformanceMonitor configured for alert testing."""
        monitor = PerformanceMonitor(
            event_bus=mock_event_bus,
            sample_interval=0.05,  # Fast sampling for tests
        )

        # Configure test thresholds to be more sensitive for testing
        monitor.cpu_threshold = 70.0
        monitor.memory_threshold = 75.0
        monitor.queue_size_threshold = 50
        monitor.error_rate_threshold = 5.0

        yield monitor

        # Cleanup
        await monitor.stop_monitoring()
        monitor.clear_metrics()

    @pytest_asyncio.fixture
    async def notification_handler_for_alerts(self):
        """Create notification handler for alert integration testing."""
        handler = InMemoryNotificationHandler(
            max_queue_size=500,
            max_history_size=1000,
            history_retention_hours=1.0,
            cleanup_interval_seconds=30.0,
            max_retry_attempts=3,
            simulate_failure_rate=0.0,
            processing_delay_seconds=0.0,
        )

        yield handler

        # Cleanup
        await handler.close()

    @pytest.mark.asyncio
    async def test_complete_alert_workflow(
        self, alert_monitor, notification_handler_for_alerts, alert_collector, mock_event_bus
    ):
        """Test complete alert workflow from threshold breach to notification delivery."""
        monitor = alert_monitor
        handler = notification_handler_for_alerts
        notifications_sent = []

        # Connect components
        monitor.add_alert_callback(alert_collector)

        async def alert_to_notification_bridge(alert_type: str, alert_data: Dict[str, Any]):
            notification = {
                "id": f"workflow_test_{alert_type}_{int(time.time() * 1000)}",
                "rule_name": f"threshold_breach_{alert_type}",
                "severity": "critical",
                "message": f"Alert: {alert_type} - {alert_data['current']} exceeds {alert_data['threshold']}",
                "metadata": {
                    "alert_type": alert_type,
                    "current_value": alert_data["current"],
                    "threshold_value": alert_data["threshold"],
                    "breach_percentage": ((alert_data["current"] - alert_data["threshold"]) / alert_data["threshold"])
                    * 100,
                },
            }
            success, message = await handler.send_notification(notification)
            notifications_sent.append((alert_type, success, message))

        monitor.add_alert_callback(alert_to_notification_bridge)

        # Trigger comprehensive alert scenario
        mock_event_bus.update_stats(
            queue_size=100,  # Above threshold (50)
            processed_count=200,
            error_count=20,  # 10% error rate, above threshold (5%)
        )

        with (
            patch.object(monitor.process, "cpu_percent", return_value=85.0),  # Above threshold (70)
            patch.object(monitor.process, "memory_percent", return_value=85.0),  # Above threshold (75)
        ):
            # Start monitoring
            await monitor.start_monitoring()

            # Wait for alert generation and notification processing
            await asyncio.sleep(1.5)

            # Stop monitoring
            await monitor.stop_monitoring()

        # Wait for all notifications to be processed
        await wait_for_notification_count(handler, 4, timeout=10.0)

        # Verify alert collection
        actual_alert_count = alert_collector.count()
        assert actual_alert_count >= 4, f"Expected at least 4 alerts, got {actual_alert_count}"

        # Verify notification attempts
        successful_notifications = [n for n in notifications_sent if n[1] is True]
        assert len(successful_notifications) >= 4, f"Expected at least 4 successful notifications, got {len(successful_notifications)}"

        # Verify notification delivery
        stats = await handler.get_notification_statistics()
        assert stats["total_sent"] >= 4, f"Expected at least 4 notifications sent, got {stats['total_sent']}"
        assert stats["total_failed"] == 0, f"Expected 0 failed notifications, got {stats['total_failed']}"

        # Verify notification content
        history = await handler.get_notification_history()
        assert len(history) >= 4, f"Expected at least 4 notifications in history, got {len(history)}"

        # Verify each notification has proper structure
        for notification_record in history:
            alert_data = notification_record["alert_data"]
            assert "rule_name" in alert_data
            assert alert_data["rule_name"].startswith("threshold_breach_")
            assert "severity" in alert_data
            assert alert_data["severity"] == "critical"
            assert "metadata" in alert_data
            assert "alert_type" in alert_data["metadata"]
            assert "current_value" in alert_data["metadata"]
            assert "threshold_value" in alert_data["metadata"]
            assert "breach_percentage" in alert_data["metadata"]

    @pytest.mark.asyncio
    async def test_alert_system_resilience(self, alert_monitor, alert_collector, mock_event_bus):
        """Test alert system resilience under various failure conditions."""
        monitor = alert_monitor

        # Add mix of successful and failing callbacks
        successful_calls = []

        def successful_callback(alert_type: str, alert_data: Dict[str, Any]):
            successful_calls.append((alert_type, alert_data))

        def failing_callback(alert_type: str, alert_data: Dict[str, Any]):
            raise RuntimeError("Simulated callback failure")

        def slow_callback(alert_type: str, alert_data: Dict[str, Any]):
            time.sleep(0.1)  # Simulate slow processing

        # Add callbacks
        monitor.add_alert_callback(successful_callback)
        monitor.add_alert_callback(failing_callback)
        monitor.add_alert_callback(slow_callback)
        monitor.add_alert_callback(alert_collector)

        # Trigger alerts
        mock_event_bus.update_stats(queue_size=100)

        with patch.object(monitor.process, "cpu_percent", return_value=85.0):
            await monitor.start_monitoring()
            await asyncio.sleep(0.5)
            await monitor.stop_monitoring()

        # Verify system remained resilient
        assert len(successful_calls) > 0
        assert alert_collector.count() > 0

        # Verify alerts were still generated despite failures
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        queue_alerts = alert_collector.get_by_type("high_queue_size")

        assert len(cpu_alerts) > 0
        assert len(queue_alerts) > 0
