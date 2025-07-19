# ABOUTME: Infrastructure validation tests for observability integration testing framework
# ABOUTME: Verifies that test fixtures, configurations, and monitoring components work correctly in isolation

import pytest
import asyncio
from pathlib import Path

# Infrastructure tests - verifying test fixtures and configurations

from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from ...fixtures.performance_metrics import PerformanceMonitor
from .fixtures.monitoring_config import (
    ObservabilityTestConfigFactory,
    validate_test_config,
    DEFAULT_TEST_CONFIG,
    PERFORMANCE_TEST_CONFIG,
    STRESS_TEST_CONFIG,
    DEBUG_TEST_CONFIG,
)


class TestObservabilityTestInfrastructure:
    """Test suite for observability testing infrastructure."""

    @pytest.mark.asyncio
    async def test_notification_handler_fixture(self, mock_notification_handler):
        """Test that notification handler fixture works correctly."""
        # Verify handler is properly initialized
        assert isinstance(mock_notification_handler, InMemoryNotificationHandler)
        assert not mock_notification_handler.is_closed

        # Test basic functionality
        test_alert = {"id": "test-001", "rule_name": "test_rule", "severity": "info", "message": "Test notification"}

        success, message = await mock_notification_handler.send_notification(test_alert)
        assert success
        assert "successfully" in message.lower()

        # Verify statistics
        stats = await mock_notification_handler.get_notification_statistics()
        assert stats["total_sent"] == 1
        assert stats["total_failed"] == 0

    @pytest.mark.asyncio
    async def test_failing_notification_handler_fixture(self, failing_notification_handler):
        """Test that failing notification handler fixture simulates failures."""
        # Verify handler is configured for failures
        assert isinstance(failing_notification_handler, InMemoryNotificationHandler)
        assert failing_notification_handler.simulate_failure_rate > 0

        # Send multiple notifications to trigger failures
        test_alert = {
            "id": "test-002",
            "rule_name": "test_rule",
            "severity": "error",
            "message": "Test failure notification",
        }

        success_count = 0
        failure_count = 0

        for i in range(10):
            test_alert["id"] = f"test-{i:03d}"
            success, _ = await failing_notification_handler.send_notification(test_alert)
            if success:
                success_count += 1
            else:
                failure_count += 1

        # Should have some failures due to configured failure rate
        assert failure_count > 0

        # Verify statistics match
        stats = await failing_notification_handler.get_notification_statistics()
        assert stats["total_sent"] == success_count
        assert stats["total_failed"] == failure_count

    def test_logger_fixture(self, test_logger):
        """Test that logger fixture is properly configured."""
        # Test logging functionality
        test_logger.info("Test info message")
        test_logger.warning("Test warning message")
        test_logger.error("Test error message")

        # Verify logger captures logs correctly
        logs = test_logger.get_logs()
        assert len(logs) == 3
        
        # Verify log levels and messages
        levels = [log[0] for log in logs]
        messages = [log[1] for log in logs]
        assert "INFO" in levels
        assert "WARNING" in levels  
        assert "ERROR" in levels
        assert "Test info message" in messages
        assert "Test warning message" in messages
        assert "Test error message" in messages

    @pytest.mark.asyncio
    async def test_performance_monitor_fixture(self, performance_monitor, mock_event_bus):
        """Test that performance monitor fixture works correctly."""
        # Verify monitor is properly initialized
        assert isinstance(performance_monitor, PerformanceMonitor)
        assert performance_monitor.event_bus == mock_event_bus
        assert performance_monitor.sample_interval == 0.05  # Fast sampling for tests

        # Test metrics collection
        performance_monitor._collect_system_metrics()
        performance_monitor._collect_event_bus_metrics()

        # Test monitoring start/stop
        await performance_monitor.start_monitoring()
        assert performance_monitor._is_monitoring

        # Wait briefly for some metrics collection
        await asyncio.sleep(0.2)

        await performance_monitor.stop_monitoring()
        assert not performance_monitor._is_monitoring

        # Verify metrics were collected
        assert len(performance_monitor.snapshots) > 0

    def test_mock_event_bus_fixture(self, mock_event_bus):
        """Test that mock event bus fixture provides expected interface."""
        # Verify default statistics
        stats = mock_event_bus.get_statistics()
        expected_keys = [
            "published_count",
            "processed_count",
            "error_count",
            "timeout_count",
            "dropped_count",
            "queue_size",
            "total_subscriptions",
        ]
        for key in expected_keys:
            assert key in stats
            assert stats[key] == 0

        # Test stats update functionality
        mock_event_bus.update_stats(published_count=10, processed_count=8, error_count=2)
        updated_stats = mock_event_bus.get_statistics()
        assert updated_stats["published_count"] == 10
        assert updated_stats["processed_count"] == 8
        assert updated_stats["error_count"] == 2

    def test_alert_collector_fixture(self, alert_collector):
        """Test that alert collector fixture works correctly."""
        # Test initial state
        assert alert_collector.count() == 0
        assert len(alert_collector.alerts) == 0

        # Test alert collection
        alert_collector("high_cpu", {"current": 85.0, "threshold": 80.0})
        alert_collector("high_memory", {"current": 90.0, "threshold": 80.0})

        assert alert_collector.count() == 2

        # Test filtering by type
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0]["type"] == "high_cpu"
        assert cpu_alerts[0]["data"]["current"] == 85.0

        # Test clearing
        alert_collector.clear()
        assert alert_collector.count() == 0

    def test_notification_test_data_fixture(self, notification_test_data):
        """Test that notification test data fixture provides expected data."""
        # Verify required test data entries
        required_entries = [
            "valid_alert",
            "warning_alert",
            "info_alert",
            "invalid_alert_missing_fields",
            "invalid_alert_wrong_type",
        ]

        for entry in required_entries:
            assert entry in notification_test_data

        # Verify valid alert structure
        valid_alert = notification_test_data["valid_alert"]
        required_fields = ["id", "rule_name", "severity", "message"]
        for field in required_fields:
            assert field in valid_alert

        # Verify invalid cases
        invalid_missing = notification_test_data["invalid_alert_missing_fields"]
        assert "rule_name" not in invalid_missing or "message" not in invalid_missing

        invalid_type = notification_test_data["invalid_alert_wrong_type"]
        assert not isinstance(invalid_type, dict)

    def test_observability_test_config_fixture(self, observability_test_config):
        """Test that observability test config fixture provides expected configuration."""
        # Verify configuration sections
        required_sections = ["notification_handler", "performance_monitor", "logging"]

        for section in required_sections:
            assert section in observability_test_config

        # Verify notification handler config
        notification_config = observability_test_config["notification_handler"]
        assert "max_queue_size" in notification_config
        assert "max_history_size" in notification_config
        assert "history_retention_hours" in notification_config

        # Verify performance monitor config
        performance_config = observability_test_config["performance_monitor"]
        assert "sample_interval" in performance_config
        assert "cpu_threshold" in performance_config
        assert "memory_threshold" in performance_config

    @pytest.mark.asyncio
    async def test_isolated_test_environment_fixture(self, isolated_test_environment):
        """Test that isolated test environment fixture provides complete setup."""
        env = isolated_test_environment

        # Verify all components are present
        required_components = [
            "notification_handler",
            "logger",
            "performance_monitor",
            "alert_collector",
            "test_data",
            "config",
            "workspace",
        ]

        for component in required_components:
            assert component in env

        # Verify workspace is created
        assert env["workspace"].exists()
        assert env["workspace"].is_dir()

        # Test integrated functionality
        handler = env["notification_handler"]
        alert_collector = env["alert_collector"]
        test_data = env["test_data"]

        # Send notification and verify alert collection
        valid_alert = test_data["valid_alert"]
        success, _ = await handler.send_notification(valid_alert)
        assert success

        # Performance monitor should be configured with alert collector
        monitor = env["performance_monitor"]
        assert alert_collector in monitor.alert_callbacks


class TestMonitoringConfigFactory:
    """Test suite for monitoring configuration factory."""

    def test_config_factory_scenarios(self):
        """Test that config factory creates all required scenarios."""
        scenarios = ["basic", "performance", "stress", "debug"]

        for scenario in scenarios:
            config = ObservabilityTestConfigFactory.create_config(scenario)
            assert hasattr(config, scenario)
            scenario_config = getattr(config, scenario)

            # Verify scenario has required components
            required_components = ["notification", "performance", "logging", "alerts"]
            for component in required_components:
                assert component in scenario_config

    def test_config_validation(self):
        """Test that configuration validation works correctly."""
        # Test predefined configs
        configs = [DEFAULT_TEST_CONFIG, PERFORMANCE_TEST_CONFIG, STRESS_TEST_CONFIG, DEBUG_TEST_CONFIG]

        for config in configs:
            validation_results = validate_test_config(config)

            # All validations should pass
            for component, is_valid in validation_results.items():
                assert is_valid, f"Validation failed for {component}"

    def test_config_factory_overrides(self):
        """Test that config factory properly applies overrides."""
        # Test notification config overrides
        custom_notification = ObservabilityTestConfigFactory.create_notification_config(
            "basic", max_queue_size=2000, simulate_failure_rate=0.1
        )

        assert custom_notification.max_queue_size == 2000
        assert custom_notification.simulate_failure_rate == 0.1

        # Test performance config overrides
        custom_performance = ObservabilityTestConfigFactory.create_performance_config(
            "basic", sample_interval=0.2, cpu_threshold=70.0
        )

        assert custom_performance.sample_interval == 0.2
        assert custom_performance.cpu_threshold == 70.0

    def test_config_factory_error_handling(self):
        """Test that config factory handles invalid scenarios correctly."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            ObservabilityTestConfigFactory.create_config("invalid_scenario")


class TestAsyncUtilities:
    """Test suite for async test utilities."""

    @pytest.mark.asyncio
    async def test_wait_for_condition(self):
        """Test wait_for_condition utility function."""
        from .conftest import wait_for_condition

        # Test condition that becomes true
        counter = {"value": 0}

        async def increment_counter():
            await asyncio.sleep(0.1)
            counter["value"] += 1

        # Start background task
        task = asyncio.create_task(increment_counter())

        # Wait for condition
        result = await wait_for_condition(lambda: counter["value"] > 0, timeout=1.0, interval=0.05)

        assert result is True
        assert counter["value"] == 1

        await task

    @pytest.mark.asyncio
    async def test_wait_for_condition_timeout(self):
        """Test wait_for_condition timeout behavior."""
        from .conftest import wait_for_condition

        # Test condition that never becomes true
        result = await wait_for_condition(lambda: False, timeout=0.2, interval=0.05)

        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_notification_count(self, mock_notification_handler):
        """Test wait_for_notification_count utility function."""
        from .conftest import wait_for_notification_count

        # Send notifications in background
        async def send_notifications():
            for i in range(3):
                await asyncio.sleep(0.05)
                await mock_notification_handler.send_notification(
                    {"id": f"test-{i}", "rule_name": "test_rule", "severity": "info", "message": f"Test message {i}"}
                )

        # Start background task
        task = asyncio.create_task(send_notifications())

        # Wait for notifications
        result = await wait_for_notification_count(mock_notification_handler, expected_count=3, timeout=1.0)

        assert result is True

        await task

    @pytest.mark.asyncio
    async def test_wait_for_alerts(self, alert_collector):
        """Test wait_for_alerts utility function."""
        from .conftest import wait_for_alerts

        # Generate alerts in background
        async def generate_alerts():
            for i in range(2):
                await asyncio.sleep(0.05)
                alert_collector(f"test_alert_{i}", {"value": i})

        # Start background task
        task = asyncio.create_task(generate_alerts())

        # Wait for alerts
        result = await wait_for_alerts(alert_collector, expected_count=2, timeout=1.0)

        assert result is True
        assert alert_collector.count() == 2

        await task
