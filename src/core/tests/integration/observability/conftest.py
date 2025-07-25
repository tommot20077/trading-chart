# ABOUTME: Comprehensive test fixtures and configuration for observability integration testing
# ABOUTME: Provides reusable components including mock handlers, loggers, and performance monitors with proper isolation

import asyncio
import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, AsyncGenerator
from unittest.mock import Mock
from loguru import logger

from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from ...fixtures.performance_metrics import PerformanceMonitor


@pytest_asyncio.fixture(scope="function")
async def mock_notification_handler() -> AsyncGenerator[InMemoryNotificationHandler, None]:
    """
    Create a mock notification handler for testing.

    Provides an in-memory notification handler with test-specific configuration
    for isolated testing of notification functionality.
    """
    handler = InMemoryNotificationHandler(
        max_queue_size=1000,
        max_history_size=5000,
        history_retention_hours=1.0,
        cleanup_interval_seconds=30.0,
        max_retry_attempts=3,
        simulate_failure_rate=0.0,  # No failures by default
        processing_delay_seconds=0.0,  # No delay by default
    )

    try:
        yield handler
    finally:
        # Close handler with timeout to prevent pytest hanging
        try:
            await asyncio.wait_for(handler.close(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            # If close times out, just continue - pytest will clean up
            pass


@pytest_asyncio.fixture(scope="function")
async def failing_notification_handler() -> AsyncGenerator[InMemoryNotificationHandler, None]:
    """
    Create a notification handler configured to simulate failures for testing error scenarios.
    """
    handler = InMemoryNotificationHandler(
        max_queue_size=100,
        max_history_size=500,
        history_retention_hours=0.5,
        cleanup_interval_seconds=60.0,
        max_retry_attempts=2,
        simulate_failure_rate=0.5,  # 50% failure rate
        processing_delay_seconds=0.1,  # Small delay
    )

    try:
        yield handler
    finally:
        # Close handler with timeout to prevent pytest hanging
        try:
            await asyncio.wait_for(handler.close(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            # If close times out, just continue - pytest will clean up
            pass


@pytest.fixture(scope="function", autouse=True)
def disable_loguru_for_tests(request):
    """
    Completely disable loguru for observability tests to prevent thread hanging issues.

    This fixture removes all loguru handlers and replaces loguru with a no-op implementation
    during tests, preventing the complex threading issues that cause test timeouts.
    
    Can be bypassed by using pytest.mark.enable_loguru_file_io marker.
    """
    # Check if the test has the enable_loguru_file_io marker
    if request.node.get_closest_marker("enable_loguru_file_io"):
        # Skip disabling loguru for tests that need file I/O
        yield
        return
    
    # Store original loguru functions
    original_loguru = {
        "info": logger.info,
        "error": logger.error,
        "debug": logger.debug,
        "warning": logger.warning,
        "remove": logger.remove,
        "add": logger.add,
        "complete": logger.complete,
    }

    # Create no-op functions to replace loguru methods
    def noop(*args, **kwargs):
        pass

    def noop_add(*args, **kwargs):
        return 1  # Return a dummy handler ID

    def noop_remove(*args, **kwargs):
        pass

    # Replace loguru methods with no-ops
    logger.info = noop
    logger.error = noop
    logger.debug = noop
    logger.warning = noop
    logger.add = noop_add
    logger.remove = noop_remove
    logger.complete = noop

    try:
        # Remove all existing handlers to prevent background threads
        logger.remove()
    except Exception:
        pass

    yield

    # Restore original loguru functions
    try:
        for method_name, original_func in original_loguru.items():
            setattr(logger, method_name, original_func)
    except Exception:
        pass


@pytest.fixture(scope="function")
def test_logger():
    """
    Provide a simple test logger that doesn't cause threading issues.

    Returns a mock logger that stores log messages for test verification
    without using loguru's complex threading system.
    """

    class SimpleTestLogger:
        def __init__(self):
            self.logs = []

        def info(self, message, **kwargs):
            self.logs.append(("INFO", message, kwargs))

        def error(self, message, **kwargs):
            self.logs.append(("ERROR", message, kwargs))

        def debug(self, message, **kwargs):
            self.logs.append(("DEBUG", message, kwargs))

        def warning(self, message, **kwargs):
            self.logs.append(("WARNING", message, kwargs))

        def get_logs(self):
            return self.logs.copy()

        def clear(self):
            self.logs.clear()

    return SimpleTestLogger()


@pytest_asyncio.fixture(scope="function")
async def performance_monitor(mock_event_bus):
    """
    Create a performance monitor for testing with mock event bus.

    Provides a configured performance monitor for testing monitoring
    and alerting functionality in isolation.
    """
    monitor = PerformanceMonitor(
        event_bus=mock_event_bus,
        sample_interval=0.05,  # Fast sampling for tests
    )

    # Configure test-appropriate thresholds
    monitor.cpu_threshold = 95.0  # High threshold to avoid false alarms in tests
    monitor.memory_threshold = 95.0
    monitor.queue_size_threshold = 500
    monitor.error_rate_threshold = 10.0

    yield monitor

    # Cleanup
    await monitor.stop_monitoring()
    monitor.clear_metrics()


@pytest.fixture(scope="function")
def mock_event_bus():
    """
    Create a mock event bus for testing observability components.

    Provides a mock event bus with realistic statistics for testing
    monitoring and notification functionality.
    """
    from unittest.mock import AsyncMock
    
    mock_bus = AsyncMock()

    # Default statistics
    default_stats = {
        "published_count": 0,
        "processed_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "dropped_count": 0,
        "queue_size": 0,
        "total_subscriptions": 0,
    }

    # Configure as a regular function, not async
    def get_stats():
        return default_stats.copy()
    
    mock_bus.get_statistics = get_stats

    # Add method to update stats for testing
    def update_stats(**kwargs):
        current_stats = default_stats
        current_stats.update(kwargs)
        
        def get_updated_stats():
            return current_stats.copy()
        
        mock_bus.get_statistics = get_updated_stats

    mock_bus.update_stats = update_stats
    
    # Configure async methods with realistic behavior
    mock_bus.publish.return_value = None  # publish returns None when successful
    mock_bus.flush_queue.return_value = None  # flush_queue returns None when successful
    mock_bus.close.return_value = None  # close returns None when successful
    mock_bus.subscribe.return_value = "mock_subscription_id"  # subscribe returns subscription id
    mock_bus.unsubscribe.return_value = None  # unsubscribe returns None
    
    # Add is_closed property 
    mock_bus.is_closed = False

    return mock_bus


@pytest.fixture(scope="function")
def alert_collector():
    """
    Create an alert collector for testing alert generation.

    Provides a callable that collects alerts fired during testing
    for verification and analysis.
    """
    collected_alerts = []

    def collect_alert(alert_type: str, alert_data: Dict[str, Any]):
        collected_alerts.append({"type": alert_type, "data": alert_data, "timestamp": asyncio.get_event_loop().time()})

    collect_alert.alerts = collected_alerts
    collect_alert.clear = lambda: collected_alerts.clear()
    collect_alert.count = lambda: len(collected_alerts)
    collect_alert.get_by_type = lambda alert_type: [alert for alert in collected_alerts if alert["type"] == alert_type]

    return collect_alert


@pytest.fixture(scope="function")
def notification_test_data():
    """
    Provide test data for notification testing.

    Returns structured test data for creating notifications
    with various scenarios and configurations.
    """
    return {
        "valid_alert": {
            "id": "test-alert-001",
            "rule_name": "high_cpu_usage",
            "severity": "critical",
            "message": "CPU usage exceeded 90% threshold",
            "metadata": {"current_value": 95.5, "threshold": 90.0, "duration": "5m"},
        },
        "warning_alert": {
            "id": "test-alert-002",
            "rule_name": "memory_warning",
            "severity": "warning",
            "message": "Memory usage is elevated",
            "metadata": {"current_value": 75.0, "threshold": 70.0, "duration": "2m"},
        },
        "info_alert": {
            "id": "test-alert-003",
            "rule_name": "system_startup",
            "severity": "info",
            "message": "System monitoring started",
            "metadata": {"startup_time": "2024-01-01T10:00:00Z"},
        },
        "invalid_alert_missing_fields": {
            "id": "test-alert-004",
            "severity": "error",
            # Missing required fields: rule_name, message
        },
        "invalid_alert_wrong_type": "not_a_dict",
    }


@pytest.fixture(scope="function")
def observability_test_config():
    """
    Provide test configuration for observability components.

    Returns configuration dictionaries for various observability
    components used in integration testing.
    """
    return {
        "notification_handler": {
            "max_queue_size": 500,
            "max_history_size": 2000,
            "history_retention_hours": 2.0,
            "cleanup_interval_seconds": 60.0,
            "max_retry_attempts": 3,
        },
        "performance_monitor": {
            "sample_interval": 0.1,
            "cpu_threshold": 80.0,
            "memory_threshold": 80.0,
            "queue_size_threshold": 100,
            "error_rate_threshold": 5.0,
        },
        "logging": {
            "level": "DEBUG",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            "rotation": None,
            "retention": None,
        },
    }


@pytest_asyncio.fixture(scope="function")
async def isolated_test_environment(
    mock_notification_handler,
    test_logger,
    performance_monitor,
    alert_collector,
    notification_test_data,
    observability_test_config,
):
    """
    Create a complete isolated test environment for observability testing.

    Combines all necessary fixtures into a single comprehensive environment
    for integration testing with proper cleanup and isolation.
    """
    # Create test workspace
    test_workspace = tempfile.mkdtemp(prefix="observability_test_")

    environment = {
        "notification_handler": mock_notification_handler,
        "logger": test_logger,
        "performance_monitor": performance_monitor,
        "alert_collector": alert_collector,
        "test_data": notification_test_data,
        "config": observability_test_config,
        "workspace": Path(test_workspace),
    }

    # Setup alert collection
    performance_monitor.add_alert_callback(alert_collector)

    yield environment

    # Cleanup
    try:
        shutil.rmtree(test_workspace, ignore_errors=True)
    except Exception:
        pass


@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_async_tasks():
    """
    Automatic cleanup of async tasks after each test.

    Ensures that all async tasks are properly cleaned up after
    each test to prevent interference between tests.
    """
    yield

    # Get the current event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, nothing to clean up
        return

    # Cancel any remaining tasks with timeout to prevent hanging
    try:
        # Get tasks that belong to this event loop
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks(loop) 
                if not task.done() and task is not current_task]
        
        if not tasks:
            return
            
        # Cancel tasks
        for task in tasks:
            if not task.cancelled():
                task.cancel()

        # Wait for cancellation with timeout to prevent hanging during pytest teardown
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=2.0
            )
    except (asyncio.TimeoutError, Exception):
        # If cleanup times out or fails, just continue
        # pytest will handle any remaining cleanup
        pass


# Async test utilities


async def wait_for_condition(condition_func: Callable[[], bool], timeout: float = 5.0, interval: float = 0.1) -> bool:
    """
    Wait for a condition to become true within a timeout period.

    Args:
        condition_func: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds

    Returns:
        True if condition was met, False if timeout occurred
    """
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(interval)

    return False


async def wait_for_notification_count(handler, expected_count: int, timeout: float = 5.0) -> bool:
    """
    Wait for notification handler to reach expected count.

    Args:
        handler: Notification handler instance
        expected_count: Expected number of notifications
        timeout: Maximum time to wait in seconds

    Returns:
        True if count was reached, False if timeout occurred
    """

    async def check_count():
        stats = await handler.get_notification_statistics()
        total = stats["total_sent"] + stats["total_failed"]
        return total >= expected_count

    return await wait_for_condition(check_count, timeout)


async def wait_for_alerts(alert_collector, expected_count: int, timeout: float = 5.0) -> bool:
    """
    Wait for alert collector to collect expected number of alerts.

    Args:
        alert_collector: Alert collector fixture
        expected_count: Expected number of alerts
        timeout: Maximum time to wait in seconds

    Returns:
        True if count was reached, False if timeout occurred
    """
    return await wait_for_condition(lambda: alert_collector.count() >= expected_count, timeout)


# Mark integration tests
pytestmark = pytest.mark.integration
