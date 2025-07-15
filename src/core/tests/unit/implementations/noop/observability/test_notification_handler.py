# ABOUTME: Unit tests for NoOpNotificationHandler implementation
# ABOUTME: Tests minimal notification handling that always succeeds without side effects

import pytest
import pytest_asyncio

from core.implementations.noop.observability.notification_handler import NoOpNotificationHandler


class TestNoOpNotificationHandler:
    """Test the NoOpNotificationHandler class."""

    @pytest_asyncio.fixture
    async def notification_handler(self):
        """Create a NoOp notification handler for testing."""
        handler = NoOpNotificationHandler()
        yield handler
        # No cleanup needed for NoOp implementation

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test NoOp notification handler initialization."""
        handler = NoOpNotificationHandler()

        # Should initialize without any configuration
        assert handler is not None

        # No cleanup needed

    @pytest.mark.asyncio
    async def test_send_notification_success(self, notification_handler):
        """Test that send_notification always succeeds."""
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
        assert message == "NoOp notification sent successfully"

    @pytest.mark.asyncio
    async def test_send_notification_with_minimal_data(self, notification_handler):
        """Test send_notification with minimal alert data."""
        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "info",
            "message": "Test message",
        }

        success, message = await notification_handler.send_notification(alert_data)

        assert success is True
        assert message == "NoOp notification sent successfully"

    @pytest.mark.asyncio
    async def test_send_notification_with_empty_data(self, notification_handler):
        """Test send_notification with empty alert data."""
        alert_data = {}

        success, message = await notification_handler.send_notification(alert_data)

        assert success is True
        assert message == "NoOp notification sent successfully"

    @pytest.mark.asyncio
    async def test_send_notification_with_invalid_data(self, notification_handler):
        """Test send_notification with invalid alert data types."""
        # Test with None
        success, message = await notification_handler.send_notification(None)
        assert success is True
        assert message == "NoOp notification sent successfully"

        # Test with string
        success, message = await notification_handler.send_notification("invalid")
        assert success is True
        assert message == "NoOp notification sent successfully"

        # Test with number
        success, message = await notification_handler.send_notification(123)
        assert success is True
        assert message == "NoOp notification sent successfully"

    @pytest.mark.asyncio
    async def test_send_notification_with_large_data(self, notification_handler):
        """Test send_notification with large alert data."""
        # Create a large alert data dictionary
        alert_data = {
            "id": "alert-001",
            "rule_name": "test_rule",
            "severity": "critical",
            "message": "A" * 10000,  # Large message
            "labels": {f"label_{i}": f"value_{i}" for i in range(1000)},  # Many labels
            "annotations": {f"annotation_{i}": f"value_{i}" for i in range(1000)},  # Many annotations
            "fired_at": "2024-01-01T12:00:00Z",
            "trace_id": "trace-123",
        }

        success, message = await notification_handler.send_notification(alert_data)

        assert success is True
        assert message == "NoOp notification sent successfully"

    @pytest.mark.asyncio
    async def test_multiple_notifications(self, notification_handler):
        """Test sending multiple notifications in sequence."""
        alert_data_list = [
            {
                "id": f"alert-{i:03d}",
                "rule_name": f"rule_{i}",
                "severity": "warning" if i % 2 == 0 else "error",
                "message": f"Test message {i}",
            }
            for i in range(10)
        ]

        for alert_data in alert_data_list:
            success, message = await notification_handler.send_notification(alert_data)
            assert success is True
            assert message == "NoOp notification sent successfully"

    @pytest.mark.asyncio
    async def test_concurrent_notifications(self, notification_handler):
        """Test sending notifications concurrently."""
        import asyncio

        async def send_notification(i):
            alert_data = {
                "id": f"alert-{i:03d}",
                "rule_name": f"rule_{i}",
                "severity": "info",
                "message": f"Test message {i}",
            }
            return await notification_handler.send_notification(alert_data)

        # Send 100 notifications concurrently
        tasks = [send_notification(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(success for success, _ in results)
        assert all(message == "NoOp notification sent successfully" for _, message in results)

    @pytest.mark.asyncio
    async def test_notification_performance(self, notification_handler):
        """Test NoOp notification handler performance."""
        import time

        alert_data = {
            "id": "alert-001",
            "rule_name": "performance_test",
            "severity": "info",
            "message": "Performance test message",
        }

        # Measure time for many notifications
        start_time = time.time()

        for i in range(1000):
            await notification_handler.send_notification(alert_data)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Should be very fast (less than 1 second for 1000 notifications)
        assert elapsed_time < 1.0

        # Calculate average time per notification
        avg_time = elapsed_time / 1000
        assert avg_time < 0.001  # Less than 1ms per notification

    @pytest.mark.asyncio
    async def test_notification_handler_consistency(self, notification_handler):
        """Test that NoOp handler behaves consistently."""
        alert_data = {
            "id": "alert-001",
            "rule_name": "consistency_test",
            "severity": "warning",
            "message": "Consistency test message",
        }

        # Send the same notification multiple times
        results = []
        for i in range(10):
            result = await notification_handler.send_notification(alert_data)
            results.append(result)

        # All results should be identical
        assert all(result == (True, "NoOp notification sent successfully") for result in results)

    @pytest.mark.asyncio
    async def test_notification_handler_no_side_effects(self, notification_handler):
        """Test that NoOp handler has no side effects."""
        alert_data = {
            "id": "alert-001",
            "rule_name": "side_effect_test",
            "severity": "error",
            "message": "Side effect test message",
        }

        # Send notification and verify no state is maintained
        success, message = await notification_handler.send_notification(alert_data)
        assert success is True

        # Handler should not maintain any state
        # (This is more of a design verification than a functional test)
        assert hasattr(notification_handler, "__dict__")  # Has attributes dict

        # Should not have any significant state attributes
        # (NoOp implementation should be minimal)
        handler_dict = notification_handler.__dict__
        assert len(handler_dict) == 0 or all(attr.startswith("_") or attr in [] for attr in handler_dict)

    @pytest.mark.asyncio
    async def test_notification_handler_exception_safety(self, notification_handler):
        """Test that NoOp handler doesn't raise exceptions."""
        # Test with various potentially problematic inputs
        problematic_inputs = [
            None,
            [],
            {},
            {"circular_ref": None},
            {"nested": {"very": {"deep": {"data": "structure"}}}},
            {"special_chars": "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`"},
            {"unicode": "测试数据 🚀 العربية"},
        ]

        # Add circular reference to one of the test cases
        circular_data = {"key": "value"}
        circular_data["circular_ref"] = circular_data
        problematic_inputs.append(circular_data)

        for test_input in problematic_inputs:
            try:
                success, message = await notification_handler.send_notification(test_input)
                assert success is True
                assert message == "NoOp notification sent successfully"
            except Exception as e:
                pytest.fail(f"NoOp handler raised exception with input {test_input}: {e}")

    @pytest.mark.asyncio
    async def test_notification_handler_memory_usage(self, notification_handler):
        """Test that NoOp handler doesn't consume excessive memory."""
        import sys

        # Get initial memory usage
        initial_size = sys.getsizeof(notification_handler)

        alert_data = {
            "id": "alert-001",
            "rule_name": "memory_test",
            "severity": "info",
            "message": "Memory test message",
        }

        # Send many notifications
        for i in range(1000):
            await notification_handler.send_notification(alert_data)

        # Memory usage should not increase significantly
        final_size = sys.getsizeof(notification_handler)
        assert final_size == initial_size or final_size - initial_size < 1024  # Allow small variance


@pytest.mark.asyncio
async def test_noop_notification_handler_integration():
    """Integration test for NoOp notification handler in realistic scenarios."""
    handler = NoOpNotificationHandler()

    # Simulate a monitoring system sending various alerts
    alert_scenarios = [
        # High-frequency alerts
        *[
            {
                "id": f"high_freq_{i}",
                "rule_name": "high_frequency_check",
                "severity": "info",
                "message": f"High frequency alert {i}",
            }
            for i in range(100)
        ],
        # Critical alerts
        {
            "id": "critical_001",
            "rule_name": "system_down",
            "severity": "critical",
            "message": "System is completely down",
            "labels": {"service": "core", "environment": "production"},
        },
        # Warning alerts
        {
            "id": "warning_001",
            "rule_name": "high_cpu",
            "severity": "warning",
            "message": "CPU usage is above 80%",
            "labels": {"service": "web", "environment": "production"},
        },
        # Complex alert with nested data
        {
            "id": "complex_001",
            "rule_name": "complex_check",
            "severity": "error",
            "message": "Complex system error",
            "labels": {"service": "database", "environment": "staging"},
            "annotations": {
                "runbook": "https://example.com/runbook",
                "dashboard": "https://grafana.example.com/dashboard",
            },
            "context": {
                "stack_trace": "Exception in thread 'main'...",
                "metrics": {
                    "cpu": 85.5,
                    "memory": 92.3,
                    "disk": 78.1,
                },
            },
        },
    ]

    success_count = 0
    total_count = len(alert_scenarios)

    # Process all alerts
    for alert_data in alert_scenarios:
        success, message = await handler.send_notification(alert_data)
        if success:
            success_count += 1

    # All should succeed with NoOp handler
    assert success_count == total_count

    # Verify no state is maintained
    assert len(handler.__dict__) == 0  # Should have no state


@pytest.mark.asyncio
async def test_noop_vs_real_handler_interface_compatibility():
    """Test that NoOp handler maintains interface compatibility."""
    from core.interfaces.observability.notification_handler import AbstractNotificationHandler

    # Verify NoOp handler is properly implementing the interface
    handler = NoOpNotificationHandler()
    assert isinstance(handler, AbstractNotificationHandler)

    # Verify all required methods exist
    assert hasattr(handler, "send_notification")
    assert callable(handler.send_notification)

    # Test method signature compatibility
    import inspect

    # Get the abstract method signature
    abstract_method = AbstractNotificationHandler.send_notification
    noop_method = handler.send_notification

    # Both should have the same parameter names (excluding 'self')
    abstract_sig = inspect.signature(abstract_method)
    noop_sig = inspect.signature(noop_method)

    # Remove 'self' parameter from abstract method for comparison
    abstract_params = [p for p in abstract_sig.parameters.keys() if p != "self"]
    noop_params = list(noop_sig.parameters.keys())

    assert abstract_params == noop_params

    # Test return type compatibility
    alert_data = {
        "id": "compatibility_test",
        "rule_name": "interface_test",
        "severity": "info",
        "message": "Interface compatibility test",
    }

    result = await handler.send_notification(alert_data)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)
