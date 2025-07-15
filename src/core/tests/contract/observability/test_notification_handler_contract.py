# ABOUTME: Contract tests for AbstractNotificationHandler interface
# ABOUTME: Verifies that all implementations comply with the notification handler contract

import pytest
from typing import Type, List

from core.interfaces.observability.notification_handler import AbstractNotificationHandler
from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from core.implementations.noop.observability.notification_handler import NoOpNotificationHandler
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestAbstractNotificationHandlerContract(ContractTestBase[AbstractNotificationHandler], AsyncContractTestMixin):
    """Contract tests for AbstractNotificationHandler interface."""

    @property
    def interface_class(self) -> Type[AbstractNotificationHandler]:
        """The interface class being tested."""
        return AbstractNotificationHandler

    @property
    def implementations(self) -> List[Type[AbstractNotificationHandler]]:
        """List of concrete implementations to test against the interface."""
        return [
            InMemoryNotificationHandler,
            NoOpNotificationHandler,
        ]

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_send_notification_method_contract(self):
        """Test that send_notification method follows the contract."""
        test_alert_data = {
            "id": "test-alert-001",
            "rule_name": "test_rule",
            "severity": "warning",
            "message": "Test alert message",
            "labels": {"service": "test"},
            "annotations": {"description": "Test alert"},
            "fired_at": "2024-01-01T12:00:00Z",
            "trace_id": "trace-123",
        }

        for impl_class in self.implementations:
            # Create instance
            if impl_class == InMemoryNotificationHandler:
                instance = impl_class(
                    max_queue_size=10,
                    max_history_size=50,
                    simulate_failure_rate=0.0,
                )
            else:
                instance = impl_class()

            try:
                # Test method exists and is callable
                assert hasattr(instance, "send_notification")
                assert callable(instance.send_notification)

                # Test method is async
                import asyncio

                assert asyncio.iscoroutinefunction(instance.send_notification)

                # Test method signature
                import inspect

                sig = inspect.signature(instance.send_notification)
                params = list(sig.parameters.keys())
                assert "alert_data" in params

                # Test method returns correct type
                result = await instance.send_notification(test_alert_data)
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], bool)
                assert isinstance(result[1], str)

                # Test with valid data
                success, message = await instance.send_notification(test_alert_data)
                assert isinstance(success, bool)
                assert isinstance(message, str)
                assert len(message) > 0

            finally:
                # Cleanup if needed
                if hasattr(instance, "close"):
                    await instance.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_send_notification_parameter_validation(self):
        """Test that implementations handle parameter validation consistently."""
        for impl_class in self.implementations:
            # Create instance
            if impl_class == InMemoryNotificationHandler:
                instance = impl_class(
                    max_queue_size=10,
                    max_history_size=50,
                    simulate_failure_rate=0.0,
                )
            else:
                instance = impl_class()

            try:
                # Test with different parameter types
                test_cases = [
                    # Valid minimal data
                    {
                        "id": "test-001",
                        "rule_name": "test_rule",
                        "severity": "info",
                        "message": "Test message",
                    },
                    # Valid comprehensive data
                    {
                        "id": "test-002",
                        "rule_name": "comprehensive_rule",
                        "severity": "critical",
                        "message": "Comprehensive test message",
                        "labels": {"service": "test", "environment": "staging"},
                        "annotations": {"runbook": "https://example.com/runbook"},
                        "fired_at": "2024-01-01T12:00:00Z",
                        "trace_id": "trace-456",
                    },
                ]

                for test_data in test_cases:
                    result = await instance.send_notification(test_data)
                    assert isinstance(result, tuple)
                    assert len(result) == 2
                    assert isinstance(result[0], bool)
                    assert isinstance(result[1], str)

            finally:
                # Cleanup if needed
                if hasattr(instance, "close"):
                    await instance.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_send_notification_return_value_consistency(self):
        """Test that return values are consistent across implementations."""
        test_alert_data = {
            "id": "consistency-test-001",
            "rule_name": "consistency_rule",
            "severity": "warning",
            "message": "Consistency test message",
        }

        results = []
        instances = []

        try:
            for impl_class in self.implementations:
                # Create instance
                if impl_class == InMemoryNotificationHandler:
                    instance = impl_class(
                        max_queue_size=10,
                        max_history_size=50,
                        simulate_failure_rate=0.0,
                    )
                else:
                    instance = impl_class()

                instances.append(instance)
                result = await instance.send_notification(test_alert_data)
                results.append(result)

            # Verify all results have the same structure
            for result in results:
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], bool)
                assert isinstance(result[1], str)

            # For this test, we don't enforce that all implementations return the same values,
            # only that they follow the same contract structure

        finally:
            # Cleanup all instances
            for instance in instances:
                if hasattr(instance, "close"):
                    await instance.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_send_notification_concurrent_access(self):
        """Test that implementations handle concurrent access properly."""
        import asyncio

        test_alert_data = {
            "id": "concurrent-test-001",
            "rule_name": "concurrent_rule",
            "severity": "info",
            "message": "Concurrent test message",
        }

        for impl_class in self.implementations:
            # Create instance
            if impl_class == InMemoryNotificationHandler:
                instance = impl_class(
                    max_queue_size=100,
                    max_history_size=500,
                    simulate_failure_rate=0.0,
                )
            else:
                instance = impl_class()

            try:
                # Send multiple notifications concurrently
                async def send_notification_with_id(i):
                    alert_data = test_alert_data.copy()
                    alert_data["id"] = f"concurrent-test-{i:03d}"
                    return await instance.send_notification(alert_data)

                # Run 10 concurrent notifications
                tasks = [send_notification_with_id(i) for i in range(10)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Verify no exceptions were raised
                for result in results:
                    assert not isinstance(result, Exception), f"Exception in concurrent test: {result}"
                    assert isinstance(result, tuple)
                    assert len(result) == 2
                    assert isinstance(result[0], bool)
                    assert isinstance(result[1], str)

            finally:
                # Cleanup if needed
                if hasattr(instance, "close"):
                    await instance.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_send_notification_error_handling(self):
        """Test that implementations handle errors gracefully."""
        # Test error handling with different types of invalid data
        error_test_cases = [
            # Test implementation-specific error handling
            # Note: Different implementations may handle errors differently
            # This test ensures they don't crash and return proper error format
        ]

        for impl_class in self.implementations:
            # Create instance
            if impl_class == InMemoryNotificationHandler:
                instance = impl_class(
                    max_queue_size=10,
                    max_history_size=50,
                    simulate_failure_rate=0.0,
                )
            else:
                instance = impl_class()

            try:
                # Test with edge cases that might cause issues
                edge_cases = [
                    # Empty dictionary
                    {},
                    # Very large message
                    {
                        "id": "large-test-001",
                        "rule_name": "large_rule",
                        "severity": "info",
                        "message": "A" * 100000,  # 100KB message
                    },
                    # Unicode characters
                    {
                        "id": "unicode-test-001",
                        "rule_name": "unicode_rule",
                        "severity": "info",
                        "message": "Test with unicode: 测试数据 🚀 العربية",
                    },
                ]

                for test_data in edge_cases:
                    try:
                        result = await instance.send_notification(test_data)
                        # Should always return a tuple, even for errors
                        assert isinstance(result, tuple)
                        assert len(result) == 2
                        assert isinstance(result[0], bool)
                        assert isinstance(result[1], str)
                    except Exception as e:
                        # If an exception is raised, it should be documented behavior
                        # For now, we'll allow it but log it
                        pytest.fail(f"Implementation {impl_class.__name__} raised unexpected exception: {e}")

            finally:
                # Cleanup if needed
                if hasattr(instance, "close"):
                    await instance.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_implementation_instantiation(self):
        """Test that all implementations can be instantiated correctly."""
        instances = []

        try:
            for impl_class in self.implementations:
                # Test default instantiation
                if impl_class == InMemoryNotificationHandler:
                    # Test with default parameters
                    instance = impl_class()
                    instances.append(instance)

                    # Test with custom parameters
                    instance2 = impl_class(
                        max_queue_size=50,
                        max_history_size=200,
                        history_retention_hours=2.0,
                        simulate_failure_rate=0.0,
                    )
                    instances.append(instance2)
                else:
                    # NoOp implementation should instantiate without parameters
                    instance = impl_class()
                    instances.append(instance)

                # Verify instance is of correct type
                assert isinstance(instance, impl_class)
                assert isinstance(instance, AbstractNotificationHandler)

        finally:
            # Cleanup all instances
            for instance in instances:
                if hasattr(instance, "close"):
                    await instance.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_resource_cleanup_contract(self):
        """Test that implementations properly handle resource cleanup."""
        for impl_class in self.implementations:
            # Create instance
            if impl_class == InMemoryNotificationHandler:
                instance = impl_class(
                    max_queue_size=10,
                    max_history_size=50,
                    simulate_failure_rate=0.0,
                )
            else:
                instance = impl_class()

            # Test that instance can be used
            test_alert_data = {
                "id": "cleanup-test-001",
                "rule_name": "cleanup_rule",
                "severity": "info",
                "message": "Cleanup test message",
            }

            result = await instance.send_notification(test_alert_data)
            assert isinstance(result, tuple)

            # Test cleanup (if supported)
            if hasattr(instance, "close"):
                await instance.close()

                # Verify that instance indicates it's closed (if supported)
                if hasattr(instance, "is_closed"):
                    assert instance.is_closed is True

    @pytest.mark.contract
    def test_interface_documentation(self):
        """Test that the interface has proper documentation."""
        assert AbstractNotificationHandler.__doc__ is not None
        assert len(AbstractNotificationHandler.__doc__.strip()) > 0

        # Test that send_notification has documentation
        assert AbstractNotificationHandler.send_notification.__doc__ is not None
        assert len(AbstractNotificationHandler.send_notification.__doc__.strip()) > 0

    @pytest.mark.contract
    def test_implementation_documentation(self):
        """Test that implementations have proper documentation."""
        for impl_class in self.implementations:
            assert impl_class.__doc__ is not None
            assert len(impl_class.__doc__.strip()) > 0

            # Test that send_notification implementation has documentation
            assert impl_class.send_notification.__doc__ is not None
            assert len(impl_class.send_notification.__doc__.strip()) > 0


@pytest.mark.asyncio
async def test_notification_handler_contract_integration():
    """Integration test for notification handler contract compliance."""
    # Test that all implementations can work together in a realistic scenario
    implementations = [
        InMemoryNotificationHandler(simulate_failure_rate=0.0),
        NoOpNotificationHandler(),
    ]

    test_scenarios = [
        {
            "id": "integration-001",
            "rule_name": "cpu_threshold",
            "severity": "warning",
            "message": "CPU usage above 80%",
        },
        {
            "id": "integration-002",
            "rule_name": "memory_threshold",
            "severity": "critical",
            "message": "Memory usage above 95%",
        },
        {
            "id": "integration-003",
            "rule_name": "disk_threshold",
            "severity": "error",
            "message": "Disk usage above 90%",
        },
    ]

    try:
        for impl in implementations:
            for scenario in test_scenarios:
                result = await impl.send_notification(scenario)
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], bool)
                assert isinstance(result[1], str)

    finally:
        # Cleanup
        for impl in implementations:
            if hasattr(impl, "close"):
                await impl.close()
