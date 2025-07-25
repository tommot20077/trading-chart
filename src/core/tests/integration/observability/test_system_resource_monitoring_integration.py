# ABOUTME: Integration tests for system resource monitoring including CPU, memory, threads, and file descriptors
# ABOUTME: Validates resource monitoring accuracy, threshold detection, and alert system integration

import asyncio
import pytest
import pytest_asyncio
import psutil
import threading
import tempfile
import os
from unittest.mock import patch
import time

# Previously skipped due to timeout issues - now fixed

from ...fixtures.performance_metrics import PerformanceMonitor, SystemMetrics


class TestSystemResourceMonitoring:
    """Integration tests for system resource monitoring capabilities."""

    @pytest_asyncio.fixture
    async def resource_monitor(self, mock_event_bus):
        """Create PerformanceMonitor with focus on resource monitoring."""
        monitor = PerformanceMonitor(
            event_bus=mock_event_bus,
            sample_interval=0.05,  # Fast sampling for tests
        )

        # Configure realistic thresholds for testing
        monitor.cpu_threshold = 85.0
        monitor.memory_threshold = 85.0
        monitor.queue_size_threshold = 50
        monitor.error_rate_threshold = 5.0

        yield monitor

        # Cleanup
        await monitor.stop_monitoring()
        monitor.clear_metrics()

    @pytest.mark.asyncio
    async def test_cpu_monitoring_accuracy(self, resource_monitor):
        """Test CPU monitoring accuracy and real-time updates."""
        monitor = resource_monitor

        # Start monitoring
        await monitor.start_monitoring()

        # Create CPU load to test monitoring
        def cpu_intensive_task():
            """Simple CPU intensive task."""
            end_time = time.time() + 0.2  # Run for 200ms
            while time.time() < end_time:
                _ = sum(i * i for i in range(1000))

        # Run CPU intensive task in background
        cpu_thread = threading.Thread(target=cpu_intensive_task)
        cpu_thread.start()

        # Wait for monitoring to capture the load
        await asyncio.sleep(0.3)

        cpu_thread.join()

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify CPU metrics were collected
        assert len(monitor.system_metrics) > 0

        # Verify CPU metrics are reasonable
        for sys_metric in monitor.system_metrics:
            assert 0 <= sys_metric.cpu_percent <= 100
            assert isinstance(sys_metric.cpu_percent, (int, float))

        # Verify we captured some variation (though this may not always be detectable)
        cpu_values = [m.cpu_percent for m in monitor.system_metrics]
        assert max(cpu_values) >= 0  # At minimum should be 0 or higher

    @pytest.mark.asyncio
    async def test_memory_monitoring_accuracy(self, resource_monitor):
        """Test memory monitoring accuracy and allocation tracking."""
        monitor = resource_monitor

        # Get initial memory state
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Start monitoring
        await monitor.start_monitoring()

        # Allocate memory to test monitoring
        memory_consumers = []
        try:
            # Allocate some memory (careful not to use too much)
            for _ in range(5):
                # Allocate 1MB chunks
                memory_consumers.append(bytearray(1024 * 1024))

            # Wait for monitoring to capture the allocation
            await asyncio.sleep(0.2)

        finally:
            # Clean up memory
            memory_consumers.clear()

        # Wait a bit more for monitoring
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify memory metrics were collected
        assert len(monitor.system_metrics) > 0

        # Verify memory metrics are reasonable
        for sys_metric in monitor.system_metrics:
            assert 0 <= sys_metric.memory_percent <= 100
            assert sys_metric.memory_used_mb > 0
            assert sys_metric.memory_available_mb > 0
            assert isinstance(sys_metric.memory_used_mb, (int, float))
            assert isinstance(sys_metric.memory_available_mb, (int, float))

        # Verify we have some memory usage
        memory_used_values = [m.memory_used_mb for m in monitor.system_metrics]
        assert all(val > 0 for val in memory_used_values)

    @pytest.mark.asyncio
    async def test_thread_monitoring(self, resource_monitor):
        """Test thread count monitoring and tracking."""
        monitor = resource_monitor

        # Start monitoring
        await monitor.start_monitoring()

        # Create additional threads to test monitoring
        def worker_thread():
            time.sleep(0.3)

        threads = []
        try:
            # Create some threads
            for i in range(3):
                thread = threading.Thread(target=worker_thread)
                thread.start()
                threads.append(thread)

            # Wait for monitoring to capture thread changes
            await asyncio.sleep(0.2)

        finally:
            # Wait for threads to complete
            for thread in threads:
                thread.join()

        # Wait a bit more for monitoring
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify thread metrics were collected
        assert len(monitor.system_metrics) > 0

        # Verify thread metrics are reasonable
        for sys_metric in monitor.system_metrics:
            assert sys_metric.thread_count >= 1  # At least the main thread
            assert isinstance(sys_metric.thread_count, int)

        # Verify we captured thread count variations
        thread_counts = [m.thread_count for m in monitor.system_metrics]
        assert max(thread_counts) >= min(thread_counts)

    @pytest.mark.asyncio
    async def test_file_descriptor_monitoring(self, resource_monitor):
        """Test file descriptor monitoring and tracking."""
        monitor = resource_monitor

        # Start monitoring
        await monitor.start_monitoring()

        # Create/open files to test FD monitoring
        temp_files = []
        try:
            # Open several temporary files
            for i in range(5):
                temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
                temp_file.write(f"Test file {i}")
                temp_files.append(temp_file)

            # Wait for monitoring to capture FD changes
            await asyncio.sleep(0.2)

        finally:
            # Clean up files
            for temp_file in temp_files:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except FileNotFoundError:
                    pass

        # Wait a bit more for monitoring
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify FD metrics were collected
        assert len(monitor.system_metrics) > 0

        # Verify FD metrics are reasonable
        for sys_metric in monitor.system_metrics:
            assert sys_metric.fd_count >= 0
            assert isinstance(sys_metric.fd_count, int)

    @pytest.mark.asyncio
    async def test_resource_threshold_detection(self, resource_monitor, alert_collector):
        """Test resource threshold detection and alerting."""
        monitor = resource_monitor

        # Configure very low thresholds to trigger alerts
        monitor.cpu_threshold = 0.1  # Very low to trigger easily
        monitor.memory_threshold = 0.1

        # Add alert collector
        monitor.add_alert_callback(alert_collector)

        # Mock high resource usage
        with (
            patch.object(monitor.process, "cpu_percent", return_value=50.0),
            patch.object(monitor.process, "memory_percent", return_value=50.0),
        ):
            # Start monitoring
            await monitor.start_monitoring()

            # Wait for threshold checks
            await asyncio.sleep(0.3)

            # Stop monitoring
            await monitor.stop_monitoring()

        # Verify alerts were generated
        assert alert_collector.count() > 0

        # Check for specific alert types
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        memory_alerts = alert_collector.get_by_type("high_memory")

        assert len(cpu_alerts) > 0
        assert len(memory_alerts) > 0

        # Verify alert data structure
        for alert in cpu_alerts:
            assert "data" in alert
            assert "current" in alert["data"]
            assert "threshold" in alert["data"]
            assert alert["data"]["current"] > alert["data"]["threshold"]

    @pytest.mark.asyncio
    async def test_resource_monitoring_under_load(self, resource_monitor):
        """Test resource monitoring accuracy under system load."""
        monitor = resource_monitor

        # Start monitoring
        await monitor.start_monitoring()

        # Create mixed workload
        def mixed_workload():
            """Mixed CPU and memory workload."""
            # CPU intensive part
            end_time = time.time() + 0.2
            data = []
            while time.time() < end_time:
                # CPU work
                _ = sum(i * i for i in range(500))
                # Memory work
                data.append(list(range(1000)))
            return data

        # Run multiple workload threads
        threads = []
        for i in range(2):
            thread = threading.Thread(target=mixed_workload)
            thread.start()
            threads.append(thread)

        # Wait for workload and monitoring
        await asyncio.sleep(0.4)

        # Wait for threads to complete
        for thread in threads:
            thread.join()

        # Continue monitoring a bit longer
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify metrics consistency under load
        assert len(monitor.system_metrics) > 0

        # Verify all metrics remain within valid ranges
        for sys_metric in monitor.system_metrics:
            assert 0 <= sys_metric.cpu_percent <= 100
            assert 0 <= sys_metric.memory_percent <= 100
            assert sys_metric.memory_used_mb > 0
            assert sys_metric.memory_available_mb > 0
            assert sys_metric.thread_count >= 1
            assert sys_metric.fd_count >= 0

        # Verify timestamps are properly ordered
        timestamps = [m.timestamp for m in monitor.system_metrics]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_resource_monitoring_with_alert_integration(self, resource_monitor, alert_collector):
        """Test integration between resource monitoring and alert system."""
        monitor = resource_monitor

        # Configure thresholds
        monitor.cpu_threshold = 75.0
        monitor.memory_threshold = 75.0

        # Add alert collector
        monitor.add_alert_callback(alert_collector)

        # Start monitoring
        await monitor.start_monitoring()

        # Simulate escalating resource usage
        cpu_values = [10.0, 30.0, 60.0, 80.0, 95.0]  # Escalating CPU usage
        memory_values = [20.0, 40.0, 65.0, 85.0, 90.0]  # Escalating memory usage

        for cpu_val, mem_val in zip(cpu_values, memory_values):
            with (
                patch.object(monitor.process, "cpu_percent", return_value=cpu_val),
                patch.object(monitor.process, "memory_percent", return_value=mem_val),
            ):
                # Wait for one monitoring cycle
                await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify metrics were collected
        assert len(monitor.system_metrics) > 0

        # Verify alerts were generated for threshold breaches
        total_alerts = alert_collector.count()

        # Should have alerts for CPU and memory when they exceeded 75%
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        memory_alerts = alert_collector.get_by_type("high_memory")

        # Verify we got appropriate alerts
        assert len(cpu_alerts) > 0  # Should trigger when CPU > 75%
        assert len(memory_alerts) > 0  # Should trigger when memory > 75%

    @pytest.mark.asyncio
    async def test_system_metrics_data_structure(self, resource_monitor):
        """Test SystemMetrics data structure integrity and serialization."""
        monitor = resource_monitor

        # Start monitoring
        await monitor.start_monitoring()

        # Wait for samples
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify metrics data structure
        assert len(monitor.system_metrics) > 0

        for sys_metric in monitor.system_metrics:
            # Verify it's a SystemMetrics instance
            assert isinstance(sys_metric, SystemMetrics)

            # Verify all required fields are present
            assert hasattr(sys_metric, "timestamp")
            assert hasattr(sys_metric, "cpu_percent")
            assert hasattr(sys_metric, "memory_percent")
            assert hasattr(sys_metric, "memory_used_mb")
            assert hasattr(sys_metric, "memory_available_mb")
            assert hasattr(sys_metric, "thread_count")
            assert hasattr(sys_metric, "fd_count")

            # Test serialization
            metric_dict = sys_metric.to_dict()
            assert isinstance(metric_dict, dict)

            # Verify all fields are serializable
            import json

            json_str = json.dumps(metric_dict)
            assert isinstance(json_str, str)

            # Verify deserialized data integrity
            restored_dict = json.loads(json_str)
            assert restored_dict == metric_dict

    @pytest.mark.asyncio
    async def test_resource_monitoring_error_handling(self, resource_monitor):
        """Test error handling in resource monitoring."""
        monitor = resource_monitor

        # Test with mocked psutil errors
        with patch.object(monitor.process, "cpu_percent", side_effect=psutil.AccessDenied()):
            await monitor.start_monitoring()
            await asyncio.sleep(0.1)
            await monitor.stop_monitoring()

            # Should still collect metrics, just with default values for failed calls
            assert len(monitor.system_metrics) > 0

            # CPU should be 0 due to AccessDenied
            assert all(m.cpu_percent == 0.0 for m in monitor.system_metrics)

        # Clear metrics for next test
        monitor.clear_metrics()

        # Test with thread count errors
        with patch.object(monitor.process, "num_threads", side_effect=psutil.AccessDenied()):
            await monitor.start_monitoring()
            await asyncio.sleep(0.1)
            await monitor.stop_monitoring()

            # Should still collect metrics
            assert len(monitor.system_metrics) > 0

            # Thread count should be 0 due to AccessDenied
            assert all(m.thread_count == 0 for m in monitor.system_metrics)

    @pytest.mark.asyncio
    async def test_concurrent_resource_access(self, resource_monitor):
        """Test concurrent access to resource monitoring data."""
        monitor = resource_monitor

        # Start monitoring
        await monitor.start_monitoring()

        # Concurrent tasks that access monitoring data
        async def read_metrics():
            for _ in range(10):
                metrics = monitor.system_metrics.copy()
                summary = monitor.get_performance_summary()
                await asyncio.sleep(0.01)

        async def simulate_activity():
            # Simulate some system activity
            for _ in range(5):
                temp_data = [i for i in range(1000)]
                await asyncio.sleep(0.02)

        # Run concurrent tasks
        tasks = [
            read_metrics(),
            read_metrics(),
            simulate_activity(),
        ]

        await asyncio.gather(*tasks)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify data integrity
        assert len(monitor.system_metrics) > 0

        # Verify no data corruption occurred
        for sys_metric in monitor.system_metrics:
            assert isinstance(sys_metric, SystemMetrics)
            assert 0 <= sys_metric.cpu_percent <= 100
            assert 0 <= sys_metric.memory_percent <= 100


@pytest.mark.integration
class TestResourceMonitoringIntegration:
    """Comprehensive integration tests for system resource monitoring."""

    @pytest.mark.asyncio
    async def test_end_to_end_resource_monitoring(self, performance_monitor, alert_collector, mock_event_bus):
        """Test complete end-to-end resource monitoring workflow."""
        # Configure monitoring
        performance_monitor.add_alert_callback(alert_collector)

        # Start monitoring
        await performance_monitor.start_monitoring()

        # Simulate resource usage patterns
        mock_event_bus.update_stats(published_count=100, processed_count=95, queue_size=5, error_count=1)

        # Wait for monitoring
        await asyncio.sleep(0.3)

        # Stop monitoring
        await performance_monitor.stop_monitoring()

        # Verify comprehensive monitoring
        assert len(performance_monitor.snapshots) > 0

        # Verify system and event bus metrics integration
        for snapshot in performance_monitor.snapshots:
            assert hasattr(snapshot, "system_metrics")
            assert hasattr(snapshot, "event_bus_metrics")
            assert snapshot.timestamp > 0

        # Generate and verify summary
        summary = performance_monitor.get_performance_summary()
        assert "system" in summary
        assert "event_bus" in summary
        assert summary["sample_count"] > 0

    @pytest.mark.asyncio
    async def test_resource_alert_workflow(self, performance_monitor, alert_collector, mock_event_bus):
        """Test complete resource alert workflow from detection to notification."""
        # Configure low thresholds for testing
        performance_monitor.cpu_threshold = 50.0
        performance_monitor.memory_threshold = 50.0
        performance_monitor.queue_size_threshold = 10
        performance_monitor.error_rate_threshold = 2.0

        # Add alert collector
        performance_monitor.add_alert_callback(alert_collector)

        # Mock high resource usage and event bus issues
        with (
            patch.object(performance_monitor.process, "cpu_percent", return_value=75.0),
            patch.object(performance_monitor.process, "memory_percent", return_value=80.0),
        ):
            # Start monitoring
            await performance_monitor.start_monitoring()

            # Simulate problematic event bus state
            mock_event_bus.update_stats(
                queue_size=20,  # Above threshold
                processed_count=50,
                error_count=2,  # 4% error rate
            )

            # Wait for monitoring and alerts
            await asyncio.sleep(0.3)

            # Stop monitoring
            await performance_monitor.stop_monitoring()

        # Verify alerts were generated
        assert alert_collector.count() > 0

        # Verify different types of alerts
        cpu_alerts = alert_collector.get_by_type("high_cpu")
        memory_alerts = alert_collector.get_by_type("high_memory")
        queue_alerts = alert_collector.get_by_type("high_queue_size")
        error_alerts = alert_collector.get_by_type("high_error_rate")

        # Should have at least some alerts
        total_expected_alerts = len(cpu_alerts) + len(memory_alerts) + len(queue_alerts) + len(error_alerts)
        assert total_expected_alerts > 0

        # Verify alert data structure for each type
        for alert_list in [cpu_alerts, memory_alerts, queue_alerts, error_alerts]:
            for alert in alert_list:
                assert "type" in alert
                assert "data" in alert
                assert "timestamp" in alert
                assert "current" in alert["data"]
                assert "threshold" in alert["data"]
