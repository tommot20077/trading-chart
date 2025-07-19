# ABOUTME: Integration tests for PerformanceMonitor and EventBus monitoring capabilities
# ABOUTME: Validates system resource monitoring, event bus metrics collection, and real-time performance tracking

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock

# Previously skipped due to timeout issues - now fixed

from ...fixtures.performance_metrics import PerformanceMonitor, EventBusProfiler


@pytest_asyncio.fixture
async def realistic_event_bus():
    """Create a more realistic mock event bus with dynamic statistics."""
    mock_bus = Mock()

    # Statistics that change over time
    stats = {
        "published_count": 0,
        "processed_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "dropped_count": 0,
        "queue_size": 0,
        "total_subscriptions": 0,
    }

    def get_stats():
        return stats.copy()

    mock_bus.get_statistics.side_effect = get_stats

    # Add async publish method
    async def async_publish(event):
        stats["published_count"] += 1
        stats["queue_size"] += 1
        # Simulate processing delay
        await asyncio.sleep(0.01)
        stats["processed_count"] += 1
        stats["queue_size"] = max(0, stats["queue_size"] - 1)

    mock_bus.publish = async_publish

    # Add subscribe/unsubscribe methods
    def subscribe(event_type, handler):
        stats["total_subscriptions"] += 1
        return f"sub_{stats['total_subscriptions']}"

    def unsubscribe(subscription_id):
        stats["total_subscriptions"] = max(0, stats["total_subscriptions"] - 1)

    mock_bus.subscribe = subscribe
    mock_bus.unsubscribe = unsubscribe

    yield mock_bus


@pytest_asyncio.fixture
async def performance_monitor_with_realistic_bus(realistic_event_bus):
    """Create PerformanceMonitor with realistic event bus."""
    monitor = PerformanceMonitor(
        event_bus=realistic_event_bus,
        sample_interval=0.05,  # Fast sampling for tests
    )

    # Configure test-appropriate thresholds
    monitor.cpu_threshold = 95.0  # High threshold to avoid false alarms
    monitor.memory_threshold = 95.0
    monitor.queue_size_threshold = 100
    monitor.error_rate_threshold = 10.0

    yield monitor

    # Cleanup
    await monitor.stop_monitoring()
    monitor.clear_metrics()


@pytest_asyncio.fixture
async def event_bus_profiler(realistic_event_bus):
    """Create EventBusProfiler for testing."""
    profiler = EventBusProfiler(realistic_event_bus)
    yield profiler

    # Cleanup
    await profiler.monitor.stop_monitoring()


class TestPerformanceEventBusIntegration:
    """Integration tests for PerformanceMonitor with EventBus."""

    @pytest.mark.asyncio
    async def test_performance_monitor_basic_metrics_collection(self, performance_monitor_with_realistic_bus):
        """Test basic performance metrics collection from EventBus."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()

        # Wait for some samples
        await asyncio.sleep(0.3)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify metrics were collected
        assert len(monitor.snapshots) > 0
        assert len(monitor.system_metrics) > 0
        assert len(monitor.event_bus_metrics) > 0

        # Verify snapshot structure
        snapshot = monitor.snapshots[0]
        assert hasattr(snapshot, "timestamp")
        assert hasattr(snapshot, "system_metrics")
        assert hasattr(snapshot, "event_bus_metrics")

        # Verify system metrics
        sys_metrics = snapshot.system_metrics
        assert sys_metrics.cpu_percent >= 0
        assert sys_metrics.memory_percent >= 0
        assert sys_metrics.memory_used_mb >= 0
        assert sys_metrics.thread_count >= 0

        # Verify event bus metrics
        eb_metrics = snapshot.event_bus_metrics
        assert eb_metrics.published_count >= 0
        assert eb_metrics.processed_count >= 0
        assert eb_metrics.queue_size >= 0

    @pytest.mark.asyncio
    async def test_real_time_metrics_updates(self, performance_monitor_with_realistic_bus, realistic_event_bus):
        """Test real-time updates of event bus metrics during operation."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()

        # Simulate event bus activity
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType
        from core.models.event.event_priority import EventPriority

        # Create some events
        events = []
        for i in range(5):
            event = BaseEvent(
                event_type=EventType.TRADE, source="test", data={"test": i}, priority=EventPriority.NORMAL
            )
            events.append(event)

        # Publish events to trigger statistics changes
        for event in events:
            await realistic_event_bus.publish(event)

        # Wait for monitoring to capture changes
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify that metrics show activity
        assert len(monitor.snapshots) > 0

        # Check if any snapshots captured the activity
        final_metrics = monitor.snapshots[-1].event_bus_metrics
        assert final_metrics.published_count >= 5
        assert final_metrics.processed_count >= 5

    @pytest.mark.asyncio
    async def test_system_resource_monitoring_accuracy(self, performance_monitor_with_realistic_bus):
        """Test accuracy of system resource monitoring."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()

        # Wait for samples
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify resource metrics are reasonable
        assert len(monitor.system_metrics) > 0

        for sys_metric in monitor.system_metrics:
            # CPU percentage should be between 0 and 100
            assert 0 <= sys_metric.cpu_percent <= 100

            # Memory percentage should be between 0 and 100
            assert 0 <= sys_metric.memory_percent <= 100

            # Memory used should be positive
            assert sys_metric.memory_used_mb > 0

            # Memory available should be positive
            assert sys_metric.memory_available_mb > 0

            # Thread count should be positive
            assert sys_metric.thread_count > 0

            # File descriptor count should be non-negative
            assert sys_metric.fd_count >= 0

    @pytest.mark.asyncio
    async def test_performance_summary_generation(self, performance_monitor_with_realistic_bus):
        """Test performance summary statistics generation."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()

        # Wait for samples
        await asyncio.sleep(0.3)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Generate summary
        summary = monitor.get_performance_summary()

        # Verify summary structure
        assert "monitoring_duration" in summary
        assert "sample_count" in summary
        assert "system" in summary
        assert "event_bus" in summary

        # Verify system summary
        sys_summary = summary["system"]
        assert "cpu" in sys_summary
        assert "memory" in sys_summary
        assert "memory_used_mb" in sys_summary

        # Verify each metric has min, max, avg
        for metric_name in ["cpu", "memory", "memory_used_mb"]:
            metric = sys_summary[metric_name]
            assert "min" in metric
            assert "max" in metric
            assert "avg" in metric

            # Verify logical relationships
            assert metric["min"] <= metric["avg"] <= metric["max"]

        # Verify event bus summary
        eb_summary = summary["event_bus"]
        assert "queue_size" in eb_summary
        assert "final_stats" in eb_summary

        # Verify monitoring duration is positive
        assert summary["monitoring_duration"] > 0
        assert summary["sample_count"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_monitoring_stability(self, performance_monitor_with_realistic_bus, realistic_event_bus):
        """Test monitoring stability under concurrent operations."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()

        # Create concurrent tasks
        async def publish_events():
            from core.models.data.event import BaseEvent
            from core.models.event.event_type import EventType
            from core.models.event.event_priority import EventPriority

            for i in range(10):
                event = BaseEvent(
                    event_type=EventType.TRADE,
                    source="concurrent_test",
                    data={"concurrent": i},
                    priority=EventPriority.NORMAL,
                )
                await realistic_event_bus.publish(event)
                await asyncio.sleep(0.01)

        # Run multiple concurrent publishing tasks
        tasks = [publish_events() for _ in range(3)]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Wait for monitoring to capture all activity
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify monitoring remained stable
        assert len(monitor.snapshots) > 0
        assert len(monitor.system_metrics) > 0
        assert len(monitor.event_bus_metrics) > 0

        # Verify metrics captured the concurrent activity
        final_snapshot = monitor.snapshots[-1]
        assert final_snapshot.event_bus_metrics.published_count >= 30

    @pytest.mark.asyncio
    async def test_monitoring_resource_cleanup(self, performance_monitor_with_realistic_bus):
        """Test proper resource cleanup after monitoring."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()
        assert monitor._is_monitoring is True
        assert monitor._monitoring_task is not None

        # Wait briefly
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify cleanup
        assert monitor._is_monitoring is False
        assert monitor._monitoring_task.cancelled() or monitor._monitoring_task.done()

        # Verify we can restart monitoring
        await monitor.start_monitoring()
        assert monitor._is_monitoring is True
        await monitor.stop_monitoring()
        assert monitor._is_monitoring is False

    @pytest.mark.asyncio
    async def test_metrics_export_functionality(self, performance_monitor_with_realistic_bus, tmp_path):
        """Test metrics export to file functionality."""
        monitor = performance_monitor_with_realistic_bus

        # Start monitoring
        await monitor.start_monitoring()

        # Wait for samples
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Export metrics
        export_file = tmp_path / "test_metrics.json"
        monitor.export_metrics(str(export_file))

        # Verify file was created
        assert export_file.exists()

        # Load and verify exported data
        import json

        with open(export_file) as f:
            exported_data = json.load(f)

        # Verify structure
        assert "monitoring_config" in exported_data
        assert "summary" in exported_data
        assert "snapshots" in exported_data

        # Verify monitoring config
        config = exported_data["monitoring_config"]
        assert "sample_interval" in config
        assert "thresholds" in config

        # Verify thresholds
        thresholds = config["thresholds"]
        assert "cpu" in thresholds
        assert "memory" in thresholds
        assert "queue_size" in thresholds
        assert "error_rate" in thresholds

        # Verify snapshots are properly serialized
        snapshots = exported_data["snapshots"]
        assert len(snapshots) > 0

        for snapshot in snapshots:
            assert "timestamp" in snapshot
            assert "system" in snapshot
            assert "event_bus" in snapshot


class TestEventBusProfilerIntegration:
    """Integration tests for EventBusProfiler with PerformanceMonitor."""

    @pytest.mark.asyncio
    async def test_publishing_performance_profiling(self, event_bus_profiler):
        """Test event publishing performance profiling."""
        # Profile publishing performance
        results = await event_bus_profiler.profile_publishing_performance(num_events=20)

        # Verify results
        assert "num_events" in results
        assert "total_published" in results
        assert results["num_events"] == 20
        assert results["total_published"] == 20

    @pytest.mark.asyncio
    async def test_handler_performance_profiling(self, event_bus_profiler):
        """Test event handler performance profiling."""

        # Simple handler
        def simple_handler(event):
            return event.data

        # Profile handler performance
        results = await event_bus_profiler.profile_handler_performance(simple_handler, num_events=10)

        # Verify results
        assert "num_events" in results
        assert "processed_count" in results
        assert results["num_events"] == 10
        # Note: processed_count might be less due to async processing timing

    @pytest.mark.asyncio
    async def test_benchmark_suite_execution(self, event_bus_profiler):
        """Test complete benchmark suite execution."""
        # Run benchmark suite
        benchmarks = await event_bus_profiler.run_benchmark_suite()

        # Verify benchmark results
        assert "publishing" in benchmarks
        assert "simple_handler" in benchmarks
        assert "complex_handler" in benchmarks
        assert "system_performance" in benchmarks

        # Verify publishing benchmark
        pub_benchmark = benchmarks["publishing"]
        assert pub_benchmark["num_events"] == 100
        assert pub_benchmark["total_published"] == 100

        # Verify handler benchmarks
        for handler_type in ["simple_handler", "complex_handler"]:
            handler_benchmark = benchmarks[handler_type]
            assert "num_events" in handler_benchmark
            assert "processed_count" in handler_benchmark

        # Verify system performance data
        sys_perf = benchmarks["system_performance"]
        assert "monitoring_duration" in sys_perf
        assert "sample_count" in sys_perf
        assert "system" in sys_perf
        assert "event_bus" in sys_perf

    @pytest.mark.asyncio
    async def test_performance_report_generation(self, event_bus_profiler):
        """Test performance report generation."""
        # Run benchmarks
        benchmarks = await event_bus_profiler.run_benchmark_suite()

        # Generate report
        report = event_bus_profiler.generate_performance_report(benchmarks)

        # Verify report contains expected sections
        assert "Event Bus Performance Report" in report
        assert "Publishing Performance" in report
        assert "Simple Handler Performance" in report
        assert "Complex Handler Performance" in report
        assert "System Performance" in report

        # Verify report contains metrics
        assert "Events:" in report
        assert "Throughput:" in report
        assert "CPU Usage:" in report
        assert "Memory Usage:" in report


@pytest.mark.integration
class TestPerformanceMonitoringIntegration:
    """Comprehensive integration tests for performance monitoring system."""

    @pytest.mark.asyncio
    async def test_end_to_end_performance_monitoring(self, performance_monitor, alert_collector, mock_event_bus):
        """Test complete end-to-end performance monitoring workflow."""
        # Add alert collector to monitor
        performance_monitor.add_alert_callback(alert_collector)

        # Start monitoring
        await performance_monitor.start_monitoring()

        # Simulate some activity
        mock_event_bus.update_stats(published_count=50, processed_count=45, queue_size=5, error_count=2)

        # Wait for monitoring
        await asyncio.sleep(0.2)

        # Stop monitoring
        await performance_monitor.stop_monitoring()

        # Verify monitoring captured metrics
        assert len(performance_monitor.snapshots) > 0

        # Generate summary
        summary = performance_monitor.get_performance_summary()
        assert summary["sample_count"] > 0
        assert summary["monitoring_duration"] > 0

        # Verify system metrics were captured
        assert "system" in summary
        assert "event_bus" in summary

    @pytest.mark.asyncio
    async def test_integration_with_alert_system(self, performance_monitor, alert_collector, mock_event_bus):
        """Test integration between performance monitoring and alert system."""
        # Configure low thresholds for testing
        performance_monitor.cpu_threshold = 1.0  # Very low to trigger alerts
        performance_monitor.memory_threshold = 1.0
        performance_monitor.queue_size_threshold = 1
        performance_monitor.error_rate_threshold = 1.0

        # Add alert collector
        performance_monitor.add_alert_callback(alert_collector)

        # Start monitoring
        await performance_monitor.start_monitoring()

        # Simulate conditions that should trigger alerts
        mock_event_bus.update_stats(
            queue_size=10,  # Above threshold
            processed_count=10,
            error_count=2,  # 20% error rate
        )

        # Wait for monitoring and alert processing
        await asyncio.sleep(0.3)

        # Stop monitoring
        await performance_monitor.stop_monitoring()

        # Verify alerts were generated
        # Note: CPU/Memory alerts might not trigger in test environment
        # but queue_size and error_rate should trigger
        assert alert_collector.count() > 0

        # Check for specific alert types
        queue_alerts = alert_collector.get_by_type("high_queue_size")
        error_alerts = alert_collector.get_by_type("high_error_rate")

        assert len(queue_alerts) > 0 or len(error_alerts) > 0
