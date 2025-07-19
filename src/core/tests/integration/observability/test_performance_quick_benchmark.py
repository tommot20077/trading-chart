# ABOUTME: Quick performance benchmark tests for observability system without timeouts
# ABOUTME: Tests core notification handling, logging, and monitoring performance with simplified scenarios

import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import Mock

# Quick performance benchmark tests - simplified performance scenarios

from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from ...fixtures.performance_metrics import PerformanceMonitor


@pytest_asyncio.fixture
async def benchmark_notification_handler():
    """Create notification handler optimized for benchmarking."""
    handler = InMemoryNotificationHandler(
        max_queue_size=1000,
        max_history_size=2000,
        history_retention_hours=1.0,
        cleanup_interval_seconds=60.0,
        max_retry_attempts=1,  # Reduced for faster testing
        simulate_failure_rate=0.0,
        processing_delay_seconds=0.0,
    )

    try:
        yield handler
    finally:
        await handler.close()


@pytest.fixture
def mock_event_bus_for_benchmark():
    """Create optimized mock event bus for benchmarking."""
    mock_bus = Mock()
    mock_bus.get_statistics.return_value = {
        "published_count": 100,
        "processed_count": 95,
        "error_count": 2,
        "timeout_count": 1,
        "dropped_count": 0,
        "queue_size": 5,
        "total_subscriptions": 3,
    }
    return mock_bus


@pytest_asyncio.fixture
async def performance_monitor_quick(mock_event_bus_for_benchmark):
    """Create performance monitor for quick benchmarking."""
    monitor = PerformanceMonitor(
        event_bus=mock_event_bus_for_benchmark,
        sample_interval=0.1,  # Fast sampling
    )

    # Set high thresholds to avoid false alerts
    monitor.cpu_threshold = 95.0
    monitor.memory_threshold = 95.0
    monitor.queue_size_threshold = 500
    monitor.error_rate_threshold = 10.0

    yield monitor

    await monitor.stop_monitoring()


class TestQuickPerformanceBenchmarks:
    """Quick performance benchmarks for observability system."""

    @pytest.mark.asyncio
    async def test_notification_handler_throughput(self, benchmark_notification_handler):
        """Test notification handler throughput with minimal latency."""
        handler = benchmark_notification_handler

        # Test data
        test_notifications = [
            {
                "id": f"test-{i}",
                "rule_name": "benchmark_test",
                "severity": "info",
                "message": f"Benchmark notification {i}",
                "metadata": {"test_id": i},
            }
            for i in range(50)  # Reduced count for speed
        ]

        # Measure throughput
        start_time = time.time()

        for notification in test_notifications:
            await handler.send_notification(notification)

        # Wait for processing
        await asyncio.sleep(0.5)

        end_time = time.time()
        processing_time = end_time - start_time

        # Get statistics
        stats = await handler.get_notification_statistics()

        # Calculate metrics
        throughput = len(test_notifications) / processing_time

        # Assert performance criteria
        assert throughput > 10, f"Throughput too low: {throughput:.2f} notifications/sec"
        assert stats["total_sent"] >= 45, f"Too few notifications sent: {stats['total_sent']}"
        assert processing_time < 10.0, f"Processing time too high: {processing_time:.2f}s"

        # Performance report
        performance_data = {
            "notifications_count": len(test_notifications),
            "processing_time_seconds": processing_time,
            "throughput_per_second": throughput,
            "success_rate_percent": (stats["total_sent"] / len(test_notifications)) * 100,
            "statistics": stats,
        }

        print("\\nNotification Handler Performance Report:")
        print(f"Count: {performance_data['notifications_count']}")
        print(f"Time: {performance_data['processing_time_seconds']:.3f}s")
        print(f"Throughput: {performance_data['throughput_per_second']:.2f} notifications/sec")
        print(f"Success Rate: {performance_data['success_rate_percent']:.1f}%")

        return performance_data

    @pytest.mark.asyncio
    async def test_performance_monitoring_overhead(self, performance_monitor_quick):
        """Test performance monitoring system overhead."""
        monitor = performance_monitor_quick

        # Start monitoring
        start_time = time.time()
        await monitor.start_monitoring()

        # Let it collect samples
        await asyncio.sleep(1.0)  # 1 second of monitoring

        # Stop monitoring
        await monitor.stop_monitoring()
        end_time = time.time()

        # Get performance summary
        summary = monitor.get_performance_summary()

        monitoring_duration = end_time - start_time
        sample_count = summary.get("sample_count", 0)

        # Calculate overhead metrics
        overhead_per_sample = monitoring_duration / max(sample_count, 1)
        samples_per_second = sample_count / monitoring_duration

        # Assert performance criteria (relaxed thresholds for testing environment)
        assert sample_count > 5, f"Too few samples collected: {sample_count}"
        assert overhead_per_sample < 0.15, f"High overhead per sample: {overhead_per_sample:.4f}s"
        assert samples_per_second > 5, f"Low sampling rate: {samples_per_second:.2f} samples/sec"

        # Performance report
        performance_data = {
            "monitoring_duration_seconds": monitoring_duration,
            "samples_collected": sample_count,
            "overhead_per_sample_ms": overhead_per_sample * 1000,
            "sampling_rate_per_second": samples_per_second,
            "summary": summary,
        }

        print("\\nPerformance Monitor Overhead Report:")
        print(f"Duration: {performance_data['monitoring_duration_seconds']:.3f}s")
        print(f"Samples: {performance_data['samples_collected']}")
        print(f"Overhead per sample: {performance_data['overhead_per_sample_ms']:.2f}ms")
        print(f"Sampling rate: {performance_data['sampling_rate_per_second']:.2f} samples/sec")

        return performance_data

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, benchmark_notification_handler):
        """Test memory usage stability under load."""
        handler = benchmark_notification_handler

        # Initial memory check
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Send notifications in batches
        batch_size = 20
        num_batches = 3

        memory_measurements = [initial_memory]

        for batch in range(num_batches):
            # Send batch of notifications
            for i in range(batch_size):
                await handler.send_notification(
                    {
                        "id": f"batch-{batch}-{i}",
                        "rule_name": "memory_test",
                        "severity": "info",
                        "message": f"Memory test notification {batch}-{i}",
                        "metadata": {"batch": batch, "item": i},
                    }
                )

            # Wait for processing
            await asyncio.sleep(0.2)

            # Measure memory
            current_memory = process.memory_info().rss
            memory_measurements.append(current_memory)

        # Calculate memory metrics
        max_memory = max(memory_measurements)
        memory_growth = max_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)

        # Get handler statistics
        stats = await handler.get_notification_statistics()

        # Assert memory criteria
        assert memory_growth_mb < 50, f"Memory growth too high: {memory_growth_mb:.2f}MB"
        assert stats["total_sent"] >= batch_size * num_batches * 0.9, "Too many failed notifications"

        # Performance report
        performance_data = {
            "initial_memory_mb": initial_memory / (1024 * 1024),
            "max_memory_mb": max_memory / (1024 * 1024),
            "memory_growth_mb": memory_growth_mb,
            "notifications_processed": stats["total_sent"],
            "memory_per_notification_kb": (memory_growth / max(stats["total_sent"], 1)) / 1024,
            "statistics": stats,
        }

        print("\\nMemory Usage Stability Report:")
        print(f"Initial Memory: {performance_data['initial_memory_mb']:.2f}MB")
        print(f"Max Memory: {performance_data['max_memory_mb']:.2f}MB")
        print(f"Growth: {performance_data['memory_growth_mb']:.2f}MB")
        print(f"Per Notification: {performance_data['memory_per_notification_kb']:.3f}KB")

        return performance_data


@pytest.mark.integration
class TestPerformanceBenchmarkSuite:
    """Complete performance benchmark suite for observability system."""

    @pytest.mark.asyncio
    async def test_comprehensive_performance_benchmark(
        self, benchmark_notification_handler, performance_monitor_quick, mock_event_bus_for_benchmark
    ):
        """Run comprehensive performance benchmark combining all metrics."""

        # Initialize components
        handler = benchmark_notification_handler
        monitor = performance_monitor_quick

        # Start performance monitoring
        await monitor.start_monitoring()

        try:
            # Phase 1: Notification throughput test
            notification_start = time.time()

            notifications = [
                {
                    "id": f"comprehensive-{i}",
                    "rule_name": "comprehensive_test",
                    "severity": "info" if i % 2 == 0 else "warning",
                    "message": f"Comprehensive test notification {i}",
                    "metadata": {"phase": "throughput", "index": i},
                }
                for i in range(30)  # Moderate count for comprehensive test
            ]

            for notification in notifications:
                await handler.send_notification(notification)

            # Wait for processing
            await asyncio.sleep(0.5)
            notification_end = time.time()

            # Phase 2: Concurrent operations test
            concurrent_start = time.time()

            # Simulate concurrent operations
            tasks = []
            for i in range(5):
                task = asyncio.create_task(
                    handler.send_notification(
                        {
                            "id": f"concurrent-{i}",
                            "rule_name": "concurrent_test",
                            "severity": "info",
                            "message": f"Concurrent notification {i}",
                            "metadata": {"phase": "concurrent", "index": i},
                        }
                    )
                )
                tasks.append(task)

            await asyncio.gather(*tasks)
            await asyncio.sleep(0.3)
            concurrent_end = time.time()

            # Phase 3: Statistics gathering test
            stats_start = time.time()

            stats = await handler.get_notification_statistics()
            history = await handler.get_notification_history(limit=10)
            pending = await handler.get_pending_notifications()

            stats_end = time.time()

        finally:
            # Stop monitoring
            await monitor.stop_monitoring()

        # Compile performance metrics
        monitor_summary = monitor.get_performance_summary()

        comprehensive_metrics = {
            "notification_throughput": {
                "count": len(notifications),
                "duration_seconds": notification_end - notification_start,
                "throughput_per_second": len(notifications) / (notification_end - notification_start),
            },
            "concurrent_operations": {
                "count": len(tasks),
                "duration_seconds": concurrent_end - concurrent_start,
                "throughput_per_second": len(tasks) / (concurrent_end - concurrent_start),
            },
            "statistics_performance": {
                "operations": 3,  # stats, history, pending
                "duration_seconds": stats_end - stats_start,
                "operations_per_second": 3 / (stats_end - stats_start),
            },
            "system_performance": monitor_summary,
            "final_statistics": stats,
            "history_count": len(history),
            "pending_count": len(pending),
        }

        # Performance assertions
        assert comprehensive_metrics["notification_throughput"]["throughput_per_second"] > 10
        assert comprehensive_metrics["concurrent_operations"]["throughput_per_second"] > 5
        assert comprehensive_metrics["statistics_performance"]["operations_per_second"] > 10
        assert stats["total_sent"] >= 30  # Most notifications should succeed

        # Generate performance report
        report = []
        report.append("=== Comprehensive Performance Benchmark Report ===")
        report.append(
            f"Notification Throughput: {comprehensive_metrics['notification_throughput']['throughput_per_second']:.2f} ops/sec"
        )
        report.append(
            f"Concurrent Operations: {comprehensive_metrics['concurrent_operations']['throughput_per_second']:.2f} ops/sec"
        )
        report.append(
            f"Statistics Performance: {comprehensive_metrics['statistics_performance']['operations_per_second']:.2f} ops/sec"
        )
        report.append(f"Success Rate: {(stats['total_sent'] / (len(notifications) + len(tasks))) * 100:.1f}%")
        report.append(f"System Samples: {monitor_summary.get('sample_count', 0)}")

        print("\\n" + "\\n".join(report))

        return comprehensive_metrics
