# ABOUTME: Performance metrics collection and analysis tools for event bus testing
# ABOUTME: Provides system resource monitoring, performance profiling, and benchmark utilities

import asyncio
import psutil
import os
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, UTC
from collections import defaultdict
import time


@dataclass
class SystemMetrics:
    """System resource metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    thread_count: int
    fd_count: int  # File descriptor count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "thread_count": self.thread_count,
            "fd_count": self.fd_count,
        }


@dataclass
class EventBusMetrics:
    """Event bus specific metrics."""

    timestamp: float
    published_count: int
    processed_count: int
    error_count: int
    timeout_count: int
    dropped_count: int
    queue_size: int
    subscription_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "published_count": self.published_count,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "dropped_count": self.dropped_count,
            "queue_size": self.queue_size,
            "subscription_count": self.subscription_count,
        }


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot."""

    timestamp: float
    system_metrics: SystemMetrics
    event_bus_metrics: EventBusMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system": self.system_metrics.to_dict(),
            "event_bus": self.event_bus_metrics.to_dict(),
        }


class PerformanceMonitor:
    """Monitors system and event bus performance."""

    def __init__(self, event_bus, sample_interval: float = 0.1):
        self.event_bus = event_bus
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())

        # Metrics storage
        self.snapshots: List[PerformanceSnapshot] = []
        self.system_metrics: List[SystemMetrics] = []
        self.event_bus_metrics: List[EventBusMetrics] = []

        # Monitoring control
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Performance baselines
        self._baseline_cpu = 0.0
        self._baseline_memory = 0.0

        # Alerting thresholds
        self.cpu_threshold = 80.0  # percent
        self.memory_threshold = 80.0  # percent
        self.queue_size_threshold = 1000
        self.error_rate_threshold = 5.0  # percent

        # Alert callbacks (can be sync or async)
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], Union[None, Any]]] = []

    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], Union[None, Any]]):
        """Add alert callback function (can be sync or async)."""
        self.alert_callbacks.append(callback)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        try:
            cpu_percent = self.process.cpu_percent()
            # Normalize CPU usage to 0-100% range for consistency
            # psutil.Process.cpu_percent() can return > 100% on multi-core systems
            # when the process uses multiple threads across different cores
            # We cap it at 100% to maintain consistent metric ranges
            cpu_percent = min(cpu_percent, 100.0)
        except psutil.AccessDenied:
            cpu_percent = 0.0

        try:
            thread_count = self.process.num_threads()
        except psutil.AccessDenied:
            thread_count = 0

        try:
            fd_count = self.process.num_fds() if hasattr(self.process, "num_fds") else 0
        except (psutil.AccessDenied, AttributeError):
            fd_count = 0

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_info.rss / 1024 / 1024,
            memory_available_mb=psutil.virtual_memory().available / 1024 / 1024,
            thread_count=thread_count,
            fd_count=fd_count,
        )

    def _collect_event_bus_metrics(self) -> EventBusMetrics:
        """Collect current event bus metrics."""
        stats = self.event_bus.get_statistics()

        return EventBusMetrics(
            timestamp=time.time(),
            published_count=stats["published_count"],
            processed_count=stats["processed_count"],
            error_count=stats["error_count"],
            timeout_count=stats["timeout_count"],
            dropped_count=stats["dropped_count"],
            queue_size=stats["queue_size"],
            subscription_count=stats["total_subscriptions"],
        )

    def _check_thresholds(self, system_metrics: SystemMetrics, event_bus_metrics: EventBusMetrics):
        """Check performance thresholds and trigger alerts."""
        alerts = []

        # CPU threshold
        if system_metrics.cpu_percent > self.cpu_threshold:
            alerts.append(("high_cpu", {"current": system_metrics.cpu_percent, "threshold": self.cpu_threshold}))

        # Memory threshold
        if system_metrics.memory_percent > self.memory_threshold:
            alerts.append(
                ("high_memory", {"current": system_metrics.memory_percent, "threshold": self.memory_threshold})
            )

        # Queue size threshold
        if event_bus_metrics.queue_size > self.queue_size_threshold:
            alerts.append(
                ("high_queue_size", {"current": event_bus_metrics.queue_size, "threshold": self.queue_size_threshold})
            )

        # Error rate threshold
        if event_bus_metrics.processed_count > 0:
            error_rate = (event_bus_metrics.error_count / event_bus_metrics.processed_count) * 100
            if error_rate > self.error_rate_threshold:
                alerts.append(("high_error_rate", {"current": error_rate, "threshold": self.error_rate_threshold}))

        # Trigger alerts
        for alert_type, alert_data in alerts:
            for callback in self.alert_callbacks:
                try:
                    # Handle both sync and async callbacks
                    result = callback(alert_type, alert_data)
                    # If callback returns a coroutine, schedule it
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    print(f"Alert callback error: {e}")

    async def start_monitoring(self):
        """Start performance monitoring."""
        if self._is_monitoring:
            return

        self._is_monitoring = True

        # Collect baseline metrics with error handling
        try:
            self._baseline_cpu = self.process.cpu_percent()
            # Normalize baseline CPU the same way as monitoring data
            self._baseline_cpu = min(self._baseline_cpu, 100.0)
        except psutil.AccessDenied:
            self._baseline_cpu = 0.0

        try:
            self._baseline_memory = self.process.memory_percent()
        except psutil.AccessDenied:
            self._baseline_memory = 0.0

        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                # Collect metrics
                system_metrics = self._collect_system_metrics()
                event_bus_metrics = self._collect_event_bus_metrics()

                # Store metrics
                self.system_metrics.append(system_metrics)
                self.event_bus_metrics.append(event_bus_metrics)

                # Create snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(), system_metrics=system_metrics, event_bus_metrics=event_bus_metrics
                )
                self.snapshots.append(snapshot)

                # Check thresholds
                self._check_thresholds(system_metrics, event_bus_metrics)

                # Sleep until next sample
                await asyncio.sleep(self.sample_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(self.sample_interval)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.snapshots:
            return {}

        # System metrics summary
        cpu_values = [s.system_metrics.cpu_percent for s in self.snapshots]
        memory_values = [s.system_metrics.memory_percent for s in self.snapshots]
        memory_used_values = [s.system_metrics.memory_used_mb for s in self.snapshots]

        # Event bus metrics summary
        queue_sizes = [s.event_bus_metrics.queue_size for s in self.snapshots]

        return {
            "monitoring_duration": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "sample_count": len(self.snapshots),
            "system": {
                "cpu": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values),
                    "baseline": self._baseline_cpu,
                },
                "memory": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": sum(memory_values) / len(memory_values),
                    "baseline": self._baseline_memory,
                },
                "memory_used_mb": {
                    "min": min(memory_used_values),
                    "max": max(memory_used_values),
                    "avg": sum(memory_used_values) / len(memory_used_values),
                },
            },
            "event_bus": {
                "queue_size": {
                    "min": min(queue_sizes),
                    "max": max(queue_sizes),
                    "avg": sum(queue_sizes) / len(queue_sizes),
                },
                "final_stats": self.snapshots[-1].event_bus_metrics.to_dict(),
            },
        }

    def export_metrics(self, filename: str):
        """Export collected metrics to file."""
        import json

        data = {
            "monitoring_config": {
                "sample_interval": self.sample_interval,
                "thresholds": {
                    "cpu": self.cpu_threshold,
                    "memory": self.memory_threshold,
                    "queue_size": self.queue_size_threshold,
                    "error_rate": self.error_rate_threshold,
                },
            },
            "summary": self.get_performance_summary(),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.snapshots.clear()
        self.system_metrics.clear()
        self.event_bus_metrics.clear()


class EventBusProfiler:
    """Profiles event bus performance and identifies bottlenecks."""

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.monitor = PerformanceMonitor(event_bus)

        # Profiling data
        self.handler_times: Dict[str, List[float]] = defaultdict(list)
        self.publish_times: List[float] = []
        self.processing_times: List[float] = []

        # Performance benchmarks
        self.benchmarks: Dict[str, float] = {}

    async def profile_publishing_performance(self, num_events: int = 100) -> Dict[str, Any]:
        """Profile event publishing performance."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType
        from core.models.event.event_priority import EventPriority

        # Reduced for better performance in benchmarks
        total_published = 0

        for i in range(num_events):
            event = BaseEvent(
                event_type=EventType.TRADE, source="profiler", data={"test": i}, priority=EventPriority.NORMAL
            )
            await self.event_bus.publish(event)
            total_published += 1

        return {
            "num_events": num_events,
            "total_published": total_published,
        }

    async def profile_handler_performance(self, handler_func: Callable, num_events: int = 50) -> Dict[str, Any]:
        """Profile event handler performance."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType
        from core.models.event.event_priority import EventPriority

        processed_count = 0

        def counting_handler(event):
            nonlocal processed_count
            result = handler_func(event)
            processed_count += 1
            return result

        # Subscribe counting handler
        subscription_id = self.event_bus.subscribe(EventType.TRADE, counting_handler)

        try:
            # Publish events
            for i in range(num_events):
                event = BaseEvent(
                    event_type=EventType.TRADE, source="profiler", data={"test": i}, priority=EventPriority.NORMAL
                )
                await self.event_bus.publish(event)

            # Wait for processing
            await asyncio.sleep(0.5)  # Reduced wait time

            return {
                "num_events": num_events,
                "processed_count": processed_count,
            }

        finally:
            self.event_bus.unsubscribe(subscription_id)

    async def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        benchmarks = {}

        # Start monitoring
        await self.monitor.start_monitoring()

        try:
            # Publishing benchmark
            benchmarks["publishing"] = await self.profile_publishing_performance(100)

            # Simple handler benchmark
            def simple_handler(event):
                return event.data

            benchmarks["simple_handler"] = await self.profile_handler_performance(simple_handler, 50)

            # Complex handler benchmark
            def complex_handler(event):
                # Simulate complex processing
                import json

                data = json.dumps(event.data)
                parsed = json.loads(data)
                return len(str(parsed))

            benchmarks["complex_handler"] = await self.profile_handler_performance(complex_handler, 30)

            # Wait for final processing
            await asyncio.sleep(0.5)  # Reduced wait time

            # System performance summary
            benchmarks["system_performance"] = self.monitor.get_performance_summary()

        finally:
            await self.monitor.stop_monitoring()

        return benchmarks

    def generate_performance_report(self, benchmarks: Dict[str, Any]) -> str:
        """Generate performance report from benchmarks."""
        report = []
        report.append("# Event Bus Performance Report")
        report.append("=" * 50)
        report.append("")

        # Publishing performance
        if "publishing" in benchmarks:
            pub = benchmarks["publishing"]
            report.append("## Publishing Performance")
            report.append(f"Events: {pub['num_events']}")
            report.append(f"Total Published: {pub['total_published']}")
            
            # Calculate basic throughput if we have the data
            if "system_performance" in benchmarks:
                duration = benchmarks["system_performance"].get("monitoring_duration", 1.0)
                throughput = pub["total_published"] / duration if duration > 0 else 0
                report.append(f"Throughput: {throughput:.2f} events/sec")
            report.append("")

        # Handler performance
        for handler_type in ["simple_handler", "complex_handler"]:
            if handler_type in benchmarks:
                handler = benchmarks[handler_type]
                report.append(f"## {handler_type.title().replace('_', ' ')} Performance")
                report.append(f"Events: {handler['num_events']}")
                report.append(f"Processed: {handler['processed_count']}")
                
                # Calculate processing rate
                if handler['num_events'] > 0:
                    processing_rate = (handler['processed_count'] / handler['num_events']) * 100
                    report.append(f"Processing Rate: {processing_rate:.1f}%")
                report.append("")

        # System performance
        if "system_performance" in benchmarks:
            sys_perf = benchmarks["system_performance"]
            report.append("## System Performance")
            report.append(f"Monitoring Duration: {sys_perf['monitoring_duration']:.2f}s")
            report.append(f"Sample Count: {sys_perf['sample_count']}")
            
            if "system" in sys_perf and sys_perf["system"]:
                system = sys_perf["system"]
                if "cpu" in system:
                    report.append(
                        f"CPU Usage: {system['cpu']['avg']:.1f}% (peak: {system['cpu']['max']:.1f}%)"
                    )
                if "memory" in system:
                    report.append(
                        f"Memory Usage: {system['memory']['avg']:.1f}% (peak: {system['memory']['max']:.1f}%)"
                    )
            
            if "event_bus" in sys_perf and sys_perf["event_bus"]:
                event_bus = sys_perf["event_bus"]
                if "queue_size" in event_bus:
                    report.append(f"Peak Queue Size: {event_bus['queue_size']['max']}")
            report.append("")

        return "\n".join(report)


def create_alert_logger() -> Callable[[str, Dict[str, Any]], None]:
    """Create an alert logger function."""

    def alert_logger(alert_type: str, alert_data: Dict[str, Any]):
        timestamp = datetime.now(UTC).isoformat()
        print(f"[{timestamp}] ALERT: {alert_type.upper()} - {alert_data}")

    return alert_logger


async def run_performance_analysis(event_bus, duration: float = 30.0) -> Dict[str, Any]:
    """Run comprehensive performance analysis."""
    profiler = EventBusProfiler(event_bus)

    # Add alert logger
    profiler.monitor.add_alert_callback(create_alert_logger())

    # Run benchmark suite
    benchmarks = await profiler.run_benchmark_suite()

    # Generate report
    report = profiler.generate_performance_report(benchmarks)

    return {"benchmarks": benchmarks, "report": report, "profiler": profiler}
