# ABOUTME: Load testing fixtures and utilities for event bus testing
# ABOUTME: Provides load generators, metrics collectors, and test data factories

import asyncio
import random
import statistics
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, UTC
from dataclasses import dataclass, field
import time

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""

    # Throughput metrics
    events_published: int = 0
    events_processed: int = 0
    events_dropped: int = 0
    events_errored: int = 0
    events_timed_out: int = 0

    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration: float = 0.0

    # Latency metrics (in milliseconds)
    latencies: List[float] = field(default_factory=list)
    min_latency: float = 0.0
    max_latency: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    # Performance metrics
    publishing_rate: float = 0.0  # events/second
    processing_rate: float = 0.0  # events/second
    error_rate: float = 0.0  # percentage
    timeout_rate: float = 0.0  # percentage

    # Resource metrics
    peak_queue_size: int = 0
    avg_queue_size: float = 0.0
    queue_size_samples: List[int] = field(default_factory=list)

    def calculate_derived_metrics(self):
        """Calculate derived metrics from raw data."""
        if self.total_duration > 0:
            self.publishing_rate = self.events_published / self.total_duration
            self.processing_rate = self.events_processed / self.total_duration

        if self.events_published > 0:
            self.error_rate = (self.events_errored / self.events_published) * 100
            self.timeout_rate = (self.events_timed_out / self.events_published) * 100

        if self.latencies:
            self.min_latency = min(self.latencies)
            self.max_latency = max(self.latencies)
            self.avg_latency = statistics.mean(self.latencies)
            self.p95_latency = statistics.quantiles(self.latencies, n=20)[18]  # 95th percentile
            self.p99_latency = statistics.quantiles(self.latencies, n=100)[98]  # 99th percentile

        if self.queue_size_samples:
            self.peak_queue_size = max(self.queue_size_samples)
            self.avg_queue_size = statistics.mean(self.queue_size_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "throughput": {
                "events_published": self.events_published,
                "events_processed": self.events_processed,
                "events_dropped": self.events_dropped,
                "events_errored": self.events_errored,
                "events_timed_out": self.events_timed_out,
                "publishing_rate": self.publishing_rate,
                "processing_rate": self.processing_rate,
                "error_rate": self.error_rate,
                "timeout_rate": self.timeout_rate,
            },
            "latency": {
                "min_latency": self.min_latency,
                "max_latency": self.max_latency,
                "avg_latency": self.avg_latency,
                "p95_latency": self.p95_latency,
                "p99_latency": self.p99_latency,
            },
            "timing": {"total_duration": self.total_duration, "start_time": self.start_time, "end_time": self.end_time},
            "resources": {"peak_queue_size": self.peak_queue_size, "avg_queue_size": self.avg_queue_size},
        }


class EventLoadGenerator:
    """Generates load for event bus testing."""

    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.metrics = LoadTestMetrics()
        self._monitoring_task: Optional[asyncio.Task] = None

    def create_event(
        self,
        event_type: EventType = EventType.TRADE,
        priority: EventPriority = EventPriority.NORMAL,
        symbol: str = "BTCUSDT",
    ) -> BaseEvent:
        """Create a test event with realistic data."""
        return BaseEvent(
            event_type=event_type,
            source="load_test",
            symbol=symbol,
            priority=priority,
            data={
                "price": random.uniform(1000, 50000),
                "volume": random.uniform(0.1, 100.0),
                "timestamp": datetime.now(UTC).isoformat(),
                "bid": random.uniform(1000, 50000),
                "ask": random.uniform(1000, 50000),
                "spread": random.uniform(0.01, 1.0),
            },
        )

    async def start_monitoring(self, interval: float = 0.1):
        """Start monitoring queue size and other metrics."""
        self._monitoring_task = asyncio.create_task(self._monitor_queue_size(interval))

    async def stop_monitoring(self):
        """Stop monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitor_queue_size(self, interval: float):
        """Monitor queue size periodically."""
        while True:
            try:
                queue_size = self.event_bus.get_queue_size()
                self.metrics.queue_size_samples.append(queue_size)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def constant_rate_load(
        self,
        rate: float,
        duration: float,
        event_type: EventType = EventType.TRADE,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> LoadTestMetrics:
        """Generate constant rate load."""
        self.metrics = LoadTestMetrics()
        self.metrics.start_time = time.time()

        await self.start_monitoring()

        try:
            interval = 1.0 / rate
            end_time = time.time() + duration

            while time.time() < end_time:
                start_publish = time.time()

                event = self.create_event(event_type=event_type, priority=priority)

                try:
                    await self.event_bus.publish(event)
                    self.metrics.events_published += 1
                except RuntimeError:
                    # Queue full
                    self.metrics.events_dropped += 1
                except Exception:
                    self.metrics.events_errored += 1

                # Maintain rate
                elapsed = time.time() - start_publish
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            self.metrics.end_time = time.time()
            self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
            await self.stop_monitoring()

            # Get final bus statistics
            stats = self.event_bus.get_statistics()
            self.metrics.events_processed = stats["processed_count"]
            self.metrics.events_errored += stats["error_count"]
            self.metrics.events_timed_out = stats["timeout_count"]
            self.metrics.events_dropped += stats["dropped_count"]

            self.metrics.calculate_derived_metrics()

        return self.metrics

    async def burst_load(
        self,
        burst_size: int,
        burst_interval: float,
        num_bursts: int,
        event_type: EventType = EventType.TRADE,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> LoadTestMetrics:
        """Generate burst load pattern."""
        self.metrics = LoadTestMetrics()
        self.metrics.start_time = time.time()

        await self.start_monitoring()

        try:
            for burst in range(num_bursts):
                # Generate burst
                burst_tasks = []
                for i in range(burst_size):
                    event = self.create_event(event_type=event_type, priority=priority)
                    task = asyncio.create_task(self._publish_with_metrics(event))
                    burst_tasks.append(task)

                # Wait for burst to complete
                await asyncio.gather(*burst_tasks, return_exceptions=True)

                # Wait between bursts
                if burst < num_bursts - 1:
                    await asyncio.sleep(burst_interval)

        finally:
            self.metrics.end_time = time.time()
            self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
            await self.stop_monitoring()

            # Get final bus statistics
            stats = self.event_bus.get_statistics()
            self.metrics.events_processed = stats["processed_count"]
            self.metrics.events_errored += stats["error_count"]
            self.metrics.events_timed_out = stats["timeout_count"]
            self.metrics.events_dropped += stats["dropped_count"]

            self.metrics.calculate_derived_metrics()

        return self.metrics

    async def random_load(
        self,
        min_rate: float,
        max_rate: float,
        duration: float,
        rate_change_interval: float = 1.0,
        event_type: EventType = EventType.TRADE,
    ) -> LoadTestMetrics:
        """Generate random load with varying rates."""
        self.metrics = LoadTestMetrics()
        self.metrics.start_time = time.time()

        await self.start_monitoring()

        try:
            end_time = time.time() + duration
            current_rate = random.uniform(min_rate, max_rate)
            next_rate_change = time.time() + rate_change_interval

            while time.time() < end_time:
                # Change rate if needed
                if time.time() >= next_rate_change:
                    current_rate = random.uniform(min_rate, max_rate)
                    next_rate_change = time.time() + rate_change_interval

                # Publish event
                start_publish = time.time()
                event = self.create_event(event_type=event_type)

                try:
                    await self.event_bus.publish(event)
                    self.metrics.events_published += 1
                except RuntimeError:
                    self.metrics.events_dropped += 1
                except Exception:
                    self.metrics.events_errored += 1

                # Sleep based on current rate
                interval = 1.0 / current_rate
                elapsed = time.time() - start_publish
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            self.metrics.end_time = time.time()
            self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
            await self.stop_monitoring()

            # Get final bus statistics
            stats = self.event_bus.get_statistics()
            self.metrics.events_processed = stats["processed_count"]
            self.metrics.events_errored += stats["error_count"]
            self.metrics.events_timed_out = stats["timeout_count"]
            self.metrics.events_dropped += stats["dropped_count"]

            self.metrics.calculate_derived_metrics()

        return self.metrics

    async def mixed_priority_load(
        self, rate: float, duration: float, priority_distribution: Dict[EventPriority, float] = None
    ) -> LoadTestMetrics:
        """Generate load with mixed event priorities."""
        if priority_distribution is None:
            priority_distribution = {
                EventPriority.CRITICAL: 0.1,
                EventPriority.HIGH: 0.2,
                EventPriority.NORMAL: 0.5,
                EventPriority.LOW: 0.2,
            }

        # Create weighted priority list
        priorities = []
        for priority, weight in priority_distribution.items():
            priorities.extend([priority] * int(weight * 100))

        self.metrics = LoadTestMetrics()
        self.metrics.start_time = time.time()

        await self.start_monitoring()

        try:
            interval = 1.0 / rate
            end_time = time.time() + duration

            while time.time() < end_time:
                start_publish = time.time()

                # Select random priority
                priority = random.choice(priorities)
                event = self.create_event(priority=priority)

                try:
                    await self.event_bus.publish(event)
                    self.metrics.events_published += 1
                except RuntimeError:
                    self.metrics.events_dropped += 1
                except Exception:
                    self.metrics.events_errored += 1

                # Maintain rate
                elapsed = time.time() - start_publish
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            self.metrics.end_time = time.time()
            self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
            await self.stop_monitoring()

            # Get final bus statistics
            stats = self.event_bus.get_statistics()
            self.metrics.events_processed = stats["processed_count"]
            self.metrics.events_errored += stats["error_count"]
            self.metrics.events_timed_out = stats["timeout_count"]
            self.metrics.events_dropped += stats["dropped_count"]

            self.metrics.calculate_derived_metrics()

        return self.metrics

    async def _publish_with_metrics(self, event: BaseEvent):
        """Publish event and collect metrics."""
        try:
            await self.event_bus.publish(event)
            self.metrics.events_published += 1
        except RuntimeError:
            self.metrics.events_dropped += 1
        except Exception:
            self.metrics.events_errored += 1


class EventLatencyCollector:
    """Collects latency metrics for event processing."""

    def __init__(self, event_bus: InMemoryEventBus):
        self.event_bus = event_bus
        self.latencies: List[float] = []
        self.event_timestamps: Dict[str, float] = {}
        self._subscription_id: Optional[str] = None

    def start_collection(self, event_type: EventType = EventType.TRADE) -> str:
        """Start collecting latency metrics."""

        def latency_handler(event: BaseEvent):
            if event.event_id in self.event_timestamps:
                publish_time = self.event_timestamps[event.event_id]
                process_time = time.time()
                latency = (process_time - publish_time) * 1000  # Convert to ms
                self.latencies.append(latency)

        self._subscription_id = self.event_bus.subscribe(event_type, latency_handler)
        return self._subscription_id

    def stop_collection(self):
        """Stop collecting latency metrics."""
        if self._subscription_id:
            self.event_bus.unsubscribe(self._subscription_id)
            self._subscription_id = None

    async def publish_with_timing(self, event: BaseEvent):
        """Publish event and record timing."""
        self.event_timestamps[event.event_id] = time.time()
        await self.event_bus.publish(event)

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {}

        return {
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies),
            "avg_latency": statistics.mean(self.latencies),
            "p95_latency": statistics.quantiles(self.latencies, n=20)[18],
            "p99_latency": statistics.quantiles(self.latencies, n=100)[98],
            "sample_count": len(self.latencies),
        }

    def reset(self):
        """Reset collected data."""
        self.latencies.clear()
        self.event_timestamps.clear()


def create_load_test_suite(event_bus: InMemoryEventBus) -> Dict[str, Callable]:
    """Create a suite of load tests for the event bus."""
    generator = EventLoadGenerator(event_bus)

    return {
        "constant_rate": lambda: generator.constant_rate_load(rate=100, duration=10),
        "burst_load": lambda: generator.burst_load(burst_size=50, burst_interval=1.0, num_bursts=10),
        "random_load": lambda: generator.random_load(min_rate=50, max_rate=200, duration=10),
        "mixed_priority": lambda: generator.mixed_priority_load(rate=100, duration=10),
        "high_throughput": lambda: generator.constant_rate_load(rate=1000, duration=5),
        "sustained_load": lambda: generator.constant_rate_load(rate=500, duration=30),
    }


async def run_load_test_suite(event_bus: InMemoryEventBus) -> Dict[str, LoadTestMetrics]:
    """Run a complete load test suite."""
    test_suite = create_load_test_suite(event_bus)
    results = {}

    for test_name, test_func in test_suite.items():
        print(f"Running {test_name}...")
        try:
            metrics = await test_func()
            results[test_name] = metrics
            print(f"✓ {test_name} completed - Rate: {metrics.publishing_rate:.2f} events/sec")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = None

    return results


def generate_load_test_report(results: Dict[str, LoadTestMetrics]) -> str:
    """Generate a comprehensive load test report."""
    report = []
    report.append("# Event Bus Load Test Report")
    report.append("=" * 50)
    report.append("")

    for test_name, metrics in results.items():
        if metrics is None:
            report.append(f"## {test_name.upper()} - FAILED")
            continue

        report.append(f"## {test_name.upper()}")
        report.append(f"Duration: {metrics.total_duration:.2f}s")
        report.append(f"Events Published: {metrics.events_published}")
        report.append(f"Events Processed: {metrics.events_processed}")
        report.append(f"Events Dropped: {metrics.events_dropped}")
        report.append(f"Events Errored: {metrics.events_errored}")
        report.append(f"Publishing Rate: {metrics.publishing_rate:.2f} events/sec")
        report.append(f"Processing Rate: {metrics.processing_rate:.2f} events/sec")
        report.append(f"Error Rate: {metrics.error_rate:.2f}%")
        report.append(f"Peak Queue Size: {metrics.peak_queue_size}")
        report.append(f"Avg Queue Size: {metrics.avg_queue_size:.2f}")

        if metrics.latencies:
            report.append(f"Avg Latency: {metrics.avg_latency:.2f}ms")
            report.append(f"P95 Latency: {metrics.p95_latency:.2f}ms")
            report.append(f"P99 Latency: {metrics.p99_latency:.2f}ms")

        report.append("")

    return "\n".join(report)
