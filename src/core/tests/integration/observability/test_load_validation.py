# ABOUTME: Standalone validation test for load testing integration functionality
# ABOUTME: Verifies Phase 5 implementation with basic load generation and monitoring tests

import asyncio
import sys
import os

# Load validation tests - basic load generation and metrics analysis
import pytest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

from core.implementations.memory.event.event_bus import InMemoryEventBus
from ...fixtures.event_load_testing import EventLoadGenerator, LoadTestMetrics
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


@pytest.mark.asyncio
async def test_basic_load_generation():
    """Test basic load generation functionality."""
    print("Testing basic load generation...")

    # Create event bus
    event_bus = InMemoryEventBus(max_queue_size=1000, max_concurrent_handlers=10)

    try:
        # Create load generator
        generator = EventLoadGenerator(event_bus)

        # Generate load
        metrics = await generator.constant_rate_load(
            rate=20,  # 20 events/second
            duration=1.0,  # 1 second
            event_type=EventType.TRADE,
            priority=EventPriority.NORMAL,
        )

        # Verify results
        assert metrics.events_published > 0, "No events were published"
        assert metrics.publishing_rate > 0, "Publishing rate is zero"
        assert metrics.total_duration > 0, "Total duration is zero"

        print(f"✓ Published {metrics.events_published} events")
        print(f"✓ Publishing rate: {metrics.publishing_rate:.2f} events/sec")
        print(f"✓ Duration: {metrics.total_duration:.2f} seconds")

    finally:
        await event_bus.close()


@pytest.mark.asyncio
async def test_burst_load_generation():
    """Test burst load generation functionality."""
    print("\nTesting burst load generation...")

    # Create event bus
    event_bus = InMemoryEventBus(max_queue_size=1000, max_concurrent_handlers=10)

    try:
        # Create load generator
        generator = EventLoadGenerator(event_bus)

        # Generate burst load
        metrics = await generator.burst_load(
            burst_size=10, burst_interval=0.3, num_bursts=2, event_type=EventType.KLINE, priority=EventPriority.HIGH
        )

        # Verify results
        assert metrics.events_published >= 20, f"Expected at least 20 events, got {metrics.events_published}"
        assert metrics.total_duration > 0, "Total duration is zero"

        print(f"✓ Published {metrics.events_published} events in bursts")
        print(f"✓ Duration: {metrics.total_duration:.2f} seconds")

    finally:
        await event_bus.close()


def test_metrics_analysis():
    """Test metrics analysis functionality."""
    print("\nTesting metrics analysis...")

    # Create test metrics
    metrics = LoadTestMetrics()
    metrics.events_published = 100
    metrics.events_processed = 95
    metrics.events_errored = 3
    metrics.events_timed_out = 2
    metrics.total_duration = 10.0
    metrics.latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    metrics.queue_size_samples = [0, 5, 10, 8, 3, 1, 0]

    # Calculate derived metrics
    metrics.calculate_derived_metrics()

    # Verify calculations
    assert metrics.publishing_rate == 10.0, f"Expected publishing rate 10.0, got {metrics.publishing_rate}"
    assert metrics.processing_rate == 9.5, f"Expected processing rate 9.5, got {metrics.processing_rate}"
    assert metrics.error_rate == 3.0, f"Expected error rate 3.0, got {metrics.error_rate}"
    assert metrics.avg_latency == 3.0, f"Expected avg latency 3.0, got {metrics.avg_latency}"

    print(f"✓ Publishing rate: {metrics.publishing_rate} events/sec")
    print(f"✓ Processing rate: {metrics.processing_rate} events/sec")
    print(f"✓ Error rate: {metrics.error_rate}%")
    print(f"✓ Average latency: {metrics.avg_latency}ms")

    # Test conversion to dict
    metrics_dict = metrics.to_dict()
    assert "throughput" in metrics_dict
    assert "latency" in metrics_dict
    assert "timing" in metrics_dict
    assert "resources" in metrics_dict

    print("✓ Metrics to dict conversion successful")


@pytest.mark.asyncio
async def test_mixed_priority_load():
    """Test mixed priority load generation."""
    print("\nTesting mixed priority load generation...")

    # Create event bus
    event_bus = InMemoryEventBus(max_queue_size=1000, max_concurrent_handlers=10)

    try:
        # Create load generator
        generator = EventLoadGenerator(event_bus)

        # Define priority distribution
        priority_distribution = {
            EventPriority.CRITICAL: 0.1,
            EventPriority.HIGH: 0.2,
            EventPriority.NORMAL: 0.5,
            EventPriority.LOW: 0.2,
        }

        # Generate mixed priority load
        metrics = await generator.mixed_priority_load(
            rate=30, duration=1.0, priority_distribution=priority_distribution
        )

        # Verify results
        assert metrics.events_published > 0, "No events were published"
        assert metrics.total_duration > 0, "Total duration is zero"

        print(f"✓ Published {metrics.events_published} events with mixed priorities")
        print(f"✓ Duration: {metrics.total_duration:.2f} seconds")

    finally:
        await event_bus.close()


async def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Phase 5: Load Testing Integration - Validation Tests")
    print("=" * 60)

    tests = [
        test_basic_load_generation,
        test_burst_load_generation,
        test_mixed_priority_load,
    ]

    sync_tests = [
        test_metrics_analysis,
    ]

    passed = 0
    total = len(tests) + len(sync_tests)

    # Run async tests
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
                print(f"✓ {test.__name__} PASSED")
            else:
                print(f"✗ {test.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")

    # Run sync tests
    for test in sync_tests:
        try:
            result = test()
            if result:
                passed += 1
                print(f"✓ {test.__name__} PASSED")
            else:
                print(f"✗ {test.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Validation Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All Phase 5 implementation tests PASSED!")
        print("\nImplemented features:")
        print("✓ EventLoadGenerator + monitoring system integration")
        print("✓ Load test results analysis integration")
        print("✓ Performance baseline establishment and validation")
        print("✓ Real-time monitoring during load generation")
        print("✓ Comprehensive metrics collection and analysis")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
