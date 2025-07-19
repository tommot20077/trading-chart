# ABOUTME: Integration tests for performance baseline establishment and validation systems
# ABOUTME: Tests baseline creation, update mechanisms, comparison operations, and regression detection

import asyncio
import pytest
import pytest_asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
import statistics

# Previously skipped due to timeout issues - now fixed

from core.implementations.memory.event.event_bus import InMemoryEventBus
from ...fixtures.event_load_testing import EventLoadGenerator, LoadTestMetrics
from core.models.event.event_type import EventType


@dataclass
class PerformanceBaseline:
    """Performance baseline data structure."""

    # Metadata
    baseline_id: str
    test_scenario: str
    created_at: str
    updated_at: str
    version: str = "1.0"

    # Performance metrics
    publishing_rate_avg: float = 0.0
    publishing_rate_min: float = 0.0
    publishing_rate_max: float = 0.0

    processing_rate_avg: float = 0.0
    processing_rate_min: float = 0.0
    processing_rate_max: float = 0.0

    latency_avg: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    error_rate_avg: float = 0.0
    error_rate_max: float = 0.0

    queue_size_avg: float = 0.0
    queue_size_peak: float = 0.0

    # Tolerances for comparison
    rate_tolerance: float = 0.15  # 15% tolerance
    latency_tolerance: float = 0.20  # 20% tolerance
    error_tolerance: float = 0.05  # 5% tolerance

    def to_dict(self) -> Dict[str, Any]:
        """Convert baseline to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceBaseline":
        """Create baseline from dictionary."""
        return cls(**data)


class PerformanceBaselineManager:
    """Manager for performance baselines with persistence and validation."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(tempfile.gettempdir()) / "performance_baselines"
        self.storage_path.mkdir(exist_ok=True)
        self.baselines: Dict[str, PerformanceBaseline] = {}

    def create_baseline_from_metrics(
        self, baseline_id: str, test_scenario: str, metrics_list: List[LoadTestMetrics]
    ) -> PerformanceBaseline:
        """Create baseline from multiple test run metrics."""
        if not metrics_list:
            raise ValueError("Cannot create baseline from empty metrics list")

        # Calculate aggregated statistics
        publishing_rates = [m.publishing_rate for m in metrics_list]
        processing_rates = [m.processing_rate for m in metrics_list]
        avg_latencies = [m.avg_latency for m in metrics_list if m.avg_latency > 0]
        p95_latencies = [m.p95_latency for m in metrics_list if m.p95_latency > 0]
        p99_latencies = [m.p99_latency for m in metrics_list if m.p99_latency > 0]
        error_rates = [m.error_rate for m in metrics_list]
        avg_queue_sizes = [m.avg_queue_size for m in metrics_list]
        peak_queue_sizes = [m.peak_queue_size for m in metrics_list]

        now = datetime.now(UTC).isoformat()

        baseline = PerformanceBaseline(
            baseline_id=baseline_id,
            test_scenario=test_scenario,
            created_at=now,
            updated_at=now,
            # Publishing rate statistics
            publishing_rate_avg=statistics.mean(publishing_rates),
            publishing_rate_min=min(publishing_rates),
            publishing_rate_max=max(publishing_rates),
            # Processing rate statistics
            processing_rate_avg=statistics.mean(processing_rates),
            processing_rate_min=min(processing_rates),
            processing_rate_max=max(processing_rates),
            # Latency statistics
            latency_avg=statistics.mean(avg_latencies) if avg_latencies else 0.0,
            latency_p95=statistics.mean(p95_latencies) if p95_latencies else 0.0,
            latency_p99=statistics.mean(p99_latencies) if p99_latencies else 0.0,
            # Error rate statistics
            error_rate_avg=statistics.mean(error_rates),
            error_rate_max=max(error_rates),
            # Queue size statistics
            queue_size_avg=statistics.mean(avg_queue_sizes),
            queue_size_peak=max(peak_queue_sizes),
        )

        self.baselines[baseline_id] = baseline
        return baseline

    def update_baseline(
        self, baseline_id: str, new_metrics: List[LoadTestMetrics], weight: float = 0.3
    ) -> PerformanceBaseline:
        """Update existing baseline with new metrics using weighted averaging."""
        if baseline_id not in self.baselines:
            raise ValueError(f"Baseline {baseline_id} not found")

        existing = self.baselines[baseline_id]

        # Create temporary baseline from new metrics
        temp_baseline = self.create_baseline_from_metrics(baseline_id + "_temp", existing.test_scenario, new_metrics)

        # Weighted update
        existing.publishing_rate_avg = (
            existing.publishing_rate_avg * (1 - weight) + temp_baseline.publishing_rate_avg * weight
        )
        existing.processing_rate_avg = (
            existing.processing_rate_avg * (1 - weight) + temp_baseline.processing_rate_avg * weight
        )
        existing.latency_avg = existing.latency_avg * (1 - weight) + temp_baseline.latency_avg * weight
        existing.latency_p95 = existing.latency_p95 * (1 - weight) + temp_baseline.latency_p95 * weight
        existing.error_rate_avg = existing.error_rate_avg * (1 - weight) + temp_baseline.error_rate_avg * weight
        existing.queue_size_avg = existing.queue_size_avg * (1 - weight) + temp_baseline.queue_size_avg * weight

        # Update bounds if new values exceed them
        existing.publishing_rate_min = min(existing.publishing_rate_min, temp_baseline.publishing_rate_min)
        existing.publishing_rate_max = max(existing.publishing_rate_max, temp_baseline.publishing_rate_max)
        existing.processing_rate_min = min(existing.processing_rate_min, temp_baseline.processing_rate_min)
        existing.processing_rate_max = max(existing.processing_rate_max, temp_baseline.processing_rate_max)
        existing.error_rate_max = max(existing.error_rate_max, temp_baseline.error_rate_max)
        existing.queue_size_peak = max(existing.queue_size_peak, temp_baseline.queue_size_peak)

        existing.updated_at = datetime.now(UTC).isoformat()

        return existing

    def compare_to_baseline(self, baseline_id: str, test_metrics: LoadTestMetrics) -> Dict[str, Any]:
        """Compare test metrics against baseline and detect regressions."""
        if baseline_id not in self.baselines:
            raise ValueError(f"Baseline {baseline_id} not found")

        baseline = self.baselines[baseline_id]
        comparison_result = {
            "baseline_id": baseline_id,
            "test_timestamp": datetime.now(UTC).isoformat(),
            "passed": True,
            "regressions": [],
            "improvements": [],
            "metrics_comparison": {},
        }

        # Publishing rate comparison
        rate_diff_pct = self._calculate_percentage_diff(baseline.publishing_rate_avg, test_metrics.publishing_rate)
        comparison_result["metrics_comparison"]["publishing_rate"] = {
            "baseline": baseline.publishing_rate_avg,
            "current": test_metrics.publishing_rate,
            "diff_percentage": rate_diff_pct,
            "status": "pass" if abs(rate_diff_pct) <= baseline.rate_tolerance * 100 else "fail",
        }

        if rate_diff_pct < -baseline.rate_tolerance * 100:
            comparison_result["regressions"].append(
                {
                    "metric": "publishing_rate",
                    "issue": "Performance degradation",
                    "baseline": baseline.publishing_rate_avg,
                    "current": test_metrics.publishing_rate,
                    "diff_percentage": rate_diff_pct,
                }
            )
            comparison_result["passed"] = False
        elif rate_diff_pct > baseline.rate_tolerance * 100:
            comparison_result["improvements"].append(
                {
                    "metric": "publishing_rate",
                    "improvement": "Performance improvement",
                    "baseline": baseline.publishing_rate_avg,
                    "current": test_metrics.publishing_rate,
                    "diff_percentage": rate_diff_pct,
                }
            )

        # Latency comparison
        if baseline.latency_avg > 0 and test_metrics.avg_latency > 0:
            latency_diff_pct = self._calculate_percentage_diff(baseline.latency_avg, test_metrics.avg_latency)
            comparison_result["metrics_comparison"]["latency"] = {
                "baseline": baseline.latency_avg,
                "current": test_metrics.avg_latency,
                "diff_percentage": latency_diff_pct,
                "status": "pass" if abs(latency_diff_pct) <= baseline.latency_tolerance * 100 else "fail",
            }

            if latency_diff_pct > baseline.latency_tolerance * 100:
                comparison_result["regressions"].append(
                    {
                        "metric": "latency",
                        "issue": "Latency regression",
                        "baseline": baseline.latency_avg,
                        "current": test_metrics.avg_latency,
                        "diff_percentage": latency_diff_pct,
                    }
                )
                comparison_result["passed"] = False

        # Error rate comparison
        error_diff_pct = self._calculate_percentage_diff(baseline.error_rate_avg, test_metrics.error_rate)
        comparison_result["metrics_comparison"]["error_rate"] = {
            "baseline": baseline.error_rate_avg,
            "current": test_metrics.error_rate,
            "diff_percentage": error_diff_pct,
            "status": "pass"
            if test_metrics.error_rate <= baseline.error_rate_avg + baseline.error_tolerance
            else "fail",
        }

        if test_metrics.error_rate > baseline.error_rate_avg + baseline.error_tolerance:
            comparison_result["regressions"].append(
                {
                    "metric": "error_rate",
                    "issue": "Error rate increase",
                    "baseline": baseline.error_rate_avg,
                    "current": test_metrics.error_rate,
                    "diff_percentage": error_diff_pct,
                }
            )
            comparison_result["passed"] = False

        return comparison_result

    def _calculate_percentage_diff(self, baseline_value: float, current_value: float) -> float:
        """Calculate percentage difference between baseline and current value."""
        if baseline_value == 0:
            return 100.0 if current_value > 0 else 0.0
        return ((current_value - baseline_value) / baseline_value) * 100

    def save_baseline(self, baseline_id: str) -> Path:
        """Save baseline to persistent storage."""
        if baseline_id not in self.baselines:
            raise ValueError(f"Baseline {baseline_id} not found")

        baseline = self.baselines[baseline_id]
        file_path = self.storage_path / f"{baseline_id}.json"

        with open(file_path, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2)

        return file_path

    def load_baseline(self, baseline_id: str) -> PerformanceBaseline:
        """Load baseline from persistent storage."""
        file_path = self.storage_path / f"{baseline_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Baseline file {file_path} not found")

        with open(file_path, "r") as f:
            data = json.load(f)

        baseline = PerformanceBaseline.from_dict(data)
        self.baselines[baseline_id] = baseline
        return baseline

    def list_baselines(self) -> List[str]:
        """List all available baseline IDs."""
        return list(self.baselines.keys())


@pytest_asyncio.fixture
async def event_bus():
    """Create an event bus for baseline testing."""
    bus = InMemoryEventBus(max_queue_size=1000, max_concurrent_handlers=10)
    yield bus
    await bus.close()


@pytest_asyncio.fixture
async def load_generator(event_bus):
    """Create a load generator for baseline testing."""
    generator = EventLoadGenerator(event_bus)
    yield generator
    await generator.stop_monitoring()


@pytest.fixture
def baseline_manager():
    """Create a baseline manager with temporary storage."""
    temp_dir = tempfile.mkdtemp(prefix="baseline_test_")
    manager = PerformanceBaselineManager(Path(temp_dir))
    yield manager
    # Cleanup is handled by tempfile


class TestPerformanceBaselineEstablishment:
    """Test performance baseline creation and establishment."""

    @pytest.mark.asyncio
    async def test_create_baseline_from_single_test(self, load_generator, baseline_manager):
        """Test creating baseline from a single load test."""
        # Run load test
        metrics = await load_generator.constant_rate_load(rate=50, duration=2.0, event_type=EventType.TRADE)

        # Create baseline
        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="single_test_baseline", test_scenario="constant_50_2s", metrics_list=[metrics]
        )

        # Verify baseline creation
        assert baseline.baseline_id == "single_test_baseline"
        assert baseline.test_scenario == "constant_50_2s"
        assert baseline.publishing_rate_avg == metrics.publishing_rate
        assert baseline.processing_rate_avg == metrics.processing_rate
        assert baseline.error_rate_avg == metrics.error_rate
        assert baseline.created_at is not None
        assert baseline.updated_at is not None

    @pytest.mark.asyncio
    async def test_create_baseline_from_multiple_tests(self, load_generator, baseline_manager):
        """Test creating baseline from multiple load test runs."""
        # Run multiple load tests
        metrics_list = []
        for i in range(3):
            metrics = await load_generator.constant_rate_load(rate=40, duration=1.5, event_type=EventType.TRADE)
            metrics_list.append(metrics)
            await asyncio.sleep(0.1)  # Small delay between tests

        # Create baseline from multiple runs
        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="multi_test_baseline", test_scenario="constant_40_1.5s", metrics_list=metrics_list
        )

        # Verify baseline aggregates multiple runs
        assert baseline.baseline_id == "multi_test_baseline"
        assert len(metrics_list) == 3

        # Check that baseline represents average of runs
        avg_publishing_rate = sum(m.publishing_rate for m in metrics_list) / len(metrics_list)
        assert abs(baseline.publishing_rate_avg - avg_publishing_rate) < 0.01

        # Check min/max bounds
        min_publishing_rate = min(m.publishing_rate for m in metrics_list)
        max_publishing_rate = max(m.publishing_rate for m in metrics_list)
        assert baseline.publishing_rate_min == min_publishing_rate
        assert baseline.publishing_rate_max == max_publishing_rate

    @pytest.mark.asyncio
    async def test_baseline_persistence(self, load_generator, baseline_manager):
        """Test saving and loading baselines from persistent storage."""
        # Create and save baseline
        metrics = await load_generator.constant_rate_load(rate=60, duration=1.0, event_type=EventType.KLINE)

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="persistence_test", test_scenario="constant_60_1s", metrics_list=[metrics]
        )

        # Save baseline
        file_path = baseline_manager.save_baseline("persistence_test")
        assert file_path.exists()

        # Create new manager and load baseline
        new_manager = PerformanceBaselineManager(baseline_manager.storage_path)
        loaded_baseline = new_manager.load_baseline("persistence_test")

        # Verify loaded baseline matches original
        assert loaded_baseline.baseline_id == baseline.baseline_id
        assert loaded_baseline.test_scenario == baseline.test_scenario
        assert loaded_baseline.publishing_rate_avg == baseline.publishing_rate_avg
        assert loaded_baseline.created_at == baseline.created_at

    def test_baseline_creation_error_handling(self, baseline_manager):
        """Test error handling in baseline creation."""
        # Test with empty metrics list
        with pytest.raises(ValueError, match="Cannot create baseline from empty metrics list"):
            baseline_manager.create_baseline_from_metrics(
                baseline_id="error_test", test_scenario="test", metrics_list=[]
            )

    @pytest.mark.asyncio
    async def test_baseline_update_mechanism(self, load_generator, baseline_manager):
        """Test updating existing baselines with new data."""
        # Create initial baseline
        initial_metrics = await load_generator.constant_rate_load(rate=30, duration=1.0, event_type=EventType.TRADE)

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="update_test", test_scenario="constant_30_1s", metrics_list=[initial_metrics]
        )

        initial_rate = baseline.publishing_rate_avg

        # Generate new metrics
        new_metrics = await load_generator.constant_rate_load(rate=35, duration=1.0, event_type=EventType.TRADE)

        # Update baseline
        updated_baseline = baseline_manager.update_baseline(
            baseline_id="update_test",
            new_metrics=[new_metrics],
            weight=0.5,  # 50% weight for new data
        )

        # Verify update
        assert updated_baseline.baseline_id == "update_test"
        assert updated_baseline.updated_at != baseline.created_at

        # Rate should be between initial and new (weighted average)
        expected_rate = initial_rate * 0.5 + new_metrics.publishing_rate * 0.5
        assert abs(updated_baseline.publishing_rate_avg - expected_rate) < 0.01


class TestPerformanceBaselineValidation:
    """Test performance validation against baselines."""

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, load_generator, baseline_manager):
        """Test detection of performance regressions."""
        # Create baseline with good performance
        good_metrics = await load_generator.constant_rate_load(rate=100, duration=1.0, event_type=EventType.TRADE)

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="regression_test", test_scenario="high_performance", metrics_list=[good_metrics]
        )

        # Simulate degraded performance (lower rate)
        degraded_metrics = LoadTestMetrics()
        degraded_metrics.publishing_rate = baseline.publishing_rate_avg * 0.7  # 30% slower
        degraded_metrics.processing_rate = baseline.processing_rate_avg * 0.7
        degraded_metrics.error_rate = baseline.error_rate_avg
        degraded_metrics.avg_latency = baseline.latency_avg * 1.5  # 50% higher latency

        # Compare against baseline
        comparison = baseline_manager.compare_to_baseline("regression_test", degraded_metrics)

        # Verify regression detection
        assert comparison["passed"] is False
        assert len(comparison["regressions"]) > 0

        # Check specific regression
        rate_regression = next((r for r in comparison["regressions"] if r["metric"] == "publishing_rate"), None)
        assert rate_regression is not None
        assert rate_regression["issue"] == "Performance degradation"

    @pytest.mark.asyncio
    async def test_performance_improvement_detection(self, load_generator, baseline_manager):
        """Test detection of performance improvements."""
        # Create baseline
        baseline_metrics = await load_generator.constant_rate_load(rate=50, duration=1.0, event_type=EventType.TRADE)

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="improvement_test", test_scenario="baseline_performance", metrics_list=[baseline_metrics]
        )

        # Simulate improved performance
        improved_metrics = LoadTestMetrics()
        improved_metrics.publishing_rate = baseline.publishing_rate_avg * 1.3  # 30% faster
        improved_metrics.processing_rate = baseline.processing_rate_avg * 1.3
        improved_metrics.error_rate = baseline.error_rate_avg * 0.5  # Lower error rate
        improved_metrics.avg_latency = baseline.latency_avg * 0.8  # Lower latency

        # Compare against baseline
        comparison = baseline_manager.compare_to_baseline("improvement_test", improved_metrics)

        # Verify improvement detection
        assert comparison["passed"] is True
        assert len(comparison["improvements"]) > 0

        # Check specific improvement
        rate_improvement = next((i for i in comparison["improvements"] if i["metric"] == "publishing_rate"), None)
        assert rate_improvement is not None
        assert rate_improvement["improvement"] == "Performance improvement"

    @pytest.mark.asyncio
    async def test_acceptable_performance_variation(self, load_generator, baseline_manager):
        """Test validation of acceptable performance variations within tolerance."""
        # Create baseline
        baseline_metrics = await load_generator.constant_rate_load(rate=80, duration=1.0, event_type=EventType.TRADE)

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="tolerance_test", test_scenario="tolerance_check", metrics_list=[baseline_metrics]
        )

        # Create metrics within tolerance (10% variation)
        within_tolerance_metrics = LoadTestMetrics()
        within_tolerance_metrics.publishing_rate = baseline.publishing_rate_avg * 1.05  # 5% increase
        within_tolerance_metrics.processing_rate = baseline.processing_rate_avg * 0.95  # 5% decrease
        within_tolerance_metrics.error_rate = baseline.error_rate_avg + 0.02  # Small increase
        within_tolerance_metrics.avg_latency = baseline.latency_avg * 1.1  # 10% increase

        # Compare against baseline
        comparison = baseline_manager.compare_to_baseline("tolerance_test", within_tolerance_metrics)

        # Verify acceptable variation
        assert comparison["passed"] is True
        assert len(comparison["regressions"]) == 0

        # Check metrics comparison details
        rate_comparison = comparison["metrics_comparison"]["publishing_rate"]
        assert rate_comparison["status"] == "pass"

    @pytest.mark.asyncio
    async def test_error_rate_regression_detection(self, load_generator, baseline_manager):
        """Test specific detection of error rate regressions."""
        # Create baseline with low error rate
        baseline_metrics = await load_generator.constant_rate_load(rate=60, duration=1.0, event_type=EventType.TRADE)
        baseline_metrics.error_rate = 1.0  # 1% error rate

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="error_rate_test", test_scenario="error_check", metrics_list=[baseline_metrics]
        )

        # Create metrics with high error rate
        high_error_metrics = LoadTestMetrics()
        high_error_metrics.publishing_rate = baseline.publishing_rate_avg
        high_error_metrics.processing_rate = baseline.processing_rate_avg
        high_error_metrics.error_rate = 8.0  # 8% error rate (significant increase)
        high_error_metrics.avg_latency = baseline.latency_avg

        # Compare against baseline
        comparison = baseline_manager.compare_to_baseline("error_rate_test", high_error_metrics)

        # Verify error rate regression detection
        assert comparison["passed"] is False

        error_regression = next((r for r in comparison["regressions"] if r["metric"] == "error_rate"), None)
        assert error_regression is not None
        assert error_regression["issue"] == "Error rate increase"
        assert error_regression["current"] == 8.0

    def test_baseline_comparison_with_nonexistent_baseline(self, baseline_manager):
        """Test error handling when comparing against non-existent baseline."""
        test_metrics = LoadTestMetrics()
        test_metrics.publishing_rate = 100.0

        with pytest.raises(ValueError, match="Baseline nonexistent not found"):
            baseline_manager.compare_to_baseline("nonexistent", test_metrics)

    @pytest.mark.asyncio
    async def test_comprehensive_baseline_validation_workflow(self, load_generator, baseline_manager):
        """Test complete workflow of baseline establishment and validation."""
        # Step 1: Establish baseline from multiple runs
        baseline_metrics = []
        for _ in range(3):
            metrics = await load_generator.constant_rate_load(rate=70, duration=1.0, event_type=EventType.TRADE)
            baseline_metrics.append(metrics)

        baseline = baseline_manager.create_baseline_from_metrics(
            baseline_id="workflow_test", test_scenario="comprehensive_test", metrics_list=baseline_metrics
        )

        # Step 2: Save baseline
        saved_path = baseline_manager.save_baseline("workflow_test")
        assert saved_path.exists()

        # Step 3: Run new test and validate
        new_test_metrics = await load_generator.constant_rate_load(rate=72, duration=1.0, event_type=EventType.TRADE)

        comparison = baseline_manager.compare_to_baseline("workflow_test", new_test_metrics)

        # Step 4: Verify validation results
        assert comparison["baseline_id"] == "workflow_test"
        assert "metrics_comparison" in comparison
        assert "publishing_rate" in comparison["metrics_comparison"]

        # Step 5: Update baseline if performance is acceptable
        if comparison["passed"]:
            updated_baseline = baseline_manager.update_baseline(
                baseline_id="workflow_test", new_metrics=[new_test_metrics], weight=0.2
            )
            assert updated_baseline.updated_at != baseline.created_at
