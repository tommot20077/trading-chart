# ABOUTME: Benchmark test configuration for core configuration system
# ABOUTME: Provides pytest-benchmark setup and fixtures for performance testing

import pytest


def pytest_configure(config):
    """Configure benchmark test markers."""
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Add benchmark marker to all tests in benchmark directory."""
    for item in items:
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)


@pytest.fixture
def benchmark_config():
    """Benchmark configuration for consistent testing."""
    return {
        "min_rounds": 10,
        "max_time": 1.0,
        "min_time": 0.1,
        "warmup": True,
        "warmup_iterations": 3,
    }
