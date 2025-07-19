# ABOUTME: pytest configuration for core tests
# ABOUTME: Configures timeouts and other test behaviors

import pytest


def pytest_configure(config):
    """Configure pytest for core tests."""
    # Update timeout descriptions
    config.addinivalue_line("markers", "unit: Unit tests with 20-second timeout")
    config.addinivalue_line("markers", "integration: Integration tests with 60-second timeout")
    config.addinivalue_line("markers", "contract: Contract tests with 60-second timeout")
    config.addinivalue_line("markers", "benchmark: Benchmark tests with 60-second timeout")


def pytest_collection_modifyitems(config, items):
    """Modify test items to add appropriate timeouts based on test type."""
    for item in items:
        # Check for existing timeout marker - if it exists, respect it
        existing_timeout = item.get_closest_marker("timeout")
        if existing_timeout:
            # If there's already a timeout marker, don't override it
            continue

        # Apply timeout based on test type
        if item.get_closest_marker("unit"):
            # Unit tests: 20 seconds timeout
            item.add_marker(pytest.mark.timeout(20))
        elif any(item.get_closest_marker(mark) for mark in ["integration", "contract", "benchmark"]):
            # Integration/Contract/Benchmark tests: 60 seconds timeout
            item.add_marker(pytest.mark.timeout(60))
        # Other tests will use the global default (60 seconds from pyproject.toml)
