# ABOUTME: pytest configuration for core tests
# ABOUTME: Configures timeouts and other test behaviors

import pytest


def pytest_configure(config):
    """Configure pytest for core tests."""
    # Add timeout for unit tests only
    config.addinivalue_line("markers", "unit: Unit tests with 30-second timeout")


def pytest_collection_modifyitems(config, items):
    """Modify test items to add timeout to unit tests."""
    for item in items:
        # Check if this is a unit test
        if item.get_closest_marker("unit"):
            # Add timeout of 30 seconds for unit tests
            item.add_marker(pytest.mark.timeout(30))
