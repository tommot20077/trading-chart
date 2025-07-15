# ABOUTME: Contract test configuration and fixtures
# ABOUTME: Provides common setup and utilities for all contract tests

import pytest
import asyncio
from typing import Any, Dict


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def mock_auth_token():
    """Create a mock auth token for testing."""

    class MockAuthToken:
        def __init__(self, user_id: str = "test_user", permissions: set = None, roles: set = None):
            self._user_id = user_id
            self._permissions = permissions or set()
            self._roles = roles or set()

        @property
        def user_id(self) -> str:
            return self._user_id

        @property
        def permissions(self) -> set:
            return self._permissions

        @property
        def roles(self) -> set:
            return self._roles

    return MockAuthToken


@pytest.fixture
def mock_auth_request():
    """Create a mock auth request for testing."""

    class MockAuthRequest:
        def __init__(self, headers: Dict[str, str] = None, client_id: str = None):
            self._headers = headers or {}
            self._client_id = client_id

        def get_header(self, name: str) -> str | None:
            return self._headers.get(name.lower())

        @property
        def client_id(self) -> str | None:
            return self._client_id

    return MockAuthRequest


@pytest.fixture
def mock_time_series_data():
    """Create mock time series data for testing."""
    from core.models.storage.time_series_data import TimeSeriesData
    from datetime import datetime

    class MockTimeSeriesData(TimeSeriesData):
        def __init__(self, symbol: str = "TEST", timestamp: datetime = None, data: Dict[str, Any] = None):
            super().__init__(symbol=symbol, timestamp=timestamp or datetime.now(), data=data or {"test": "value"})

    return MockTimeSeriesData


# Contract test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "contract: mark test as a contract test")
    config.addinivalue_line("markers", "interface: mark test as testing interface compliance")
    config.addinivalue_line("markers", "implementation: mark test as testing implementation behavior")


# Configure asyncio mode for contract tests
# Note: pytest_asyncio is configured at the project level
