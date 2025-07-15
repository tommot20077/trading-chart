# ABOUTME: pytest configuration and fixtures for authentication integration tests
# ABOUTME: Provides shared fixtures and utilities for auth integration testing

import pytest
import pytest_asyncio
import tempfile
import os
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
from core.implementations.memory.auth.authorizer import InMemoryAuthorizer
from core.implementations.memory.auth.models import MemoryAuthToken


@dataclass
class AuthTestConfig:
    """Configuration settings for authentication integration tests."""

    # Test users configuration
    test_users: List[Dict[str, Any]]

    # Token settings
    token_expiry_seconds: int
    token_refresh_threshold_seconds: int

    # Test timeouts
    test_timeout_seconds: int
    async_operation_timeout_seconds: int

    # Database settings
    test_database_url: str
    test_database_cleanup: bool

    # Security settings
    password_min_length: int
    max_login_attempts: int
    lockout_duration_seconds: int

    # Performance settings
    max_concurrent_users: int
    token_cleanup_interval_seconds: int


@pytest.fixture(scope="session")
def auth_test_config() -> AuthTestConfig:
    """Configuration for authentication integration tests."""
    return AuthTestConfig(
        test_users=[
            {"username": "test_admin", "password": "admin_test_123", "roles": ["ADMIN"], "is_active": True},
            {"username": "test_user", "password": "user_test_123", "roles": ["USER"], "is_active": True},
            {"username": "test_guest", "password": "guest_test_123", "roles": [], "is_active": True},
            {"username": "test_inactive", "password": "inactive_test_123", "roles": ["USER"], "is_active": False},
        ],
        # Token settings - shorter for testing
        token_expiry_seconds=300,  # 5 minutes
        token_refresh_threshold_seconds=60,  # 1 minute before expiry
        # Test timeouts
        test_timeout_seconds=30,
        async_operation_timeout_seconds=10,
        # Database settings
        test_database_url="sqlite:///:memory:",
        test_database_cleanup=True,
        # Security settings
        password_min_length=8,
        max_login_attempts=3,
        lockout_duration_seconds=300,  # 5 minutes
        # Performance settings
        max_concurrent_users=100,
        token_cleanup_interval_seconds=60,
    )


@pytest.fixture(scope="session")
def performance_test_config(auth_test_config: AuthTestConfig) -> AuthTestConfig:
    """Configuration optimized for performance testing."""
    config = AuthTestConfig(
        test_users=auth_test_config.test_users,
        token_expiry_seconds=3600,  # 1 hour
        token_refresh_threshold_seconds=300,  # 5 minutes
        test_timeout_seconds=60,
        async_operation_timeout_seconds=30,
        test_database_url=auth_test_config.test_database_url,
        test_database_cleanup=auth_test_config.test_database_cleanup,
        password_min_length=auth_test_config.password_min_length,
        max_login_attempts=auth_test_config.max_login_attempts,
        lockout_duration_seconds=auth_test_config.lockout_duration_seconds,
        max_concurrent_users=1000,
        token_cleanup_interval_seconds=60,
    )
    return config


@pytest.fixture(scope="session")
def security_test_config(auth_test_config: AuthTestConfig) -> AuthTestConfig:
    """Configuration optimized for security testing."""
    config = AuthTestConfig(
        test_users=auth_test_config.test_users,
        token_expiry_seconds=auth_test_config.token_expiry_seconds,
        token_refresh_threshold_seconds=auth_test_config.token_refresh_threshold_seconds,
        test_timeout_seconds=auth_test_config.test_timeout_seconds,
        async_operation_timeout_seconds=auth_test_config.async_operation_timeout_seconds,
        test_database_url=auth_test_config.test_database_url,
        test_database_cleanup=auth_test_config.test_database_cleanup,
        password_min_length=12,
        max_login_attempts=2,
        lockout_duration_seconds=60,
        max_concurrent_users=auth_test_config.max_concurrent_users,
        token_cleanup_interval_seconds=auth_test_config.token_cleanup_interval_seconds,
    )
    return config


@pytest.fixture(scope="session")
def test_database_url(auth_test_config: AuthTestConfig) -> str:
    """Get test database URL."""
    return auth_test_config.test_database_url


@pytest.fixture(scope="function")
def test_database_cleanup(auth_test_config: AuthTestConfig):
    """Database cleanup fixture."""
    # Setup - create temporary database if needed
    temp_db_file = None
    if auth_test_config.test_database_cleanup and "memory" not in auth_test_config.test_database_url:
        temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db_file.close()

    yield temp_db_file

    # Cleanup - remove temporary database
    if temp_db_file and os.path.exists(temp_db_file.name):
        os.unlink(temp_db_file.name)


@pytest_asyncio.fixture
async def authenticator(auth_test_config: AuthTestConfig):
    """Create an authenticator instance for testing."""
    auth = InMemoryAuthenticator()
    # Note: Default users (admin/admin123, user/user123) are automatically created
    # Additional test users can be added if needed

    # Add test users from config
    for user_info in auth_test_config.test_users:
        try:
            await auth.create_user(
                username=user_info["username"], password=user_info["password"], roles=user_info["roles"]
            )
        except Exception:
            # User might already exist, skip
            pass

    return auth


@pytest.fixture
def authorizer():
    """Create an authorizer instance for testing."""
    return InMemoryAuthorizer()


@pytest.fixture
def mock_auth_request():
    """Create a mock authentication request."""

    class MockAuthRequest:
        def __init__(self, token: str = None, client_id: str = None):
            self._token = token
            self._client_id = client_id

        def get_header(self, name: str) -> str | None:
            if name.lower() == "authorization" and self._token:
                return f"Bearer {self._token}"
            return None

        @property
        def client_id(self) -> str | None:
            return self._client_id

    return MockAuthRequest


@pytest.fixture
def valid_admin_token(authenticator, auth_test_config: AuthTestConfig):
    """Create a valid admin token for testing."""
    # Create admin token with configured expiry
    token = MemoryAuthToken(
        id="test_admin_token",
        username="test_admin",
        roles=["ADMIN"],
        expires_at=datetime.now() + timedelta(seconds=auth_test_config.token_expiry_seconds),
        issued_at=datetime.now(),
    )

    # Store token in authenticator
    authenticator._tokens[token.id] = token

    return token


@pytest.fixture
def valid_user_token(authenticator, auth_test_config: AuthTestConfig):
    """Create a valid user token for testing."""
    token = MemoryAuthToken(
        id="test_user_token",
        username="test_user",
        roles=["USER"],
        expires_at=datetime.now() + timedelta(seconds=auth_test_config.token_expiry_seconds),
        issued_at=datetime.now(),
    )

    # Store token in authenticator
    authenticator._tokens[token.id] = token

    return token


@pytest.fixture
def expired_token():
    """Create an expired token for testing."""
    return MemoryAuthToken(
        id="expired_token",
        username="test_user",
        roles=["USER"],
        expires_at=datetime.now() - timedelta(hours=1),
        issued_at=datetime.now() - timedelta(hours=2),
    )


@pytest.fixture
def integration_test_cleanup():
    """Cleanup fixture for integration tests."""
    # Setup
    yield
    # Teardown - cleanup any test data
    pass


@pytest.fixture
def auth_integration_environment(authenticator, authorizer, auth_test_config: AuthTestConfig, test_database_cleanup):
    """Complete authentication integration test environment."""
    # Setup complete test environment
    environment = {
        "authenticator": authenticator,
        "authorizer": authorizer,
        "config": auth_test_config,
        "test_users": auth_test_config.test_users,
        "database": test_database_cleanup,
    }

    yield environment

    # Cleanup
    try:
        # Clear all tokens
        authenticator._tokens.clear()
        # Clear all users except defaults
        for user_info in auth_test_config.test_users:
            try:
                authenticator.delete_user(user_info["username"])
            except Exception:
                pass
    except Exception:
        pass
