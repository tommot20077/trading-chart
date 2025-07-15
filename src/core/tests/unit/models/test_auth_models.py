# ABOUTME: Unit tests for authentication models including Role, Permission enums and Protocol classes
# ABOUTME: Tests cover normal cases, exception cases, and boundary cases following TDD principles

import pytest
from typing import Optional

from core.models.auth.enum import Role, Permission
from core.models.auth.auth_request import AuthRequest
from core.models.auth.auth_token import AuthToken


class TestRole:
    """Test cases for Role enum."""

    def test_role_values_normal_case(self):
        """Test that Role enum has correct string values."""
        assert Role.ADMIN == "admin"
        assert Role.USER == "user"

    def test_role_string_representation(self):
        """Test string representation of Role enum members."""
        assert str(Role.ADMIN) == "Role.ADMIN"
        assert str(Role.USER) == "Role.USER"

    def test_role_equality(self):
        """Test equality comparison of Role enum members."""
        assert Role.ADMIN == Role.ADMIN
        assert Role.USER == Role.USER
        assert Role.ADMIN != Role.USER

    def test_role_membership(self):
        """Test membership checking for Role enum."""
        assert "admin" in [role.value for role in Role]
        assert "user" in [role.value for role in Role]
        assert "invalid_role" not in [role.value for role in Role]

    def test_role_iteration(self):
        """Test iteration over Role enum members."""
        roles = list(Role)
        assert len(roles) == 2
        assert Role.ADMIN in roles
        assert Role.USER in roles

    def test_role_from_string(self):
        """Test creating Role from string value."""
        assert Role("admin") == Role.ADMIN
        assert Role("user") == Role.USER

    def test_role_invalid_value_raises_exception(self):
        """Test that invalid role value raises ValueError."""
        with pytest.raises(ValueError):
            Role("invalid_role")


class TestPermission:
    """Test cases for Permission enum."""

    def test_permission_values_normal_case(self):
        """Test that Permission enum has correct string values."""
        assert Permission.READ == "read"
        assert Permission.WRITE == "write"
        assert Permission.DELETE == "delete"
        assert Permission.ADMIN == "admin"

    def test_permission_string_representation(self):
        """Test string representation of Permission enum members."""
        assert str(Permission.READ) == "Permission.READ"
        assert str(Permission.WRITE) == "Permission.WRITE"
        assert str(Permission.DELETE) == "Permission.DELETE"
        assert str(Permission.ADMIN) == "Permission.ADMIN"

    def test_permission_equality(self):
        """Test equality comparison of Permission enum members."""
        assert Permission.READ == Permission.READ
        assert Permission.WRITE != Permission.READ
        assert Permission.DELETE != Permission.ADMIN

    def test_permission_membership(self):
        """Test membership checking for Permission enum."""
        permission_values = [perm.value for perm in Permission]
        assert "read" in permission_values
        assert "write" in permission_values
        assert "delete" in permission_values
        assert "admin" in permission_values
        assert "invalid_permission" not in permission_values

    def test_permission_iteration(self):
        """Test iteration over Permission enum members."""
        permissions = list(Permission)
        assert len(permissions) == 4
        assert Permission.READ in permissions
        assert Permission.WRITE in permissions
        assert Permission.DELETE in permissions
        assert Permission.ADMIN in permissions

    def test_permission_from_string(self):
        """Test creating Permission from string value."""
        assert Permission("read") == Permission.READ
        assert Permission("write") == Permission.WRITE
        assert Permission("delete") == Permission.DELETE
        assert Permission("admin") == Permission.ADMIN

    def test_permission_invalid_value_raises_exception(self):
        """Test that invalid permission value raises ValueError."""
        with pytest.raises(ValueError):
            Permission("invalid_permission")


class MockAuthRequest:
    """Mock implementation of AuthRequest protocol for testing."""

    def __init__(self, headers: dict[str, str], client_id: str | None = None):
        self._headers = {k.lower(): v for k, v in headers.items()}
        self._client_id = client_id

    def get_header(self, name: str) -> str | None:
        return self._headers.get(name.lower())

    @property
    def client_id(self) -> str | None:
        return self._client_id


class TestAuthRequestProtocol:
    """Test cases for AuthRequest protocol implementation."""

    def test_get_header_normal_case(self):
        """Test getting header value with valid header name."""
        headers = {"Authorization": "Bearer token123", "Content-Type": "application/json"}
        request = MockAuthRequest(headers)
        
        assert request.get_header("Authorization") == "Bearer token123"
        assert request.get_header("Content-Type") == "application/json"

    def test_get_header_case_insensitive(self):
        """Test that header retrieval is case-insensitive."""
        headers = {"Authorization": "Bearer token123"}
        request = MockAuthRequest(headers)
        
        assert request.get_header("authorization") == "Bearer token123"
        assert request.get_header("AUTHORIZATION") == "Bearer token123"
        assert request.get_header("Authorization") == "Bearer token123"

    def test_get_header_not_found_returns_none(self):
        """Test that non-existent header returns None."""
        headers = {"Authorization": "Bearer token123"}
        request = MockAuthRequest(headers)
        
        assert request.get_header("X-Custom-Header") is None
        assert request.get_header("") is None

    def test_get_header_empty_headers(self):
        """Test getting header from empty headers dict."""
        request = MockAuthRequest({})
        
        assert request.get_header("Authorization") is None
        assert request.get_header("any-header") is None

    def test_client_id_normal_case(self):
        """Test client_id property with valid client ID."""
        request = MockAuthRequest({}, client_id="client123")
        assert request.client_id == "client123"

    def test_client_id_none_case(self):
        """Test client_id property when no client ID is set."""
        request = MockAuthRequest({})
        assert request.client_id is None

    def test_client_id_empty_string(self):
        """Test client_id property with empty string."""
        request = MockAuthRequest({}, client_id="")
        assert request.client_id == ""


class MockAuthToken:
    """Mock implementation of AuthToken protocol for testing."""

    def __init__(
        self,
        id: str,
        username: str,
        roles: list[str],
        expires_at: float | None = None,
        issued_at: float | None = None,
    ):
        self._id = id
        self._username = username
        self._roles = roles
        self._expires_at = expires_at
        self._issued_at = issued_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def username(self) -> str:
        return self._username

    @property
    def roles(self) -> list[str]:
        return self._roles

    @property
    def expires_at(self) -> float | None:
        return self._expires_at

    @property
    def issued_at(self) -> float | None:
        return self._issued_at


class TestAuthTokenProtocol:
    """Test cases for AuthToken protocol implementation."""

    def test_token_properties_normal_case(self):
        """Test all token properties with valid values."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user", "admin"],
            expires_at=1640995200.0,  # 2022-01-01 00:00:00 UTC
            issued_at=1640908800.0,   # 2021-12-31 00:00:00 UTC
        )
        
        assert token.id == "user123"
        assert token.username == "testuser"
        assert token.roles == ["user", "admin"]
        assert token.expires_at == 1640995200.0
        assert token.issued_at == 1640908800.0

    def test_token_with_none_timestamps(self):
        """Test token with None timestamps (non-expiring token)."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user"],
            expires_at=None,
            issued_at=None,
        )
        
        assert token.id == "user123"
        assert token.username == "testuser"
        assert token.roles == ["user"]
        assert token.expires_at is None
        assert token.issued_at is None

    def test_token_with_empty_roles(self):
        """Test token with empty roles list."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=[],
        )
        
        assert token.id == "user123"
        assert token.username == "testuser"
        assert token.roles == []

    def test_token_with_single_role(self):
        """Test token with single role."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["admin"],
        )
        
        assert token.roles == ["admin"]
        assert len(token.roles) == 1

    def test_token_boundary_cases(self):
        """Test token with boundary values."""
        # Test with minimal valid values
        token = MockAuthToken(
            id="1",
            username="a",
            roles=["r"],
        )
        
        assert token.id == "1"
        assert token.username == "a"
        assert token.roles == ["r"]

    def test_token_with_zero_timestamps(self):
        """Test token with zero timestamps (epoch time)."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user"],
            expires_at=0.0,
            issued_at=0.0,
        )
        
        assert token.expires_at == 0.0
        assert token.issued_at == 0.0

    def test_token_with_negative_timestamps(self):
        """Test token with negative timestamps (before epoch)."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user"],
            expires_at=-1.0,
            issued_at=-1.0,
        )
        
        assert token.expires_at == -1.0
        assert token.issued_at == -1.0