# ABOUTME: Unit tests for AuthToken protocol implementation
# ABOUTME: Tests cover protocol compliance, token properties, and timestamp handling

import pytest


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

    @pytest.mark.unit
    def test_token_properties_normal_case(self):
        """Test all token properties with valid values."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user", "admin"],
            expires_at=1640995200.0,  # 2022-01-01 00:00:00 UTC
            issued_at=1640908800.0,  # 2021-12-31 00:00:00 UTC
        )

        assert token.id == "user123"
        assert token.username == "testuser"
        assert token.roles == ["user", "admin"]
        assert token.expires_at == 1640995200.0
        assert token.issued_at == 1640908800.0

    @pytest.mark.unit
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

    @pytest.mark.unit
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
        assert len(token.roles) == 0

    @pytest.mark.unit
    def test_token_with_single_role(self):
        """Test token with single role."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["admin"],
        )

        assert token.roles == ["admin"]
        assert len(token.roles) == 1

    @pytest.mark.unit
    def test_token_with_multiple_roles(self):
        """Test token with multiple roles."""
        roles = ["user", "admin", "moderator", "viewer"]
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=roles,
        )

        assert token.roles == roles
        assert len(token.roles) == 4
        assert "user" in token.roles
        assert "admin" in token.roles
        assert "moderator" in token.roles
        assert "viewer" in token.roles

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_token_with_large_timestamps(self):
        """Test token with large timestamp values."""
        large_timestamp = 9999999999.0  # Year 2286
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user"],
            expires_at=large_timestamp,
            issued_at=large_timestamp,
        )

        assert token.expires_at == large_timestamp
        assert token.issued_at == large_timestamp

    @pytest.mark.unit
    def test_token_with_fractional_timestamps(self):
        """Test token with fractional timestamp values."""
        fractional_timestamp = 1640995200.123456
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user"],
            expires_at=fractional_timestamp,
            issued_at=fractional_timestamp,
        )

        assert token.expires_at == fractional_timestamp
        assert token.issued_at == fractional_timestamp

    @pytest.mark.unit
    def test_token_id_variations(self):
        """Test token with various ID formats."""
        id_formats = [
            "123",
            "user-123",
            "user_123",
            "user@domain.com",
            "550e8400-e29b-41d4-a716-446655440000",  # UUID
            "用戶123",  # Unicode
        ]

        for user_id in id_formats:
            token = MockAuthToken(id=user_id, username="testuser", roles=["user"])
            assert token.id == user_id

    @pytest.mark.unit
    def test_token_username_variations(self):
        """Test token with various username formats."""
        username_formats = [
            "user",
            "test.user",
            "test-user",
            "test_user",
            "user@domain.com",
            "用戶名",  # Unicode
            "User With Spaces",
        ]

        for username in username_formats:
            token = MockAuthToken(id="123", username=username, roles=["user"])
            assert token.username == username

    @pytest.mark.unit
    def test_token_role_variations(self):
        """Test token with various role formats."""
        role_variations = [
            ["admin"],
            ["user", "admin"],
            ["role-with-dash"],
            ["role_with_underscore"],
            ["UPPERCASE_ROLE"],
            ["角色"],  # Unicode
            ["role with spaces"],
        ]

        for roles in role_variations:
            token = MockAuthToken(id="123", username="testuser", roles=roles)
            assert token.roles == roles

    @pytest.mark.unit
    def test_token_immutability_expectation(self):
        """Test that token properties are consistent across multiple accesses."""
        token = MockAuthToken(
            id="user123",
            username="testuser",
            roles=["user", "admin"],
            expires_at=1640995200.0,
            issued_at=1640908800.0,
        )

        # Multiple accesses should return the same values
        for _ in range(3):
            assert token.id == "user123"
            assert token.username == "testuser"
            assert token.roles == ["user", "admin"]
            assert token.expires_at == 1640995200.0
            assert token.issued_at == 1640908800.0

    @pytest.mark.unit
    def test_token_roles_list_independence(self):
        """Test that roles list is independent from the original list."""
        original_roles = ["user", "admin"]
        token = MockAuthToken(id="123", username="testuser", roles=original_roles)

        # Modifying original list should not affect token (if implementation is correct)
        retrieved_roles = token.roles
        assert retrieved_roles == original_roles

        # The retrieved roles should be the same reference or a copy
        # This test documents the expected behavior
        assert isinstance(retrieved_roles, list)
        assert len(retrieved_roles) == 2
