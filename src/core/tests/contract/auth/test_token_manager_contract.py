# ABOUTME: Contract tests for AbstractTokenManager interface
# ABOUTME: Verifies all token manager implementations comply with the interface contract

import pytest
from typing import Type, List, Any, Dict

from core.interfaces.auth.token_manager import AbstractTokenManager
from core.models.auth.auth_token import AuthToken
from core.exceptions import AuthenticationException
from ..base_contract_test import ContractTestBase


class MockAuthToken:
    """Mock implementation of AuthToken protocol for testing."""

    def __init__(
        self,
        id: str = "test_user",
        username: str = "testuser",
        roles: List[str] = None,
        expires_at: float = None,
        issued_at: float = None,
    ):
        self._id = id
        self._username = username
        self._roles = roles or ["user"]
        self._expires_at = expires_at
        self._issued_at = issued_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def username(self) -> str:
        return self._username

    @property
    def roles(self) -> List[str]:
        return self._roles

    @property
    def expires_at(self) -> float | None:
        return self._expires_at

    @property
    def issued_at(self) -> float | None:
        return self._issued_at


class MockTokenManager(AbstractTokenManager):
    """Mock implementation of AbstractTokenManager for contract testing."""

    def __init__(self):
        self._tokens: Dict[str, MockAuthToken] = {}
        self._revoked_tokens: set = set()

    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate a mock token."""
        if not isinstance(user_data, dict):
            raise ValueError("user_data must be a dictionary")

        if not user_data.get("user_id"):
            raise ValueError("user_data must contain user_id")

        token_id = f"token_{user_data['user_id']}"
        auth_token = MockAuthToken(
            id=user_data["user_id"],
            username=user_data.get("username", "unknown"),
            roles=user_data.get("roles", ["user"]),
        )
        self._tokens[token_id] = auth_token
        return token_id

    def validate_token(self, token: str) -> AuthToken:
        """Validate a mock token."""
        if not isinstance(token, str):
            raise ValueError("token must be a string")

        if not token:
            raise AuthenticationException("Empty token")

        if token in self._revoked_tokens:
            raise AuthenticationException("Token has been revoked")

        if token not in self._tokens:
            raise AuthenticationException("Invalid token")

        return self._tokens[token]

    def refresh_token(self, token: str) -> str:
        """Refresh a mock token."""
        if not isinstance(token, str):
            raise ValueError("token must be a string")

        if not token:
            raise AuthenticationException("Empty token")

        if token in self._revoked_tokens:
            raise AuthenticationException("Cannot refresh revoked token")

        if token not in self._tokens:
            raise AuthenticationException("Invalid token for refresh")

        # Create new token with same user data
        auth_token = self._tokens[token]
        new_token = f"refreshed_{token}"
        self._tokens[new_token] = auth_token

        # Optionally revoke old token
        self._revoked_tokens.add(token)

        return new_token

    def revoke_token(self, token: str) -> None:
        """Revoke a mock token."""
        if not isinstance(token, str):
            raise ValueError("token must be a string")

        if not token:
            raise AuthenticationException("Empty token")

        if token not in self._tokens:
            raise AuthenticationException("Cannot revoke non-existent token")

        if token in self._revoked_tokens:
            raise AuthenticationException("Token already revoked")

        self._revoked_tokens.add(token)


class TestTokenManagerContract(ContractTestBase[AbstractTokenManager]):
    """Contract tests for AbstractTokenManager interface."""

    @property
    def interface_class(self) -> Type[AbstractTokenManager]:
        return AbstractTokenManager

    @property
    def implementations(self) -> List[Type[AbstractTokenManager]]:
        return [
            MockTokenManager,
            # TODO: Add actual implementations when they exist
            # JWTTokenManager,
            # DatabaseTokenManager,
        ]

    @pytest.mark.contract
    def test_generate_token_method_signature(self):
        """Test generate_token method has correct signature and behavior."""
        method = getattr(self.interface_class, "generate_token")
        assert hasattr(method, "__isabstractmethod__")

        # Test with mock implementation
        token_manager = MockTokenManager()

        # Valid user data should return string token
        user_data = {"user_id": "test123", "username": "testuser", "roles": ["user"]}
        token = token_manager.generate_token(user_data)
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.contract
    def test_generate_token_contract_validation(self):
        """Test generate_token input validation contract."""
        token_manager = MockTokenManager()

        # Test with invalid input types
        with pytest.raises((ValueError, TypeError)):
            token_manager.generate_token("not_a_dict")

        with pytest.raises((ValueError, TypeError)):
            token_manager.generate_token(None)

        # Test with missing required fields
        with pytest.raises(ValueError):
            token_manager.generate_token({})  # Missing user_id

        with pytest.raises(ValueError):
            token_manager.generate_token({"username": "test"})  # Missing user_id

    @pytest.mark.contract
    def test_validate_token_method_signature(self):
        """Test validate_token method has correct signature and behavior."""
        method = getattr(self.interface_class, "validate_token")
        assert hasattr(method, "__isabstractmethod__")

        # Test with mock implementation
        token_manager = MockTokenManager()

        # Generate a valid token first
        user_data = {"user_id": "test123", "username": "testuser"}
        token = token_manager.generate_token(user_data)

        # Validate should return AuthToken protocol object
        auth_token = token_manager.validate_token(token)
        assert hasattr(auth_token, "id")
        assert hasattr(auth_token, "username")
        assert hasattr(auth_token, "roles")
        assert auth_token.id == "test123"
        assert auth_token.username == "testuser"

    @pytest.mark.contract
    def test_validate_token_contract_exceptions(self):
        """Test validate_token exception handling contract."""
        token_manager = MockTokenManager()

        # Test with invalid token types
        with pytest.raises((ValueError, TypeError)):
            token_manager.validate_token(None)

        with pytest.raises((ValueError, TypeError)):
            token_manager.validate_token(123)

        # Test with invalid tokens
        with pytest.raises(AuthenticationException):
            token_manager.validate_token("")  # Empty token

        with pytest.raises(AuthenticationException):
            token_manager.validate_token("invalid_token")  # Non-existent token

        # Test with revoked token
        user_data = {"user_id": "test123", "username": "testuser"}
        token = token_manager.generate_token(user_data)
        token_manager.revoke_token(token)

        with pytest.raises(AuthenticationException):
            token_manager.validate_token(token)  # Revoked token

    @pytest.mark.contract
    def test_refresh_token_method_signature(self):
        """Test refresh_token method has correct signature and behavior."""
        method = getattr(self.interface_class, "refresh_token")
        assert hasattr(method, "__isabstractmethod__")

        # Test with mock implementation
        token_manager = MockTokenManager()

        # Generate a valid token first
        user_data = {"user_id": "test123", "username": "testuser"}
        original_token = token_manager.generate_token(user_data)

        # Refresh should return new string token
        new_token = token_manager.refresh_token(original_token)
        assert isinstance(new_token, str)
        assert len(new_token) > 0
        assert new_token != original_token  # Should be different

    @pytest.mark.contract
    def test_refresh_token_contract_exceptions(self):
        """Test refresh_token exception handling contract."""
        token_manager = MockTokenManager()

        # Test with invalid token types
        with pytest.raises((ValueError, TypeError)):
            token_manager.refresh_token(None)

        with pytest.raises((ValueError, TypeError)):
            token_manager.refresh_token(123)

        # Test with invalid tokens
        with pytest.raises(AuthenticationException):
            token_manager.refresh_token("")  # Empty token

        with pytest.raises(AuthenticationException):
            token_manager.refresh_token("invalid_token")  # Non-existent token

        # Test with revoked token
        user_data = {"user_id": "test123", "username": "testuser"}
        token = token_manager.generate_token(user_data)
        token_manager.revoke_token(token)

        with pytest.raises(AuthenticationException):
            token_manager.refresh_token(token)  # Revoked token

    @pytest.mark.contract
    def test_revoke_token_method_signature(self):
        """Test revoke_token method has correct signature and behavior."""
        method = getattr(self.interface_class, "revoke_token")
        assert hasattr(method, "__isabstractmethod__")

        # Test with mock implementation
        token_manager = MockTokenManager()

        # Generate a valid token first
        user_data = {"user_id": "test123", "username": "testuser"}
        token = token_manager.generate_token(user_data)

        # Revoke should not raise exception and return None
        result = token_manager.revoke_token(token)
        assert result is None

        # Token should be invalid after revocation
        with pytest.raises(AuthenticationException):
            token_manager.validate_token(token)

    @pytest.mark.contract
    def test_revoke_token_contract_exceptions(self):
        """Test revoke_token exception handling contract."""
        token_manager = MockTokenManager()

        # Test with invalid token types
        with pytest.raises((ValueError, TypeError)):
            token_manager.revoke_token(None)

        with pytest.raises((ValueError, TypeError)):
            token_manager.revoke_token(123)

        # Test with invalid tokens
        with pytest.raises(AuthenticationException):
            token_manager.revoke_token("")  # Empty token

        with pytest.raises(AuthenticationException):
            token_manager.revoke_token("invalid_token")  # Non-existent token

    @pytest.mark.contract
    @pytest.mark.concurrency
    def test_token_lifecycle_contract(self):
        """Test complete token lifecycle contract behavior."""
        token_manager = MockTokenManager()

        # 1. Generate token
        user_data = {"user_id": "lifecycle_test", "username": "testuser", "roles": ["user", "admin"]}
        token = token_manager.generate_token(user_data)
        assert isinstance(token, str)

        # 2. Validate token
        auth_token = token_manager.validate_token(token)
        assert auth_token.id == "lifecycle_test"
        assert auth_token.username == "testuser"
        assert "user" in auth_token.roles
        assert "admin" in auth_token.roles

        # 3. Refresh token
        new_token = token_manager.refresh_token(token)
        assert isinstance(new_token, str)
        assert new_token != token

        # 4. Validate new token
        new_auth_token = token_manager.validate_token(new_token)
        assert new_auth_token.id == "lifecycle_test"
        assert new_auth_token.username == "testuser"

        # 5. Revoke new token
        token_manager.revoke_token(new_token)

        # 6. Validation should fail after revocation
        with pytest.raises(AuthenticationException):
            token_manager.validate_token(new_token)

    @pytest.mark.contract
    @pytest.mark.concurrency
    def test_concurrent_token_operations_contract(self):
        """Test contract behavior with multiple tokens."""
        token_manager = MockTokenManager()

        # Generate multiple tokens
        tokens = []
        for i in range(3):
            user_data = {"user_id": f"user_{i}", "username": f"user{i}"}
            token = token_manager.generate_token(user_data)
            tokens.append(token)

        # All tokens should be valid
        for i, token in enumerate(tokens):
            auth_token = token_manager.validate_token(token)
            assert auth_token.id == f"user_{i}"
            assert auth_token.username == f"user{i}"

        # Revoke one token
        token_manager.revoke_token(tokens[1])

        # First and third should still be valid
        token_manager.validate_token(tokens[0])  # Should not raise
        token_manager.validate_token(tokens[2])  # Should not raise

        # Second should be invalid
        with pytest.raises(AuthenticationException):
            token_manager.validate_token(tokens[1])

    @pytest.mark.contract
    def test_edge_cases_contract(self):
        """Test contract behavior with edge cases."""
        token_manager = MockTokenManager()

        # Test with minimal user data
        minimal_data = {"user_id": "min"}
        token = token_manager.generate_token(minimal_data)
        auth_token = token_manager.validate_token(token)
        assert auth_token.id == "min"

        # Test with complex user data
        complex_data = {
            "user_id": "complex_user_123",
            "username": "complex@example.com",
            "roles": ["admin", "moderator", "user"],
            "extra_field": "should_be_ignored_or_handled",
        }
        token = token_manager.generate_token(complex_data)
        auth_token = token_manager.validate_token(token)
        assert auth_token.id == "complex_user_123"
        assert auth_token.username == "complex@example.com"

        # Test double revocation
        token_manager.revoke_token(token)
        # Second revocation should raise AuthenticationException for already revoked token
        with pytest.raises(AuthenticationException):
            token_manager.revoke_token(token)
