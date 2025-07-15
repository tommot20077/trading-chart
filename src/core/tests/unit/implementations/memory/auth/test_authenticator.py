# ABOUTME: Unit tests for InMemoryAuthenticator implementation
# ABOUTME: Tests authentication, user management, and token handling functionality

import pytest
import asyncio
import time

import time_machine

from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
from core.implementations.memory.auth.models import MemoryAuthToken
from core.models.auth.enum import Role, Permission
from core.exceptions import AuthenticationException


class MockAuthRequest:
    """Mock implementation of AuthRequest protocol for testing."""

    def __init__(self, headers: dict = None, client_id: str = None):
        self.headers = headers or {}
        self._client_id = client_id

    def get_header(self, name: str) -> str | None:
        """Get header value (case-insensitive)."""
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return None

    @property
    def client_id(self) -> str | None:
        """Get client ID."""
        return self._client_id


@pytest.mark.unit
class TestInMemoryAuthenticator:
    """Test cases for InMemoryAuthenticator."""

    @pytest.fixture
    def authenticator(self):
        """Create a fresh authenticator instance for each test."""
        return InMemoryAuthenticator(default_token_ttl=3600)

    @pytest.fixture
    def short_ttl_authenticator(self):
        """Create an authenticator with short TTL for expiration tests."""
        return InMemoryAuthenticator(default_token_ttl=1)

    @pytest.fixture
    def mock_auth_request(self):
        """Create a mock authentication request."""
        return MockAuthRequest()

    # === Initialization Tests ===

    @pytest.mark.unit
    @pytest.mark.concurrency
    def test_init_creates_default_users(self, authenticator):
        """Test that initialization creates default admin and user accounts."""
        assert authenticator._users["admin"]
        assert authenticator._users["user"]

        admin_user = authenticator._users["admin"]
        assert admin_user.username == "admin"
        assert Role.ADMIN in admin_user.roles
        assert admin_user.is_active

        user = authenticator._users["user"]
        assert user.username == "user"
        assert Role.USER in user.roles
        assert user.is_active

    @pytest.mark.unit
    @pytest.mark.concurrency
    def test_init_with_custom_ttl(self):
        """Test initialization with custom token TTL."""
        custom_ttl = 7200
        authenticator = InMemoryAuthenticator(default_token_ttl=custom_ttl)
        assert authenticator.default_token_ttl == custom_ttl

    # === Authentication Tests ===

    @pytest.mark.asyncio
    async def test_authenticate_success(self, authenticator):
        """Test successful authentication with valid Bearer token."""
        # First login to get a token
        token, auth_token = await authenticator.login("admin", "admin123")

        # Create request with Bearer token
        request = MockAuthRequest(headers={"Authorization": f"Bearer {token}"})

        # Authenticate
        result = await authenticator.authenticate(request)

        assert result.username == "admin"
        assert result.id == auth_token.id
        assert "admin" in result.roles

    @pytest.mark.asyncio
    async def test_authenticate_missing_header(self, authenticator, mock_auth_request):
        """Test authentication failure when Authorization header is missing."""
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_auth_request)

        assert exc_info.value.code == "MISSING_AUTH_HEADER"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_bearer_format(self, authenticator):
        """Test authentication failure with invalid Bearer token format."""
        # Test various invalid formats
        invalid_headers = [
            {"Authorization": "InvalidFormat"},
            {"Authorization": "Bearer"},
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {"Authorization": ""},
        ]

        for headers in invalid_headers:
            request = MockAuthRequest(headers=headers)
            with pytest.raises(AuthenticationException) as exc_info:
                await authenticator.authenticate(request)

            assert exc_info.value.code == "INVALID_AUTH_FORMAT"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self, authenticator):
        """Test authentication failure with invalid token."""
        request = MockAuthRequest(headers={"Authorization": "Bearer invalid_token_123"})

        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(request)

        assert exc_info.value.code == "INVALID_TOKEN"

    @pytest.mark.asyncio
    async def test_authenticate_expired_token(self, short_ttl_authenticator):
        """Test authentication failure with expired token."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Login and get token
            token, _ = await short_ttl_authenticator.login("admin", "admin123")

            # Wait for token to expire
            traveller.shift(1.1)

            # Try to authenticate with expired token
            request = MockAuthRequest(headers={"Authorization": f"Bearer {token}"})

            with pytest.raises(AuthenticationException) as exc_info:
                await short_ttl_authenticator.authenticate(request)

            assert exc_info.value.code == "TOKEN_EXPIRED"

    @pytest.mark.asyncio
    async def test_authenticate_inactive_user(self, authenticator):
        """Test authentication failure when user is inactive."""
        # Create user and get token
        await authenticator.create_user("testuser", "password123")
        token, _ = await authenticator.login("testuser", "password123")

        # Deactivate user
        user_info = await authenticator.get_user("testuser")
        user_info.is_active = False

        # Try to authenticate
        request = MockAuthRequest(headers={"Authorization": f"Bearer {token}"})

        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(request)

        assert exc_info.value.code == "USER_INACTIVE"

    @pytest.mark.asyncio
    async def test_authenticate_updates_last_login(self, authenticator):
        """Test that authentication updates the user's last login time."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Login and get token
            token, _ = await authenticator.login("admin", "admin123")

            # Get initial last login time
            user_info = await authenticator.get_user("admin")
            initial_last_login = user_info.last_login

            # Wait a bit and authenticate
            traveller.shift(0.1)
            request = MockAuthRequest(headers={"Authorization": f"Bearer {token}"})
            await authenticator.authenticate(request)

            # Check that last login was updated
            user_info = await authenticator.get_user("admin")
            assert user_info.last_login > initial_last_login

    # === Login Tests ===

    @pytest.mark.asyncio
    async def test_login_success(self, authenticator):
        """Test successful login with valid credentials."""
        token, auth_token = await authenticator.login("admin", "admin123")

        assert isinstance(token, str)
        assert len(token) > 0
        assert isinstance(auth_token, MemoryAuthToken)
        assert auth_token.username == "admin"
        assert Role.ADMIN in auth_token.user_roles
        assert auth_token.expires_at > time.time()

    @pytest.mark.asyncio
    async def test_login_invalid_username(self, authenticator):
        """Test login failure with invalid username."""
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.login("nonexistent", "password")

        assert exc_info.value.code == "INVALID_CREDENTIALS"

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, authenticator):
        """Test login failure with invalid password."""
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.login("admin", "wrongpassword")

        assert exc_info.value.code == "INVALID_CREDENTIALS"

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, authenticator):
        """Test login failure when user account is inactive."""
        # Create and then deactivate user
        await authenticator.create_user("testuser", "password123")
        user_info = await authenticator.get_user("testuser")
        user_info.is_active = False

        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.login("testuser", "password123")

        assert exc_info.value.code == "USER_INACTIVE"

    @pytest.mark.asyncio
    async def test_login_stores_token(self, authenticator):
        """Test that login stores the token for future authentication."""
        token, auth_token = await authenticator.login("admin", "admin123")

        # Verify token is stored
        assert token in authenticator._tokens
        assert authenticator._tokens[token] == auth_token

    # === Logout Tests ===

    @pytest.mark.asyncio
    async def test_logout_success(self, authenticator):
        """Test successful logout removes token."""
        token, _ = await authenticator.login("admin", "admin123")

        # Verify token exists
        assert token in authenticator._tokens

        # Logout
        await authenticator.logout(token)

        # Verify token is removed
        assert token not in authenticator._tokens

    @pytest.mark.asyncio
    async def test_logout_invalid_token(self, authenticator):
        """Test logout with invalid token doesn't raise error."""
        # Should not raise exception
        await authenticator.logout("invalid_token")

    # === User Management Tests ===

    @pytest.mark.asyncio
    async def test_create_user_success(self, authenticator):
        """Test successful user creation."""
        user_info = await authenticator.create_user("newuser", "password123", {Role.USER})

        assert user_info.username == "newuser"
        assert Role.USER in user_info.roles
        assert user_info.is_active
        assert user_info.permissions == {Permission.READ, Permission.WRITE}

        # Verify user is stored
        stored_user = await authenticator.get_user("newuser")
        assert stored_user == user_info

    @pytest.mark.asyncio
    async def test_create_user_default_role(self, authenticator):
        """Test user creation with default role."""
        user_info = await authenticator.create_user("newuser", "password123")

        assert Role.USER in user_info.roles
        assert user_info.permissions == {Permission.READ, Permission.WRITE}

    @pytest.mark.asyncio
    async def test_create_user_admin_role(self, authenticator):
        """Test user creation with admin role."""
        user_info = await authenticator.create_user("newadmin", "password123", {Role.ADMIN})

        assert Role.ADMIN in user_info.roles
        assert Permission.ADMIN in user_info.permissions

    @pytest.mark.asyncio
    async def test_create_user_invalid_username(self, authenticator):
        """Test user creation with invalid username."""
        invalid_usernames = ["", "ab", "x" * 51, "user@test", "user space"]

        for username in invalid_usernames:
            with pytest.raises(AuthenticationException) as exc_info:
                await authenticator.create_user(username, "password123")

            assert exc_info.value.code == "INVALID_USERNAME"

    @pytest.mark.asyncio
    async def test_create_user_invalid_password(self, authenticator):
        """Test user creation with invalid password."""
        invalid_passwords = ["", "short", "12345"]

        for password in invalid_passwords:
            with pytest.raises(AuthenticationException) as exc_info:
                await authenticator.create_user("testuser", password)

            assert exc_info.value.code == "INVALID_PASSWORD"

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, authenticator):
        """Test user creation with duplicate username."""
        await authenticator.create_user("testuser", "password123")

        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.create_user("testuser", "password456")

        assert exc_info.value.code == "USER_EXISTS"

    @pytest.mark.asyncio
    async def test_delete_user_success(self, authenticator):
        """Test successful user deletion."""
        # Create user and login
        await authenticator.create_user("testuser", "password123")
        token, _ = await authenticator.login("testuser", "password123")

        # Verify user exists and has token
        assert await authenticator.get_user("testuser") is not None
        assert token in authenticator._tokens

        # Delete user
        await authenticator.delete_user("testuser")

        # Verify user is removed and tokens are invalidated
        assert await authenticator.get_user("testuser") is None
        assert token not in authenticator._tokens

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, authenticator):
        """Test deletion of non-existent user."""
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.delete_user("nonexistent")

        assert exc_info.value.code == "USER_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_update_user_password_success(self, authenticator):
        """Test successful password update."""
        # Create user
        await authenticator.create_user("testuser", "password123")
        token, _ = await authenticator.login("testuser", "password123")

        # Update password
        await authenticator.update_user_password("testuser", "newpassword123")

        # Verify old token is invalidated
        assert token not in authenticator._tokens

        # Verify new password works
        new_token, _ = await authenticator.login("testuser", "newpassword123")
        assert new_token is not None

        # Verify old password doesn't work
        with pytest.raises(AuthenticationException):
            await authenticator.login("testuser", "password123")

    @pytest.mark.asyncio
    async def test_update_user_password_invalid_password(self, authenticator):
        """Test password update with invalid new password."""
        await authenticator.create_user("testuser", "password123")

        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.update_user_password("testuser", "short")

        assert exc_info.value.code == "INVALID_PASSWORD"

    @pytest.mark.asyncio
    async def test_update_user_password_user_not_found(self, authenticator):
        """Test password update for non-existent user."""
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.update_user_password("nonexistent", "newpassword123")

        assert exc_info.value.code == "USER_NOT_FOUND"

    # === User Query Tests ===

    @pytest.mark.asyncio
    async def test_get_user_success(self, authenticator):
        """Test successful user retrieval."""
        user_info = await authenticator.get_user("admin")

        assert user_info.username == "admin"
        assert Role.ADMIN in user_info.roles

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, authenticator):
        """Test retrieval of non-existent user."""
        user_info = await authenticator.get_user("nonexistent")
        assert user_info is None

    @pytest.mark.asyncio
    async def test_list_users(self, authenticator):
        """Test listing all users."""
        # Create additional user
        await authenticator.create_user("testuser", "password123")

        users = await authenticator.list_users()

        assert len(users) == 3  # admin, user, testuser
        assert "admin" in users
        assert "user" in users
        assert "testuser" in users

    # === Token Management Tests ===

    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens(self, short_ttl_authenticator):
        """Test cleanup of expired tokens."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Create multiple tokens
            token1, _ = await short_ttl_authenticator.login("admin", "admin123")
            token2, _ = await short_ttl_authenticator.login("user", "user123")

            # Wait for tokens to expire
            traveller.shift(1.1)

            # Cleanup expired tokens
            removed_count = await short_ttl_authenticator.cleanup_expired_tokens()

            assert removed_count == 2
            assert token1 not in short_ttl_authenticator._tokens
            assert token2 not in short_ttl_authenticator._tokens

    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens_no_expired(self, authenticator):
        """Test cleanup when no tokens are expired."""
        # Create token
        await authenticator.login("admin", "admin123")

        # Cleanup (no tokens should be expired)
        removed_count = await authenticator.cleanup_expired_tokens()

        assert removed_count == 0

    @pytest.mark.asyncio
    async def test_get_token_count(self, authenticator):
        """Test getting active token count."""
        initial_count = await authenticator.get_token_count()
        assert initial_count == 0

        # Create tokens
        await authenticator.login("admin", "admin123")
        await authenticator.login("user", "user123")

        count = await authenticator.get_token_count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_user_count(self, authenticator):
        """Test getting user count."""
        initial_count = await authenticator.get_user_count()
        assert initial_count == 2  # admin, user

        # Create additional user
        await authenticator.create_user("testuser", "password123")

        count = await authenticator.get_user_count()
        assert count == 3

    # === Thread Safety Tests ===

    @pytest.mark.asyncio
    async def test_concurrent_user_creation(self, authenticator):
        """Test concurrent user creation operations."""

        async def create_user(username):
            try:
                await authenticator.create_user(username, "password123")
                return True
            except AuthenticationException:
                return False

        # Create multiple users concurrently
        tasks = [create_user(f"user{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # Verify all users were created
        users = await authenticator.list_users()
        assert len(users) == 12  # 2 default + 10 created

    @pytest.mark.asyncio
    async def test_concurrent_login_logout(self, authenticator):
        """Test concurrent login and logout operations."""

        async def login_logout_cycle():
            try:
                token, _ = await authenticator.login("admin", "admin123")
                await authenticator.logout(token)
                return True
            except AuthenticationException:
                return False

        # Perform multiple login/logout cycles concurrently
        tasks = [login_logout_cycle() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # No tokens should remain
        count = await authenticator.get_token_count()
        assert count == 0

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_multiple_logins_same_user(self, authenticator):
        """Test multiple login sessions for the same user."""
        # Login multiple times
        token1, _ = await authenticator.login("admin", "admin123")
        token2, _ = await authenticator.login("admin", "admin123")
        token3, _ = await authenticator.login("admin", "admin123")

        # All tokens should be different and valid
        assert token1 != token2 != token3
        assert token1 in authenticator._tokens
        assert token2 in authenticator._tokens
        assert token3 in authenticator._tokens

    @pytest.mark.asyncio
    async def test_token_expiration_during_auth(self, short_ttl_authenticator):
        """Test token expiration during authentication process."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Login and get token
            token, _ = await short_ttl_authenticator.login("admin", "admin123")

            # Create request
            request = MockAuthRequest(headers={"Authorization": f"Bearer {token}"})

            # Authenticate immediately (should work)
            result = await short_ttl_authenticator.authenticate(request)
            assert result.username == "admin"

            # Wait for expiration
            traveller.shift(1.1)

            # Authenticate again (should fail)
            with pytest.raises(AuthenticationException) as exc_info:
                await short_ttl_authenticator.authenticate(request)

            assert exc_info.value.code == "TOKEN_EXPIRED"

    @pytest.mark.asyncio
    async def test_password_hashing_security(self, authenticator):
        """Test that passwords are properly hashed and not stored in plaintext."""
        await authenticator.create_user("testuser", "password123")
        user_info = await authenticator.get_user("testuser")

        # Password should be hashed, not plaintext
        assert user_info.password_hash != "password123"
        assert "$" in user_info.password_hash  # Salt separator
        assert len(user_info.password_hash) > 50  # Reasonable hash length

    @pytest.mark.asyncio
    async def test_user_creation_with_multiple_roles(self, authenticator):
        """Test user creation with multiple roles."""
        roles = {Role.ADMIN, Role.USER}
        user_info = await authenticator.create_user("multiuser", "password123", roles)

        assert user_info.roles == roles
        # Should have permissions from all roles
        expected_permissions = {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        assert user_info.permissions == expected_permissions
