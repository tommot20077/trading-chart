# ABOUTME: In-memory implementation of AbstractAuthenticator
# ABOUTME: Provides authentication using in-memory user database

from typing import Set
import time
import threading

from core.interfaces.auth.authenticator import AbstractAuthenticator
from core.models.auth.auth_request import AuthRequest
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Role
from core.exceptions import AuthenticationException

from .models import UserInfo, MemoryAuthToken
from .utils import (
    hash_password,
    verify_password,
    generate_user_id,
    generate_session_token,
    extract_bearer_token,
    get_default_permissions_for_role,
    validate_username,
    validate_password,
)


class InMemoryAuthenticator(AbstractAuthenticator):
    """
    In-memory implementation of AbstractAuthenticator.

    This authenticator maintains a simple in-memory user database and supports
    Bearer token authentication. It's designed for testing and development
    purposes where no external authentication service is available.

    Features:
    - User management (create, update, delete users)
    - Password authentication with secure hashing
    - Bearer token authentication
    - Role-based permissions
    - Thread-safe operations
    - Token expiration support
    """

    def __init__(self, default_token_ttl: int = 3600):
        """
        Initialize the in-memory authenticator.

        Args:
            default_token_ttl: Default token time-to-live in seconds (default: 1 hour)
        """
        self.default_token_ttl = default_token_ttl
        self._users: dict[str, UserInfo] = {}
        self._tokens: dict[str, MemoryAuthToken] = {}
        self._lock = threading.RLock()

        # Create default admin user
        self._create_default_users()

    def _create_default_users(self) -> None:
        """Create default users for testing and development."""
        # Create admin user
        admin_user = UserInfo(
            username="admin",
            password_hash=hash_password("admin123"),
            roles={Role.ADMIN},
            permissions=get_default_permissions_for_role(Role.ADMIN),
        )

        # Create regular user
        user = UserInfo(
            username="user",
            password_hash=hash_password("user123"),
            roles={Role.USER},
            permissions=get_default_permissions_for_role(Role.USER),
        )

        with self._lock:
            self._users["admin"] = admin_user
            self._users["user"] = user

    async def authenticate(self, request: AuthRequest) -> AuthToken:
        """
        Authenticate an incoming request and return the associated user token.

        This method supports Bearer token authentication. It extracts the token
        from the Authorization header and validates it against the stored tokens.

        Args:
            request: The incoming authentication request.

        Returns:
            An AuthToken representing the authenticated user.

        Raises:
            AuthenticationException: If authentication fails.
        """
        try:
            # Get Authorization header
            auth_header = request.get_header("Authorization")
            if auth_header is None:
                raise AuthenticationException(message="Missing Authorization header", code="MISSING_AUTH_HEADER")

            # Extract Bearer token
            try:
                token = extract_bearer_token(auth_header)
            except ValueError as e:
                raise AuthenticationException(
                    message="Invalid Authorization header format", code="INVALID_AUTH_FORMAT", details={"error": str(e)}
                )

            # Validate token
            auth_token = await self._validate_token(token)

            # Update last login time
            with self._lock:
                if auth_token.username in self._users:
                    self._users[auth_token.username].last_login = time.time()

            return auth_token

        except AuthenticationException:
            raise
        except Exception as e:
            raise AuthenticationException(
                message="Authentication failed due to internal error", code="INTERNAL_ERROR", details={"error": str(e)}
            )

    async def _validate_token(self, token: str) -> MemoryAuthToken:
        """
        Validate a Bearer token.

        Args:
            token: The token to validate.

        Returns:
            The validated MemoryAuthToken.

        Raises:
            AuthenticationException: If the token is invalid or expired.
        """
        with self._lock:
            if token not in self._tokens:
                raise AuthenticationException(message="Invalid or expired token", code="INVALID_TOKEN")

            auth_token = self._tokens[token]

            # Check if token is expired
            if auth_token.is_expired():
                # Remove expired token
                del self._tokens[token]
                raise AuthenticationException(
                    message="Token has expired", code="TOKEN_EXPIRED", details={"expires_at": auth_token.expires_at}
                )

            # Check if user is still active
            if auth_token.username in self._users:
                user_info = self._users[auth_token.username]
                if not user_info.is_active:
                    raise AuthenticationException(
                        message="User account is inactive",
                        code="USER_INACTIVE",
                        details={"username": auth_token.username},
                    )

            return auth_token

    async def login(self, username: str, password: str) -> tuple[str, MemoryAuthToken]:
        """
        Authenticate a user with username and password.

        Args:
            username: The username.
            password: The password.

        Returns:
            A tuple containing the token string and the MemoryAuthToken.

        Raises:
            AuthenticationException: If authentication fails.
        """
        with self._lock:
            # Check if user exists
            if username not in self._users:
                raise AuthenticationException(message="Invalid username or password", code="INVALID_CREDENTIALS")

            user_info = self._users[username]

            # Check if user is active
            if not user_info.is_active:
                raise AuthenticationException(
                    message="User account is inactive", code="USER_INACTIVE", details={"username": username}
                )

            # Verify password
            if not verify_password(password, user_info.password_hash):
                raise AuthenticationException(message="Invalid username or password", code="INVALID_CREDENTIALS")

            # Generate token
            token = generate_session_token()
            expires_at = time.time() + self.default_token_ttl

            # Create auth token
            auth_token = MemoryAuthToken(
                user_id=generate_user_id(),
                username=username,
                user_roles=user_info.roles.copy(),
                user_permissions=user_info.permissions.copy(),
                expires_at=expires_at,
            )

            # Store token
            self._tokens[token] = auth_token

            # Update last login
            user_info.last_login = time.time()

            return token, auth_token

    async def logout(self, token: str) -> None:
        """
        Logout a user by invalidating their token.

        Args:
            token: The token to invalidate.
        """
        with self._lock:
            self._tokens.pop(token, None)

    async def create_user(self, username: str, password: str, roles: Set[Role] | None = None) -> UserInfo:
        """
        Create a new user.

        Args:
            username: The username.
            password: The password.
            roles: The user roles (default: {Role.USER}).

        Returns:
            The created UserInfo.

        Raises:
            AuthenticationException: If user creation fails.
        """
        if roles is None:
            roles = {Role.USER}

        # Validate input
        if not validate_username(username):
            raise AuthenticationException(
                message="Invalid username format", code="INVALID_USERNAME", details={"username": username}
            )

        if not validate_password(password):
            raise AuthenticationException(message="Invalid password format", code="INVALID_PASSWORD")

        with self._lock:
            # Check if user already exists
            if username in self._users:
                raise AuthenticationException(
                    message="User already exists", code="USER_EXISTS", details={"username": username}
                )

            # Calculate permissions based on roles
            permissions = set()
            for role in roles:
                permissions.update(get_default_permissions_for_role(role))

            # Create user
            user_info = UserInfo(
                username=username, password_hash=hash_password(password), roles=roles, permissions=permissions
            )

            self._users[username] = user_info

            return user_info

    async def delete_user(self, username: str) -> None:
        """
        Delete a user.

        Args:
            username: The username to delete.

        Raises:
            AuthenticationException: If the user doesn't exist.
        """
        with self._lock:
            if username not in self._users:
                raise AuthenticationException(
                    message="User not found", code="USER_NOT_FOUND", details={"username": username}
                )

            # Remove user
            del self._users[username]

            # Invalidate all tokens for this user
            tokens_to_remove = [token for token, auth_token in self._tokens.items() if auth_token.username == username]

            for token in tokens_to_remove:
                del self._tokens[token]

    # Security-enhanced method to update user password
    async def update_user_password(self, username: str, new_password: str) -> None:
        """
        Update a user's password.

        Args:
            username: The username.
            new_password: The new password.

        Raises:
            AuthenticationException: If the user doesn't exist or password is invalid.
        """
        if not validate_password(new_password):
            raise AuthenticationException(message="Invalid password format", code="INVALID_PASSWORD")

        with self._lock:
            if username not in self._users:
                raise AuthenticationException(
                    message="User not found", code="USER_NOT_FOUND", details={"username": username}
                )

            # Update password
            self._users[username].password_hash = hash_password(new_password)

            # Invalidate all tokens for this user (force re-authentication)
            tokens_to_remove = [token for token, auth_token in self._tokens.items() if auth_token.username == username]

            for token in tokens_to_remove:
                del self._tokens[token]

    async def get_user(self, username: str) -> UserInfo | None:
        """
        Get user information.

        Args:
            username: The username.

        Returns:
            UserInfo if found, None otherwise.
        """
        with self._lock:
            return self._users.get(username)

    async def list_users(self) -> dict[str, UserInfo]:
        """
        List all users.

        Returns:
            A dictionary mapping usernames to UserInfo objects.
        """
        with self._lock:
            return self._users.copy()

    async def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens.

        Returns:
            The number of expired tokens removed.
        """
        with self._lock:
            expired_tokens = [token for token, auth_token in self._tokens.items() if auth_token.is_expired()]

            for token in expired_tokens:
                del self._tokens[token]

            return len(expired_tokens)

    async def get_token_count(self) -> int:
        """
        Get the current number of active tokens.

        Returns:
            The number of active tokens.
        """
        with self._lock:
            return len(self._tokens)

    async def get_user_count(self) -> int:
        """
        Get the current number of users.

        Returns:
            The number of users.
        """
        with self._lock:
            return len(self._users)
