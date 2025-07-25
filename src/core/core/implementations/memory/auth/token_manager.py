# ABOUTME: In-memory implementation of AbstractTokenManager using UUID tokens
# ABOUTME: Provides simple UUID-based token management for testing and development environments

import time
import threading
import uuid
from typing import Any, Dict, Optional

from core.interfaces.auth.token_manager import AbstractTokenManager
from core.models.auth.auth_token import AuthToken
from core.models.types import UserData
from core.exceptions.base import AuthenticationException

from .models import MemoryAuthToken


class TokenData:
    """
    Internal token data structure for UUID-based tokens.

    Stores all token-related information in memory including user data,
    timestamps, and token status.
    """

    def __init__(
        self, user_id: str, username: str, roles: list[str], permissions: list[str], issued_at: float, expires_at: float
    ):
        self.user_id = user_id
        self.username = username
        self.roles = roles
        self.permissions = permissions
        self.issued_at = issued_at
        self.expires_at = expires_at
        self.is_active = True

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert token data to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
        }


class InMemoryTokenManager(AbstractTokenManager):
    """
    In-memory implementation of AbstractTokenManager using UUID tokens.

    This token manager uses simple UUID tokens for authentication with
    all token data stored in memory. It's designed for testing and development
    environments where no external dependencies are desired.

    Features:
    - UUID-based token generation (no external dependencies)
    - In-memory token storage and validation
    - Token expiration checking
    - Token refresh with configurable TTL
    - Token revocation using status flags
    - Thread-safe operations
    - Automatic cleanup of expired tokens

    The implementation maintains an internal mapping of UUID tokens to
    user data and provides all standard token management operations
    without any external library dependencies.

    Note:
        This implementation stores all data in memory and will lose
        all tokens when the process restarts. For production use,
        consider implementing persistent storage in the integrations layer.
    """

    def __init__(
        self,
        default_ttl: int = 3600,
        refresh_ttl: int = 7200,
    ):
        """
        Initialize the in-memory UUID token manager.

        Args:
            default_ttl: Default token time-to-live in seconds (default: 1 hour)
            refresh_ttl: Refresh token time-to-live in seconds (default: 2 hours)
        """
        self.default_ttl = default_ttl
        self.refresh_ttl = refresh_ttl

        # Thread-safe storage for active tokens
        self._tokens: Dict[str, TokenData] = {}
        self._lock = threading.RLock()

    def generate_token(self, user_data: UserData) -> str:
        """
        Generates a new UUID authentication token based on provided user data.

        This method creates a UUID token and stores the associated user data
        in memory. The token is a simple UUID string that can be used for
        subsequent authentication requests.

        Args:
            user_data: A dictionary containing user information such as:
                      - user_id: Unique user identifier (required)
                      - username: User's username (required)
                      - roles: List of user roles (optional, default: [])
                      - permissions: List of user permissions (optional, default: [])

        Returns:
            A UUID token string ready for use in authentication.

        Raises:
            AuthenticationException: If token generation fails due to invalid
                                   user data or missing required fields.
        """
        try:
            # Validate required fields
            user_id = user_data.get("user_id")
            username = user_data.get("username")

            if not user_id or not username:
                raise AuthenticationException(
                    message="Missing required user data (user_id or username)",
                    code="INVALID_USER_DATA",
                    details={"provided_data": list(user_data.keys())},
                )

            # Generate UUID token
            token = str(uuid.uuid4())

            # Prepare token data
            current_time = time.time()
            roles = user_data.get("roles", [])
            permissions = user_data.get("permissions", [])

            # Ensure roles and permissions are lists of strings
            if not isinstance(roles, list):
                roles = []
            if not isinstance(permissions, list):
                permissions = []

            roles = [str(role) for role in roles]
            permissions = [str(perm) for perm in permissions]

            # Create token data
            token_data = TokenData(
                user_id=str(user_id),
                username=str(username),
                roles=roles,
                permissions=permissions,
                issued_at=current_time,
                expires_at=current_time + self.default_ttl,
            )

            # Store token in memory
            with self._lock:
                self._tokens[token] = token_data

            return token

        except AuthenticationException:
            raise  # Re-raise our own exceptions
        except Exception as e:
            raise AuthenticationException(
                message="Token generation failed due to internal error",
                code="INTERNAL_ERROR",
                details={"error": str(e)},
            )

    def validate_token(self, token: str) -> AuthToken:
        """
        Validates a UUID authentication token and extracts user information.

        This method performs comprehensive token validation including:
        - Token existence checking
        - Expiration time checking
        - Active status checking
        - User data extraction

        Args:
            token: The UUID token string to validate.

        Returns:
            An AuthToken object containing the validated user's identity and roles.

        Raises:
            AuthenticationException: If the token is invalid, expired, revoked,
                                   or not found in the token store.
        """
        try:
            if not token or not isinstance(token, str):
                raise AuthenticationException(
                    message="Invalid token format", code="INVALID_TOKEN", details={"token_type": type(token).__name__}
                )

            # Retrieve token data
            with self._lock:
                token_data = self._tokens.get(token)

            if token_data is None:
                raise AuthenticationException(message="Token not found", code="TOKEN_NOT_FOUND")

            # Check if token is active
            if not token_data.is_active:
                raise AuthenticationException(message="Token has been revoked", code="TOKEN_REVOKED")

            # Check expiration
            if token_data.is_expired():
                # Clean up expired token
                with self._lock:
                    del self._tokens[token]

                raise AuthenticationException(message="Token has expired", code="TOKEN_EXPIRED")

            # Convert to enum types
            from core.models.auth.enum import Role, Permission

            user_roles = set()
            for role_str in token_data.roles:
                try:
                    user_roles.add(Role(role_str))
                except ValueError:
                    # Skip invalid roles but continue
                    continue

            user_permissions = set()
            for perm_str in token_data.permissions:
                try:
                    user_permissions.add(Permission(perm_str))
                except ValueError:
                    # Skip invalid permissions but continue
                    continue

            # Create and return AuthToken
            auth_token = MemoryAuthToken(
                user_id=token_data.user_id,
                username=token_data.username,
                user_roles=user_roles,
                user_permissions=user_permissions,
                expires_at=token_data.expires_at,
                issued_at=token_data.issued_at,
            )

            return auth_token

        except AuthenticationException:
            raise
        except Exception as e:
            raise AuthenticationException(
                message="Token validation failed due to internal error",
                code="INTERNAL_ERROR",
                details={"error": str(e)},
            )

    def refresh_token(self, token: str) -> str:
        """
        Refreshes an existing UUID authentication token.

        This method validates the current token and generates a new token
        with extended expiration time. The original token is revoked to
        prevent reuse.

        Args:
            token: The existing UUID token string to be refreshed.

        Returns:
            A new UUID token string with extended validity period.

        Raises:
            AuthenticationException: If the token cannot be refreshed due to
                                   invalid token, expiration, or other errors.
        """
        try:
            # Validate the current token first
            auth_token = self.validate_token(token)

            # Prepare user data for new token generation
            user_data = {
                "user_id": auth_token.id,
                "username": auth_token.username,
                "roles": [str(role) for role in auth_token.roles],
                "permissions": [str(perm) for perm in getattr(auth_token, "permissions", [])],
            }

            # Generate new token with refresh TTL
            current_time = time.time()
            new_token = str(uuid.uuid4())

            # Create new token data with extended expiration
            user_id_val = user_data["user_id"]
            username_val = user_data["username"]
            roles_val = user_data["roles"]
            permissions_val = user_data["permissions"]

            new_token_data = TokenData(
                user_id=user_id_val[0] if isinstance(user_id_val, (list, tuple)) else str(user_id_val),
                username=username_val[0] if isinstance(username_val, (list, tuple)) else str(username_val),
                roles=list(roles_val) if not isinstance(roles_val, list) else roles_val,
                permissions=list(permissions_val) if not isinstance(permissions_val, list) else permissions_val,
                issued_at=current_time,
                expires_at=current_time + self.refresh_ttl,
            )

            # Revoke old token and store new token
            with self._lock:
                # Revoke old token
                if token in self._tokens:
                    self._tokens[token].is_active = False

                # Store new token
                self._tokens[new_token] = new_token_data

            return new_token

        except AuthenticationException:
            raise
        except Exception as e:
            raise AuthenticationException(
                message="Token refresh failed due to internal error",
                code="INTERNAL_ERROR",
                details={"error": str(e)},
            )

    def revoke_token(self, token: str) -> None:
        """
        Revokes a UUID authentication token, making it invalid for future use.

        This method marks the token as inactive in the internal storage.
        Once revoked, the token will fail validation even if it hasn't expired yet.

        Args:
            token: The UUID token string to be revoked.

        Raises:
            AuthenticationException: If the token cannot be revoked due to
                                   invalid format or token not found.
        """
        try:
            if not token or not isinstance(token, str):
                raise AuthenticationException(
                    message="Invalid token format for revocation",
                    code="INVALID_TOKEN",
                    details={"token_type": type(token).__name__},
                )

            # Find and revoke token
            with self._lock:
                token_data = self._tokens.get(token)

                if token_data is None:
                    raise AuthenticationException(
                        message="Cannot revoke token: token not found", code="TOKEN_NOT_FOUND"
                    )

                # Mark token as inactive
                token_data.is_active = False

        except AuthenticationException:
            raise
        except Exception as e:
            raise AuthenticationException(
                message="Token revocation failed due to internal error",
                code="INTERNAL_ERROR",
                details={"error": str(e)},
            )

    # Additional utility methods for token management

    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens from internal storage.

        This method removes expired tokens from memory and can be called
        periodically to free up memory resources.

        Returns:
            The number of expired tokens that were cleaned up.
        """
        current_time = time.time()
        cleaned_count = 0

        with self._lock:
            # Find expired tokens
            expired_tokens = [token for token, data in self._tokens.items() if data.expires_at < current_time]

            # Remove expired tokens
            for token in expired_tokens:
                del self._tokens[token]
                cleaned_count += 1

        return cleaned_count

    def get_active_token_count(self) -> int:
        """
        Get the current number of active tokens in storage.

        Returns:
            The number of active (non-revoked, non-expired) tokens.
        """
        current_time = time.time()

        with self._lock:
            active_count = sum(1 for data in self._tokens.values() if data.is_active and data.expires_at > current_time)

        return active_count

    def get_total_token_count(self) -> int:
        """
        Get the total number of tokens in storage (active + inactive).

        Returns:
            The total number of tokens currently stored.
        """
        with self._lock:
            return len(self._tokens)

    def get_revoked_token_count(self) -> int:
        """
        Get the current number of revoked tokens in storage.

        Returns:
            The number of tokens currently marked as revoked.
        """
        with self._lock:
            revoked_count = sum(1 for data in self._tokens.values() if not data.is_active)

        return revoked_count

    def clear_all_tokens(self) -> None:
        """
        Clear all tokens from storage.

        This method removes all tokens from memory. Use with caution
        as this will invalidate all currently active sessions.
        """
        with self._lock:
            self._tokens.clear()

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific token (for debugging/admin purposes).

        Args:
            token: The token to get information about.

        Returns:
            Dictionary containing token information, or None if token not found.
        """
        with self._lock:
            token_data = self._tokens.get(token)

            if token_data is None:
                return None

            return {
                **token_data.to_dict(),
                "is_expired": token_data.is_expired(),
                "time_until_expiry": max(0, token_data.expires_at - time.time()),
            }

    def get_user_tokens(self, user_id: str) -> list[str]:
        """
        Get all active tokens for a specific user.

        Args:
            user_id: The user ID to search for.

        Returns:
            List of active token strings for the user.
        """
        current_time = time.time()
        user_tokens = []

        with self._lock:
            for token, data in self._tokens.items():
                if data.user_id == user_id and data.is_active and data.expires_at > current_time:
                    user_tokens.append(token)

        return user_tokens

    def revoke_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a specific user.

        Args:
            user_id: The user ID whose tokens should be revoked.

        Returns:
            The number of tokens that were revoked.
        """
        revoked_count = 0

        with self._lock:
            for data in self._tokens.values():
                if data.user_id == user_id and data.is_active:
                    data.is_active = False
                    revoked_count += 1

        return revoked_count
