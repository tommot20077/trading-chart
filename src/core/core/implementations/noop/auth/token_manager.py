# ABOUTME: NoOp implementation of AbstractTokenManager that provides fake token operations
# ABOUTME: Provides minimal token management functionality for testing scenarios

from typing import Any

from core.interfaces.auth.token_manager import AbstractTokenManager
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Role, Permission
from core.implementations.memory.auth.models import MemoryAuthToken


class NoOpTokenManager(AbstractTokenManager):
    """
    No-operation implementation of AbstractTokenManager.

    This implementation provides minimal token management functionality that
    returns fake tokens without performing any actual token operations. It's
    useful for testing, performance benchmarking, and scenarios where token
    management is not required.

    Features:
    - Always returns fake successful tokens
    - No actual token generation, validation, or storage
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where token management should be bypassed
    - Performance benchmarking without token management overhead
    - Development environments where token management is not needed
    - Fallback when token management systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation token manager."""
        # No initialization needed for NoOp implementation
        pass

    def generate_token(self, user_data: dict[str, Any]) -> str:
        """
        Generate a token - returns a fake token string.

        This implementation always returns a fake token without performing
        any actual token generation.

        Args:
            user_data: User data for token generation (ignored)

        Returns:
            A fake token string
        """
        return "noop-token-fake"

    def validate_token(self, token: str) -> AuthToken:
        """
        Validate a token - always returns a fake successful auth token.

        This implementation always returns a successful auth token without
        performing any actual token validation.

        Args:
            token: The token to validate (ignored)

        Returns:
            A fake AuthToken with default permissions
        """
        return MemoryAuthToken(
            user_id="noop-user",
            username="noop-user",
            user_roles={Role.USER},
            user_permissions={Permission.READ},
            expires_at=None,  # Never expires in NoOp
        )

    def refresh_token(self, token: str) -> str:
        """
        Refresh a token - returns a fake refreshed token.

        This implementation always returns a fake refreshed token without
        performing any actual token refresh operations.

        Args:
            token: The token to refresh (ignored)

        Returns:
            A fake refreshed token string
        """
        return "noop-token-refreshed"

    def revoke_token(self, token: str) -> None:
        """
        Revoke a token - always succeeds without actual revocation.

        This implementation always succeeds without performing any
        actual token revocation.

        Args:
            token: The token to revoke (ignored)

        Returns:
            None (always succeeds)
        """
        # Always succeed in NoOp implementation
        pass
