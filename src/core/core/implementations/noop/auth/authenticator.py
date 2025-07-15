# ABOUTME: NoOp implementation of AbstractAuthenticator that always succeeds
# ABOUTME: Provides minimal authentication functionality for testing scenarios

from core.interfaces.auth.authenticator import AbstractAuthenticator
from core.models.auth.auth_request import AuthRequest
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Role, Permission
from core.implementations.memory.auth.models import MemoryAuthToken


class NoOpAuthenticator(AbstractAuthenticator):
    """
    No-operation implementation of AbstractAuthenticator.

    This implementation provides minimal authentication functionality that always
    succeeds without performing any actual authentication operations. It's useful
    for testing, performance benchmarking, and scenarios where authentication
    is not required.

    Features:
    - Always returns successful authentication
    - Creates fake auth tokens with default permissions
    - No actual credential validation
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where authentication should be bypassed
    - Performance benchmarking without authentication overhead
    - Development environments where authentication is not needed
    - Fallback when authentication systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation authenticator."""
        # No initialization needed for NoOp implementation
        pass

    async def authenticate(self, request: AuthRequest) -> AuthToken:
        """
        Authenticate a request - always returns a fake successful token.

        This implementation always returns a successful authentication token
        without performing any actual credential validation.

        Args:
            request: The authentication request (ignored)

        Returns:
            A fake AuthToken with default permissions
        """
        # Create a fake auth token that always succeeds
        return MemoryAuthToken(
            user_id="noop-user",
            username="noop-user",
            user_roles={Role.USER},
            user_permissions={Permission.READ},
            expires_at=None,  # Never expires in NoOp
        )
