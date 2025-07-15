# ABOUTME: NoOp implementation of AbstractAuthorizer that always allows access
# ABOUTME: Provides minimal authorization functionality for testing scenarios

from core.interfaces.auth.authorizer import AbstractAuthorizer
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Permission, Role


class NoOpAuthorizer(AbstractAuthorizer):
    """
    No-operation implementation of AbstractAuthorizer.

    This implementation provides minimal authorization functionality that always
    allows access without performing any actual authorization checks. It's useful
    for testing, performance benchmarking, and scenarios where authorization
    is not required.

    Features:
    - Always allows access (never raises authorization errors)
    - No actual permission or role validation
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where authorization should be bypassed
    - Performance benchmarking without authorization overhead
    - Development environments where authorization is not needed
    - Fallback when authorization systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation authorizer."""
        # No initialization needed for NoOp implementation
        pass

    async def authorize_permission(self, token: AuthToken, permission: Permission) -> None:
        """
        Check permission - always allows access without validation.

        This implementation always allows access without performing any
        actual permission checks.

        Args:
            token: The auth token (ignored)
            permission: The permission to check (ignored)

        Returns:
            None (always succeeds, never raises authorization errors)
        """
        # Always allow access in NoOp implementation
        pass

    async def authorize_role(self, token: AuthToken, role: Role) -> None:
        """
        Check role - always allows access without validation.

        This implementation always allows access without performing any
        actual role checks.

        Args:
            token: The auth token (ignored)
            role: The role to check (ignored)

        Returns:
            None (always succeeds, never raises authorization errors)
        """
        # Always allow access in NoOp implementation
        pass
