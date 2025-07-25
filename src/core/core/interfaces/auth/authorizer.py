# ABOUTME: Abstract authorizer interface for user permission and role-based access control
# ABOUTME: Defines the contract for components that check user permissions and roles after authentication

from abc import abstractmethod, ABC

from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Permission, Role


class AbstractAuthorizer(ABC):
    """
    Abstract authorizer for checking user permissions and roles.

    This abstract class defines the contract for components responsible for
    determining if an authenticated user (represented by an `AuthToken`) has
    the necessary permissions or roles to perform a specific action or access
    a particular resource.
    """

    @abstractmethod
    async def authorize_permission(self, token: AuthToken, permission: Permission) -> None:
        """
        Checks if the user associated with the given token has a specific permission.

        This asynchronous method determines if the authenticated user, represented by the
        `AuthToken`, possesses the specified `Permission`. If the user lacks the
        required permission, an `AuthorizationError` is raised, preventing unauthorized
        access or operations.

        Args:
            token (AuthToken): The `AuthToken` representing the authenticated user.
            permission (Permission): The specific permission to check (e.g., `Permission.READ`).

        Returns:
            None: This method does not return a value; it raises an exception on failure.

        Raises:
            AuthorizationError: If the user does not have the specified permission.
        """
        pass

    @abstractmethod
    async def authorize_role(self, token: AuthToken, role: Role) -> None:
        """
        Checks if the user associated with the given token has a specific role.

        This asynchronous method verifies if the authenticated user, identified by the
        `AuthToken`, is assigned the specified `Role`. If the user does not possess
        the required role, an `AuthorizationError` is raised, ensuring role-based
        access control.

        Args:
            token (AuthToken): The `AuthToken` representing the authenticated user.
            role (Role): The specific role enum value to check (e.g., `Role.ADMIN`).

        Returns:
            None: This method does not return a value; it raises an exception on failure.

        Raises:
            AuthorizationError: If the user does not have the specified role.
        """
        pass
