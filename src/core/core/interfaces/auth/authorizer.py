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

        This method use permission formats:
        - Permission enum (e.g., Permission.READ)

        If the user does not possess the required permission, an authorization
        failure exception should be raised.

        Args:
            token: The `AuthToken` representing the authenticated user.
            permission: The permission to check. Can be:
                - Permission enum value (e.g., Permission.READ)

        Raises:
            AuthorizationError: If the user does not have the specified permission.
        """
        pass

    @abstractmethod
    async def authorize_role(self, token: AuthToken, role: Role) -> None:
        """
        Checks if the user associated with the given token has a specific role.

        If the user does not possess the required role, an authorization
        failure exception should be raised.

        Args:
            token: The `AuthToken` representing the authenticated user.
            role: The specific role enum value to check. Can be:
                - Role enum value (e.g., Role.ADMIN)

        Raises:
            AuthorizationError: If the user does not have the specified role.
        """
        pass
