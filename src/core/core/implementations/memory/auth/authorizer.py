# ABOUTME: In-memory implementation of AbstractAuthorizer
# ABOUTME: Provides authorization using role-based access control

from typing import Dict, Set, Optional
import threading

from core.interfaces.auth.authorizer import AbstractAuthorizer
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Role, Permission
from core.exceptions import AuthorizationError

from .models import MemoryAuthToken


class InMemoryAuthorizer(AbstractAuthorizer):
    """
    In-memory implementation of AbstractAuthorizer.

    This authorizer provides role-based access control (RBAC) using in-memory
    configuration. It supports both role-based and permission-based authorization
    checks with configurable permission mappings.

    Features:
    - Role-based authorization
    - Permission-based authorization
    - Configurable role-permission mappings
    - Thread-safe operations
    - Hierarchical role support
    - Resource-based permissions (future extension)
    """

    def __init__(self, custom_role_permissions: Optional[Dict[Role, Set[Permission]]] = None):
        """
        Initialize the in-memory authorizer.

        Args:
            custom_role_permissions: Optional custom role-permission mappings.
                                   If not provided, default mappings are used.
        """
        self._lock = threading.RLock()

        # Initialize role-permission mappings
        self._role_permissions = self._init_role_permissions(custom_role_permissions)

        # Initialize role hierarchy (for future extension)
        self._role_hierarchy = self._init_role_hierarchy()

    def _init_role_permissions(
        self, custom_mappings: Optional[Dict[Role, Set[Permission]]]
    ) -> Dict[Role, Set[Permission]]:
        """
        Initialize role-permission mappings.

        Args:
            custom_mappings: Optional custom role-permission mappings.

        Returns:
            Dictionary mapping roles to their permissions.
        """
        if custom_mappings:
            return custom_mappings.copy()

        # Default role-permission mappings
        return {
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
            Role.USER: {Permission.READ, Permission.WRITE},
        }

    def _init_role_hierarchy(self) -> Dict[Role, Set[Role]]:
        """
        Initialize role hierarchy for inheritance.

        Returns:
            Dictionary mapping roles to their inherited roles.
        """
        # Default hierarchy: ADMIN inherits all USER permissions
        return {Role.ADMIN: {Role.USER}, Role.USER: set()}

    async def authorize_permission(self, token: AuthToken, permission: Permission) -> None:
        """
        Check if the user associated with the given token has a specific permission.

        This method checks both direct permissions assigned to the user and
        permissions inherited from their roles.

        Args:
            token: The AuthToken representing the authenticated user.
            permission: The permission to check.

        Raises:
            AuthorizationError: If the user does not have the specified permission.
        """
        try:
            # Get user permissions
            user_permissions = self._get_user_permissions(token)

            # Check if user has the required permission
            if permission not in user_permissions:
                raise AuthorizationError(
                    message=f"User '{token.username}' does not have permission '{permission.value}'",
                    code="INSUFFICIENT_PERMISSION",
                    details={
                        "user_id": token.id,
                        "username": token.username,
                        "required_permission": permission.value,
                        "user_permissions": [p.value for p in user_permissions],
                        "user_roles": token.roles,
                    },
                )

        except AuthorizationError:
            raise
        except Exception as e:
            try:
                user_id = token.id
            except Exception:
                user_id = "unknown"

            raise AuthorizationError(
                message="Authorization check failed due to internal error",
                code="INTERNAL_ERROR",
                details={"error": str(e), "user_id": user_id, "permission": permission.value},
            )

    async def authorize_role(self, token: AuthToken, role: Role) -> None:
        """
        Check if the user associated with the given token has a specific role.

        This method checks both direct roles assigned to the user and
        roles inherited through the role hierarchy.

        Args:
            token: The AuthToken representing the authenticated user.
            role: The role to check.

        Raises:
            AuthorizationError: If the user does not have the specified role.
        """
        try:
            # Get user roles (including inherited)
            user_roles = self._get_user_roles(token)

            # Check if user has the required role
            if role not in user_roles:
                raise AuthorizationError(
                    message=f"User '{token.username}' does not have role '{role.value}'",
                    code="INSUFFICIENT_ROLE",
                    details={
                        "user_id": token.id,
                        "username": token.username,
                        "required_role": role.value,
                        "user_roles": [r.value for r in user_roles],
                        "direct_roles": token.roles,
                    },
                )

        except AuthorizationError:
            raise
        except Exception as e:
            raise AuthorizationError(
                message="Authorization check failed due to internal error",
                code="INTERNAL_ERROR",
                details={"error": str(e), "user_id": token.id, "role": role.value},
            )

    def _get_user_permissions(self, token: AuthToken) -> Set[Permission]:
        """
        Get all permissions for a user (direct + role-based).

        Args:
            token: The AuthToken representing the user.

        Returns:
            Set of all permissions the user has.
        """
        permissions = set()

        # Add direct permissions if token is MemoryAuthToken
        if isinstance(token, MemoryAuthToken):
            permissions.update(token.user_permissions)

        # Add role-based permissions
        user_roles = self._get_user_roles(token)

        with self._lock:
            for role in user_roles:
                role_permissions = self._role_permissions.get(role, set())
                permissions.update(role_permissions)

        return permissions

    def _get_user_roles(self, token: AuthToken) -> Set[Role]:
        """
        Get all roles for a user (direct + inherited).

        Args:
            token: The AuthToken representing the user.

        Returns:
            Set of all roles the user has.
        """
        roles = set()

        # Convert string roles to Role enum
        if isinstance(token, MemoryAuthToken):
            roles.update(token.user_roles)
        else:
            # Handle generic AuthToken protocol
            for role_str in token.roles:
                try:
                    role = Role(role_str)
                    roles.add(role)
                except ValueError:
                    # Skip invalid roles
                    continue

        # Add inherited roles
        with self._lock:
            inherited_roles = set()
            for role in roles:
                inherited_roles.update(self._get_inherited_roles(role))

            roles.update(inherited_roles)

        return roles

    def _get_inherited_roles(self, role: Role) -> Set[Role]:
        """
        Get all roles inherited by a given role.

        Args:
            role: The role to get inherited roles for.

        Returns:
            Set of inherited roles.
        """
        inherited = set()

        # Get direct inheritance
        direct_inherited = self._role_hierarchy.get(role, set())
        inherited.update(direct_inherited)

        # Get transitive inheritance (recursive)
        for inherited_role in direct_inherited:
            inherited.update(self._get_inherited_roles(inherited_role))

        return inherited

    async def has_permission(self, token: AuthToken, permission: Permission) -> bool:
        """
        Check if a user has a specific permission (non-throwing version).

        Args:
            token: The AuthToken representing the user.
            permission: The permission to check.

        Returns:
            True if the user has the permission, False otherwise.
        """
        try:
            await self.authorize_permission(token, permission)
            return True
        except AuthorizationError:
            return False

    async def has_role(self, token: AuthToken, role: Role) -> bool:
        """
        Check if a user has a specific role (non-throwing version).

        Args:
            token: The AuthToken representing the user.
            role: The role to check.

        Returns:
            True if the user has the role, False otherwise.
        """
        try:
            await self.authorize_role(token, role)
            return True
        except AuthorizationError:
            return False

    async def get_user_permissions(self, token: AuthToken) -> Set[Permission]:
        """
        Get all permissions for a user.

        Args:
            token: The AuthToken representing the user.

        Returns:
            Set of all permissions the user has.
        """
        return self._get_user_permissions(token)

    async def get_user_roles(self, token: AuthToken) -> Set[Role]:
        """
        Get all roles for a user.

        Args:
            token: The AuthToken representing the user.

        Returns:
            Set of all roles the user has.
        """
        return self._get_user_roles(token)

    async def add_role_permission(self, role: Role, permission: Permission) -> None:
        """
        Add a permission to a role.

        Args:
            role: The role to add the permission to.
            permission: The permission to add.
        """
        with self._lock:
            if role not in self._role_permissions:
                self._role_permissions[role] = set()

            self._role_permissions[role].add(permission)

    async def remove_role_permission(self, role: Role, permission: Permission) -> None:
        """
        Remove a permission from a role.

        Args:
            role: The role to remove the permission from.
            permission: The permission to remove.
        """
        with self._lock:
            if role in self._role_permissions:
                self._role_permissions[role].discard(permission)

    async def set_role_permissions(self, role: Role, permissions: Set[Permission]) -> None:
        """
        Set all permissions for a role.

        Args:
            role: The role to set permissions for.
            permissions: The set of permissions to assign.
        """
        with self._lock:
            self._role_permissions[role] = permissions.copy()

    async def get_role_permissions(self, role: Role) -> Set[Permission]:
        """
        Get all permissions for a role.

        Args:
            role: The role to get permissions for.

        Returns:
            Set of permissions assigned to the role.
        """
        with self._lock:
            return self._role_permissions.get(role, set()).copy()

    async def add_role_inheritance(self, parent_role: Role, child_role: Role) -> None:
        """
        Add role inheritance (parent inherits child's permissions).

        Args:
            parent_role: The role that will inherit permissions.
            child_role: The role whose permissions will be inherited.
        """
        with self._lock:
            if parent_role not in self._role_hierarchy:
                self._role_hierarchy[parent_role] = set()

            self._role_hierarchy[parent_role].add(child_role)

    async def remove_role_inheritance(self, parent_role: Role, child_role: Role) -> None:
        """
        Remove role inheritance.

        Args:
            parent_role: The role that will no longer inherit permissions.
            child_role: The role whose permissions will no longer be inherited.
        """
        with self._lock:
            if parent_role in self._role_hierarchy:
                self._role_hierarchy[parent_role].discard(child_role)

    async def get_role_hierarchy(self) -> Dict[Role, Set[Role]]:
        """
        Get the current role hierarchy.

        Returns:
            Dictionary mapping roles to their inherited roles.
        """
        with self._lock:
            return {role: inherited_roles.copy() for role, inherited_roles in self._role_hierarchy.items()}

    async def reset_to_defaults(self) -> None:
        """
        Reset the authorizer to default configuration.
        """
        with self._lock:
            self._role_permissions = self._init_role_permissions(None)
            self._role_hierarchy = self._init_role_hierarchy()
