# ABOUTME: Authentication data models for in-memory implementations
# ABOUTME: Provides MemoryAuthToken and UserInfo classes

import time
from dataclasses import dataclass, field
from typing import Set

from core.models.auth.enum import Role, Permission


@dataclass
class UserInfo:
    """
    User information stored in memory for authentication.

    This class represents user data stored in the in-memory authentication system.
    It contains the essential information needed for authentication and authorization.
    """

    username: str
    password_hash: str
    roles: Set[Role]
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: float = field(default_factory=lambda: time.time())
    last_login: float | None = None

    def has_role(self, role: Role) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def add_role(self, role: Role) -> None:
        """Add a role to the user."""
        self.roles.add(role)

    def remove_role(self, role: Role) -> None:
        """Remove a role from the user."""
        self.roles.discard(role)

    def add_permission(self, permission: Permission) -> None:
        """Add a permission to the user."""
        self.permissions.add(permission)

    def remove_permission(self, permission: Permission) -> None:
        """Remove a permission from the user."""
        self.permissions.discard(permission)


@dataclass
class MemoryAuthToken:
    """
    In-memory implementation of AuthToken protocol.

    This class provides a concrete implementation of the AuthToken protocol
    for use in the in-memory authentication system.
    """

    user_id: str
    username: str
    user_roles: Set[Role]
    user_permissions: Set[Permission]
    expires_at: float | None = None
    issued_at: float | None = field(default_factory=lambda: time.time())

    @property
    def id(self) -> str:
        """The unique identifier for the authenticated user."""
        return self.user_id

    @property
    def roles(self) -> list[str]:
        """A list of roles assigned to the authenticated user."""
        return [role.value for role in self.user_roles]

    @property
    def permissions(self) -> Set[Permission]:
        """A set of permissions assigned to the authenticated user."""
        return self.user_permissions

    @property
    def token(self) -> str:
        """The token string representation."""
        # For NoOp/testing purposes, return a fixed token value
        if self.user_id == "noop-user":
            return "noop-token"
        return f"memory-token-{self.user_id}-{self.issued_at}"

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def has_role(self, role: Role) -> bool:
        """Check if the token has a specific role."""
        return role in self.user_roles

    def has_permission(self, permission: Permission) -> bool:
        """Check if the token has a specific permission."""
        return permission in self.user_permissions

    def time_until_expiry(self) -> float | None:
        """Get the time in seconds until the token expires."""
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - time.time())

    def __str__(self) -> str:
        """String representation of the token."""
        return f"MemoryAuthToken(user_id={self.user_id}, username={self.username})"

    def __repr__(self) -> str:
        """Detailed string representation of the token."""
        return (
            f"MemoryAuthToken("
            f"user_id={self.user_id!r}, "
            f"username={self.username!r}, "
            f"roles={self.roles!r}, "
            f"expires_at={self.expires_at}, "
            f"issued_at={self.issued_at}"
            f")"
        )
