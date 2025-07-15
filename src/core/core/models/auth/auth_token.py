from typing import Protocol


class AuthToken(Protocol):
    """
    Protocol for authentication tokens.

    This protocol defines the essential attributes that any object representing
    an authenticated user's token must possess. It allows for flexible token
    implementations (e.g., JWT, session tokens) while ensuring a consistent
    interface for accessing user identity and roles.
    """

    @property
    def id(self) -> str:
        """
        The unique identifier for the authenticated user.

        This ID should be stable and uniquely identify a user across the system.
        """
        ...

    @property
    def username(self) -> str:
        """
        The human-readable username of the authenticated user.

        This is typically used for display purposes or logging.
        """
        ...

    @property
    def roles(self) -> list[str]:
        """
        A list of roles assigned to the authenticated user.

        Roles are typically used for coarse-grained authorization checks (e.g., "admin", "viewer").
        """
        ...

    @property
    def expires_at(self) -> float | None:
        """
        The Unix timestamp (seconds since epoch) when the token is set to expire.

        If the token does not expire (e.g., a long-lived API key), this property
        should return `None`. This is crucial for managing token lifetimes.
        """
        ...

    @property
    def issued_at(self) -> float | None:
        """
        The Unix timestamp (seconds since epoch) when the token was issued.

        This is useful for tracking token creation time and implementing
        token rotation strategies.
        """
        ...
