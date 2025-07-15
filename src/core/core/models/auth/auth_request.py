from typing import Protocol


class AuthRequest(Protocol):
    """
    Protocol for framework-agnostic authentication requests.

    This protocol defines the minimal interface required for an incoming request
    object to be processed by an `AbstractAuthenticator` or `AbstractSecurityMiddleware`.
    It abstracts away framework-specific request details (e.g., Flask's `request`,
    FastAPI's `Request`) to enable reusable security logic.
    """

    def get_header(self, name: str) -> str | None:
        """
        Retrieves the value of a specific HTTP header from the request.

        Args:
            name: The name of the HTTP header to retrieve (case-insensitive).

        Returns:
            The string value of the header if found, otherwise `None`.
        """
        ...

    @property
    def client_id(self) -> str | None:
        """
        An identifier for the client making the request, primarily used for rate limiting.

        This could be an IP address, an API key, or any other unique client identifier.
        Returns `None` if no client ID can be determined.
        """
        ...
