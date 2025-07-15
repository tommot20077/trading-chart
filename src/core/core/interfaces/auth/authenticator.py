from abc import ABC, abstractmethod

from core.models.auth.auth_request import AuthRequest
from core.models.auth.auth_token import AuthToken


class AbstractAuthenticator(ABC):
    """
    Abstract authenticator for validating incoming requests.

    This abstract class defines the contract for components responsible for
    determining the identity of the user or client making a request. It typically
    extracts credentials from the request (e.g., HTTP headers) and uses a
    `TokenManager` to validate them.
    """

    @abstractmethod
    async def authenticate(self, request: AuthRequest) -> AuthToken:
        """
        Authenticates an incoming request and returns the associated user token.

        This asynchronous method should extract authentication credentials from the
        `AuthRequest` (e.g., an Authorization header), validate them, and
        return an `AuthToken` representing the authenticated principal.

        Args:
            request: An object conforming to the `AuthRequest` protocol, representing
                     the incoming request to be authenticated.

        Returns:
            An object conforming to the `AuthToken` protocol, representing the
            successfully authenticated user or client.

        Raises:
            AuthenticationError: If authentication fails (e.g., missing credentials,
                                 invalid token, expired token).
        """
        pass
