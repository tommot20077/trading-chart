from abc import ABC, abstractmethod
from typing import Any

from core.models.auth.auth_token import AuthToken


class AbstractTokenManager(ABC):
    """
    Abstract token manager for authentication.

    This abstract class defines the contract for components responsible for
    the lifecycle management of authentication tokens, including their generation,
    validation, and refreshing. Concrete implementations would handle specific
    token types (e.g., JWT, OAuth tokens).
    """

    @abstractmethod
    def generate_token(self, user_data: dict[str, Any]) -> str:
        """
        Generates a new authentication token based on provided user data.

        This method should encapsulate the logic for creating a valid token,
        including signing, encryption, and embedding necessary claims (like user ID, roles).

        Args:
            user_data: A dictionary containing data about the user for whom the token is being generated.
                       This typically includes `user_id`, `username`, `roles`, etc.

        Returns:
            A string representing the newly generated authentication token.
        """
        pass

    @abstractmethod
    def validate_token(self, token: str) -> AuthToken:
        """
        Validates an authentication token and extracts the user information it contains.

        This method should verify the token's signature, expiration, and other claims.
        If validation fails, it should raise an appropriate authentication error.

        Args:
            token: The authentication token string to validate.

        Returns:
            An object conforming to the `AuthToken` protocol, containing the validated
            user's identity and roles.

        Raises:
            AuthenticationError: If the token is invalid, expired, malformed, or otherwise
                                 fails validation.
        """
        pass

    @abstractmethod
    def refresh_token(self, token: str) -> str:
        """
        Refreshes an existing authentication token, typically extending its validity period.

        This is often used with refresh tokens to obtain a new access token without
        requiring the user to re-authenticate.

        Args:
            token: The existing token string to be refreshed. This might be an access token
                   or a dedicated refresh token, depending on the authentication scheme.

        Returns:
            A new, refreshed token string.

        Raises:
            AuthenticationError: If the token cannot be refreshed (e.g., invalid, expired refresh token).
        """
        pass

    @abstractmethod
    def revoke_token(self, token: str) -> None:
        """
        Revokes an authentication token, making it invalid for future use.

        This is typically used when a user logs out or when a token is compromised.

        Args:
            token: The token string to be revoked.

        Raises:
            AuthenticationError: If the token cannot be revoked (e.g., it does not exist).
        """
        pass
