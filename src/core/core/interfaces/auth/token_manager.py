# ABOUTME: Abstract token manager interface for authentication token lifecycle management
# ABOUTME: Defines the contract for components that generate, validate, refresh, and revoke authentication tokens

from abc import ABC, abstractmethod

from core.models.auth.auth_token import AuthToken
from core.models.types import UserData


class AbstractTokenManager(ABC):
    """
    Abstract token manager for authentication token lifecycle management.

    This abstract class defines the contract for components responsible for
    the lifecycle management of authentication tokens, including their generation,
    validation, and refreshing. Concrete implementations would handle specific
    token types (e.g., JWT, OAuth tokens).

    Note: Methods are designed to be synchronous for simplicity. Async variants
    can be implemented in concrete classes if needed for I/O operations.
    """

    @abstractmethod
    def generate_token(self, user_data: UserData) -> str:
        """
        Generates a new authentication token based on provided user data.

        This method encapsulates the logic for creating a valid token, including
        signing, encryption, and embedding necessary claims (like user ID, roles).
        The generated token is a string representation suitable for transmission
        and subsequent validation.

        Args:
            user_data (dict[str, Any]): A dictionary containing data about the user for whom
                                       the token is being generated. This typically includes
                                       `user_id`, `username`, `roles`, etc.

        Returns:
            str: A string representing the newly generated authentication token.

        Raises:
            AuthenticationError: If token generation fails (e.g., invalid user data, configuration error).
            ValueError: If required user data fields are missing or invalid.
        """
        pass

    @abstractmethod
    def validate_token(self, token: str) -> AuthToken:
        """
        Validates an authentication token and extracts the user information it contains.

        This method verifies the token's signature, expiration, and other claims.
        If validation fails (e.g., the token is invalid, expired, or malformed),
        it raises an `AuthenticationError`.

        Args:
            token (str): The authentication token string to validate.

        Returns:
            AuthToken: An object conforming to the `AuthToken` protocol, containing the validated
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

        This method is often used with refresh tokens to obtain a new access token
        without requiring the user to re-authenticate. It handles the logic for
        generating a new, valid token based on the provided existing token.

        Args:
            token (str): The existing token string to be refreshed. This might be an access token
                         or a dedicated refresh token, depending on the authentication scheme.

        Returns:
            str: A new, refreshed token string.

        Raises:
            AuthenticationError: If the token cannot be refreshed (e.g., invalid or expired refresh token).
        """
        pass

    @abstractmethod
    def revoke_token(self, token: str) -> None:
        """
        Revokes an authentication token, making it invalid for future use.

        This method is typically used when a user logs out or when a token is compromised.
        It ensures that the specified token can no longer be used for authentication or authorization.

        Args:
            token (str): The token string to be revoked.

        Returns:
            None: This method does not return a value; it raises an exception on failure.

        Raises:
            AuthenticationError: If the token cannot be revoked (e.g., it does not exist or is already invalid).
        """
        pass
