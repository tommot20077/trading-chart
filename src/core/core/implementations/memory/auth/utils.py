# ABOUTME: Utility functions for in-memory authentication implementations
# ABOUTME: Provides password hashing and token generation utilities

import hashlib
import secrets
import string
from typing import Set, Dict

from core.models.auth.enum import Role, Permission


def hash_password(password: str, salt: str | None = None) -> str:
    """
    Hash a password using SHA-256 with salt.

    Args:
        password: The plain text password to hash.
        salt: Optional salt. If not provided, a random salt is generated.

    Returns:
        The hashed password in format "salt$hash".
    """
    if salt is None:
        salt = secrets.token_hex(16)

    # Combine salt and password
    salted_password = salt + password

    # Hash the salted password
    password_hash = hashlib.sha256(salted_password.encode("utf-8")).hexdigest()

    return f"{salt}${password_hash}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: The plain text password to verify.
        password_hash: The stored hash in format "salt$hash".

    Returns:
        True if the password matches the hash, False otherwise.
    """
    try:
        salt, stored_hash = password_hash.split("$", 1)
        return hash_password(password, salt) == password_hash
    except ValueError:
        # Invalid hash format
        return False


def generate_user_id() -> str:
    """
    Generate a unique user ID.

    Returns:
        A unique identifier string.
    """
    return secrets.token_hex(12)


def generate_session_token() -> str:
    """
    Generate a session token for authentication.

    Returns:
        A secure random token string.
    """
    return secrets.token_urlsafe(32)


def get_default_permissions_for_role(role: Role) -> Set[Permission]:
    """
    Get default permissions for a given role.

    Args:
        role: The role to get permissions for.

    Returns:
        A set of default permissions for the role.
    """
    role_permissions: Dict[Role, Set[Permission]] = {
        Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
        Role.USER: {Permission.READ, Permission.WRITE},
    }

    return role_permissions.get(role, set())


def create_bearer_token(token: str) -> str:
    """
    Create a Bearer token string.

    Args:
        token: The token value.

    Returns:
        A Bearer token string.
    """
    return f"Bearer {token}"


def extract_bearer_token(auth_header: str) -> str:
    """
    Extract token from Bearer authorization header.

    Args:
        auth_header: The Authorization header value.

    Returns:
        The extracted token.

    Raises:
        ValueError: If the header is not a valid Bearer token.
    """
    if not auth_header:
        raise ValueError("Invalid Bearer token format")

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ValueError("Invalid Bearer token format")

    # Check if token part is empty
    if not parts[1].strip():
        raise ValueError("Invalid Bearer token format")

    return parts[1]


def validate_username(username: str) -> bool:
    """
    Validate username format.

    Args:
        username: The username to validate.

    Returns:
        True if the username is valid, False otherwise.
    """
    if not username:
        return False

    # Username should be 3-50 characters, alphanumeric and underscores
    if len(username) < 3 or len(username) > 50:
        return False

    allowed_chars = set(string.ascii_letters + string.digits + "_")
    return all(c in allowed_chars for c in username)


def validate_password(password: str) -> bool:
    """
    Validate password strength.

    Args:
        password: The password to validate.

    Returns:
        True if the password is valid, False otherwise.
    """
    if not password:
        return False

    # Password should be at least 6 characters for this simple implementation
    return len(password) >= 6
