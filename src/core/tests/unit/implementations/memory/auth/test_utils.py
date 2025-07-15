# ABOUTME: Unit tests for authentication utilities in memory implementation
# ABOUTME: Tests password hashing, token generation, and validation functions

import pytest

from core.implementations.memory.auth.utils import (
    hash_password,
    verify_password,
    generate_user_id,
    generate_session_token,
    get_default_permissions_for_role,
    create_bearer_token,
    extract_bearer_token,
    validate_username,
    validate_password,
)
from core.models.auth.enum import Role, Permission


class TestPasswordHashing:
    """Test cases for password hashing functions."""

    @pytest.mark.unit
    def test_hash_password_with_salt(self):
        """Test hash_password with provided salt."""
        password = "testpassword"
        salt = "testsalt"

        hashed = hash_password(password, salt)

        assert "$" in hashed
        salt_part, hash_part = hashed.split("$", 1)
        assert salt_part == salt
        assert len(hash_part) == 64  # SHA-256 produces 64 character hex string

    @pytest.mark.unit
    def test_hash_password_without_salt(self):
        """Test hash_password generates salt when not provided."""
        password = "testpassword"

        hashed = hash_password(password)

        assert "$" in hashed
        salt_part, hash_part = hashed.split("$", 1)
        assert len(salt_part) == 32  # secrets.token_hex(16) produces 32 character string
        assert len(hash_part) == 64  # SHA-256 produces 64 character hex string

    @pytest.mark.unit
    def test_hash_password_consistent_with_same_salt(self):
        """Test hash_password produces same result with same salt."""
        password = "testpassword"
        salt = "testsalt"

        hash1 = hash_password(password, salt)
        hash2 = hash_password(password, salt)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_password_different_with_different_salt(self):
        """Test hash_password produces different results with different salts."""
        password = "testpassword"
        salt1 = "salt1"
        salt2 = "salt2"

        hash1 = hash_password(password, salt1)
        hash2 = hash_password(password, salt2)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_password_different_without_salt(self):
        """Test hash_password produces different results when no salt provided."""
        password = "testpassword"

        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_verify_password_correct(self):
        """Test verify_password returns True for correct password."""
        password = "testpassword"
        salt = "testsalt"
        password_hash = hash_password(password, salt)

        assert verify_password(password, password_hash) is True

    @pytest.mark.unit
    def test_verify_password_incorrect(self):
        """Test verify_password returns False for incorrect password."""
        password = "testpassword"
        wrong_password = "wrongpassword"
        salt = "testsalt"
        password_hash = hash_password(password, salt)

        assert verify_password(wrong_password, password_hash) is False

    @pytest.mark.unit
    def test_verify_password_invalid_hash_format(self):
        """Test verify_password returns False for invalid hash format."""
        password = "testpassword"
        invalid_hash = "invalidhash"

        assert verify_password(password, invalid_hash) is False

    @pytest.mark.unit
    def test_verify_password_empty_hash(self):
        """Test verify_password returns False for empty hash."""
        password = "testpassword"

        assert verify_password(password, "") is False

    @pytest.mark.unit
    def test_verify_password_hash_without_salt(self):
        """Test verify_password returns False for hash without salt separator."""
        password = "testpassword"
        hash_without_salt = "abcdef123456"

        assert verify_password(password, hash_without_salt) is False


class TestTokenGeneration:
    """Test cases for token generation functions."""

    @pytest.mark.unit
    def test_generate_user_id_format(self):
        """Test generate_user_id returns proper format."""
        user_id = generate_user_id()

        assert isinstance(user_id, str)
        assert len(user_id) == 24  # secrets.token_hex(12) produces 24 character string

    @pytest.mark.unit
    def test_generate_user_id_uniqueness(self):
        """Test generate_user_id generates unique IDs."""
        id1 = generate_user_id()
        id2 = generate_user_id()

        assert id1 != id2

    @pytest.mark.unit
    def test_generate_session_token_format(self):
        """Test generate_session_token returns proper format."""
        token = generate_session_token()

        assert isinstance(token, str)
        # secrets.token_urlsafe(32) produces variable length string, but should be > 32
        assert len(token) > 32

    @pytest.mark.unit
    def test_generate_session_token_uniqueness(self):
        """Test generate_session_token generates unique tokens."""
        token1 = generate_session_token()
        token2 = generate_session_token()

        assert token1 != token2

    @pytest.mark.unit
    def test_generate_session_token_url_safe(self):
        """Test generate_session_token produces URL-safe characters."""
        token = generate_session_token()

        # URL-safe characters: A-Z, a-z, 0-9, -, _
        url_safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert all(c in url_safe_chars for c in token)


class TestRolePermissions:
    """Test cases for role permissions functions."""

    @pytest.mark.unit
    def test_get_default_permissions_for_admin(self):
        """Test get_default_permissions_for_role returns admin permissions."""
        permissions = get_default_permissions_for_role(Role.ADMIN)

        expected_permissions = {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        assert permissions == expected_permissions

    @pytest.mark.unit
    def test_get_default_permissions_for_user(self):
        """Test get_default_permissions_for_role returns user permissions."""
        permissions = get_default_permissions_for_role(Role.USER)

        expected_permissions = {Permission.READ, Permission.WRITE}
        assert permissions == expected_permissions

    @pytest.mark.unit
    def test_get_default_permissions_for_unknown_role(self):
        """Test get_default_permissions_for_role returns empty set for unknown role."""

        # Create a mock role that doesn't exist in the mapping
        class MockRole:
            pass

        mock_role = MockRole()
        permissions = get_default_permissions_for_role(mock_role)

        assert permissions == set()


class TestBearerToken:
    """Test cases for Bearer token functions."""

    @pytest.mark.unit
    def test_create_bearer_token(self):
        """Test create_bearer_token creates proper Bearer token."""
        token = "abc123"
        bearer_token = create_bearer_token(token)

        assert bearer_token == "Bearer abc123"

    @pytest.mark.unit
    def test_create_bearer_token_empty(self):
        """Test create_bearer_token with empty token."""
        token = ""
        bearer_token = create_bearer_token(token)

        assert bearer_token == "Bearer "

    @pytest.mark.unit
    def test_extract_bearer_token_valid(self):
        """Test extract_bearer_token extracts token from valid header."""
        auth_header = "Bearer abc123def456"
        token = extract_bearer_token(auth_header)

        assert token == "abc123def456"

    @pytest.mark.unit
    def test_extract_bearer_token_case_insensitive(self):
        """Test extract_bearer_token is case insensitive."""
        auth_header = "bearer abc123def456"
        token = extract_bearer_token(auth_header)

        assert token == "abc123def456"

    @pytest.mark.unit
    def test_extract_bearer_token_mixed_case(self):
        """Test extract_bearer_token with mixed case."""
        auth_header = "BeArEr abc123def456"
        token = extract_bearer_token(auth_header)

        assert token == "abc123def456"

    @pytest.mark.unit
    def test_extract_bearer_token_empty_header(self):
        """Test extract_bearer_token raises ValueError for empty header."""
        with pytest.raises(ValueError, match="Invalid Bearer token format"):
            extract_bearer_token("")

    @pytest.mark.unit
    def test_extract_bearer_token_none_header(self):
        """Test extract_bearer_token raises ValueError for None header."""
        with pytest.raises(ValueError, match="Invalid Bearer token format"):
            extract_bearer_token(None)

    @pytest.mark.unit
    def test_extract_bearer_token_invalid_format(self):
        """Test extract_bearer_token raises ValueError for invalid format."""
        with pytest.raises(ValueError, match="Invalid Bearer token format"):
            extract_bearer_token("Basic abc123")

    @pytest.mark.unit
    def test_extract_bearer_token_no_space(self):
        """Test extract_bearer_token raises ValueError for no space."""
        with pytest.raises(ValueError, match="Invalid Bearer token format"):
            extract_bearer_token("Bearerabc123")

    @pytest.mark.unit
    def test_extract_bearer_token_empty_token(self):
        """Test extract_bearer_token raises ValueError for empty token part."""
        with pytest.raises(ValueError, match="Invalid Bearer token format"):
            extract_bearer_token("Bearer ")

    @pytest.mark.unit
    def test_extract_bearer_token_whitespace_only_token(self):
        """Test extract_bearer_token raises ValueError for whitespace-only token."""
        with pytest.raises(ValueError, match="Invalid Bearer token format"):
            extract_bearer_token("Bearer   ")

    @pytest.mark.unit
    def test_extract_bearer_token_multiple_spaces(self):
        """Test extract_bearer_token handles multiple spaces correctly."""
        auth_header = "Bearer  abc123  def456"
        token = extract_bearer_token(auth_header)

        assert token == " abc123  def456"


class TestValidation:
    """Test cases for validation functions."""

    @pytest.mark.unit
    def test_validate_username_valid(self):
        """Test validate_username returns True for valid usernames."""
        valid_usernames = [
            "user123",
            "test_user",
            "ABC123",
            "user_123_test",
            "a12",  # minimum length
            "a" * 50,  # maximum length
        ]

        for username in valid_usernames:
            assert validate_username(username) is True, f"Username {username} should be valid"

    @pytest.mark.unit
    def test_validate_username_invalid_length(self):
        """Test validate_username returns False for invalid length."""
        invalid_usernames = [
            "ab",  # too short
            "a" * 51,  # too long
            "",  # empty
        ]

        for username in invalid_usernames:
            assert validate_username(username) is False, f"Username {username} should be invalid"

    @pytest.mark.unit
    def test_validate_username_invalid_characters(self):
        """Test validate_username returns False for invalid characters."""
        invalid_usernames = [
            "user-123",  # hyphen not allowed
            "user@test",  # @ not allowed
            "user.test",  # dot not allowed
            "user 123",  # space not allowed
            "user#123",  # # not allowed
            "user+123",  # + not allowed
        ]

        for username in invalid_usernames:
            assert validate_username(username) is False, f"Username {username} should be invalid"

    @pytest.mark.unit
    def test_validate_username_none(self):
        """Test validate_username returns False for None."""
        assert validate_username(None) is False

    @pytest.mark.unit
    def test_validate_password_valid(self):
        """Test validate_password returns True for valid passwords."""
        valid_passwords = [
            "123456",  # minimum length
            "password123",
            "P@ssw0rd!",
            "a" * 100,  # long password
            "abc def",  # with space
        ]

        for password in valid_passwords:
            assert validate_password(password) is True, f"Password {password} should be valid"

    @pytest.mark.unit
    def test_validate_password_invalid_length(self):
        """Test validate_password returns False for invalid length."""
        invalid_passwords = [
            "12345",  # too short
            "abc",  # too short
            "",  # empty
        ]

        for password in invalid_passwords:
            assert validate_password(password) is False, f"Password {password} should be invalid"

    @pytest.mark.unit
    def test_validate_password_none(self):
        """Test validate_password returns False for None."""
        assert validate_password(None) is False


class TestIntegration:
    """Integration tests for multiple functions working together."""

    @pytest.mark.unit
    def test_password_hash_verify_integration(self):
        """Test hash_password and verify_password work together."""
        password = "mypassword123"

        # Hash the password
        hashed = hash_password(password)

        # Verify the password
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False

    @pytest.mark.unit
    def test_bearer_token_create_extract_integration(self):
        """Test create_bearer_token and extract_bearer_token work together."""
        token = "abc123def456"

        # Create bearer token
        bearer_token = create_bearer_token(token)

        # Extract token back
        extracted_token = extract_bearer_token(bearer_token)

        assert extracted_token == token

    @pytest.mark.unit
    def test_user_id_generation_uniqueness_large_sample(self):
        """Test user ID generation uniqueness with large sample."""
        user_ids = set()

        for _ in range(1000):
            user_id = generate_user_id()
            assert user_id not in user_ids, "Generated duplicate user ID"
            user_ids.add(user_id)

    @pytest.mark.unit
    def test_session_token_generation_uniqueness_large_sample(self):
        """Test session token generation uniqueness with large sample."""
        tokens = set()

        for _ in range(1000):
            token = generate_session_token()
            assert token not in tokens, "Generated duplicate session token"
            tokens.add(token)
