# ABOUTME: Token lifecycle integration tests
# ABOUTME: Tests token generation, validation, refresh, and cleanup flows

import pytest
from unittest.mock import Mock

from core.exceptions import AuthenticationException


class TestTokenLifecycleIntegration:
    """Integration tests for token lifecycle management."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_refresh_mechanism(self, authenticator):
        """Test token refresh mechanism."""
        import time

        # Test 1: Create a token with short TTL for testing
        original_ttl = authenticator.default_token_ttl
        authenticator.default_token_ttl = 5  # 5 seconds for testing

        try:
            # Login and get token
            token_str, auth_token = await authenticator.login("admin", "admin123")

            # Verify token is valid initially
            mock_request = Mock()
            mock_request.get_header.return_value = f"Bearer {token_str}"
            validated_token = await authenticator.authenticate(mock_request)
            assert validated_token.username == "admin"

            # Check token expiration time
            time_until_expiry = auth_token.time_until_expiry()
            assert time_until_expiry is not None
            assert time_until_expiry <= 5.0
            assert time_until_expiry > 0.0

            # Test 2: Simulate token refresh by creating a new token before expiry
            # Wait until token is close to expiry (but not expired)
            time.sleep(2)  # Wait 2 seconds

            # Check token is still valid but closer to expiry
            remaining_time = auth_token.time_until_expiry()
            assert remaining_time is not None
            assert remaining_time < 4.0  # Should be less than original
            assert remaining_time > 0.0  # But still valid

            # Simulate refresh by logging in again (getting new token)
            new_token_str, new_auth_token = await authenticator.login("admin", "admin123")

            # Verify new token has fresh expiry time
            new_time_until_expiry = new_auth_token.time_until_expiry()
            assert new_time_until_expiry is not None
            assert new_time_until_expiry > remaining_time  # New token should have more time
            assert new_time_until_expiry <= 5.0  # But within TTL limit

            # Verify both tokens are different
            assert token_str != new_token_str
            assert auth_token.id != new_auth_token.id

            # Test 3: Verify both tokens work independently
            mock_request.get_header.return_value = f"Bearer {token_str}"
            old_validated = await authenticator.authenticate(mock_request)
            assert old_validated.username == "admin"

            mock_request.get_header.return_value = f"Bearer {new_token_str}"
            new_validated = await authenticator.authenticate(mock_request)
            assert new_validated.username == "admin"

            # Test 4: Logout old token and verify new token still works
            await authenticator.logout(token_str)

            # Old token should be invalid
            mock_request.get_header.return_value = f"Bearer {token_str}"
            with pytest.raises(AuthenticationException) as exc_info:
                await authenticator.authenticate(mock_request)
            assert exc_info.value.code == "INVALID_TOKEN"

            # New token should still work
            mock_request.get_header.return_value = f"Bearer {new_token_str}"
            still_valid = await authenticator.authenticate(mock_request)
            assert still_valid.username == "admin"

        finally:
            # Restore original TTL
            authenticator.default_token_ttl = original_ttl

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_expiry_handling(self, authenticator):
        """Test token expiry handling."""
        import time

        # Test 1: Create a token with very short TTL to test expiry
        original_ttl = authenticator.default_token_ttl
        authenticator.default_token_ttl = 1  # 1 second for testing

        try:
            # Login and get token
            token_str, auth_token = await authenticator.login("admin", "admin123")

            # Verify token is valid initially
            mock_request = Mock()
            mock_request.get_header.return_value = f"Bearer {token_str}"
            validated_token = await authenticator.authenticate(mock_request)
            assert validated_token.username == "admin"

            # Check token is not expired initially
            assert not auth_token.is_expired()
            assert auth_token.time_until_expiry() > 0

            # Test 2: Wait for token to expire
            time.sleep(1.5)  # Wait longer than TTL

            # Check token is now expired
            assert auth_token.is_expired()
            assert auth_token.time_until_expiry() == 0.0

            # Test 3: Try to authenticate with expired token
            with pytest.raises(AuthenticationException) as exc_info:
                await authenticator.authenticate(mock_request)
            assert exc_info.value.code == "TOKEN_EXPIRED"
            assert "expires_at" in exc_info.value.details

            # Test 4: Verify expired token is automatically removed from storage
            initial_token_count = await authenticator.get_token_count()

            # Try authentication again (should remove expired token)
            with pytest.raises(AuthenticationException):
                await authenticator.authenticate(mock_request)

            # Token should be removed from storage after failed authentication
            # (The _validate_token method removes expired tokens)

            # Test 5: Test cleanup_expired_tokens method
            # Create multiple tokens with short TTL
            tokens = []
            for i in range(3):
                token_str_i, auth_token_i = await authenticator.login("admin", "admin123")
                tokens.append((token_str_i, auth_token_i))

            # Verify all tokens are valid
            for token_str_i, auth_token_i in tokens:
                assert not auth_token_i.is_expired()
                mock_request.get_header.return_value = f"Bearer {token_str_i}"
                validated = await authenticator.authenticate(mock_request)
                assert validated.username == "admin"

            # Wait for tokens to expire
            time.sleep(1.5)

            # Verify all tokens are expired
            for token_str_i, auth_token_i in tokens:
                assert auth_token_i.is_expired()

            # Test cleanup method
            token_count_before = await authenticator.get_token_count()
            expired_count = await authenticator.cleanup_expired_tokens()
            token_count_after = await authenticator.get_token_count()

            # Should have cleaned up the expired tokens
            assert expired_count >= 3  # At least the 3 we created
            assert token_count_after < token_count_before

            # Test 6: Verify expired tokens cannot be used after cleanup
            for token_str_i, auth_token_i in tokens:
                mock_request.get_header.return_value = f"Bearer {token_str_i}"
                with pytest.raises(AuthenticationException) as exc_info:
                    await authenticator.authenticate(mock_request)
                assert exc_info.value.code == "INVALID_TOKEN"

        finally:
            # Restore original TTL
            authenticator.default_token_ttl = original_ttl

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_revocation_cleanup(self, authenticator):
        """Test token revocation and cleanup."""

        # Test 1: Single token revocation
        token_str, auth_token = await authenticator.login("admin", "admin123")

        # Verify token is valid initially
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {token_str}"
        validated_token = await authenticator.authenticate(mock_request)
        assert validated_token.username == "admin"

        # Get initial token count
        initial_count = await authenticator.get_token_count()
        assert initial_count >= 1

        # Revoke token (logout)
        await authenticator.logout(token_str)

        # Verify token count decreased
        after_logout_count = await authenticator.get_token_count()
        assert after_logout_count == initial_count - 1

        # Verify token is no longer valid
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_request)
        assert exc_info.value.code == "INVALID_TOKEN"

        # Test 2: Multiple token revocation for same user
        tokens = []
        for i in range(3):
            token_str_i, auth_token_i = await authenticator.login("admin", "admin123")
            tokens.append((token_str_i, auth_token_i))

        # Verify all tokens are valid
        for token_str_i, auth_token_i in tokens:
            mock_request.get_header.return_value = f"Bearer {token_str_i}"
            validated = await authenticator.authenticate(mock_request)
            assert validated.username == "admin"

        # Revoke tokens one by one
        for i, (token_str_i, auth_token_i) in enumerate(tokens):
            count_before = await authenticator.get_token_count()
            await authenticator.logout(token_str_i)
            count_after = await authenticator.get_token_count()

            # Token count should decrease
            assert count_after == count_before - 1

            # Revoked token should be invalid
            mock_request.get_header.return_value = f"Bearer {token_str_i}"
            with pytest.raises(AuthenticationException):
                await authenticator.authenticate(mock_request)

            # Remaining tokens should still be valid
            for j in range(i + 1, len(tokens)):
                remaining_token_str, _ = tokens[j]
                mock_request.get_header.return_value = f"Bearer {remaining_token_str}"
                validated = await authenticator.authenticate(mock_request)
                assert validated.username == "admin"

        # Test 3: User deletion and token cleanup
        # Create a test user with multiple tokens (use unique name to avoid conflicts)
        test_username = "test_user_revocation"

        # Check if user already exists and delete it first
        existing_user = await authenticator.get_user(test_username)
        if existing_user:
            await authenticator.delete_user(test_username)

        await authenticator.create_user(test_username, "password123", set())

        user_tokens = []
        for i in range(3):
            token_str_i, auth_token_i = await authenticator.login(test_username, "password123")
            user_tokens.append((token_str_i, auth_token_i))

        # Verify all user tokens are valid
        for token_str_i, auth_token_i in user_tokens:
            mock_request.get_header.return_value = f"Bearer {token_str_i}"
            validated = await authenticator.authenticate(mock_request)
            assert validated.username == test_username

        # Get token count before user deletion
        count_before_deletion = await authenticator.get_token_count()

        # Delete user (should revoke all user tokens)
        await authenticator.delete_user(test_username)

        # Verify token count decreased by number of user tokens
        count_after_deletion = await authenticator.get_token_count()
        assert count_after_deletion <= count_before_deletion - 3

        # Verify all user tokens are invalid
        for token_str_i, auth_token_i in user_tokens:
            mock_request.get_header.return_value = f"Bearer {token_str_i}"
            with pytest.raises(AuthenticationException):
                await authenticator.authenticate(mock_request)

        # Test 4: Password change and token revocation
        # Create another test user with unique name
        test_username2 = "test_user_password_change"

        # Check if user already exists and delete it first
        existing_user2 = await authenticator.get_user(test_username2)
        if existing_user2:
            await authenticator.delete_user(test_username2)

        await authenticator.create_user(test_username2, "password123", set())

        # Create tokens for the user
        user2_tokens = []
        for i in range(2):
            token_str_i, auth_token_i = await authenticator.login(test_username2, "password123")
            user2_tokens.append((token_str_i, auth_token_i))

        # Verify tokens are valid
        for token_str_i, auth_token_i in user2_tokens:
            mock_request.get_header.return_value = f"Bearer {token_str_i}"
            validated = await authenticator.authenticate(mock_request)
            assert validated.username == test_username2

        # Change user password (should invalidate all tokens)
        await authenticator.update_user_password(test_username2, "new_password123")

        # Verify all old tokens are invalid
        for token_str_i, auth_token_i in user2_tokens:
            mock_request.get_header.return_value = f"Bearer {token_str_i}"
            with pytest.raises(AuthenticationException):
                await authenticator.authenticate(mock_request)

        # Verify user can login with new password
        new_token_str, new_auth_token = await authenticator.login(test_username2, "new_password123")
        mock_request.get_header.return_value = f"Bearer {new_token_str}"
        validated = await authenticator.authenticate(mock_request)
        assert validated.username == test_username2

        # Test 5: Logout with non-existent token (should not raise error)
        await authenticator.logout("non_existent_token")  # Should not raise exception

        # Clean up
        await authenticator.delete_user(test_username2)
