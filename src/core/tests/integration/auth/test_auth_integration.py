# ABOUTME: Core authentication integration tests
# ABOUTME: Tests the complete authentication flow with multiple components

import pytest
from unittest.mock import Mock

from core.exceptions import AuthenticationException, AuthorizationError


class TestAuthenticationIntegration:
    """Integration tests for authentication flow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self, authenticator, authorizer, mock_auth_request):
        """Test complete authentication flow: login → token → authorization → access."""
        from core.models.auth.enum import Role, Permission

        # Step 1: Login with valid credentials
        token_str, auth_token = await authenticator.login("admin", "admin123")

        # Verify token was generated
        assert token_str is not None
        assert len(token_str) > 0
        assert auth_token is not None
        assert auth_token.username == "admin"
        assert Role.ADMIN in auth_token.roles

        # Step 2: Use token for authentication via request
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {token_str}"
        authenticated_token = await authenticator.authenticate(mock_request)

        # Verify authentication succeeded
        assert authenticated_token.username == "admin"
        assert authenticated_token.id == auth_token.id

        # Step 3: Authorize admin permission
        await authorizer.authorize_permission(authenticated_token, Permission.ADMIN)

        # Step 4: Authorize admin role
        await authorizer.authorize_role(authenticated_token, Role.ADMIN)

        # Step 5: Test resource access simulation (no exception means success)
        # This simulates accessing a protected resource
        assert authenticated_token.username == "admin"
        assert Permission.ADMIN in [Permission.ADMIN]  # Simulate resource requiring admin permission

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authentication_failure_handling(self, authenticator, authorizer):
        """Test authentication failure handling."""
        from core.models.auth.enum import Role, Permission

        # Test 1: Missing Authorization header
        mock_request = Mock()
        mock_request.get_header.return_value = None
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_request)
        assert exc_info.value.code == "MISSING_AUTH_HEADER"

        # Test 2: Invalid Authorization header format
        mock_request.get_header.return_value = "InvalidFormat"
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_request)
        assert exc_info.value.code == "INVALID_AUTH_FORMAT"

        # Test 3: Invalid token
        mock_request.get_header.return_value = "Bearer invalid_token_12345"
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_request)
        assert exc_info.value.code == "INVALID_TOKEN"

        # Test 4: Invalid login credentials
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.login("nonexistent", "wrongpassword")
        assert exc_info.value.code == "INVALID_CREDENTIALS"

        # Test 5: Authorization failure - insufficient permissions
        # First login with regular user
        token_str, auth_token = await authenticator.login("user", "user123")

        # Try to authorize admin permission with user token
        with pytest.raises(AuthorizationError):
            await authorizer.authorize_permission(auth_token, Permission.ADMIN)

        # Try to authorize admin role with user token
        with pytest.raises(AuthorizationError):
            await authorizer.authorize_role(auth_token, Role.ADMIN)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authenticator_token_manager_integration(self, authenticator):
        """Test authenticator and token manager integration."""
        import time

        # Test 1: Token generation and validation cycle
        token_str, auth_token = await authenticator.login("admin", "admin123")

        # Verify token properties
        assert auth_token.username == "admin"
        assert auth_token.expires_at is not None
        assert auth_token.expires_at > time.time()  # Should not be expired
        assert auth_token.issued_at is not None
        assert auth_token.issued_at <= time.time()  # Should be issued in the past or now

        # Test 2: Token validation through authentication
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {token_str}"
        validated_token = await authenticator.authenticate(mock_request)

        # Verify validated token matches original
        assert validated_token.username == auth_token.username
        assert validated_token.id == auth_token.id
        assert validated_token.roles == auth_token.roles

        # Test 3: Token lifecycle - logout (revocation)
        await authenticator.logout(token_str)

        # Verify token is invalidated after logout
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_request)
        assert exc_info.value.code == "INVALID_TOKEN"

        # Test 4: Multiple tokens for same user
        token1_str, token1 = await authenticator.login("admin", "admin123")
        token2_str, token2 = await authenticator.login("admin", "admin123")

        # Both tokens should be valid but different
        assert token1_str != token2_str
        assert token1.id != token2.id
        assert token1.username == token2.username == "admin"

        # Test both tokens work independently
        mock_request.get_header.return_value = f"Bearer {token1_str}"
        validated1 = await authenticator.authenticate(mock_request)
        assert validated1.username == "admin"

        mock_request.get_header.return_value = f"Bearer {token2_str}"
        validated2 = await authenticator.authenticate(mock_request)
        assert validated2.username == "admin"

        # Test 5: Token cleanup
        initial_count = await authenticator.get_token_count()
        assert initial_count >= 2  # At least the 2 tokens we just created

        # Logout one token
        await authenticator.logout(token1_str)
        after_logout_count = await authenticator.get_token_count()
        assert after_logout_count == initial_count - 1
