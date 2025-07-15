# ABOUTME: Authentication provider switching integration tests
# ABOUTME: Tests switching between different authentication providers

import pytest
from unittest.mock import Mock

from core.exceptions import AuthenticationException


class TestAuthProviderSwitchingIntegration:
    """Integration tests for authentication provider switching."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_auth_provider_switching(self):
        """Test switching between different authentication providers."""
        from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
        from core.implementations.noop.auth.authenticator import NoOpAuthenticator
        from core.models.auth.enum import Role

        # Test 1: Initialize different authentication providers
        memory_auth = InMemoryAuthenticator()
        noop_auth = NoOpAuthenticator()

        # Test 2: Test InMemoryAuthenticator functionality
        # Create user in memory authenticator
        await memory_auth.create_user("test_user", "password123", {Role.USER})

        # Login with memory authenticator
        memory_token_str, memory_token = await memory_auth.login("test_user", "password123")
        assert memory_token.username == "test_user"
        assert Role.USER in memory_token.user_roles

        # Authenticate with memory authenticator
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {memory_token_str}"
        validated_memory_token = await memory_auth.authenticate(mock_request)
        assert validated_memory_token.username == "test_user"

        # Test 3: Test NoOpAuthenticator functionality
        # NoOp authenticator should always succeed with fake data
        noop_token = await noop_auth.authenticate(mock_request)  # Any request works
        assert noop_token.username == "noop-user"
        assert Role.USER in noop_token.user_roles

        # Test 4: Demonstrate provider switching scenario
        # Simulate a system that can switch between providers
        class AuthProviderManager:
            def __init__(self):
                self.providers = {"memory": memory_auth, "noop": noop_auth}
                self.current_provider = "memory"

            async def authenticate_with_provider(self, provider_name: str, request):
                if provider_name not in self.providers:
                    raise ValueError(f"Unknown provider: {provider_name}")
                return await self.providers[provider_name].authenticate(request)

            def switch_provider(self, provider_name: str):
                if provider_name not in self.providers:
                    raise ValueError(f"Unknown provider: {provider_name}")
                self.current_provider = provider_name

            async def authenticate(self, request):
                return await self.authenticate_with_provider(self.current_provider, request)

        # Test provider manager
        auth_manager = AuthProviderManager()

        # Test with memory provider (default)
        mock_request.get_header.return_value = f"Bearer {memory_token_str}"
        result = await auth_manager.authenticate(mock_request)
        assert result.username == "test_user"

        # Switch to noop provider
        auth_manager.switch_provider("noop")

        # Test with noop provider (should work with any token)
        mock_request.get_header.return_value = "Bearer any_token"
        result = await auth_manager.authenticate(mock_request)
        assert result.username == "noop-user"

        # Test 5: Verify provider isolation
        # Memory provider should reject invalid tokens
        auth_manager.switch_provider("memory")
        mock_request.get_header.return_value = "Bearer invalid_token"
        with pytest.raises(AuthenticationException):
            await auth_manager.authenticate(mock_request)

        # NoOp provider should accept any token
        auth_manager.switch_provider("noop")
        result = await auth_manager.authenticate(mock_request)
        assert result.username == "noop-user"

        # Test 6: Provider-specific features
        # Memory provider supports user management
        user_count = await memory_auth.get_user_count()
        assert user_count >= 1  # At least the test user we created

        # NoOp provider doesn't support user management (would need different interface)
        # This demonstrates the differences between providers

        # Clean up
        await memory_auth.delete_user("test_user")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_auth_failure_fallback_handling(self):
        """Test authentication failure fallback handling."""
        from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
        from core.implementations.noop.auth.authenticator import NoOpAuthenticator
        from core.models.auth.enum import Role
        import logging

        # Test 1: Create a fallback authentication system
        class FallbackAuthenticator:
            def __init__(self, primary_auth, fallback_auth):
                self.primary_auth = primary_auth
                self.fallback_auth = fallback_auth
                self.fallback_count = 0
                self.primary_success_count = 0

            async def authenticate(self, request):
                try:
                    # Try primary authenticator first
                    result = await self.primary_auth.authenticate(request)
                    self.primary_success_count += 1
                    return result
                except AuthenticationException as e:
                    # Primary failed, try fallback
                    logging.info(f"Primary auth failed: {e.message}, falling back to secondary")
                    self.fallback_count += 1
                    return await self.fallback_auth.authenticate(request)

            def get_stats(self):
                return {"primary_success": self.primary_success_count, "fallback_used": self.fallback_count}

        # Test 2: Setup authenticators
        memory_auth = InMemoryAuthenticator()
        noop_auth = NoOpAuthenticator()  # Always succeeds as fallback

        # Create user in memory auth
        await memory_auth.create_user("valid_user", "password123", {Role.USER})
        valid_token_str, valid_token = await memory_auth.login("valid_user", "password123")

        # Create fallback system (memory primary, noop fallback)
        fallback_auth = FallbackAuthenticator(memory_auth, noop_auth)

        # Test 3: Test successful primary authentication
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {valid_token_str}"

        result = await fallback_auth.authenticate(mock_request)
        assert result.username == "valid_user"

        stats = fallback_auth.get_stats()
        assert stats["primary_success"] == 1
        assert stats["fallback_used"] == 0

        # Test 4: Test fallback on invalid token
        mock_request.get_header.return_value = "Bearer invalid_token"

        result = await fallback_auth.authenticate(mock_request)
        assert result.username == "noop-user"  # Fallback succeeded

        stats = fallback_auth.get_stats()
        assert stats["primary_success"] == 1  # Still 1
        assert stats["fallback_used"] == 1  # Incremented

        # Test 5: Test fallback on missing header
        mock_request.get_header.return_value = None

        result = await fallback_auth.authenticate(mock_request)
        assert result.username == "noop-user"  # Fallback succeeded

        stats = fallback_auth.get_stats()
        assert stats["primary_success"] == 1
        assert stats["fallback_used"] == 2

        # Test 6: Test multiple fallback scenarios
        invalid_scenarios = ["Bearer expired_token", "InvalidFormat", "Bearer ", ""]

        for i, invalid_auth in enumerate(invalid_scenarios):
            mock_request.get_header.return_value = invalid_auth
            result = await fallback_auth.authenticate(mock_request)
            assert result.username == "noop-user"

            stats = fallback_auth.get_stats()
            assert stats["fallback_used"] == 3 + i  # Incremented each time

        # Test 7: Test cascading fallback (multiple fallback levels)
        class CascadingAuthenticator:
            def __init__(self, auth_chain):
                self.auth_chain = auth_chain
                self.attempt_counts = [0] * len(auth_chain)

            async def authenticate(self, request):
                last_exception = None

                for i, auth in enumerate(self.auth_chain):
                    try:
                        self.attempt_counts[i] += 1
                        return await auth.authenticate(request)
                    except AuthenticationException as e:
                        last_exception = e
                        continue

                # If all failed, raise the last exception
                raise last_exception

        # Create another memory auth that will fail
        failing_memory_auth = InMemoryAuthenticator()
        # Don't create any users, so it will always fail

        # Create cascading auth: failing_memory -> memory -> noop
        cascading_auth = CascadingAuthenticator(
            [
                failing_memory_auth,  # Will fail
                memory_auth,  # Will fail for invalid tokens
                noop_auth,  # Will always succeed
            ]
        )

        # Test with invalid token - should cascade to noop
        mock_request.get_header.return_value = "Bearer invalid_token"
        result = await cascading_auth.authenticate(mock_request)
        assert result.username == "noop-user"

        # Check that all three were attempted
        assert cascading_auth.attempt_counts[0] == 1  # failing_memory tried
        assert cascading_auth.attempt_counts[1] == 1  # memory tried
        assert cascading_auth.attempt_counts[2] == 1  # noop succeeded

        # Test with valid token - should succeed on second try (memory)
        mock_request.get_header.return_value = f"Bearer {valid_token_str}"
        result = await cascading_auth.authenticate(mock_request)
        assert result.username == "valid_user"

        # Check attempt counts
        assert cascading_auth.attempt_counts[0] == 2  # failing_memory tried again
        assert cascading_auth.attempt_counts[1] == 2  # memory succeeded
        assert cascading_auth.attempt_counts[2] == 1  # noop not reached

        # Clean up
        await memory_auth.delete_user("valid_user")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_auth_state_synchronization(self):
        """Test authentication state synchronization."""
        from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
        from core.implementations.memory.auth.authorizer import InMemoryAuthorizer
        from core.models.auth.enum import Role, Permission
        import asyncio

        # Test 1: Create synchronized authentication system
        class SynchronizedAuthSystem:
            def __init__(self):
                self.authenticator = InMemoryAuthenticator()
                self.authorizer = InMemoryAuthorizer()
                self.active_sessions = {}  # Track active sessions across components
                self.session_events = []  # Log session events for verification

            async def login(self, username: str, password: str):
                # Login through authenticator
                token_str, auth_token = await self.authenticator.login(username, password)

                # Sync session state
                session_info = {
                    "token": token_str,
                    "auth_token": auth_token,
                    "login_time": auth_token.issued_at,
                    "last_activity": auth_token.issued_at,
                }
                self.active_sessions[token_str] = session_info
                self.session_events.append(f"LOGIN: {username} - {token_str[:8]}...")

                return token_str, auth_token

            async def authenticate_and_authorize(self, request, required_permission: Permission):
                # Extract token from request
                auth_header = request.get_header("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise AuthenticationException("Invalid auth header", "INVALID_AUTH_HEADER")

                token_str = auth_header.split(" ")[1]

                # Check if session is tracked
                if token_str not in self.active_sessions:
                    # Try to authenticate (might be a valid token not in our session tracking)
                    auth_token = await self.authenticator.authenticate(request)
                    # Add to session tracking if authentication succeeds
                    self.active_sessions[token_str] = {
                        "token": token_str,
                        "auth_token": auth_token,
                        "login_time": auth_token.issued_at,
                        "last_activity": auth_token.issued_at,
                    }
                else:
                    # Use cached session info but verify token is still valid
                    auth_token = await self.authenticator.authenticate(request)
                    # Update last activity
                    self.active_sessions[token_str]["last_activity"] = auth_token.issued_at

                # Authorize the action
                await self.authorizer.authorize_permission(auth_token, required_permission)

                self.session_events.append(f"ACCESS: {auth_token.username} - {required_permission.value}")
                return auth_token

            async def logout(self, token_str: str):
                # Logout from authenticator
                await self.authenticator.logout(token_str)

                # Remove from session tracking
                if token_str in self.active_sessions:
                    session_info = self.active_sessions[token_str]
                    username = session_info["auth_token"].username
                    del self.active_sessions[token_str]
                    self.session_events.append(f"LOGOUT: {username} - {token_str[:8]}...")

            async def sync_expired_sessions(self):
                # Clean up expired tokens from both authenticator and session tracking
                expired_count = await self.authenticator.cleanup_expired_tokens()

                # Remove expired sessions from tracking
                expired_sessions = []
                for token_str, session_info in self.active_sessions.items():
                    if session_info["auth_token"].is_expired():
                        expired_sessions.append(token_str)

                for token_str in expired_sessions:
                    session_info = self.active_sessions[token_str]
                    username = session_info["auth_token"].username
                    del self.active_sessions[token_str]
                    self.session_events.append(f"EXPIRED: {username} - {token_str[:8]}...")

                return expired_count, len(expired_sessions)

            def get_session_stats(self):
                return {
                    "active_sessions": len(self.active_sessions),
                    "total_events": len(self.session_events),
                    "events": self.session_events.copy(),
                }

        # Test 2: Setup synchronized system
        auth_system = SynchronizedAuthSystem()

        # Create test users
        await auth_system.authenticator.create_user("user1", "password1", {Role.USER})
        await auth_system.authenticator.create_user("user2", "password2", {Role.USER})
        await auth_system.authenticator.create_user("admin1", "password3", {Role.ADMIN})

        # Test 3: Test synchronized login and session tracking
        token1_str, token1 = await auth_system.login("user1", "password1")
        token2_str, token2 = await auth_system.login("user2", "password2")
        admin_token_str, admin_token = await auth_system.login("admin1", "password3")

        stats = auth_system.get_session_stats()
        assert stats["active_sessions"] == 3
        assert len([e for e in stats["events"] if e.startswith("LOGIN:")]) == 3

        # Test 4: Test synchronized authentication and authorization
        mock_request = Mock()

        # User1 accesses with READ permission
        mock_request.get_header.return_value = f"Bearer {token1_str}"
        result = await auth_system.authenticate_and_authorize(mock_request, Permission.READ)
        assert result.username == "user1"

        # Admin accesses with ADMIN permission
        mock_request.get_header.return_value = f"Bearer {admin_token_str}"
        result = await auth_system.authenticate_and_authorize(mock_request, Permission.ADMIN)
        assert result.username == "admin1"

        # Verify access events were logged
        stats = auth_system.get_session_stats()
        access_events = [e for e in stats["events"] if e.startswith("ACCESS:")]
        assert len(access_events) == 2
        assert "ACCESS: user1 - read" in access_events
        assert "ACCESS: admin1 - admin" in access_events

        # Test 5: Test synchronized logout
        await auth_system.logout(token1_str)

        stats = auth_system.get_session_stats()
        assert stats["active_sessions"] == 2  # One session removed
        logout_events = [e for e in stats["events"] if e.startswith("LOGOUT:")]
        assert len(logout_events) == 1
        assert "user1" in logout_events[0]

        # Verify logged out token cannot be used
        mock_request.get_header.return_value = f"Bearer {token1_str}"
        with pytest.raises(AuthenticationException):
            await auth_system.authenticate_and_authorize(mock_request, Permission.READ)

        # Test 6: Test concurrent access synchronization
        async def concurrent_access(token_str, permission, user_id):
            mock_req = Mock()
            mock_req.get_header.return_value = f"Bearer {token_str}"
            try:
                result = await auth_system.authenticate_and_authorize(mock_req, permission)
                return f"SUCCESS: {user_id} - {result.username}"
            except Exception as e:
                return f"FAILED: {user_id} - {str(e)}"

        # Simulate concurrent access from multiple users
        tasks = [
            concurrent_access(token2_str, Permission.READ, "user2_req1"),
            concurrent_access(admin_token_str, Permission.WRITE, "admin_req1"),
            concurrent_access(token2_str, Permission.WRITE, "user2_req2"),
            concurrent_access(admin_token_str, Permission.DELETE, "admin_req2"),
        ]

        results = await asyncio.gather(*tasks)

        # Verify results
        success_results = [r for r in results if r.startswith("SUCCESS:")]
        assert len(success_results) == 4  # All should succeed

        # Test 7: Test expired session synchronization
        # Create short-lived tokens for expiry testing
        original_ttl = auth_system.authenticator.default_token_ttl
        auth_system.authenticator.default_token_ttl = 1  # 1 second

        try:
            short_token_str, short_token = await auth_system.login("user2", "password2")

            # Wait for expiry
            import time

            time.sleep(1.5)

            # Sync expired sessions
            expired_auth, expired_sessions = await auth_system.sync_expired_sessions()

            stats = auth_system.get_session_stats()
            expired_events = [e for e in stats["events"] if e.startswith("EXPIRED:")]
            assert len(expired_events) >= 1  # At least the short-lived token

        finally:
            # Restore original TTL
            auth_system.authenticator.default_token_ttl = original_ttl

        # Test 8: Verify final state consistency
        final_stats = auth_system.get_session_stats()
        authenticator_token_count = await auth_system.authenticator.get_token_count()

        # Session tracking should be consistent with authenticator state
        # (allowing for some variance due to cleanup timing)
        assert abs(final_stats["active_sessions"] - authenticator_token_count) <= 1

        # Clean up
        await auth_system.authenticator.delete_user("user1")
        await auth_system.authenticator.delete_user("user2")
        await auth_system.authenticator.delete_user("admin1")
