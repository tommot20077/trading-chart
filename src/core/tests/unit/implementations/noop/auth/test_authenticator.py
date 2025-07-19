# ABOUTME: Unit tests for NoOpAuthenticator
# ABOUTME: Tests for no-operation authentication implementation

import pytest
from unittest.mock import Mock

from core.implementations.noop.auth.authenticator import NoOpAuthenticator
from core.models.auth.auth_request import AuthRequest
from core.models.auth.enum import Role, Permission


class TestNoOpAuthenticator:
    """Test cases for NoOpAuthenticator."""

    @pytest.mark.unit
    @pytest.mark.external
    def test_initialization(self):
        """Test authenticator initialization."""
        auth = NoOpAuthenticator()
        assert auth is not None

    @pytest.mark.asyncio
    async def test_authenticate_always_succeeds(self):
        """Test that authenticate always returns a valid token."""
        auth = NoOpAuthenticator()
        request = Mock(spec=AuthRequest)

        token = await auth.authenticate(request)

        assert token is not None
        assert token.user_id == "noop-user"
        assert token.username == "noop-user"
        assert Role.USER in token.roles
        assert Permission.READ in token.permissions
        assert token.token == "noop-token"
        assert token.expires_at is None

    @pytest.mark.asyncio
    async def test_authenticate_with_different_requests(self):
        """Test authenticate with different request objects."""
        auth = NoOpAuthenticator()

        request1 = Mock(spec=AuthRequest)
        request2 = Mock(spec=AuthRequest)

        token1 = await auth.authenticate(request1)
        token2 = await auth.authenticate(request2)

        # Should return identical tokens (NoOp behavior)
        assert token1.user_id == token2.user_id
        assert token1.username == token2.username
        assert token1.token == token2.token

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_authenticate_performance(self, benchmark):
        """Test that authenticate is fast (NoOp should be very fast)."""
        auth = NoOpAuthenticator()
        request = Mock(spec=AuthRequest)

        async def authenticate_operations():
            for _ in range(100):
                await auth.authenticate(request)
            return 100

        # Benchmark NoOp authenticate operations
        result = await benchmark(authenticate_operations)
        assert result == 100
