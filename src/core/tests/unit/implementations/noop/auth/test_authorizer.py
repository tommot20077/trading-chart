# ABOUTME: Unit tests for NoOpAuthorizer
# ABOUTME: Tests for no-operation authorization implementation

import pytest
from unittest.mock import Mock

from core.implementations.noop.auth.authorizer import NoOpAuthorizer
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Role, Permission


class TestNoOpAuthorizer:
    """Test cases for NoOpAuthorizer."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test authorizer initialization."""
        authorizer = NoOpAuthorizer()
        assert authorizer is not None

    @pytest.mark.asyncio
    async def test_authorize_permission_always_succeeds(self):
        """Test that permission authorization always succeeds."""
        authorizer = NoOpAuthorizer()
        token = Mock(spec=AuthToken)

        # Should not raise any exception
        await authorizer.authorize_permission(token, Permission.READ)
        await authorizer.authorize_permission(token, Permission.WRITE)
        await authorizer.authorize_permission(token, Permission.DELETE)

    @pytest.mark.asyncio
    async def test_authorize_role_always_succeeds(self):
        """Test that role authorization always succeeds."""
        authorizer = NoOpAuthorizer()
        token = Mock(spec=AuthToken)

        # Should not raise any exception
        await authorizer.authorize_role(token, Role.USER)
        await authorizer.authorize_role(token, Role.ADMIN)
        await authorizer.authorize_role(token, Role.VIEWER)

    @pytest.mark.asyncio
    async def test_authorize_with_none_token(self):
        """Test authorization with None token (should still succeed)."""
        authorizer = NoOpAuthorizer()

        # NoOp should allow even None tokens
        await authorizer.authorize_permission(None, Permission.READ)
        await authorizer.authorize_role(None, Role.USER)

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_authorize_performance(self, benchmark):
        """Test that authorization is fast (NoOp should be very fast)."""
        authorizer = NoOpAuthorizer()
        token = Mock(spec=AuthToken)

        async def authorize_operations():
            for _ in range(100):
                await authorizer.authorize_permission(token, Permission.READ)
                await authorizer.authorize_role(token, Role.USER)
            return 200  # 100 * 2 operations

        # Benchmark NoOp authorize operations
        result = await benchmark(authorize_operations)
        assert result == 200
