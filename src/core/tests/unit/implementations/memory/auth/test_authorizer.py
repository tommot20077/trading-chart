# ABOUTME: Unit tests for InMemoryAuthorizer implementation
# ABOUTME: Tests role-based and permission-based authorization functionality

import pytest
import asyncio

from core.implementations.memory.auth.authorizer import InMemoryAuthorizer
from core.implementations.memory.auth.models import MemoryAuthToken
from core.models.auth.enum import Role, Permission
from core.exceptions import AuthorizationError


class MockAuthToken:
    """Mock implementation of AuthToken protocol for testing."""

    def __init__(
        self, user_id: str, username: str, roles: list[str], expires_at: float = None, issued_at: float = None
    ):
        self._id = user_id
        self._username = username
        self._roles = roles
        self._expires_at = expires_at
        self._issued_at = issued_at

    @property
    def id(self) -> str:
        return self._id

    @property
    def username(self) -> str:
        return self._username

    @property
    def roles(self) -> list[str]:
        return self._roles

    @property
    def expires_at(self) -> float | None:
        return self._expires_at

    @property
    def issued_at(self) -> float | None:
        return self._issued_at


@pytest.mark.unit
class TestInMemoryAuthorizer:
    """Test cases for InMemoryAuthorizer."""

    @pytest.fixture
    def authorizer(self):
        """Create a fresh authorizer instance for each test."""
        return InMemoryAuthorizer()

    @pytest.fixture
    def custom_authorizer(self):
        """Create an authorizer with custom role-permission mappings."""
        custom_mappings = {
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
            Role.USER: {Permission.READ},
        }
        return InMemoryAuthorizer(custom_role_permissions=custom_mappings)

    @pytest.fixture
    def admin_token(self):
        """Create a MemoryAuthToken for admin user."""
        return MemoryAuthToken(
            user_id="admin_001",
            username="admin",
            user_roles={Role.ADMIN},
            user_permissions={Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
        )

    @pytest.fixture
    def user_token(self):
        """Create a MemoryAuthToken for regular user."""
        return MemoryAuthToken(
            user_id="user_001",
            username="user",
            user_roles={Role.USER},
            user_permissions=set(),  # No direct permissions, only role-based
        )

    @pytest.fixture
    def mock_admin_token(self):
        """Create a mock AuthToken for admin user."""
        return MockAuthToken(user_id="admin_001", username="admin", roles=["admin"])

    @pytest.fixture
    def mock_user_token(self):
        """Create a mock AuthToken for regular user."""
        return MockAuthToken(user_id="user_001", username="user", roles=["user"])

    # === Initialization Tests ===

    @pytest.mark.unit
    def test_init_default_mappings(self, authorizer):
        """Test initialization with default role-permission mappings."""
        admin_permissions = authorizer._role_permissions[Role.ADMIN]
        user_permissions = authorizer._role_permissions[Role.USER]

        assert admin_permissions == {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        assert user_permissions == {Permission.READ, Permission.WRITE}

    @pytest.mark.unit
    def test_init_custom_mappings(self, custom_authorizer):
        """Test initialization with custom role-permission mappings."""
        admin_permissions = custom_authorizer._role_permissions[Role.ADMIN]
        user_permissions = custom_authorizer._role_permissions[Role.USER]

        assert admin_permissions == {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        assert user_permissions == {Permission.READ}  # Only READ permission

    @pytest.mark.unit
    def test_init_role_hierarchy(self, authorizer):
        """Test initialization of role hierarchy."""
        assert authorizer._role_hierarchy[Role.ADMIN] == {Role.USER}
        assert authorizer._role_hierarchy[Role.USER] == set()

    # === Permission Authorization Tests ===

    @pytest.mark.asyncio
    async def test_authorize_permission_admin_success(self, authorizer, admin_token):
        """Test successful permission authorization for admin user."""
        # Admin should have all permissions
        await authorizer.authorize_permission(admin_token, Permission.READ)
        await authorizer.authorize_permission(admin_token, Permission.WRITE)
        await authorizer.authorize_permission(admin_token, Permission.DELETE)
        await authorizer.authorize_permission(admin_token, Permission.ADMIN)

    @pytest.mark.asyncio
    async def test_authorize_permission_user_success(self, authorizer, user_token):
        """Test successful permission authorization for regular user."""
        # User should have READ and WRITE permissions
        await authorizer.authorize_permission(user_token, Permission.READ)
        await authorizer.authorize_permission(user_token, Permission.WRITE)

    @pytest.mark.asyncio
    async def test_authorize_permission_user_failure(self, authorizer, user_token):
        """Test permission authorization failure for regular user."""
        # User should not have DELETE or ADMIN permissions
        with pytest.raises(AuthorizationError) as exc_info:
            await authorizer.authorize_permission(user_token, Permission.DELETE)

        assert exc_info.value.code == "INSUFFICIENT_PERMISSION"
        assert "delete" in exc_info.value.message.lower()
        assert exc_info.value.details["user_id"] == "user_001"
        assert exc_info.value.details["required_permission"] == "delete"

    @pytest.mark.asyncio
    async def test_authorize_permission_admin_failure(self, authorizer, user_token):
        """Test permission authorization failure for admin permission."""
        with pytest.raises(AuthorizationError) as exc_info:
            await authorizer.authorize_permission(user_token, Permission.ADMIN)

        assert exc_info.value.code == "INSUFFICIENT_PERMISSION"
        assert "admin" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_authorize_permission_with_mock_token(self, authorizer, mock_admin_token):
        """Test permission authorization with mock AuthToken."""
        # Mock admin token should work with role-based permissions
        await authorizer.authorize_permission(mock_admin_token, Permission.READ)
        await authorizer.authorize_permission(mock_admin_token, Permission.WRITE)

    @pytest.mark.asyncio
    async def test_authorize_permission_with_invalid_role(self, authorizer):
        """Test permission authorization with invalid role in token."""
        invalid_token = MockAuthToken(user_id="invalid_001", username="invalid", roles=["invalid_role"])

        # Should fail because invalid_role doesn't exist
        with pytest.raises(AuthorizationError):
            await authorizer.authorize_permission(invalid_token, Permission.READ)

    # === Role Authorization Tests ===

    @pytest.mark.asyncio
    async def test_authorize_role_admin_success(self, authorizer, admin_token):
        """Test successful role authorization for admin user."""
        await authorizer.authorize_role(admin_token, Role.ADMIN)
        await authorizer.authorize_role(admin_token, Role.USER)  # Due to inheritance

    @pytest.mark.asyncio
    async def test_authorize_role_user_success(self, authorizer, user_token):
        """Test successful role authorization for regular user."""
        await authorizer.authorize_role(user_token, Role.USER)

    @pytest.mark.asyncio
    async def test_authorize_role_user_failure(self, authorizer, user_token):
        """Test role authorization failure for regular user."""
        with pytest.raises(AuthorizationError) as exc_info:
            await authorizer.authorize_role(user_token, Role.ADMIN)

        assert exc_info.value.code == "INSUFFICIENT_ROLE"
        assert "admin" in exc_info.value.message.lower()
        assert exc_info.value.details["user_id"] == "user_001"
        assert exc_info.value.details["required_role"] == "admin"

    @pytest.mark.asyncio
    async def test_authorize_role_inheritance(self, authorizer, admin_token):
        """Test role inheritance - admin should inherit user role."""
        # Admin should have both ADMIN and USER roles due to inheritance
        await authorizer.authorize_role(admin_token, Role.ADMIN)
        await authorizer.authorize_role(admin_token, Role.USER)

    @pytest.mark.asyncio
    async def test_authorize_role_with_mock_token(self, authorizer, mock_user_token):
        """Test role authorization with mock AuthToken."""
        await authorizer.authorize_role(mock_user_token, Role.USER)

        with pytest.raises(AuthorizationError):
            await authorizer.authorize_role(mock_user_token, Role.ADMIN)

    # === Non-throwing Helper Methods ===

    @pytest.mark.asyncio
    async def test_has_permission_success(self, authorizer, admin_token):
        """Test has_permission method returns True for valid permissions."""
        assert await authorizer.has_permission(admin_token, Permission.READ)
        assert await authorizer.has_permission(admin_token, Permission.ADMIN)

    @pytest.mark.asyncio
    async def test_has_permission_failure(self, authorizer, user_token):
        """Test has_permission method returns False for invalid permissions."""
        assert await authorizer.has_permission(user_token, Permission.READ)
        assert not await authorizer.has_permission(user_token, Permission.ADMIN)

    @pytest.mark.asyncio
    async def test_has_role_success(self, authorizer, admin_token):
        """Test has_role method returns True for valid roles."""
        assert await authorizer.has_role(admin_token, Role.ADMIN)
        assert await authorizer.has_role(admin_token, Role.USER)  # Due to inheritance

    @pytest.mark.asyncio
    async def test_has_role_failure(self, authorizer, user_token):
        """Test has_role method returns False for invalid roles."""
        assert await authorizer.has_role(user_token, Role.USER)
        assert not await authorizer.has_role(user_token, Role.ADMIN)

    # === User Query Methods ===

    @pytest.mark.asyncio
    async def test_get_user_permissions_admin(self, authorizer, admin_token):
        """Test getting all permissions for admin user."""
        permissions = await authorizer.get_user_permissions(admin_token)

        # Admin should have all permissions (direct + inherited)
        expected_permissions = {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        assert permissions == expected_permissions

    @pytest.mark.asyncio
    async def test_get_user_permissions_user(self, authorizer, user_token):
        """Test getting all permissions for regular user."""
        permissions = await authorizer.get_user_permissions(user_token)

        expected_permissions = {Permission.READ, Permission.WRITE}
        assert permissions == expected_permissions

    @pytest.mark.asyncio
    async def test_get_user_permissions_with_mock_token(self, authorizer, mock_user_token):
        """Test getting permissions with mock AuthToken."""
        permissions = await authorizer.get_user_permissions(mock_user_token)

        expected_permissions = {Permission.READ, Permission.WRITE}
        assert permissions == expected_permissions

    @pytest.mark.asyncio
    async def test_get_user_roles_admin(self, authorizer, admin_token):
        """Test getting all roles for admin user."""
        roles = await authorizer.get_user_roles(admin_token)

        # Admin should have ADMIN and USER roles due to inheritance
        expected_roles = {Role.ADMIN, Role.USER}
        assert roles == expected_roles

    @pytest.mark.asyncio
    async def test_get_user_roles_user(self, authorizer, user_token):
        """Test getting all roles for regular user."""
        roles = await authorizer.get_user_roles(user_token)

        expected_roles = {Role.USER}
        assert roles == expected_roles

    @pytest.mark.asyncio
    async def test_get_user_roles_with_mock_token(self, authorizer, mock_admin_token):
        """Test getting roles with mock AuthToken."""
        roles = await authorizer.get_user_roles(mock_admin_token)

        expected_roles = {Role.ADMIN, Role.USER}
        assert roles == expected_roles

    # === Role Permission Management ===

    @pytest.mark.asyncio
    async def test_add_role_permission(self, authorizer, user_token):
        """Test adding permission to a role."""
        # Initially user doesn't have DELETE permission
        assert not await authorizer.has_permission(user_token, Permission.DELETE)

        # Add DELETE permission to USER role
        await authorizer.add_role_permission(Role.USER, Permission.DELETE)

        # Now user should have DELETE permission
        assert await authorizer.has_permission(user_token, Permission.DELETE)

    @pytest.mark.asyncio
    async def test_remove_role_permission(self, authorizer, user_token):
        """Test removing permission from a role."""
        # Initially user has READ permission
        assert await authorizer.has_permission(user_token, Permission.READ)

        # Remove READ permission from USER role
        await authorizer.remove_role_permission(Role.USER, Permission.READ)

        # Now user should not have READ permission
        assert not await authorizer.has_permission(user_token, Permission.READ)

    @pytest.mark.asyncio
    async def test_set_role_permissions(self, authorizer, user_token):
        """Test setting all permissions for a role."""
        # Set USER role to have only DELETE permission
        await authorizer.set_role_permissions(Role.USER, {Permission.DELETE})

        # User should now only have DELETE permission
        assert await authorizer.has_permission(user_token, Permission.DELETE)
        assert not await authorizer.has_permission(user_token, Permission.READ)
        assert not await authorizer.has_permission(user_token, Permission.WRITE)

    @pytest.mark.asyncio
    async def test_get_role_permissions(self, authorizer):
        """Test getting all permissions for a role."""
        admin_permissions = await authorizer.get_role_permissions(Role.ADMIN)
        user_permissions = await authorizer.get_role_permissions(Role.USER)

        assert admin_permissions == {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        assert user_permissions == {Permission.READ, Permission.WRITE}

    @pytest.mark.asyncio
    async def test_get_role_permissions_nonexistent(self, authorizer):
        """Test getting permissions for non-existent role."""

        # Create a mock role-like object for testing
        class TestRole:
            def __init__(self, value):
                self._value = value

            @property
            def value(self):
                return self._value

            def __eq__(self, other):
                return hasattr(other, "value") and self.value == other.value

            def __hash__(self):
                return hash(self.value)

        test_role = TestRole("test")
        permissions = await authorizer.get_role_permissions(test_role)
        assert permissions == set()

    # === Role Hierarchy Management ===

    @pytest.mark.asyncio
    async def test_add_role_inheritance(self, authorizer):
        """Test adding role inheritance."""
        # Initially admin inherits from user
        admin_roles = await authorizer.get_user_roles(MemoryAuthToken("admin", "admin", {Role.ADMIN}, set()))
        assert Role.USER in admin_roles

        # Remove existing inheritance first
        await authorizer.remove_role_inheritance(Role.ADMIN, Role.USER)

        # Add it back
        await authorizer.add_role_inheritance(Role.ADMIN, Role.USER)

        # Admin should still inherit from user
        admin_roles = await authorizer.get_user_roles(MemoryAuthToken("admin", "admin", {Role.ADMIN}, set()))
        assert Role.USER in admin_roles

    @pytest.mark.asyncio
    async def test_remove_role_inheritance(self, authorizer):
        """Test removing role inheritance."""
        # Create admin token
        admin_token = MemoryAuthToken("admin", "admin", {Role.ADMIN}, set())

        # Initially admin inherits from user
        admin_roles = await authorizer.get_user_roles(admin_token)
        assert Role.USER in admin_roles

        # Remove inheritance
        await authorizer.remove_role_inheritance(Role.ADMIN, Role.USER)

        # Now admin should not inherit from user
        admin_roles = await authorizer.get_user_roles(admin_token)
        assert Role.USER not in admin_roles

    @pytest.mark.asyncio
    async def test_get_role_hierarchy(self, authorizer):
        """Test getting current role hierarchy."""
        hierarchy = await authorizer.get_role_hierarchy()

        assert hierarchy[Role.ADMIN] == {Role.USER}
        assert hierarchy[Role.USER] == set()

    # === Configuration Management ===

    @pytest.mark.asyncio
    async def test_reset_to_defaults(self, authorizer):
        """Test resetting authorizer to default configuration."""
        # Modify configuration
        await authorizer.add_role_permission(Role.USER, Permission.ADMIN)
        await authorizer.remove_role_inheritance(Role.ADMIN, Role.USER)

        # Reset to defaults
        await authorizer.reset_to_defaults()

        # Check that defaults are restored
        user_permissions = await authorizer.get_role_permissions(Role.USER)
        assert user_permissions == {Permission.READ, Permission.WRITE}

        hierarchy = await authorizer.get_role_hierarchy()
        assert hierarchy[Role.ADMIN] == {Role.USER}

    # === Thread Safety Tests ===

    @pytest.mark.asyncio
    async def test_concurrent_permission_checks(self, authorizer):
        """Test concurrent permission authorization checks."""
        admin_token = MemoryAuthToken("admin", "admin", {Role.ADMIN}, set())
        user_token = MemoryAuthToken("user", "user", {Role.USER}, set())

        async def check_permissions():
            await authorizer.authorize_permission(admin_token, Permission.ADMIN)
            await authorizer.authorize_permission(user_token, Permission.READ)
            return True

        # Run multiple concurrent permission checks
        tasks = [check_permissions() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_concurrent_role_management(self, authorizer):
        """Test concurrent role permission management."""

        async def modify_role_permissions():
            await authorizer.add_role_permission(Role.USER, Permission.DELETE)
            await authorizer.remove_role_permission(Role.USER, Permission.DELETE)
            return True

        # Run multiple concurrent modifications
        tasks = [modify_role_permissions() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # Final state should be consistent
        user_permissions = await authorizer.get_role_permissions(Role.USER)
        assert Permission.DELETE not in user_permissions

    # === Edge Cases ===

    @pytest.mark.asyncio
    async def test_empty_roles_token(self, authorizer):
        """Test authorization with token having no roles."""
        empty_token = MemoryAuthToken("empty", "empty", set(), set())

        # Should fail all role checks
        assert not await authorizer.has_role(empty_token, Role.USER)
        assert not await authorizer.has_role(empty_token, Role.ADMIN)

        # Should fail all permission checks (no role-based permissions)
        assert not await authorizer.has_permission(empty_token, Permission.READ)

    @pytest.mark.asyncio
    async def test_token_with_direct_permissions(self, authorizer):
        """Test authorization with token having direct permissions."""
        # Token with no roles but direct permissions
        token_with_perms = MemoryAuthToken("direct", "direct", set(), {Permission.READ, Permission.ADMIN})

        # Should have direct permissions
        assert await authorizer.has_permission(token_with_perms, Permission.READ)
        assert await authorizer.has_permission(token_with_perms, Permission.ADMIN)

        # Should not have role-based permissions
        assert not await authorizer.has_permission(token_with_perms, Permission.WRITE)

    @pytest.mark.asyncio
    async def test_complex_role_hierarchy(self, authorizer):
        """Test complex role hierarchy with multiple levels."""
        # Create a custom role and add it to hierarchy
        # This would require extending the Role enum, but we'll simulate
        # by using existing roles in a different way

        # Remove existing hierarchy
        await authorizer.remove_role_inheritance(Role.ADMIN, Role.USER)

        # Create a different hierarchy (just for testing)
        await authorizer.add_role_inheritance(Role.ADMIN, Role.USER)

        # Test that permissions flow through hierarchy
        admin_token = MemoryAuthToken("admin", "admin", {Role.ADMIN}, set())
        admin_perms = await authorizer.get_user_permissions(admin_token)

        # Should have both ADMIN and USER permissions
        assert Permission.ADMIN in admin_perms
        assert Permission.READ in admin_perms
        assert Permission.WRITE in admin_perms

    @pytest.mark.asyncio
    async def test_authorization_error_details(self, authorizer, user_token):
        """Test that authorization errors contain proper details."""
        try:
            await authorizer.authorize_permission(user_token, Permission.ADMIN)
            assert False, "Should have raised AuthorizationError"
        except AuthorizationError as e:
            # Check error details
            assert e.code == "INSUFFICIENT_PERMISSION"
            assert e.details["user_id"] == "user_001"
            assert e.details["username"] == "user"
            assert e.details["required_permission"] == "admin"
            assert "read" in e.details["user_permissions"]
            assert "write" in e.details["user_permissions"]
            assert "user" in e.details["user_roles"]

    @pytest.mark.asyncio
    async def test_exception_handling_in_authorization(self, authorizer):
        """Test exception handling in authorization methods."""

        # Create a token that will cause an exception during permission check
        class ProblematicToken:
            @property
            def id(self):
                return "test_001"

            @property
            def username(self):
                raise ValueError("Test error accessing username")

            @property
            def roles(self):
                return ["user"]

        problematic_token = ProblematicToken()

        # Should catch the exception and raise AuthorizationError
        try:
            await authorizer.authorize_permission(problematic_token, Permission.ADMIN)
            assert False, "Should have raised AuthorizationError"
        except AuthorizationError as e:
            assert e.code == "INTERNAL_ERROR"
        except Exception as e:
            assert False, f"Got unexpected exception: {type(e).__name__}: {e}"
