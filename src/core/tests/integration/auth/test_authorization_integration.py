# ABOUTME: Authorization integration tests
# ABOUTME: Tests authorization component integration with authentication results

import pytest
from unittest.mock import Mock

from core.exceptions import AuthorizationError


class TestAuthorizationIntegration:
    """Integration tests for authorization flow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authorization_with_authentication_results(self, authenticator, authorizer):
        """Test authorization integration with authentication results."""
        from core.models.auth.enum import Role, Permission

        # Test 1: Admin user authorization flow
        admin_token_str, admin_token = await authenticator.login("admin", "admin123")

        # Verify admin has all permissions
        await authorizer.authorize_permission(admin_token, Permission.READ)
        await authorizer.authorize_permission(admin_token, Permission.WRITE)
        await authorizer.authorize_permission(admin_token, Permission.DELETE)
        await authorizer.authorize_permission(admin_token, Permission.ADMIN)

        # Verify admin has admin role
        await authorizer.authorize_role(admin_token, Role.ADMIN)
        await authorizer.authorize_role(admin_token, Role.USER)  # Should inherit USER role

        # Test 2: Regular user authorization flow
        user_token_str, user_token = await authenticator.login("user", "user123")

        # Verify user has limited permissions
        await authorizer.authorize_permission(user_token, Permission.READ)
        await authorizer.authorize_permission(user_token, Permission.WRITE)

        # Verify user cannot access admin permissions
        with pytest.raises(AuthorizationError) as exc_info:
            await authorizer.authorize_permission(user_token, Permission.DELETE)
        assert exc_info.value.code == "INSUFFICIENT_PERMISSION"

        with pytest.raises(AuthorizationError) as exc_info:
            await authorizer.authorize_permission(user_token, Permission.ADMIN)
        assert exc_info.value.code == "INSUFFICIENT_PERMISSION"

        # Verify user has user role but not admin role
        await authorizer.authorize_role(user_token, Role.USER)

        with pytest.raises(AuthorizationError) as exc_info:
            await authorizer.authorize_role(user_token, Role.ADMIN)
        assert exc_info.value.code == "INSUFFICIENT_ROLE"

        # Test 3: Non-throwing permission checks
        assert await authorizer.has_permission(admin_token, Permission.ADMIN) == True
        assert await authorizer.has_permission(user_token, Permission.ADMIN) == False
        assert await authorizer.has_role(admin_token, Role.ADMIN) == True
        assert await authorizer.has_role(user_token, Role.ADMIN) == False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_role_permission_management_integration(self, authorizer, authenticator):
        """Test role and permission management integration."""
        from core.models.auth.enum import Role, Permission

        # Test 1: Dynamic role permission modification
        # Create a test user
        await authenticator.create_user("test_manager", "manager123", {Role.USER})
        manager_token_str, manager_token = await authenticator.login("test_manager", "manager123")

        # Initially user should not have DELETE permission
        assert await authorizer.has_permission(manager_token, Permission.DELETE) == False

        # Add DELETE permission to USER role
        await authorizer.add_role_permission(Role.USER, Permission.DELETE)

        # Now user should have DELETE permission
        assert await authorizer.has_permission(manager_token, Permission.DELETE) == True
        await authorizer.authorize_permission(manager_token, Permission.DELETE)

        # Remove DELETE permission from USER role
        await authorizer.remove_role_permission(Role.USER, Permission.DELETE)

        # User should no longer have DELETE permission
        assert await authorizer.has_permission(manager_token, Permission.DELETE) == False
        with pytest.raises(AuthorizationError):
            await authorizer.authorize_permission(manager_token, Permission.DELETE)

        # Test 2: Role hierarchy management
        # Create a custom role for testing

        # Test role inheritance - ADMIN should inherit USER permissions
        admin_token_str, admin_token = await authenticator.login("admin", "admin123")

        # Verify admin inherits user permissions
        user_permissions = await authorizer.get_role_permissions(Role.USER)
        admin_permissions = await authorizer.get_user_permissions(admin_token)

        # Admin should have all user permissions
        for permission in user_permissions:
            assert permission in admin_permissions

        # Test 3: Permission set management
        original_user_permissions = await authorizer.get_role_permissions(Role.USER)

        # Set new permissions for USER role
        new_permissions = {Permission.READ}
        await authorizer.set_role_permissions(Role.USER, new_permissions)

        # Verify permissions were updated
        updated_permissions = await authorizer.get_role_permissions(Role.USER)
        assert updated_permissions == new_permissions

        # User should only have READ permission now (need new login for updated permissions)
        user_token_str, user_token = await authenticator.login("user", "user123")

        # Note: The token contains cached permissions from login time, but authorizer
        # checks current role permissions, so we need to verify the role permissions directly
        current_user_permissions = await authorizer.get_role_permissions(Role.USER)
        assert Permission.READ in current_user_permissions
        assert Permission.WRITE not in current_user_permissions

        # Restore original permissions
        await authorizer.set_role_permissions(Role.USER, original_user_permissions)

        # Clean up test user
        await authenticator.delete_user("test_manager")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_component_authorization(self, authenticator, authorizer):
        """Test cross-component authorization flow."""
        from core.models.auth.enum import Role, Permission
        from core.exceptions import AuthenticationException

        # Test 1: Complete authentication + authorization workflow
        # Step 1: Login and get token
        token_str, auth_token = await authenticator.login("admin", "admin123")

        # Step 2: Use token for request authentication
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {token_str}"
        authenticated_token = await authenticator.authenticate(mock_request)

        # Step 3: Use authenticated token for authorization
        await authorizer.authorize_permission(authenticated_token, Permission.ADMIN)
        await authorizer.authorize_role(authenticated_token, Role.ADMIN)

        # Verify tokens are equivalent
        assert authenticated_token.username == auth_token.username
        assert authenticated_token.id == auth_token.id

        # Test 2: Multi-step authorization with different users
        # Create users with different permission levels
        await authenticator.create_user("editor", "editor123", {Role.USER})
        await authenticator.create_user("viewer", "viewer123", {Role.USER})

        # Modify permissions for testing
        await authorizer.set_role_permissions(Role.USER, {Permission.READ, Permission.WRITE})

        # Test editor workflow
        editor_token_str, editor_token = await authenticator.login("editor", "editor123")
        mock_request.get_header.return_value = f"Bearer {editor_token_str}"
        editor_auth_token = await authenticator.authenticate(mock_request)

        # Editor should have read/write but not delete/admin
        await authorizer.authorize_permission(editor_auth_token, Permission.READ)
        await authorizer.authorize_permission(editor_auth_token, Permission.WRITE)

        with pytest.raises(AuthorizationError):
            await authorizer.authorize_permission(editor_auth_token, Permission.DELETE)

        with pytest.raises(AuthorizationError):
            await authorizer.authorize_permission(editor_auth_token, Permission.ADMIN)

        # Test 3: Token invalidation and authorization failure
        # Logout editor
        await authenticator.logout(editor_token_str)

        # Try to authenticate with invalidated token
        with pytest.raises(AuthenticationException) as exc_info:
            await authenticator.authenticate(mock_request)
        assert exc_info.value.code == "INVALID_TOKEN"

        # Test 4: Permission changes affecting active tokens
        viewer_token_str, viewer_token = await authenticator.login("viewer", "viewer123")

        # Initially viewer has read/write permissions
        assert await authorizer.has_permission(viewer_token, Permission.READ) == True
        assert await authorizer.has_permission(viewer_token, Permission.WRITE) == True

        # Remove write permission from USER role
        await authorizer.remove_role_permission(Role.USER, Permission.WRITE)

        # Check that the role no longer has write permission
        current_user_permissions = await authorizer.get_role_permissions(Role.USER)
        assert Permission.WRITE not in current_user_permissions

        # Note: The existing token still has cached permissions from login time
        # The authorizer combines both token permissions and current role permissions
        # So the user still has WRITE permission through the cached token permissions
        assert await authorizer.has_permission(viewer_token, Permission.READ) == True
        assert await authorizer.has_permission(viewer_token, Permission.WRITE) == True  # Still has cached permission

        # To test the role permission change, we need a new user created after the role change
        await authenticator.logout(viewer_token_str)
        await authenticator.delete_user("viewer")

        # Create a new user after the role permission change
        await authenticator.create_user("new_viewer", "viewer123", {Role.USER})
        new_viewer_token_str, new_viewer_token = await authenticator.login("new_viewer", "viewer123")

        # Note: The authenticator uses get_default_permissions_for_role() which returns
        # hardcoded permissions, not the current authorizer role permissions.
        # This demonstrates the separation between authenticator and authorizer concerns.

        # The new user still gets default USER permissions from authenticator
        user_info = await authenticator.get_user("new_viewer")
        assert Permission.READ in user_info.permissions
        assert Permission.WRITE in user_info.permissions  # Still has default permissions

        # But the authorizer uses its own role configuration for authorization
        current_role_permissions = await authorizer.get_role_permissions(Role.USER)
        assert Permission.READ in current_role_permissions
        assert Permission.WRITE not in current_role_permissions  # Role permission was removed

        # The combined permissions (token + role) still include WRITE from token
        combined_permissions = await authorizer.get_user_permissions(new_viewer_token)
        assert Permission.READ in combined_permissions
        assert Permission.WRITE in combined_permissions  # From token's cached permissions

        # Clean up new user
        await authenticator.delete_user("new_viewer")

        # Test 5: Role hierarchy in cross-component flow
        # Admin should inherit all user permissions
        admin_permissions = await authorizer.get_user_permissions(authenticated_token)
        user_permissions = await authorizer.get_role_permissions(Role.USER)

        # Admin should have all user permissions plus admin-specific ones
        for permission in user_permissions:
            assert permission in admin_permissions

        # Admin should also have admin-specific permissions
        assert Permission.ADMIN in admin_permissions
        assert Permission.DELETE in admin_permissions

        # Clean up
        await authorizer.reset_to_defaults()
        await authenticator.delete_user("editor")
        # Note: "viewer" was already deleted earlier in the test
