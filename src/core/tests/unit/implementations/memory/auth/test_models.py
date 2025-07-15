# ABOUTME: Unit tests for authentication models in memory implementation
# ABOUTME: Tests UserInfo and MemoryAuthToken classes

import pytest
import time

from core.implementations.memory.auth.models import UserInfo, MemoryAuthToken
from core.models.auth.enum import Role, Permission


class TestUserInfo:
    """Test cases for UserInfo class."""

    @pytest.mark.unit
    def test_user_info_creation(self):
        """Test UserInfo creation with required fields."""
        roles = {Role.USER}
        user = UserInfo(username="testuser", password_hash="hash123", roles=roles)

        assert user.username == "testuser"
        assert user.password_hash == "hash123"
        assert user.roles == roles
        assert user.permissions == set()
        assert user.is_active is True
        assert user.last_login is None
        assert isinstance(user.created_at, float)

    @pytest.mark.unit
    def test_user_info_creation_with_all_fields(self):
        """Test UserInfo creation with all fields."""
        roles = {Role.ADMIN}
        permissions = {Permission.READ, Permission.WRITE}
        created_time = time.time()
        last_login_time = time.time()

        user = UserInfo(
            username="admin",
            password_hash="adminhash",
            roles=roles,
            permissions=permissions,
            is_active=False,
            created_at=created_time,
            last_login=last_login_time,
        )

        assert user.username == "admin"
        assert user.password_hash == "adminhash"
        assert user.roles == roles
        assert user.permissions == permissions
        assert user.is_active is False
        assert user.created_at == created_time
        assert user.last_login == last_login_time

    @pytest.mark.unit
    def test_has_role_true(self):
        """Test has_role returns True when user has the role."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER, Role.ADMIN})

        assert user.has_role(Role.USER) is True
        assert user.has_role(Role.ADMIN) is True

    @pytest.mark.unit
    def test_has_role_false(self):
        """Test has_role returns False when user doesn't have the role."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER})

        assert user.has_role(Role.ADMIN) is False

    @pytest.mark.unit
    def test_has_permission_true(self):
        """Test has_permission returns True when user has the permission."""
        user = UserInfo(
            username="test", password_hash="hash", roles={Role.USER}, permissions={Permission.READ, Permission.WRITE}
        )

        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.WRITE) is True

    @pytest.mark.unit
    def test_has_permission_false(self):
        """Test has_permission returns False when user doesn't have the permission."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER}, permissions={Permission.READ})

        assert user.has_permission(Permission.WRITE) is False
        assert user.has_permission(Permission.DELETE) is False

    @pytest.mark.unit
    def test_add_role(self):
        """Test adding a role to user."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER})

        user.add_role(Role.ADMIN)
        assert Role.ADMIN in user.roles
        assert Role.USER in user.roles

    @pytest.mark.unit
    def test_add_role_duplicate(self):
        """Test adding a role that user already has."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER})

        user.add_role(Role.USER)
        assert user.roles == {Role.USER}

    @pytest.mark.unit
    def test_remove_role(self):
        """Test removing a role from user."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER, Role.ADMIN})

        user.remove_role(Role.ADMIN)
        assert Role.ADMIN not in user.roles
        assert Role.USER in user.roles

    @pytest.mark.unit
    def test_remove_role_not_present(self):
        """Test removing a role that user doesn't have."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER})

        user.remove_role(Role.ADMIN)
        assert user.roles == {Role.USER}

    @pytest.mark.unit
    def test_add_permission(self):
        """Test adding a permission to user."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER}, permissions={Permission.READ})

        user.add_permission(Permission.WRITE)
        assert Permission.WRITE in user.permissions
        assert Permission.READ in user.permissions

    @pytest.mark.unit
    def test_add_permission_duplicate(self):
        """Test adding a permission that user already has."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER}, permissions={Permission.READ})

        user.add_permission(Permission.READ)
        assert user.permissions == {Permission.READ}

    @pytest.mark.unit
    def test_remove_permission(self):
        """Test removing a permission from user."""
        user = UserInfo(
            username="test", password_hash="hash", roles={Role.USER}, permissions={Permission.READ, Permission.WRITE}
        )

        user.remove_permission(Permission.WRITE)
        assert Permission.WRITE not in user.permissions
        assert Permission.READ in user.permissions

    @pytest.mark.unit
    def test_remove_permission_not_present(self):
        """Test removing a permission that user doesn't have."""
        user = UserInfo(username="test", password_hash="hash", roles={Role.USER}, permissions={Permission.READ})

        user.remove_permission(Permission.WRITE)
        assert user.permissions == {Permission.READ}


class TestMemoryAuthToken:
    """Test cases for MemoryAuthToken class."""

    @pytest.mark.unit
    def test_memory_auth_token_creation(self):
        """Test MemoryAuthToken creation with required fields."""
        roles = {Role.USER}
        permissions = {Permission.READ}

        token = MemoryAuthToken(user_id="user123", username="testuser", user_roles=roles, user_permissions=permissions)

        assert token.user_id == "user123"
        assert token.username == "testuser"
        assert token.user_roles == roles
        assert token.user_permissions == permissions
        assert token.expires_at is None
        assert isinstance(token.issued_at, float)

    @pytest.mark.unit
    def test_memory_auth_token_creation_with_expiry(self):
        """Test MemoryAuthToken creation with expiry."""
        roles = {Role.ADMIN}
        permissions = {Permission.READ, Permission.WRITE}
        expires_time = time.time() + 3600  # 1 hour from now
        issued_time = time.time()

        token = MemoryAuthToken(
            user_id="admin123",
            username="admin",
            user_roles=roles,
            user_permissions=permissions,
            expires_at=expires_time,
            issued_at=issued_time,
        )

        assert token.user_id == "admin123"
        assert token.username == "admin"
        assert token.user_roles == roles
        assert token.user_permissions == permissions
        assert token.expires_at == expires_time
        assert token.issued_at == issued_time

    @pytest.mark.unit
    def test_id_property(self):
        """Test id property returns user_id."""
        token = MemoryAuthToken(user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set())

        assert token.id == "user123"

    @pytest.mark.unit
    def test_roles_property(self):
        """Test roles property returns role values as strings."""
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER, Role.ADMIN}, user_permissions=set()
        )

        roles = token.roles
        assert isinstance(roles, list)
        assert Role.USER.value in roles
        assert Role.ADMIN.value in roles

    @pytest.mark.unit
    def test_is_expired_false_no_expiry(self):
        """Test is_expired returns False when no expiry is set."""
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set(), expires_at=None
        )

        assert token.is_expired() is False

    @pytest.mark.unit
    def test_is_expired_false_future_expiry(self):
        """Test is_expired returns False when expiry is in the future."""
        future_time = time.time() + 3600  # 1 hour from now
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set(), expires_at=future_time
        )

        assert token.is_expired() is False

    @pytest.mark.unit
    def test_is_expired_true_past_expiry(self):
        """Test is_expired returns True when expiry is in the past."""
        past_time = time.time() - 3600  # 1 hour ago
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set(), expires_at=past_time
        )

        assert token.is_expired() is True

    @pytest.mark.unit
    def test_has_role_true(self):
        """Test has_role returns True when token has the role."""
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER, Role.ADMIN}, user_permissions=set()
        )

        assert token.has_role(Role.USER) is True
        assert token.has_role(Role.ADMIN) is True

    @pytest.mark.unit
    def test_has_role_false(self):
        """Test has_role returns False when token doesn't have the role."""
        token = MemoryAuthToken(user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set())

        assert token.has_role(Role.ADMIN) is False

    @pytest.mark.unit
    def test_has_permission_true(self):
        """Test has_permission returns True when token has the permission."""
        token = MemoryAuthToken(
            user_id="user123",
            username="test",
            user_roles={Role.USER},
            user_permissions={Permission.READ, Permission.WRITE},
        )

        assert token.has_permission(Permission.READ) is True
        assert token.has_permission(Permission.WRITE) is True

    @pytest.mark.unit
    def test_has_permission_false(self):
        """Test has_permission returns False when token doesn't have the permission."""
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions={Permission.READ}
        )

        assert token.has_permission(Permission.WRITE) is False
        assert token.has_permission(Permission.DELETE) is False

    @pytest.mark.unit
    def test_time_until_expiry_none(self):
        """Test time_until_expiry returns None when no expiry is set."""
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set(), expires_at=None
        )

        assert token.time_until_expiry() is None

    @pytest.mark.unit
    def test_time_until_expiry_future(self):
        """Test time_until_expiry returns positive value for future expiry."""
        future_time = time.time() + 1800  # 30 minutes from now
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set(), expires_at=future_time
        )

        time_left = token.time_until_expiry()
        assert time_left is not None
        assert time_left > 0
        assert time_left <= 1800

    @pytest.mark.unit
    def test_time_until_expiry_past(self):
        """Test time_until_expiry returns 0 for past expiry."""
        past_time = time.time() - 1800  # 30 minutes ago
        token = MemoryAuthToken(
            user_id="user123", username="test", user_roles={Role.USER}, user_permissions=set(), expires_at=past_time
        )

        assert token.time_until_expiry() == 0

    @pytest.mark.unit
    def test_str_representation(self):
        """Test string representation of the token."""
        token = MemoryAuthToken(user_id="user123", username="testuser", user_roles={Role.USER}, user_permissions=set())

        str_repr = str(token)
        assert "MemoryAuthToken" in str_repr
        assert "user123" in str_repr
        assert "testuser" in str_repr

    @pytest.mark.unit
    def test_repr_representation(self):
        """Test detailed string representation of the token."""
        token = MemoryAuthToken(user_id="user123", username="testuser", user_roles={Role.USER}, user_permissions=set())

        repr_str = repr(token)
        assert "MemoryAuthToken" in repr_str
        assert "user_id='user123'" in repr_str
        assert "username='testuser'" in repr_str
        assert "roles=" in repr_str
        assert "expires_at=" in repr_str
        assert "issued_at=" in repr_str
