# ABOUTME: Unit tests for authentication enum models (Role and Permission)
# ABOUTME: Tests cover normal cases, exception cases, and boundary cases following TDD principles

import pytest

from core.models.auth.enum import Role, Permission


class TestRole:
    """Test cases for Role enum."""

    @pytest.mark.unit
    def test_role_values_normal_case(self):
        """Test that Role enum has correct string values."""
        assert Role.ADMIN == "admin"
        assert Role.USER == "user"
        assert Role.VIEWER == "viewer"

    @pytest.mark.unit
    def test_role_string_representation(self):
        """Test string representation of Role enum members."""
        assert str(Role.ADMIN) == "Role.ADMIN"
        assert str(Role.USER) == "Role.USER"
        assert str(Role.VIEWER) == "Role.VIEWER"

    @pytest.mark.unit
    def test_role_equality(self):
        """Test equality comparison of Role enum members."""
        assert Role.ADMIN == Role.ADMIN
        assert Role.USER == Role.USER
        assert Role.ADMIN != Role.USER

    @pytest.mark.unit
    def test_role_membership(self):
        """Test membership checking for Role enum."""
        assert "admin" in [role.value for role in Role]
        assert "user" in [role.value for role in Role]
        assert "viewer" in [role.value for role in Role]
        assert "invalid_role" not in [role.value for role in Role]

    @pytest.mark.unit
    def test_role_iteration(self):
        """Test iteration over Role enum members."""
        roles = list(Role)
        assert len(roles) == 3
        assert Role.ADMIN in roles
        assert Role.USER in roles
        assert Role.VIEWER in roles

    @pytest.mark.unit
    def test_role_from_string(self):
        """Test creating Role from string value."""
        assert Role("admin") == Role.ADMIN
        assert Role("user") == Role.USER
        assert Role("viewer") == Role.VIEWER

    @pytest.mark.unit
    def test_role_invalid_value_raises_exception(self):
        """Test that invalid role value raises ValueError."""
        with pytest.raises(ValueError):
            Role("invalid_role")


class TestPermission:
    """Test cases for Permission enum."""

    @pytest.mark.unit
    def test_permission_values_normal_case(self):
        """Test that Permission enum has correct string values."""
        assert Permission.READ == "read"
        assert Permission.WRITE == "write"
        assert Permission.DELETE == "delete"
        assert Permission.ADMIN == "admin"

    @pytest.mark.unit
    def test_permission_string_representation(self):
        """Test string representation of Permission enum members."""
        assert str(Permission.READ) == "Permission.READ"
        assert str(Permission.WRITE) == "Permission.WRITE"
        assert str(Permission.DELETE) == "Permission.DELETE"
        assert str(Permission.ADMIN) == "Permission.ADMIN"

    @pytest.mark.unit
    def test_permission_equality(self):
        """Test equality comparison of Permission enum members."""
        assert Permission.READ == Permission.READ
        assert Permission.WRITE != Permission.READ
        assert Permission.DELETE != Permission.ADMIN

    @pytest.mark.unit
    def test_permission_membership(self):
        """Test membership checking for Permission enum."""
        permission_values = [perm.value for perm in Permission]
        assert "read" in permission_values
        assert "write" in permission_values
        assert "delete" in permission_values
        assert "admin" in permission_values
        assert "invalid_permission" not in permission_values

    @pytest.mark.unit
    def test_permission_iteration(self):
        """Test iteration over Permission enum members."""
        permissions = list(Permission)
        assert len(permissions) == 4
        assert Permission.READ in permissions
        assert Permission.WRITE in permissions
        assert Permission.DELETE in permissions
        assert Permission.ADMIN in permissions

    @pytest.mark.unit
    def test_permission_from_string(self):
        """Test creating Permission from string value."""
        assert Permission("read") == Permission.READ
        assert Permission("write") == Permission.WRITE
        assert Permission("delete") == Permission.DELETE
        assert Permission("admin") == Permission.ADMIN

    @pytest.mark.unit
    def test_permission_invalid_value_raises_exception(self):
        """Test that invalid permission value raises ValueError."""
        with pytest.raises(ValueError):
            Permission("invalid_permission")
