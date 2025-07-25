# ABOUTME: Unit tests for InMemoryTokenManager UUID-based implementation
# ABOUTME: Tests UUID-based token generation, validation, refresh, and revocation functionality

import pytest
import uuid
from datetime import datetime, timedelta, UTC
from typing import Any
import time

from core.implementations.memory.auth.token_manager import InMemoryTokenManager
from core.implementations.memory.auth.models import MemoryAuthToken
from core.models.auth.enum import Role
from core.models.types import UserData
from core.exceptions import AuthenticationException


class TestInMemoryTokenManager:
    """Test suite for InMemoryTokenManager UUID-based implementation."""
    
    @pytest.fixture
    def token_manager(self):
        """Create a token manager instance."""
        return InMemoryTokenManager(default_ttl=3600, refresh_ttl=7200)
    
    @pytest.fixture
    def sample_user_data(self) -> UserData:
        """Sample user data for testing."""
        return {
            "user_id": "user123",
            "username": "testuser",
            "roles": ["user", "admin"],
            "email": "test@example.com",
            "permissions": ["read", "write"],
        }
    
    def test_generate_token_success(self, token_manager, sample_user_data):
        """Test successful token generation."""
        token = token_manager.generate_token(sample_user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Should be a valid UUID
        uuid.UUID(token)  # This will raise ValueError if not a valid UUID
        
        # Token should be stored in manager
        assert token in token_manager._tokens
    
    def test_generate_token_minimal_data(self, token_manager):
        """Test token generation with minimal user data."""
        minimal_data: UserData = {
            "user_id": "minimal123",
            "username": "minimal",
        }
        
        token = token_manager.generate_token(minimal_data)
        uuid.UUID(token)  # Valid UUID
        
        # Validate token to check stored data
        auth_token = token_manager.validate_token(token)
        assert auth_token.user_id == "minimal123"
        assert auth_token.username == "minimal"
        assert auth_token.roles == []  # Default empty list
    
    def test_validate_token_success(self, token_manager, sample_user_data):
        """Test successful token validation."""
        token = token_manager.generate_token(sample_user_data)
        auth_token = token_manager.validate_token(token)
        
        assert isinstance(auth_token, MemoryAuthToken)
        assert auth_token.user_id == "user123"
        assert auth_token.username == "testuser"
        assert set(auth_token.roles) == {"user", "admin"}
        assert not auth_token.is_expired()
        assert auth_token.expires_at > time.time()
    
    def test_validate_token_invalid_token(self, token_manager, sample_user_data):
        """Test validation fails with invalid token."""
        # Generate token with different manager
        different_manager = InMemoryTokenManager()
        token = different_manager.generate_token(sample_user_data)
        
        with pytest.raises(AuthenticationException, match="Token not found"):
            token_manager.validate_token(token)
    
    def test_validate_token_expired(self, token_manager):
        """Test validation fails with expired token."""
        # Create token with very short TTL
        short_ttl_manager = InMemoryTokenManager(default_ttl=1)  # 1 second
        
        user_data: UserData = {
            "user_id": "expired_user",
            "username": "expired",
        }
        
        token = short_ttl_manager.generate_token(user_data)
        
        # Wait for expiration
        time.sleep(2)
        
        with pytest.raises(AuthenticationException, match="Token has expired"):
            short_ttl_manager.validate_token(token)
    
    def test_validate_token_malformed(self, token_manager):
        """Test validation fails with malformed token."""
        with pytest.raises(AuthenticationException, match="Token not found"):
            token_manager.validate_token("not-a-valid-uuid")
        
        with pytest.raises(AuthenticationException, match="Invalid token format"):
            token_manager.validate_token("")
        
        # Valid UUID but not in our token store
        random_uuid = str(uuid.uuid4())
        with pytest.raises(AuthenticationException, match="Token not found"):
            token_manager.validate_token(random_uuid)
    
    def test_validate_token_after_revocation(self, token_manager, sample_user_data):
        """Test validation fails after token revocation."""
        token = token_manager.generate_token(sample_user_data)
        
        # First validation should succeed
        auth_token = token_manager.validate_token(token)
        assert not auth_token.is_expired()
        
        # Revoke the token
        token_manager.revoke_token(token)
        
        # Validation should now fail
        with pytest.raises(AuthenticationException, match="Token has been revoked"):
            token_manager.validate_token(token)
    
    def test_refresh_token_success(self, token_manager, sample_user_data):
        """Test successful token refresh."""
        original_token = token_manager.generate_token(sample_user_data)
        
        # Wait a bit to ensure different timestamp
        time.sleep(0.1)
        
        refreshed_token = token_manager.refresh_token(original_token)
        
        assert refreshed_token != original_token
        
        # Original token should be revoked after refresh
        with pytest.raises(AuthenticationException, match="Token has been revoked"):
            token_manager.validate_token(original_token)
        
        # Refreshed token should be valid
        refreshed_auth = token_manager.validate_token(refreshed_token)
        
        # User data should be preserved
        assert refreshed_auth.user_id == "user123"
        assert refreshed_auth.username == "testuser"
        assert set(refreshed_auth.roles) == {"user", "admin"}
        
        # Refreshed token should not be expired  
        assert not refreshed_auth.is_expired()
    
    def test_refresh_token_invalid(self, token_manager):
        """Test refresh fails with invalid token."""
        with pytest.raises(AuthenticationException, match="Token not found"):
            token_manager.refresh_token("invalid-token-here")
    
    def test_refresh_token_expired(self, token_manager):
        """Test refresh fails with expired token."""
        # Create token with very short TTL
        short_ttl_manager = InMemoryTokenManager(default_ttl=1)
        
        user_data: UserData = {
            "user_id": "expired_user",
            "username": "expired",  
        }
        
        token = short_ttl_manager.generate_token(user_data)
        
        # Wait for expiration
        time.sleep(2)
        
        with pytest.raises(AuthenticationException, match="Token has expired"):
            short_ttl_manager.refresh_token(token)
    
    def test_refresh_token_revoked(self, token_manager, sample_user_data):
        """Test refresh fails with revoked token."""
        token = token_manager.generate_token(sample_user_data)
        token_manager.revoke_token(token)
        
        with pytest.raises(AuthenticationException, match="Token has been revoked"):
            token_manager.refresh_token(token)
    
    def test_revoke_token_success(self, token_manager, sample_user_data):
        """Test successful token revocation."""
        token = token_manager.generate_token(sample_user_data)
        
        # Token should be valid before revocation
        auth_token = token_manager.validate_token(token)
        assert not auth_token.is_expired()
        
        # Revoke the token
        token_manager.revoke_token(token)
        
        # Token should be marked as inactive in storage
        token_data = token_manager._tokens[token]
        assert token_data.is_active is False
        
        # Validation should fail
        with pytest.raises(AuthenticationException, match="Token has been revoked"):
            token_manager.validate_token(token)
    
    def test_revoke_token_invalid(self, token_manager):
        """Test revoking invalid token raises error."""
        with pytest.raises(AuthenticationException, match="Cannot revoke token: token not found"):
            token_manager.revoke_token("invalid-token")
    
    def test_revoke_token_already_revoked(self, token_manager, sample_user_data):
        """Test revoking already revoked token."""
        token = token_manager.generate_token(sample_user_data)
        
        # First revocation should succeed
        token_manager.revoke_token(token)
        
        # Second revocation should also succeed (idempotent)
        token_manager.revoke_token(token)
        
        # Token should still be inactive
        token_data = token_manager._tokens[token]
        assert token_data.is_active is False
    
    def test_multiple_tokens_same_user(self, token_manager, sample_user_data):
        """Test multiple tokens can be generated for same user."""
        token1 = token_manager.generate_token(sample_user_data)
        token2 = token_manager.generate_token(sample_user_data)
        
        assert token1 != token2
        
        # Both should be valid
        auth1 = token_manager.validate_token(token1)
        auth2 = token_manager.validate_token(token2)
        
        assert auth1.user_id == auth2.user_id
        assert auth1.username == auth2.username
        
        # Revoking one shouldn't affect the other
        token_manager.revoke_token(token1)
        
        with pytest.raises(AuthenticationException):
            token_manager.validate_token(token1)
        
        # token2 should still be valid
        auth2_again = token_manager.validate_token(token2)
        assert not auth2_again.is_expired()
    
    def test_token_expiration_time(self, token_manager, sample_user_data):
        """Test token expiration time is set correctly."""
        token = token_manager.generate_token(sample_user_data)
        auth_token = token_manager.validate_token(token)
        
        # Check expiration is approximately 1 hour from now (default_ttl=3600)
        expected_exp = time.time() + 3600
        time_diff = abs(auth_token.expires_at - expected_exp)
        
        # Allow 5 seconds difference for test execution time
        assert time_diff < 5
    
    def test_custom_expiration_time(self):
        """Test token manager with custom expiration time."""
        custom_manager = InMemoryTokenManager(
            default_ttl=7200  # 2 hours
        )
        
        user_data: UserData = {
            "user_id": "user123",
            "username": "testuser",
        }
        
        token = custom_manager.generate_token(user_data)
        auth_token = custom_manager.validate_token(token)
        
        # Check expiration is approximately 2 hours from now
        expected_exp = time.time() + 7200
        time_diff = abs(auth_token.expires_at - expected_exp)
        
        assert time_diff < 5
    
    def test_role_storage(self, token_manager):
        """Test role string storage and retrieval."""
        user_data: UserData = {
            "user_id": "user123",
            "username": "testuser",
            "roles": ["admin", "user", "viewer"],
        }
        
        token = token_manager.generate_token(user_data)
        auth_token = token_manager.validate_token(token)
        
        # Should store roles as strings
        assert set(auth_token.roles) == {"admin", "user", "viewer"}
    
    def test_permissions_storage(self, token_manager):
        """Test permissions storage and retrieval."""
        user_data: UserData = {
            "user_id": "user123",
            "username": "testuser",
            "permissions": ["read", "write", "delete"],
        }
        
        token = token_manager.generate_token(user_data)
        auth_token = token_manager.validate_token(token)
        
        # Check that auth_token has permissions if the implementation supports it
        token_data = token_manager._tokens[token]
        assert token_data.permissions == ["read", "write", "delete"]
    
    def test_concurrent_token_operations(self, token_manager, sample_user_data):
        """Test thread safety of token operations."""
        import threading
        import concurrent.futures
        
        tokens = []
        
        def generate_tokens():
            for _ in range(10):
                token = token_manager.generate_token(sample_user_data)
                tokens.append(token)
        
        # Generate tokens concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_tokens) for _ in range(5)]
            concurrent.futures.wait(futures)
        
        # All tokens should be unique
        assert len(tokens) == 50
        assert len(set(tokens)) == 50
        
        # All tokens should be valid
        for token in tokens:
            auth_token = token_manager.validate_token(token)
            assert not auth_token.is_expired()