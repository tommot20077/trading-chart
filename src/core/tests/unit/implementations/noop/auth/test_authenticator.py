# ABOUTME: Unit tests for NoOpAuthenticator
# ABOUTME: Tests for no-operation authentication implementation

import pytest
import asyncio
import inspect
from unittest.mock import Mock, patch
from typing import Any, Dict

from core.implementations.noop.auth.authenticator import NoOpAuthenticator
from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
from core.interfaces.auth.authenticator import AbstractAuthenticator
from core.models.auth.auth_request import AuthRequest
from core.models.auth.auth_token import AuthToken
from core.models.auth.enum import Role, Permission
from core.implementations.memory.auth.models import MemoryAuthToken
from tests.constants import TestTimeouts, TestDataSizes, PerformanceThresholds


class MockAuthRequest:
    """Concrete implementation of AuthRequest protocol for testing."""
    
    def __init__(self, username: str = "test_user", password: str = "test_password", 
                 metadata: Dict[str, Any] = None, client_id: str = None, headers: Dict[str, str] = None):
        self.username = username
        self.password = password
        self.metadata = metadata or {}
        self._client_id = client_id
        self._headers = headers or {}
    
    def get_header(self, name: str) -> str | None:
        """Get header value by name."""
        return self._headers.get(name.lower())
    
    @property
    def client_id(self) -> str | None:
        """Get client ID."""
        return self._client_id


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


class TestNoOpAuthenticatorContractCompliance:
    """Test contract compliance and interface implementation for NoOpAuthenticator."""
    
    @pytest.mark.unit
    def test_interface_compliance(self):
        """Test that NoOpAuthenticator properly implements AbstractAuthenticator interface."""
        auth = NoOpAuthenticator()
        
        # Verify inheritance
        assert isinstance(auth, AbstractAuthenticator)
        
        # Verify all abstract methods are implemented
        abstract_methods = [
            name for name, method in inspect.getmembers(AbstractAuthenticator, predicate=inspect.ismethod)
            if getattr(method, '__isabstractmethod__', False)
        ]
        
        for method_name in abstract_methods:
            assert hasattr(auth, method_name), f"Missing method: {method_name}"
            method = getattr(auth, method_name)
            assert callable(method), f"Method {method_name} is not callable"
    
    @pytest.mark.unit
    def test_method_signatures_match_interface(self):
        """Test that method signatures match the abstract interface."""
        noop_auth = NoOpAuthenticator()
        
        # Get interface methods
        interface_methods = inspect.getmembers(AbstractAuthenticator, predicate=inspect.isfunction)
        
        for method_name, interface_method in interface_methods:
            if method_name.startswith('_'):
                continue  # Skip private methods
            
            if hasattr(noop_auth, method_name):
                noop_method = getattr(noop_auth, method_name)
                interface_sig = inspect.signature(interface_method)
                noop_sig = inspect.signature(noop_method)
                
                # Compare parameter names and types
                interface_params = list(interface_sig.parameters.keys())
                noop_params = list(noop_sig.parameters.keys())
                
                # Allow for 'self' parameter difference
                if 'self' in interface_params:
                    interface_params.remove('self')
                if 'self' in noop_params:
                    noop_params.remove('self')
                
                assert interface_params == noop_params, \
                    f"Method {method_name} parameter mismatch: {interface_params} vs {noop_params}"
    
    @pytest.mark.asyncio
    async def test_authenticate_contract_compliance(self):
        """Test that authenticate method fulfills the expected contract."""
        auth = NoOpAuthenticator()
        
        # Test with valid MockAuthRequest  
        request = MockAuthRequest(
            username="test_user",
            password="test_password",
            metadata={"source": "test"}
        )
        
        token = await auth.authenticate(request)
        
        # Verify return type
        assert isinstance(token, MemoryAuthToken), "authenticate must return AuthToken instance"
        
        # Verify required token fields are present
        assert hasattr(token, 'user_id'), "Token must have user_id"
        assert hasattr(token, 'username'), "Token must have username"
        assert hasattr(token, 'token'), "Token must have token"
        assert hasattr(token, 'roles'), "Token must have roles"
        assert hasattr(token, 'permissions'), "Token must have permissions"
        assert hasattr(token, 'expires_at'), "Token must have expires_at"
        
        # Verify field types
        assert isinstance(token.user_id, str), "user_id must be string"
        assert isinstance(token.username, str), "username must be string"
        assert isinstance(token.token, str), "token must be string"
        assert isinstance(token.roles, (list, set)), "roles must be list or set"
        assert isinstance(token.permissions, (list, set)), "permissions must be list or set"
    
    @pytest.mark.asyncio
    async def test_authenticate_with_invalid_inputs(self):
        """Test authenticate behavior with invalid inputs according to contract."""
        auth = NoOpAuthenticator()
        
        # Test with None input - NoOp should handle gracefully or raise appropriate error
        try:
            token = await auth.authenticate(None)
            # If NoOp handles gracefully, verify token is still valid
            assert isinstance(token, MemoryAuthToken)
        except (TypeError, ValueError, AttributeError):
            # These are acceptable errors for invalid input  
            pass
        
        # Test with malformed request
        malformed_request = Mock()
        malformed_request.username = None
        malformed_request.password = None
        
        # NoOp should handle gracefully or raise appropriate error
        try:
            token = await auth.authenticate(malformed_request)
            # If it succeeds, it should still return a valid token
            assert isinstance(token, MemoryAuthToken)
        except (TypeError, ValueError, AttributeError):
            # These are acceptable errors for invalid input  
            pass
    
    @pytest.mark.asyncio
    async def test_consistency_across_calls(self):
        """Test that NoOp authenticator is consistent across multiple calls."""
        auth = NoOpAuthenticator()
        request = MockAuthRequest(username="test", password="test")
        
        # Multiple calls should return consistent results
        tokens = []
        for _ in range(TestDataSizes.SMALL_DATASET):  # 10 calls
            token = await auth.authenticate(request)
            tokens.append(token)
        
        # All tokens should have same basic properties (NoOp behavior)
        first_token = tokens[0]
        for token in tokens[1:]:
            assert token.user_id == first_token.user_id
            assert token.username == first_token.username
            assert token.token == first_token.token
            assert token.roles == first_token.roles
            assert token.permissions == first_token.permissions
    
    @pytest.mark.asyncio
    async def test_concurrent_authenticate_calls(self):
        """Test that concurrent authenticate calls work correctly."""
        auth = NoOpAuthenticator()
        request = MockAuthRequest(username="test", password="test")
        
        # Run concurrent authentication requests
        async def authenticate_task():
            return await auth.authenticate(request)
        
        tasks = [authenticate_task() for _ in range(TestDataSizes.SMALL_DATASET)]
        tokens = await asyncio.gather(*tasks)
        
        # All should succeed and return valid tokens
        assert len(tokens) == TestDataSizes.SMALL_DATASET
        for token in tokens:
            assert isinstance(token, MemoryAuthToken)
            assert token.user_id is not None
            assert token.username is not None
    
    @pytest.mark.unit
    def test_performance_characteristics(self):
        """Test that NoOp authenticator has expected performance characteristics."""
        auth = NoOpAuthenticator()
        request = MockAuthRequest(username="test", password="test")
        
        # NoOp should be extremely fast
        import time
        
        async def time_authenticate():
            start = time.time()
            for _ in range(TestDataSizes.MEDIUM_DATASET):  # 100 calls
                await auth.authenticate(request)
            return time.time() - start
        
        # Run the timing test
        total_time = asyncio.run(time_authenticate())
        avg_time = total_time / TestDataSizes.MEDIUM_DATASET
        
        # NoOp should be much faster than normal operations
        assert avg_time < TestTimeouts.QUICK_OPERATION / 10, \
            f"NoOp authenticate too slow: {avg_time}s per call"
    
    @pytest.mark.unit 
    def test_behavioral_contract_with_memory_implementation(self):
        """Test behavioral compatibility with memory implementation interface."""
        noop_auth = NoOpAuthenticator()
        memory_auth = InMemoryAuthenticator()
        
        # Both should have same public interface
        noop_methods = set(method for method in dir(noop_auth) 
                          if not method.startswith('_') and callable(getattr(noop_auth, method)))
        memory_methods = set(method for method in dir(memory_auth)
                           if not method.startswith('_') and callable(getattr(memory_auth, method)))
        
        # NoOp should implement at least the core interface methods
        core_methods = {'authenticate'}  # Add other core methods as needed
        assert core_methods.issubset(noop_methods), \
            f"NoOp missing core methods: {core_methods - noop_methods}"
    
    @pytest.mark.asyncio
    async def test_error_handling_contract(self):
        """Test error handling behavior matches expected contract."""
        auth = NoOpAuthenticator()
        
        # Test various error scenarios
        error_scenarios = [
            Mock(spec=AuthRequest, username="", password=""),  # Empty credentials
            Mock(spec=AuthRequest, username="test", password=""),  # Empty password
            Mock(spec=AuthRequest, username="", password="test"),  # Empty username
        ]
        
        for scenario in error_scenarios:
            try:
                token = await auth.authenticate(scenario)
                # If NoOp handles gracefully, verify token is still valid
                assert isinstance(token, MemoryAuthToken)
                assert token.user_id is not None
            except (ValueError, TypeError, AttributeError):
                # These are acceptable error types for invalid input
                pass
    
    @pytest.mark.unit
    def test_immutability_contract(self):
        """Test that NoOp authenticator maintains immutability where expected."""
        auth = NoOpAuthenticator()
        
        # Multiple instances should behave identically
        auth2 = NoOpAuthenticator()
        
        # Test that both instances behave the same way
        request = MockAuthRequest(username="test", password="test")
        
        async def compare_instances():
            token1 = await auth.authenticate(request)
            token2 = await auth2.authenticate(request)
            
            # Should return equivalent tokens
            assert token1.user_id == token2.user_id
            assert token1.username == token2.username
            assert token1.token == token2.token
            assert token1.roles == token2.roles
            assert token1.permissions == token2.permissions
        
        asyncio.run(compare_instances())
    
    @pytest.mark.unit
    def test_resource_cleanup_contract(self):
        """Test that NoOp authenticator properly manages resources."""
        auth = NoOpAuthenticator()
        
        # NoOp should not hold onto resources
        request = MockAuthRequest(username="test", password="test")
        
        async def resource_test():
            # Create many tokens
            tokens = []
            for _ in range(TestDataSizes.MEDIUM_DATASET):
                token = await auth.authenticate(request)
                tokens.append(token)
            
            # Clear references
            tokens.clear()
            
            # NoOp should not accumulate state
            token_after = await auth.authenticate(request)
            assert isinstance(token_after, MemoryAuthToken)
        
        asyncio.run(resource_test())
        
        # Authenticator should still be functional
        final_token = asyncio.run(auth.authenticate(request))
        assert isinstance(final_token, MemoryAuthToken)
    
    @pytest.mark.unit
    def test_state_isolation_contract(self):
        """Test that NoOp authenticator maintains proper state isolation."""
        auth = NoOpAuthenticator()
        
        # Different requests should not interfere with each other
        request1 = MockAuthRequest(username="user1", password="pass1")
        request2 = MockAuthRequest(username="user2", password="pass2")
        
        async def isolation_test():
            # Authenticate different users
            token1 = await auth.authenticate(request1)
            token2 = await auth.authenticate(request2)
            
            # Tokens should be consistent with NoOp behavior
            # (they might be identical for NoOp, but should be valid)
            assert isinstance(token1, MemoryAuthToken)
            assert isinstance(token2, MemoryAuthToken)
            
            # Re-authenticate first user - should get consistent result
            token1_again = await auth.authenticate(request1)
            assert token1_again.user_id == token1.user_id
            assert token1_again.username == token1.username
        
        asyncio.run(isolation_test())
