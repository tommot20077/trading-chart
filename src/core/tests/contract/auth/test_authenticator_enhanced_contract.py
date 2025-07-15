# ABOUTME: Enhanced contract tests for AbstractAuthenticator with comprehensive behavior verification
# ABOUTME: Tests security, performance, edge cases, and error handling for authenticator implementations

import pytest
import asyncio
import time
from typing import Type, List

from core.interfaces.auth.authenticator import AbstractAuthenticator
from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
from core.exceptions import AuthenticationException
from ..enhanced_contract_tests import EnhancedContractTestBase


class TestAuthenticatorEnhancedContract(EnhancedContractTestBase[AbstractAuthenticator]):
    """Enhanced contract tests for AbstractAuthenticator interface."""

    @property
    def interface_class(self) -> Type[AbstractAuthenticator]:
        return AbstractAuthenticator

    @property
    def implementations(self) -> List[Type[AbstractAuthenticator]]:
        return [InMemoryAuthenticator]

    # Performance bounds for authenticator
    max_operation_time = 0.1  # Authentication should be fast

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authentication_security_contract(self):
        """Test authentication security requirements."""
        for impl_class in self.implementations:
            if impl_class == InMemoryAuthenticator:
                auth = impl_class()

                # Test 1: Invalid bearer token format
                class InvalidBearerRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return "InvalidFormat token123"
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(InvalidBearerRequest())

                # Test 2: Missing authorization header
                class NoAuthRequest:
                    def get_header(self, name: str) -> str | None:
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(NoAuthRequest())

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authentication_performance_contract(self):
        """Test authentication performance requirements."""
        for impl_class in self.implementations:
            if impl_class == InMemoryAuthenticator:
                auth = impl_class()

                # Create a valid user first
                user_result = await auth.create_user("test_user", "password123")
                login_result = await auth.login("test_user", "password123")
                raw_token, token = login_result  # Unpack tuple result

                class ValidRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return f"Bearer {raw_token}"  # Use raw token string
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                # Test authentication performance
                start_time = time.time()
                result = await auth.authenticate(ValidRequest())
                end_time = time.time()

                # Authentication should complete within performance bounds
                assert (end_time - start_time) < self.max_operation_time
                assert result is not None

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authentication_edge_cases(self):
        """Test authentication edge case handling."""
        for impl_class in self.implementations:
            if impl_class == InMemoryAuthenticator:
                auth = impl_class()

                # Test 1: Extremely long token
                class LongTokenRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return f"Bearer {'x' * 10000}"
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(LongTokenRequest())

                # Test 2: Empty token
                class EmptyTokenRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return "Bearer "
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(EmptyTokenRequest())

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authentication_concurrent_safety(self):
        """Test authentication concurrent access safety."""
        for impl_class in self.implementations:
            if impl_class == InMemoryAuthenticator:
                auth = impl_class()

                # Create a valid user
                user_result = await auth.create_user("concurrent_user", "password123")
                login_result = await auth.login("concurrent_user", "password123")
                raw_token, token = login_result  # Unpack tuple result

                class ConcurrentRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return f"Bearer {raw_token}"  # Use raw token string
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                # Test concurrent authentication requests
                tasks = []
                for _ in range(10):
                    task = asyncio.create_task(auth.authenticate(ConcurrentRequest()))
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # All concurrent requests should succeed
                for result in results:
                    assert not isinstance(result, Exception)
                    assert result is not None

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authentication_input_sanitization(self):
        """Test authentication input sanitization."""
        for impl_class in self.implementations:
            if impl_class == InMemoryAuthenticator:
                auth = impl_class()

                # Test SQL injection attempt in token
                class SQLInjectionRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return "Bearer '; DROP TABLE users; --"
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(SQLInjectionRequest())

                # Test XSS attempt in token
                class XSSRequest:
                    def get_header(self, name: str) -> str | None:
                        if name.lower() == "authorization":
                            return "Bearer <script>alert('xss')</script>"
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(XSSRequest())
