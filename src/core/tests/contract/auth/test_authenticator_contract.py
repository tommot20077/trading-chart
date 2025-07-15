# ABOUTME: Contract tests for AbstractAuthenticator interface
# ABOUTME: Verifies all authenticator implementations comply with the interface contract

import pytest
from typing import Type, List

from core.interfaces.auth.authenticator import AbstractAuthenticator
from core.implementations.memory.auth.authenticator import InMemoryAuthenticator
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestAuthenticatorContract(ContractTestBase[AbstractAuthenticator], AsyncContractTestMixin):
    """Contract tests for AbstractAuthenticator interface."""

    @property
    def interface_class(self) -> Type[AbstractAuthenticator]:
        return AbstractAuthenticator

    @property
    def implementations(self) -> List[Type[AbstractAuthenticator]]:
        return [
            InMemoryAuthenticator,
            # Add other implementations here as they are created
            # DatabaseAuthenticator,
            # JWTAuthenticator,
        ]

    @pytest.mark.contract
    def test_authenticate_method_signature(self):
        """Test authenticate method has correct signature."""
        method = getattr(self.interface_class, "authenticate")
        assert hasattr(method, "__isabstractmethod__")

        # Verify it's async
        import asyncio

        assert asyncio.iscoroutinefunction(method)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authenticate_contract_behavior(self):
        """Test authenticate method contract behavior across implementations."""
        from core.exceptions import AuthenticationException

        # Create a simple mock implementation for testing
        class MockAuthenticator(AbstractAuthenticator):
            async def authenticate(self, request):
                from core.exceptions import AuthenticationException

                raise AuthenticationException("Mock authentication failure")

        mock_auth = MockAuthenticator()

        # Test that method exists and is callable
        assert hasattr(mock_auth, "authenticate")
        assert callable(mock_auth.authenticate)

        # For real implementations, we would test actual behavior
        for impl_class in self.implementations:
            if impl_class == InMemoryAuthenticator:
                # Test with actual implementation
                auth = impl_class()

                # Test with invalid request should raise AuthenticationException
                class MockRequest:
                    def get_header(self, name: str) -> str | None:
                        return None

                    @property
                    def client_id(self) -> str | None:
                        return None

                with pytest.raises(AuthenticationException):
                    await auth.authenticate(MockRequest())
