# ABOUTME: Contract tests for AbstractAuthorizer interface
# ABOUTME: Verifies all authorizer implementations comply with the interface contract

import pytest
from typing import Type, List

from core.interfaces.auth.authorizer import AbstractAuthorizer
from core.implementations.memory.auth.authorizer import InMemoryAuthorizer
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestAuthorizerContract(ContractTestBase[AbstractAuthorizer], AsyncContractTestMixin):
    """Contract tests for AbstractAuthorizer interface."""

    @property
    def interface_class(self) -> Type[AbstractAuthorizer]:
        return AbstractAuthorizer

    @property
    def implementations(self) -> List[Type[AbstractAuthorizer]]:
        return [
            InMemoryAuthorizer,
            # Add other implementations here as they are created
        ]

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authorize_permission_contract(self):
        """Test authorize_permission method contract behavior."""
        from core.models.auth.enum import Permission
        from core.exceptions import AuthorizationError

        for impl_class in self.implementations:
            if impl_class == InMemoryAuthorizer:
                authorizer = impl_class()

                # Test with invalid token should raise AuthorizationError
                class MockToken:
                    @property
                    def id(self) -> str:
                        return "nonexistent"

                    @property
                    def user_id(self) -> str:
                        return "nonexistent"

                    @property
                    def username(self) -> str:
                        return "test_user"

                    @property
                    def permissions(self) -> set:
                        return set()

                    @property
                    def roles(self) -> set:
                        return set()

                with pytest.raises(AuthorizationError):
                    await authorizer.authorize_permission(MockToken(), Permission.READ)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_authorize_role_contract(self):
        """Test authorize_role method contract behavior."""
        from core.models.auth.enum import Role
        from core.exceptions import AuthorizationError

        for impl_class in self.implementations:
            if impl_class == InMemoryAuthorizer:
                authorizer = impl_class()

                # Test with invalid token should raise AuthorizationError
                class MockToken:
                    @property
                    def id(self) -> str:
                        return "nonexistent"

                    @property
                    def user_id(self) -> str:
                        return "nonexistent"

                    @property
                    def username(self) -> str:
                        return "test_user"

                    @property
                    def permissions(self) -> set:
                        return set()

                    @property
                    def roles(self) -> set:
                        return set()

                with pytest.raises(AuthorizationError):
                    await authorizer.authorize_role(MockToken(), Role.USER)
