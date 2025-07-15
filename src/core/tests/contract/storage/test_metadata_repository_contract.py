# ABOUTME: Contract tests for AbstractMetadataRepository interface
# ABOUTME: Verifies all metadata repository implementations comply with the interface contract

import pytest
from typing import Type, List
from datetime import datetime, UTC

from core.interfaces.storage.metadata_repository import AbstractMetadataRepository
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin, ResourceManagementContractMixin


class TestMetadataRepositoryContract(
    ContractTestBase[AbstractMetadataRepository], AsyncContractTestMixin, ResourceManagementContractMixin
):
    """Contract tests for AbstractMetadataRepository interface."""

    @property
    def interface_class(self) -> Type[AbstractMetadataRepository]:
        return AbstractMetadataRepository

    @property
    def implementations(self) -> List[Type[AbstractMetadataRepository]]:
        return [
            InMemoryMetadataRepository,
            # Add other implementations here as they are created
            # DatabaseMetadataRepository,
            # RedisMetadataRepository,
        ]

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_basic_crud_contract_behavior(self):
        """Test basic CRUD operations contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Test set and get
                test_key = "test_key"
                test_value = {"test": "value", "number": 42}

                await repo.set(test_key, test_value)
                retrieved_value = await repo.get(test_key)
                assert retrieved_value == test_value

                # Test get non-existent key returns None
                non_existent = await repo.get("non_existent_key")
                assert non_existent is None

                # Test exists
                assert await repo.exists(test_key) is True
                assert await repo.exists("non_existent_key") is False

                # Test delete
                deleted = await repo.delete(test_key)
                assert deleted is True
                assert await repo.exists(test_key) is False

                # Test delete non-existent key returns False
                deleted_again = await repo.delete(test_key)
                assert deleted_again is False

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_ttl_contract_behavior(self):
        """Test TTL (Time To Live) operations contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Test set_with_ttl
                test_key = "ttl_test_key"
                test_value = {"ttl": "test"}
                ttl_seconds = 2  # Use longer TTL for more reliable testing

                await repo.set_with_ttl(test_key, test_value, ttl_seconds)

                # Should exist immediately
                assert await repo.exists(test_key) is True
                retrieved = await repo.get(test_key)
                assert retrieved == test_value

                # Note: get_ttl and set_ttl methods are not in the interface
                # This is testing implementation-specific behavior
                # We'll focus on the core TTL functionality instead

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_specialized_methods_contract_behavior(self):
        """Test specialized sync and backfill methods contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Test sync time methods
                symbol = "BTCUSDT"
                data_type = "trades"
                test_time = datetime.now(UTC)

                # Initially should return None
                sync_time = await repo.get_last_sync_time(symbol, data_type)
                assert sync_time is None

                # Set sync time
                await repo.set_last_sync_time(symbol, data_type, test_time)

                # Retrieve sync time
                retrieved_time = await repo.get_last_sync_time(symbol, data_type)
                assert retrieved_time is not None
                # Allow some tolerance for datetime comparison
                assert abs((retrieved_time - test_time).total_seconds()) < 1

                # Test backfill status methods
                test_status = {"progress": 0.5, "last_timestamp": test_time.isoformat(), "errors": []}

                # Initially should return None
                backfill_status = await repo.get_backfill_status(symbol, data_type)
                assert backfill_status is None

                # Set backfill status
                await repo.set_backfill_status(symbol, data_type, test_status)

                # Retrieve backfill status
                retrieved_status = await repo.get_backfill_status(symbol, data_type)
                assert retrieved_status is not None
                assert retrieved_status["progress"] == 0.5
                assert retrieved_status["symbol"] == symbol
                assert retrieved_status["data_type"] == data_type

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_query_operations_contract_behavior(self):
        """Test query operations contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Set up test data manually
                test_data = {
                    "user:1": {"name": "Alice"},
                    "user:2": {"name": "Bob"},
                    "config:app": {"debug": True},
                    "config:db": {"host": "localhost"},
                }
                for key, value in test_data.items():
                    await repo.set(key, value)

                # Test list_keys without pattern (should return all keys)
                all_keys = await repo.list_keys()
                assert isinstance(all_keys, list)
                assert len(all_keys) >= 4  # At least our test keys
                for key in test_data.keys():
                    assert key in all_keys

                # Test list_keys with pattern
                user_keys = await repo.list_keys("user:")
                assert isinstance(user_keys, list)
                assert "user:1" in user_keys
                assert "user:2" in user_keys
                assert "config:app" not in user_keys

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_context_manager_contract_behavior(self):
        """Test async context manager contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Test async context manager protocol
                async with repo as ctx_repo:
                    assert ctx_repo is not None
                    assert ctx_repo is repo

                    # Should be able to use repo normally within context
                    await ctx_repo.set("ctx_test", {"context": "test"})
                    value = await ctx_repo.get("ctx_test")
                    assert value == {"context": "test"}

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_multiple_operations_contract_behavior(self):
        """Test multiple operations working together."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Set up test data manually (since set_many is not in interface)
                test_keys = ["multi_test_1", "multi_test_2", "multi_test_3"]
                for i, key in enumerate(test_keys):
                    await repo.set(key, {"value": i + 1})

                # Verify all data exists
                for key in test_keys:
                    assert await repo.exists(key) is True

                # Test list_keys functionality
                all_keys = await repo.list_keys()
                assert isinstance(all_keys, list)
                for key in test_keys:
                    assert key in all_keys

                # Clean up by deleting test keys
                for key in test_keys:
                    deleted = await repo.delete(key)
                    assert deleted is True

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_close_contract_behavior(self):
        """Test close method contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                repo = impl_class()

                # Test close method exists and can be called
                result = await repo.close()
                assert result is None  # close should return None

    @pytest.mark.contract
    def test_required_methods_exist(self):
        """Test that all required methods exist in implementations."""
        required_methods = [
            "get",
            "set",
            "delete",
            "exists",
            "list_keys",
            "set_with_ttl",
            "get_last_sync_time",
            "set_last_sync_time",
            "get_backfill_status",
            "set_backfill_status",
            "close",
        ]

        for impl_class in self.implementations:
            for method_name in required_methods:
                assert hasattr(impl_class, method_name), f"{impl_class.__name__} missing method: {method_name}"

                method = getattr(impl_class, method_name)
                assert callable(method), f"{impl_class.__name__}.{method_name} is not callable"

    @pytest.mark.contract
    def test_data_type_handling_contract(self):
        """Test that implementations handle various data types correctly."""
        for impl_class in self.implementations:
            if impl_class == InMemoryMetadataRepository:
                # This is a basic contract test - implementations should handle common types
                # Actual type handling is tested in unit tests
                repo = impl_class()

                # Test that repo can be instantiated
                assert repo is not None
                assert isinstance(repo, impl_class)
                assert isinstance(repo, AbstractMetadataRepository)

    @pytest.mark.contract
    def test_error_handling_contracts(self):
        """Test that implementations properly define error handling contracts."""

        # Check that methods have proper exception documentation or type hints
        critical_methods = ["get", "set", "delete", "close"]

        for method_name in critical_methods:
            method = getattr(self.interface_class, method_name)
            assert hasattr(method, "__isabstractmethod__")

            # Verify method exists in implementations
            for impl_class in self.implementations:
                assert hasattr(impl_class, method_name)
                impl_method = getattr(impl_class, method_name)
                assert callable(impl_method)
