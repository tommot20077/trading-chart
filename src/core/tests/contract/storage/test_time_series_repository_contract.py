# ABOUTME: Contract tests for AbstractTimeSeriesRepository interface
# ABOUTME: Verifies all time series repository implementations comply with the interface contract

import pytest
from typing import Type, List
from datetime import datetime

from core.interfaces.storage.time_sequence_repository import AbstractTimeSeriesRepository
from core.implementations.memory.storage.time_series_repository import InMemoryTimeSeriesRepository
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin, ResourceManagementContractMixin


class TestTimeSeriesRepositoryContract(
    ContractTestBase[AbstractTimeSeriesRepository], AsyncContractTestMixin, ResourceManagementContractMixin
):
    """Contract tests for AbstractTimeSeriesRepository interface."""

    @property
    def interface_class(self) -> Type[AbstractTimeSeriesRepository]:
        return AbstractTimeSeriesRepository

    @property
    def implementations(self) -> List[Type[AbstractTimeSeriesRepository]]:
        return [
            InMemoryTimeSeriesRepository,
            # Add other implementations here as they are created
            # DatabaseTimeSeriesRepository,
            # RedisTimeSeriesRepository,
        ]

    @pytest.mark.contract
    def test_generic_type_parameter(self):
        """Test that the interface properly uses generic type parameter."""
        import inspect

        # Check that the interface is generic
        assert hasattr(self.interface_class, "__orig_bases__")

        # Check method signatures use the generic type
        save_method = getattr(self.interface_class, "save")
        sig = inspect.signature(save_method)

        # The 'item' parameter should use the generic type T
        assert "item" in sig.parameters

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_save_contract_behavior(self):
        """Test save method contract behavior."""

        for impl_class in self.implementations:
            if impl_class == InMemoryTimeSeriesRepository:
                repo = impl_class()

                # Create a mock time series data with required attributes
                class MockTimeSeriesData:
                    def __init__(self):
                        self.symbol = "TEST"
                        self.timestamp = datetime.now()
                        self.data = {"test": "value"}
                        self.primary_timestamp = self.timestamp

                # Test save method exists and is callable
                mock_data = MockTimeSeriesData()
                result = await repo.save(mock_data)

                # Save should return None according to interface
                assert result is None

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_query_contract_behavior(self):
        """Test query method contract behavior."""
        from core.models.storage.query_option import QueryOptions
        from datetime import timedelta

        for impl_class in self.implementations:
            if impl_class == InMemoryTimeSeriesRepository:
                repo = impl_class()

                start_time = datetime.now()
                end_time = start_time + timedelta(hours=1)

                # Test query method exists and returns list
                result = await repo.query("TEST", start_time, end_time)
                assert isinstance(result, list)

                # Test with options
                options = QueryOptions(limit=10)
                result_with_options = await repo.query("TEST", start_time, end_time, options=options)
                assert isinstance(result_with_options, list)

    @pytest.mark.contract
    def test_stream_contract_behavior(self):
        """Test stream method contract behavior."""
        from datetime import timedelta
        import inspect

        for impl_class in self.implementations:
            if impl_class == InMemoryTimeSeriesRepository:
                repo = impl_class()

                start_time = datetime.now()
                end_time = start_time + timedelta(hours=1)

                # Test stream method exists and returns AsyncIterator
                stream_method = getattr(repo, "stream")
                assert callable(stream_method)

                # Check return type annotation
                sig = inspect.signature(stream_method)
                # Should return AsyncIterator[T]
                assert "AsyncIterator" in str(sig.return_annotation) or sig.return_annotation == inspect.Signature.empty
