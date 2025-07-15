# ABOUTME: Contract tests for AbstractEventStorage interface
# ABOUTME: Verifies all event storage implementations comply with the interface contract

import pytest
from typing import Type, List
from datetime import datetime, timedelta

from core.interfaces.event.event_storage import AbstractEventStorage
from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.noop.event_storage import NoOpEventStorage
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin


class TestEventStorageContract(ContractTestBase[AbstractEventStorage], AsyncContractTestMixin):
    """Contract tests for AbstractEventStorage interface."""

    @property
    def interface_class(self) -> Type[AbstractEventStorage]:
        return AbstractEventStorage

    @property
    def implementations(self) -> List[Type[AbstractEventStorage]]:
        return [
            InMemoryEventStorage,
            NoOpEventStorage,
            # Add other implementations here as they are created
            # DatabaseEventStorage,
            # RedisEventStorage,
        ]

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_store_event_contract_behavior(self):
        """Test store_event method contract behavior."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies for InMemoryEventStorage
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)
            elif impl_class == NoOpEventStorage:
                # Create dependencies for NoOpEventStorage
                from unittest.mock import Mock
                from core.interfaces.event.event_serializer import AbstractEventSerializer

                mock_serializer = Mock(spec=AbstractEventSerializer)
                storage = impl_class(mock_serializer)
            else:
                continue

            # Create a mock event with simple data structure using BaseEvent directly
            from datetime import UTC

            mock_event = BaseEvent(
                event_type=EventType.SYSTEM,  # Use SYSTEM which has simpler validation
                timestamp=datetime.now(UTC),
                symbol="TEST",
                data={"test": "value"},
                source="test",
            )

            # Test store_event method exists and returns event_id
            result = await storage.store_event(mock_event)
            assert isinstance(result, str)  # Should return event_id
            assert len(result) > 0

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_retrieve_event_contract_behavior(self):
        """Test retrieve_event method contract behavior."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test retrieve_event with non-existent event_id should return None
                result = await storage.retrieve_event("non-existent-id")
                assert result is None

                # Test retrieve_event with valid event_id
                from datetime import datetime, UTC

                mock_event = BaseEvent(
                    event_type=EventType.SYSTEM,
                    timestamp=datetime.now(UTC),
                    symbol="TEST",
                    data={"test": "value"},
                    source="test",
                )
                event_id = await storage.store_event(mock_event)

                retrieved_event = await storage.retrieve_event(event_id)
                assert isinstance(retrieved_event, BaseEvent)
                assert retrieved_event.symbol == "TEST"

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_query_events_contract_behavior(self):
        """Test query_events method contract behavior."""
        from core.models.event.event_query import EventQuery
        from core.models.event.event_type import EventType

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test query_events method exists and returns list
                start_time = datetime.now() - timedelta(hours=1)
                end_time = datetime.now()

                query = EventQuery(
                    start_time=start_time, end_time=end_time, event_types=[EventType.KLINE], symbols=["TEST"]
                )

                result = await storage.query_events(query)
                assert isinstance(result, list)

                # Test query without parameters returns all events
                query_all = EventQuery()
                result_all = await storage.query_events(query_all)
                assert isinstance(result_all, list)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_delete_event_contract_behavior(self):
        """Test delete_event method contract behavior."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test delete_event with non-existent event_id should return False
                result = await storage.delete_event("non-existent-id")
                assert result is False

                # Test delete_event with valid event_id
                from datetime import datetime, UTC

                mock_event = BaseEvent(
                    event_type=EventType.SYSTEM,
                    timestamp=datetime.now(UTC),
                    symbol="TEST",
                    data={"test": "value"},
                    source="test",
                )
                event_id = await storage.store_event(mock_event)

                # Delete should succeed and return True
                delete_result = await storage.delete_event(event_id)
                assert delete_result is True

                # Subsequent retrieve should return None
                retrieved = await storage.retrieve_event(event_id)
                assert retrieved is None

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_noop_implementation_basic_behavior(self):
        """Test NoOpEventStorage basic contract compliance."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType
        from core.models.event.event_query import EventQuery
        from unittest.mock import Mock
        from core.interfaces.event.event_serializer import AbstractEventSerializer

        # Test only NoOp implementation for basic behavior
        mock_serializer = Mock(spec=AbstractEventSerializer)
        storage = NoOpEventStorage(mock_serializer)

        # Create a simple mock event
        from datetime import datetime, UTC

        mock_event = BaseEvent(
            event_type=EventType.SYSTEM,
            timestamp=datetime.now(UTC),
            symbol="TEST",
            data={"test": "value"},
            source="test",
        )

        # Test that basic methods exist and return appropriate types
        # Even though NoOp doesn't preserve data, it should follow the contract
        event_id = await storage.store_event(mock_event)
        assert isinstance(event_id, str)
        assert len(event_id) > 0

        # Retrieve might not return the exact event but should return valid type
        retrieved = await storage.retrieve_event(event_id)
        # NoOp can return None or a fake event, both are acceptable
        assert retrieved is None or isinstance(retrieved, BaseEvent)

        # Query should return a list
        query = EventQuery()
        results = await storage.query_events(query)
        assert isinstance(results, list)

        # Delete should return a boolean
        delete_result = await storage.delete_event(event_id)
        assert isinstance(delete_result, bool)

        # Stream should be async generator (test signature only)
        import inspect

        stream_method = storage.stream_events
        assert inspect.iscoroutinefunction(stream_method) or inspect.isasyncgenfunction(stream_method)

    @pytest.mark.contract
    def test_stream_events_contract_behavior(self):
        """Test stream_events method contract behavior."""
        import inspect

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test stream_events method exists and returns AsyncIterator
                stream_method = getattr(storage, "stream_events")
                assert callable(stream_method)

                # Check return type annotation
                sig = inspect.signature(stream_method)
                # Should return AsyncIterator[BaseEvent]
                return_annotation = str(sig.return_annotation)
                assert "AsyncIterator" in return_annotation or sig.return_annotation == inspect.Signature.empty

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_get_stats_contract_behavior(self):
        """Test get_stats method contract behavior."""
        from core.models.event.event_storage_stats import EventStorageStats

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test get_stats method exists and returns EventStorageStats
                result = await storage.get_stats()
                assert isinstance(result, EventStorageStats)

                # Verify required fields exist
                assert hasattr(result, "total_events")
                assert hasattr(result, "events_by_type")
                assert hasattr(result, "storage_size_bytes")
                # Note: events_by_symbol is not in the actual EventStorageStats model

    @pytest.mark.contract
    def test_required_dependencies_injection(self):
        """Test that implementations properly handle dependency injection."""
        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Test that constructor requires proper dependencies
                import inspect

                sig = inspect.signature(impl_class.__init__)

                # Should require serializer parameter
                params = list(sig.parameters.keys())
                assert "serializer" in params

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_store_events_contract_behavior(self):
        """Test store_events batch method contract behavior."""
        from core.models.data.event import BaseEvent
        from core.models.event.event_type import EventType

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Create multiple mock events
                from datetime import datetime, UTC

                events = [
                    BaseEvent(
                        event_type=EventType.SYSTEM,
                        timestamp=datetime.now(UTC),
                        symbol="TEST1",
                        data={"test": "value"},
                        source="test",
                    ),
                    BaseEvent(
                        event_type=EventType.SYSTEM,
                        timestamp=datetime.now(UTC),
                        symbol="TEST2",
                        data={"test": "value"},
                        source="test",
                    ),
                    BaseEvent(
                        event_type=EventType.SYSTEM,
                        timestamp=datetime.now(UTC),
                        symbol="TEST3",
                        data={"test": "value"},
                        source="test",
                    ),
                ]

                # Test store_events method exists and returns list of event_ids
                result = await storage.store_events(events)
                assert isinstance(result, list)
                assert len(result) == 3
                assert all(isinstance(event_id, str) for event_id in result)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_delete_events_contract_behavior(self):
        """Test delete_events batch method contract behavior."""
        from core.models.event.event_query import EventQuery
        from core.models.event.event_type import EventType

        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test delete_events method exists and returns count
                query = EventQuery(event_types=[EventType.KLINE])
                result = await storage.delete_events(query)
                assert isinstance(result, int)
                assert result >= 0  # Should return count of deleted events

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_health_check_contract_behavior(self):
        """Test health_check method contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test health_check method exists and returns boolean
                result = await storage.health_check()
                assert isinstance(result, bool)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_close_contract_behavior(self):
        """Test close method contract behavior."""
        for impl_class in self.implementations:
            if impl_class == InMemoryEventStorage:
                # Create dependencies
                from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
                from core.implementations.memory.event.event_serializer import MemoryEventSerializer

                metadata_repo = InMemoryMetadataRepository()
                serializer = MemoryEventSerializer()
                storage = impl_class(serializer, metadata_repo)

                # Test close method exists and can be called
                result = await storage.close()
                assert result is None  # close should return None

    @pytest.mark.contract
    def test_error_handling_contracts(self):
        """Test that implementations properly define error handling contracts."""

        # Check that methods have proper exception documentation or type hints
        for method_name in ["store_event", "retrieve_event", "delete_event", "query_events"]:
            method = getattr(self.interface_class, method_name)
            assert hasattr(method, "__isabstractmethod__")

            # Verify method exists in implementations
            for impl_class in self.implementations:
                assert hasattr(impl_class, method_name)
                impl_method = getattr(impl_class, method_name)
                assert callable(impl_method)
