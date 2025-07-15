# ABOUTME: Integration tests for event serialization and deserialization (E3.1-E3.3)
# ABOUTME: Tests event serialization, version compatibility, and EventSerializer + EventBus integration

import pytest
import asyncio
import json
from typing import Dict, Any

from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.interfaces.event.event_bus import AbstractEventBus
from core.models.data.event import BaseEvent
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.exceptions.base import EventSerializationError, EventDeserializationError


@pytest.mark.integration
class TestEventSerializationIntegration:
    """
    Integration tests for event serialization and deserialization.

    Tests event serialization, version compatibility, and EventSerializer + EventBus integration.
    This covers tasks E3.1-E3.3 from the integration test plan.
    """

    @pytest.mark.asyncio
    async def test_round_trip_serialization_all_event_types(
        self,
        event_serializer: AbstractEventSerializer,
        sample_trade_event: TradeEvent,
        sample_kline_event: KlineEvent,
        sample_connection_event: ConnectionEvent,
        sample_error_event: ErrorEvent,
    ):
        """Test round-trip serialization for all event types (E3.1)."""
        events = [sample_trade_event, sample_kline_event, sample_connection_event, sample_error_event]

        for original_event in events:
            # Act: Serialize and deserialize
            serialized_data = event_serializer.serialize(original_event)
            deserialized_event = event_serializer.deserialize(serialized_data)

            # Assert: Events are identical after round-trip
            assert deserialized_event.event_id == original_event.event_id
            assert deserialized_event.event_type == original_event.event_type
            assert deserialized_event.source == original_event.source
            assert deserialized_event.symbol == original_event.symbol
            assert deserialized_event.priority == original_event.priority
            assert deserialized_event.correlation_id == original_event.correlation_id
            assert deserialized_event.metadata == original_event.metadata

            # Verify timestamp precision (within 1 second due to serialization)
            time_diff = abs((deserialized_event.timestamp - original_event.timestamp).total_seconds())
            assert time_diff < 1.0

            # Type-specific assertions
            if isinstance(original_event, TradeEvent):
                assert isinstance(deserialized_event, TradeEvent)
                assert deserialized_event.data.symbol == original_event.data.symbol
                assert deserialized_event.data.price == original_event.data.price
                assert deserialized_event.data.quantity == original_event.data.quantity
                assert deserialized_event.data.side == original_event.data.side

            elif isinstance(original_event, KlineEvent):
                assert isinstance(deserialized_event, KlineEvent)
                assert deserialized_event.data.symbol == original_event.data.symbol
                assert deserialized_event.data.interval == original_event.data.interval
                assert deserialized_event.data.open_price == original_event.data.open_price
                assert deserialized_event.data.close_price == original_event.data.close_price

            elif isinstance(original_event, ConnectionEvent):
                assert isinstance(deserialized_event, ConnectionEvent)
                assert deserialized_event.status == original_event.status
                assert deserialized_event.data == original_event.data

            elif isinstance(original_event, ErrorEvent):
                assert isinstance(deserialized_event, ErrorEvent)
                assert deserialized_event.data == original_event.data

    @pytest.mark.asyncio
    async def test_serialization_with_custom_priority(self, event_serializer: AbstractEventSerializer, event_factory):
        """Test serialization preserves custom priority values."""
        # Arrange: Create event with custom priority
        custom_priority = EventPriority.custom(42)
        event = event_factory(EventType.TRADE, priority=custom_priority)

        # Act: Serialize and deserialize
        serialized_data = event_serializer.serialize(event)
        deserialized_event = event_serializer.deserialize(serialized_data)

        # Assert: Custom priority is preserved
        assert deserialized_event.priority == custom_priority
        assert deserialized_event.priority.value == 42

    @pytest.mark.asyncio
    async def test_serialization_with_metadata(self, event_serializer: AbstractEventSerializer, event_factory):
        """Test serialization preserves complex metadata."""
        # Arrange: Create event with complex metadata
        complex_metadata = {
            "string_field": "test_value",
            "number_field": 123,
            "boolean_field": True,
            "nested_object": {"inner_field": "inner_value", "inner_number": 456},
            "list_field": [1, 2, 3, "four"],
        }

        event = event_factory(EventType.TRADE, metadata=complex_metadata)

        # Act: Serialize and deserialize
        serialized_data = event_serializer.serialize(event)
        deserialized_event = event_serializer.deserialize(serialized_data)

        # Assert: Complex metadata is preserved
        assert deserialized_event.metadata == complex_metadata

    @pytest.mark.asyncio
    async def test_serialization_error_handling(self, event_serializer: AbstractEventSerializer):
        """Test serialization error handling for invalid data."""

        # Arrange: Create an event with non-serializable data
        class NonSerializableClass:
            def __init__(self):
                self.data = "test"

        # Create a mock event with non-serializable data
        event = BaseEvent(
            event_type=EventType.SYSTEM,
            source="test",
            data=NonSerializableClass(),  # This should cause serialization to fail
        )

        # Act & Assert: Serialization should raise appropriate error
        with pytest.raises(EventSerializationError):
            event_serializer.serialize(event)

    @pytest.mark.asyncio
    async def test_deserialization_error_handling(self, event_serializer: AbstractEventSerializer):
        """Test deserialization error handling for invalid data."""
        # Test cases for invalid data
        invalid_data_cases = [
            b"invalid json",  # Invalid JSON
            b'{"invalid": "structure"}',  # Valid JSON but invalid event structure
            b'{"event_type": "INVALID_TYPE"}',  # Invalid event type
            b"",  # Empty data
        ]

        for invalid_data in invalid_data_cases:
            # Act & Assert: Deserialization should raise appropriate error
            with pytest.raises(EventDeserializationError):
                event_serializer.deserialize(invalid_data)

    @pytest.mark.asyncio
    async def test_version_compatibility_simulation(
        self, event_serializer: AbstractEventSerializer, sample_trade_event: TradeEvent
    ):
        """Test version compatibility by simulating older event formats (E3.2)."""
        # Arrange: Serialize current event
        current_serialized = event_serializer.serialize(sample_trade_event)
        current_data = json.loads(current_serialized.decode("utf-8"))

        # Simulate older version by removing some fields
        older_version_data = current_data.copy()
        older_version_data.pop("correlation_id", None)  # Remove newer field
        older_version_data.pop("metadata", None)  # Remove newer field

        # Act: Try to deserialize older format
        older_serialized = json.dumps(older_version_data).encode("utf-8")
        deserialized_event = event_serializer.deserialize(older_serialized)

        # Assert: Core fields are preserved, missing fields have defaults
        assert deserialized_event.event_type == sample_trade_event.event_type
        assert deserialized_event.source == sample_trade_event.source
        assert deserialized_event.symbol == sample_trade_event.symbol
        assert deserialized_event.correlation_id is None  # Default value
        assert deserialized_event.metadata == {}  # Default value

    @pytest.mark.asyncio
    async def test_forward_compatibility_simulation(
        self, event_serializer: AbstractEventSerializer, sample_trade_event: TradeEvent
    ):
        """Test forward compatibility by simulating newer event formats."""
        # Arrange: Serialize current event and add future fields
        current_serialized = event_serializer.serialize(sample_trade_event)
        current_data = json.loads(current_serialized.decode("utf-8"))

        # Simulate newer version by adding unknown fields
        future_version_data = current_data.copy()
        future_version_data["future_field"] = "future_value"
        future_version_data["another_future_field"] = {"complex": "data"}

        # Act: Try to deserialize future format
        future_serialized = json.dumps(future_version_data).encode("utf-8")
        deserialized_event = event_serializer.deserialize(future_serialized)

        # Assert: Core fields are preserved, unknown fields are ignored
        assert deserialized_event.event_type == sample_trade_event.event_type
        assert deserialized_event.source == sample_trade_event.source
        assert deserialized_event.symbol == sample_trade_event.symbol
        # Unknown fields should be gracefully ignored

    @pytest.mark.asyncio
    async def test_serializer_event_bus_integration(
        self, event_bus: AbstractEventBus, event_serializer: AbstractEventSerializer, sample_trade_event: TradeEvent
    ):
        """Test EventSerializer integration with EventBus (E3.3)."""
        # Arrange: Track serialized events
        serialized_events = []

        def serializing_handler(event: BaseEvent):
            # Simulate serialization during event handling
            serialized_data = event_serializer.serialize(event)
            deserialized_event = event_serializer.deserialize(serialized_data)
            serialized_events.append(deserialized_event)

        subscription_id = event_bus.subscribe(EventType.TRADE, serializing_handler)

        # Act: Publish event through bus
        await event_bus.publish(sample_trade_event)
        await asyncio.sleep(0.1)

        # Assert: Event was properly serialized and deserialized
        assert len(serialized_events) == 1
        serialized_event = serialized_events[0]

        assert serialized_event.event_id == sample_trade_event.event_id
        assert serialized_event.event_type == sample_trade_event.event_type
        assert serialized_event.source == sample_trade_event.source

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_batch_serialization_performance(
        self,
        event_serializer: AbstractEventSerializer,
        event_factory,
        performance_monitor,
        event_test_config: Dict[str, Any],
    ):
        """Test serialization performance with batch operations."""
        # Arrange: Create batch of events
        events = [event_factory(EventType.TRADE) for _ in range(100)]

        # Act: Measure serialization performance
        performance_monitor.start_timer("batch_serialization")

        serialized_data = []
        for event in events:
            data = event_serializer.serialize(event)
            serialized_data.append(data)

        serialization_duration = performance_monitor.end_timer("batch_serialization")

        # Measure deserialization performance
        performance_monitor.start_timer("batch_deserialization")

        deserialized_events = []
        for data in serialized_data:
            event = event_serializer.deserialize(data)
            deserialized_events.append(event)

        deserialization_duration = performance_monitor.end_timer("batch_deserialization")

        # Assert: Performance meets requirements
        avg_serialization_ms = (serialization_duration * 1000) / len(events)
        avg_deserialization_ms = (deserialization_duration * 1000) / len(events)
        threshold_ms = event_test_config["performance_thresholds"]["serialization_ms"]

        assert avg_serialization_ms < threshold_ms, f"Serialization too slow: {avg_serialization_ms:.2f}ms"
        assert avg_deserialization_ms < threshold_ms, f"Deserialization too slow: {avg_deserialization_ms:.2f}ms"
        assert len(deserialized_events) == len(events)

    @pytest.mark.asyncio
    async def test_serializer_validation_method(
        self, event_serializer: AbstractEventSerializer, sample_trade_event: TradeEvent, event_factory
    ):
        """Test the serializer's validate_event method."""
        # Test valid event
        assert event_serializer.validate_event(sample_trade_event) is True

        # Test various event types
        for event_type in [EventType.TRADE, EventType.KLINE, EventType.CONNECTION, EventType.ERROR]:
            event = event_factory(event_type)
            assert event_serializer.validate_event(event) is True

    @pytest.mark.asyncio
    async def test_serializer_configuration_and_stats(self, event_serializer: AbstractEventSerializer):
        """Test serializer configuration and statistics methods."""
        # Test supported event types
        supported_types = event_serializer.get_supported_event_types()
        assert EventType.TRADE in supported_types
        assert EventType.KLINE in supported_types
        assert EventType.CONNECTION in supported_types
        assert EventType.ERROR in supported_types

        # Test serialization stats
        stats = event_serializer.get_serialization_stats()
        assert "serializer_type" in stats
        assert "supported_event_types" in stats
        assert "version" in stats
        assert stats["serializer_type"] == "JSON"
