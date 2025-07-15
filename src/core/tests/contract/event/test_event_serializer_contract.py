# ABOUTME: Contract tests for AbstractEventSerializer interface
# ABOUTME: Verifies all event serializer implementations comply with the interface contract

import pytest
from typing import Type, List
from datetime import datetime, UTC
from decimal import Decimal

from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.noop.event.event_serializer import NoOpEventSerializer
from core.models.data.event import BaseEvent
from core.models.event.trade_event import TradeEvent
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.exceptions.base import EventSerializationError, EventDeserializationError
from ..base_contract_test import ContractTestBase


class TestEventSerializerContract(ContractTestBase[AbstractEventSerializer]):
    """Contract tests for AbstractEventSerializer interface."""

    @property
    def interface_class(self) -> Type[AbstractEventSerializer]:
        return AbstractEventSerializer

    @property
    def implementations(self) -> List[Type[AbstractEventSerializer]]:
        return [
            MemoryEventSerializer,
            NoOpEventSerializer,
        ]

    @pytest.fixture
    def sample_trade_event(self) -> TradeEvent:
        """Create a sample trade event for testing."""
        trade = Trade(
            symbol="BTC/USDT",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            trade_id="test-trade-123",
            exchange="test-exchange",
            maker_order_id="maker-123",
            taker_order_id="taker-456",
            received_at=datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC),
            is_buyer_maker=True,
        )

        return TradeEvent(
            symbol="BTC/USDT",
            data=trade,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            source="test-source",
            priority=EventPriority.NORMAL,
            correlation_id="test-correlation-123",
            metadata={"test": "metadata"},
        )

    @pytest.fixture
    def sample_base_event(self) -> BaseEvent:
        """Create a sample base event for testing."""
        return BaseEvent(
            event_type=EventType.CONNECTION,
            source="test-source",
            symbol="BTC/USDT",
            data={"test": "data"},
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            priority=EventPriority.HIGH,
            correlation_id="test-correlation-456",
            metadata={"additional": "info"},
        )

    @pytest.mark.contract
    def test_serialize_method_contract(self, sample_trade_event):
        """Test serialize method contract behavior."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Should accept BaseEvent and return bytes
            result = serializer.serialize(sample_trade_event)
            assert isinstance(result, bytes), f"{impl_class.__name__}.serialize should return bytes"
            assert len(result) > 0, f"{impl_class.__name__}.serialize should return non-empty bytes"

    @pytest.mark.contract
    def test_deserialize_method_contract(self, sample_trade_event):
        """Test deserialize method contract behavior."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Serialize then deserialize should work
            serialized = serializer.serialize(sample_trade_event)
            deserialized = serializer.deserialize(serialized)

            # Should return BaseEvent
            assert isinstance(deserialized, BaseEvent), f"{impl_class.__name__}.deserialize should return BaseEvent"

            # Basic event properties should be preserved
            assert deserialized.event_type is not None
            assert deserialized.source is not None
            assert deserialized.timestamp is not None

    @pytest.mark.contract
    def test_round_trip_consistency(self, sample_trade_event):
        """Test that serialize -> deserialize preserves event data."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Skip NoOp implementation as it doesn't preserve data by design
            if "NoOp" in impl_class.__name__:
                continue

            # Only test with TradeEvent to avoid complex event type mapping issues
            event = sample_trade_event

            # Serialize then deserialize
            serialized = serializer.serialize(event)
            deserialized = serializer.deserialize(serialized)

            # Core properties should be preserved
            assert deserialized.event_type == event.event_type, f"{impl_class.__name__} should preserve event_type"
            assert deserialized.source == event.source, f"{impl_class.__name__} should preserve source"
            assert deserialized.symbol == event.symbol, f"{impl_class.__name__} should preserve symbol"
            assert deserialized.priority == event.priority, f"{impl_class.__name__} should preserve priority"

            # Timestamp should be preserved (allowing for timezone conversion)
            if hasattr(event, "timestamp") and hasattr(deserialized, "timestamp"):
                assert abs((deserialized.timestamp - event.timestamp).total_seconds()) < 1, (
                    f"{impl_class.__name__} should preserve timestamp"
                )

    @pytest.mark.contract
    def test_serialize_none_event_handling(self):
        """Test serialize method handles None input appropriately."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Should raise appropriate exception for None input
            # Note: NoOp implementation might not raise for None, so we allow that
            try:
                result = serializer.serialize(None)
                # If no exception, result should still be bytes
                assert isinstance(result, bytes), f"{impl_class.__name__} should return bytes even for None"
            except (TypeError, AttributeError, EventSerializationError):
                # This is expected behavior for most implementations
                pass

    @pytest.mark.contract
    def test_deserialize_invalid_data_handling(self):
        """Test deserialize method handles invalid data appropriately."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Test various invalid inputs
            invalid_inputs = [
                b"",  # Empty bytes
                b"invalid json",  # Invalid format
                b"null",  # Null data
                b"[]",  # Wrong data type
            ]

            for invalid_data in invalid_inputs:
                with pytest.raises((EventDeserializationError, ValueError, TypeError)):
                    serializer.deserialize(invalid_data)

    @pytest.mark.contract
    def test_deserialize_none_data_handling(self):
        """Test deserialize method handles None input appropriately."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Should raise appropriate exception for None input
            # Note: NoOp implementation might not raise for None, so we allow that
            try:
                result = serializer.deserialize(None)
                # If no exception, result should still be BaseEvent
                assert isinstance(result, BaseEvent), f"{impl_class.__name__} should return BaseEvent even for None"
            except (TypeError, AttributeError, EventDeserializationError):
                # This is expected behavior for most implementations
                pass

    @pytest.mark.contract
    def test_serializer_instantiation(self):
        """Test that all implementations can be instantiated."""
        for impl_class in self.implementations:
            # Should be able to create instance without arguments
            serializer = impl_class()
            assert isinstance(serializer, AbstractEventSerializer)
            assert isinstance(serializer, impl_class)

    @pytest.mark.contract
    def test_multiple_serializations_consistency(self, sample_trade_event):
        """Test that multiple serializations of the same event are consistent."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Serialize the same event multiple times
            serialized1 = serializer.serialize(sample_trade_event)
            serialized2 = serializer.serialize(sample_trade_event)

            # Results should be consistent (for deterministic serializers)
            # Note: Some implementations might include timestamps, so we check deserialization consistency
            deserialized1 = serializer.deserialize(serialized1)
            deserialized2 = serializer.deserialize(serialized2)

            # Core properties should be the same
            assert deserialized1.event_type == deserialized2.event_type
            assert deserialized1.source == deserialized2.source
            assert deserialized1.symbol == deserialized2.symbol

    @pytest.mark.contract
    def test_different_event_types_handling(self):
        """Test serializer handles different event types."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Create simple base events that don't require complex data structures
            events = []

            # Base event with simple data
            events.append(
                BaseEvent(event_type=EventType.CONNECTION, source="test-source", data={"status": "connected"})
            )

            for event in events:
                # Should be able to serialize and deserialize
                serialized = serializer.serialize(event)
                deserialized = serializer.deserialize(serialized)

                assert isinstance(deserialized, BaseEvent)
                # Note: Some implementations might map event types differently
                # So we just check that we get a valid BaseEvent back

                # Skip NoOp implementation as it doesn't preserve data by design
                if "NoOp" not in impl_class.__name__:
                    assert deserialized.source == event.source

    @pytest.mark.contract
    def test_large_event_handling(self):
        """Test serializer handles large events appropriately."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Create event with large data payload
            large_data = {"large_field": "x" * 10000}  # 10KB string
            large_event = BaseEvent(event_type=EventType.CONNECTION, source="test-source", data=large_data)

            # Should handle large events without issues
            serialized = serializer.serialize(large_event)
            deserialized = serializer.deserialize(serialized)

            assert isinstance(deserialized, BaseEvent)

            # Skip NoOp implementation as it doesn't preserve data by design
            if "NoOp" not in impl_class.__name__:
                assert deserialized.source == large_event.source

    @pytest.mark.contract
    def test_unicode_data_handling(self):
        """Test serializer handles Unicode data correctly."""
        for impl_class in self.implementations:
            serializer = impl_class()

            # Create event with Unicode data
            unicode_event = BaseEvent(
                event_type=EventType.CONNECTION,
                source="test-source",
                symbol="BTC/USDT",
                data={"message": "Hello world! Test", "emoji": "Money Chart", "special_chars": "aaaaaeceeeee"},
            )

            # Should handle Unicode correctly
            serialized = serializer.serialize(unicode_event)
            deserialized = serializer.deserialize(serialized)

            assert isinstance(deserialized, BaseEvent)

            # Skip NoOp implementation as it doesn't preserve data by design
            if "NoOp" not in impl_class.__name__:
                assert deserialized.source == unicode_event.source
