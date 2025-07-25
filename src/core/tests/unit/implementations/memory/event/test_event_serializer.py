# ABOUTME: Unit tests for MemoryEventSerializer implementation
# ABOUTME: Tests serialization/deserialization of all event types with comprehensive validation

import pytest
import json
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from uuid import uuid4

from core.implementations import MemoryEventSerializer
from core.models.data.event import BaseEvent
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval, TradeSide, AssetClass
from core.models.network.enum import ConnectionStatus
from core.exceptions.base import EventSerializationError, EventDeserializationError


class TestMemoryEventSerializer:
    """Test suite for MemoryEventSerializer."""

    @pytest.fixture
    def serializer(self):
        """Create a MemoryEventSerializer instance for testing."""
        return MemoryEventSerializer()

    @pytest.fixture
    def pretty_serializer(self):
        """Create a MemoryEventSerializer with pretty printing enabled."""
        return MemoryEventSerializer(pretty_print=True)

    @pytest.fixture
    def sample_trade(self):
        """Create a sample Trade object for testing."""
        return Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.50"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )

    @pytest.fixture
    def sample_kline(self):
        """Create a sample Kline object for testing."""
        now = datetime.now(UTC)
        open_time = now.replace(second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=1)  # 1 minute later
        return Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=open_time,
            close_time=close_time,
            open_price=Decimal("45000.00"),
            high_price=Decimal("45100.00"),
            low_price=Decimal("44900.00"),
            close_price=Decimal("45050.00"),
            volume=Decimal("100.5"),
            quote_volume=Decimal("4525000.0"),
            trades_count=150,
        )

    @pytest.fixture
    def sample_trade_event(self, sample_trade):
        """Create a sample TradeEvent for testing."""
        return TradeEvent(source="test_exchange", symbol="BTC/USDT", data=sample_trade, priority=EventPriority.NORMAL)

    @pytest.fixture
    def sample_kline_event(self, sample_kline):
        """Create a sample KlineEvent for testing."""
        return KlineEvent(source="test_exchange", symbol="BTC/USDT", data=sample_kline, priority=EventPriority.NORMAL)

    @pytest.fixture
    def sample_connection_event(self):
        """Create a sample ConnectionEvent for testing."""
        return ConnectionEvent(status=ConnectionStatus.CONNECTED, source="test_exchange", symbol="BTC/USDT")

    @pytest.fixture
    def sample_error_event(self):
        """Create a sample ErrorEvent for testing."""
        return ErrorEvent(error="Test error message", error_code="TEST_ERROR", source="test_exchange")

    @pytest.mark.unit
    @pytest.mark.config
    def test_serializer_initialization(self, serializer):
        """Test serializer initialization with default settings."""
        assert not serializer._pretty_print
        assert not serializer._ensure_ascii
        assert EventType.TRADE in serializer.get_supported_event_types()
        assert EventType.KLINE in serializer.get_supported_event_types()

    @pytest.mark.unit
    def test_serializer_initialization_with_options(self):
        """Test serializer initialization with custom options."""
        serializer = MemoryEventSerializer(pretty_print=True, ensure_ascii=True)
        assert serializer._pretty_print
        assert serializer._ensure_ascii

    @pytest.mark.unit
    def test_serialize_trade_event(self, serializer, sample_trade_event):
        """Test serialization of TradeEvent."""
        serialized = serializer.serialize(sample_trade_event)

        assert isinstance(serialized, bytes)

        # Verify it's valid JSON
        json_data = json.loads(serialized.decode("utf-8"))
        assert json_data["event_type"]["__value__"] == "trade"
        assert json_data["__serializer_version__"] == "1.0"
        assert json_data["__event_class__"] == "TradeEvent"

    @pytest.mark.unit
    def test_serialize_kline_event(self, serializer, sample_kline_event):
        """Test serialization of KlineEvent."""
        serialized = serializer.serialize(sample_kline_event)

        assert isinstance(serialized, bytes)

        # Verify it's valid JSON
        json_data = json.loads(serialized.decode("utf-8"))
        assert json_data["event_type"]["__value__"] == "kline"
        assert json_data["__serializer_version__"] == "1.0"
        assert json_data["__event_class__"] == "KlineEvent"

    @pytest.mark.unit
    def test_serialize_connection_event(self, serializer, sample_connection_event):
        """Test serialization of ConnectionEvent."""
        serialized = serializer.serialize(sample_connection_event)

        assert isinstance(serialized, bytes)

        # Verify it's valid JSON
        json_data = json.loads(serialized.decode("utf-8"))
        assert json_data["event_type"]["__value__"] == "connection"
        assert json_data["__serializer_version__"] == "1.0"
        assert json_data["__event_class__"] == "ConnectionEvent"

    @pytest.mark.unit
    def test_serialize_error_event(self, serializer, sample_error_event):
        """Test serialization of ErrorEvent."""
        serialized = serializer.serialize(sample_error_event)

        assert isinstance(serialized, bytes)

        # Verify it's valid JSON
        json_data = json.loads(serialized.decode("utf-8"))
        assert json_data["event_type"]["__value__"] == "error"
        assert json_data["__serializer_version__"] == "1.0"
        assert json_data["__event_class__"] == "ErrorEvent"

    @pytest.mark.unit
    def test_deserialize_trade_event(self, serializer, sample_trade_event):
        """Test deserialization of TradeEvent."""
        serialized = serializer.serialize(sample_trade_event)
        deserialized = serializer.deserialize(serialized)

        assert isinstance(deserialized, TradeEvent)
        assert deserialized.event_type == EventType.TRADE
        assert deserialized.source == sample_trade_event.source
        assert deserialized.symbol == sample_trade_event.symbol
        assert isinstance(deserialized.data, Trade)
        assert deserialized.data.symbol == sample_trade_event.data.symbol
        assert deserialized.data.price == sample_trade_event.data.price

    @pytest.mark.unit
    def test_deserialize_kline_event(self, serializer, sample_kline_event):
        """Test deserialization of KlineEvent."""
        serialized = serializer.serialize(sample_kline_event)
        deserialized = serializer.deserialize(serialized)

        assert isinstance(deserialized, KlineEvent)
        assert deserialized.event_type == EventType.KLINE
        assert deserialized.source == sample_kline_event.source
        assert isinstance(deserialized.data, Kline)
        assert deserialized.data.symbol == sample_kline_event.data.symbol
        assert deserialized.data.open_price == sample_kline_event.data.open_price

    @pytest.mark.unit
    def test_deserialize_connection_event(self, serializer, sample_connection_event):
        """Test deserialization of ConnectionEvent."""
        serialized = serializer.serialize(sample_connection_event)
        deserialized = serializer.deserialize(serialized)

        assert isinstance(deserialized, ConnectionEvent)
        assert deserialized.event_type == EventType.CONNECTION
        assert deserialized.source == sample_connection_event.source

    @pytest.mark.unit
    def test_deserialize_error_event(self, serializer, sample_error_event):
        """Test deserialization of ErrorEvent."""
        serialized = serializer.serialize(sample_error_event)
        deserialized = serializer.deserialize(serialized)

        assert isinstance(deserialized, ErrorEvent)
        assert deserialized.event_type == EventType.ERROR
        assert deserialized.source == sample_error_event.source
        assert deserialized.priority == EventPriority.HIGH

    @pytest.mark.unit
    def test_round_trip_serialization(self, serializer, sample_trade_event):
        """Test that serialization and deserialization preserve event data."""
        original = sample_trade_event
        serialized = serializer.serialize(original)
        deserialized = serializer.deserialize(serialized)

        # Check key fields are preserved
        assert deserialized.event_id == original.event_id
        assert deserialized.event_type == original.event_type
        assert deserialized.source == original.source
        assert deserialized.symbol == original.symbol
        assert deserialized.priority == original.priority

    @pytest.mark.unit
    def test_serialize_special_types(self, serializer):
        """Test serialization of special types like Decimal, datetime, etc."""
        # Create an event with various special types
        trade = Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.12345678"),  # 8 decimal precision decimal
            quantity=Decimal("0.00000001"),  # Small decimal with proper precision
            side=TradeSide.SELL,
            timestamp=datetime.now(UTC),
            is_buyer_maker=False,
        )

        event = TradeEvent(source="test_exchange", symbol="BTC/USDT", data=trade, priority=EventPriority.HIGH)

        # Serialize and deserialize
        serialized = serializer.serialize(event)
        deserialized = serializer.deserialize(serialized)

        # Verify special types are preserved
        assert isinstance(deserialized.data.price, Decimal)
        assert deserialized.data.price == trade.price
        assert isinstance(deserialized.data.timestamp, datetime)
        assert deserialized.data.side == TradeSide.SELL
        assert deserialized.priority == EventPriority.HIGH

    @pytest.mark.unit
    def test_pretty_print_serialization(self, pretty_serializer, sample_trade_event):
        """Test pretty printing produces formatted JSON."""
        serialized = pretty_serializer.serialize(sample_trade_event)
        json_str = serialized.decode("utf-8")

        # Pretty printed JSON should contain newlines and indentation
        assert "\n" in json_str
        assert "  " in json_str  # Indentation

    @pytest.mark.unit
    def test_serialize_invalid_event(self, serializer):
        """Test serialization error handling."""

        # Create a mock event that will cause serialization issues
        class BadEvent(BaseEvent):
            def model_dump(self):
                # Return something that can't be JSON serialized
                return {"bad_data": object()}

        bad_event = BadEvent(event_type=EventType.SYSTEM, source="test", data={})

        with pytest.raises(EventSerializationError) as exc_info:
            serializer.serialize(bad_event)

        assert exc_info.value.code == "SERIALIZATION_FAILED"

    @pytest.mark.unit
    def test_deserialize_invalid_json(self, serializer):
        """Test deserialization of invalid JSON."""
        invalid_json = b'{"invalid": json data}'

        with pytest.raises(EventDeserializationError) as exc_info:
            serializer.deserialize(invalid_json)

        assert exc_info.value.code == "INVALID_JSON"

    @pytest.mark.unit
    def test_deserialize_invalid_utf8(self, serializer):
        """Test deserialization of invalid UTF-8 bytes."""
        invalid_utf8 = b"\xff\xfe\xfd"

        with pytest.raises(EventDeserializationError) as exc_info:
            serializer.deserialize(invalid_utf8)

        assert exc_info.value.code == "INVALID_ENCODING"

    @pytest.mark.unit
    def test_deserialize_missing_event_type(self, serializer):
        """Test deserialization of data missing event_type."""
        data_without_type = {"__serializer_version__": "1.0", "__event_class__": "TradeEvent", "source": "test"}
        json_bytes = json.dumps(data_without_type).encode("utf-8")

        with pytest.raises(EventDeserializationError) as exc_info:
            serializer.deserialize(json_bytes)

        assert exc_info.value.code == "MISSING_EVENT_TYPE"

    @pytest.mark.unit
    def test_deserialize_unknown_event_type(self, serializer):
        """Test deserialization of unknown event type."""
        data_with_unknown_type = {
            "__serializer_version__": "1.0",
            "__event_class__": "TradeEvent",
            "event_type": "unknown_type",
            "source": "test",
            "data": {},
        }
        json_bytes = json.dumps(data_with_unknown_type).encode("utf-8")

        with pytest.raises(EventDeserializationError) as exc_info:
            serializer.deserialize(json_bytes)

        assert exc_info.value.code == "UNKNOWN_EVENT_TYPE"

    @pytest.mark.unit
    def test_deserialize_unsupported_version(self, serializer):
        """Test deserialization of unsupported serializer version."""
        data_with_bad_version = {
            "__serializer_version__": "2.0",
            "__event_class__": "TradeEvent",
            "event_type": "trade",
            "source": "test",
            "data": {},
        }
        json_bytes = json.dumps(data_with_bad_version).encode("utf-8")

        with pytest.raises(EventDeserializationError) as exc_info:
            serializer.deserialize(json_bytes)

        assert exc_info.value.code == "UNSUPPORTED_VERSION"

    @pytest.mark.unit
    def test_deserialize_non_dict_data(self, serializer):
        """Test deserialization of non-dictionary data."""
        non_dict_data = ["not", "a", "dictionary"]
        json_bytes = json.dumps(non_dict_data).encode("utf-8")

        with pytest.raises(EventDeserializationError) as exc_info:
            serializer.deserialize(json_bytes)

        assert exc_info.value.code == "INVALID_EVENT_FORMAT"

    @pytest.mark.unit
    def test_get_supported_event_types(self, serializer):
        """Test getting supported event types."""
        supported_types = serializer.get_supported_event_types()

        assert EventType.TRADE in supported_types
        assert EventType.KLINE in supported_types
        assert EventType.CONNECTION in supported_types
        assert EventType.ERROR in supported_types

    @pytest.mark.unit
    def test_register_event_type(self, serializer):
        """Test registering a new event type."""
        # Create a custom event type (this would normally be added to EventType enum)
        custom_type = EventType.SYSTEM

        class CustomEvent(BaseEvent):
            pass

        # Register the new type
        serializer.register_event_type(custom_type, CustomEvent)

        # Verify it's now supported
        assert custom_type in serializer.get_supported_event_types()

    @pytest.mark.unit
    def test_validate_event(self, serializer, sample_trade_event):
        """Test event validation."""
        # Valid event should pass validation
        assert serializer.validate_event(sample_trade_event)

        # Create an event that might fail validation
        class ProblematicEvent(BaseEvent):
            def model_dump(self):
                return {"problematic_field": object()}

        bad_event = ProblematicEvent(event_type=EventType.SYSTEM, source="test", data={})

        # This should fail validation due to serialization issues
        assert not serializer.validate_event(bad_event)

    @pytest.mark.unit
    def test_get_serialization_stats(self, serializer):
        """Test getting serialization statistics."""
        stats = serializer.get_serialization_stats()

        assert stats["serializer_type"] == "JSON"
        assert stats["version"] == "1.0"
        assert "supported_event_types" in stats
        assert "pretty_print" in stats
        assert "ensure_ascii" in stats

    @pytest.mark.unit
    def test_serialize_complex_metadata(self, serializer):
        """Test serialization of events with complex metadata."""
        trade = Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.50"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )

        # Create event with complex metadata
        event = TradeEvent(
            source="test_exchange",
            symbol="BTC/USDT",
            data=trade,
            metadata={
                "nested_dict": {"key": "value", "number": 42},
                "list_data": [1, 2, 3],
                "decimal_value": Decimal("123.456"),
                "datetime_value": datetime.now(UTC),
                "uuid_value": uuid4(),
            },
        )

        # Serialize and deserialize
        serialized = serializer.serialize(event)
        deserialized = serializer.deserialize(serialized)

        # Verify complex metadata is preserved
        assert deserialized.metadata["nested_dict"]["key"] == "value"
        assert deserialized.metadata["list_data"] == [1, 2, 3]
        assert isinstance(deserialized.metadata["decimal_value"], Decimal)
        assert isinstance(deserialized.metadata["datetime_value"], datetime)

    @pytest.mark.unit
    def test_serialize_all_enum_types(self, serializer):
        """Test serialization of all supported enum types."""
        # Create objects with all enum types
        kline = Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.HOUR_1,  # KlineInterval enum
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("45000.00"),
            high_price=Decimal("45100.00"),
            low_price=Decimal("44900.00"),
            close_price=Decimal("45050.00"),
            volume=Decimal("100.5"),
            quote_volume=Decimal("4525000.0"),
            trades_count=150,
            asset_class=AssetClass.DIGITAL,  # AssetClass enum
        )

        event = KlineEvent(
            source="test_exchange",
            symbol="BTC/USDT",
            data=kline,
            priority=EventPriority.CRITICAL,  # EventPriority enum
        )

        # Serialize and deserialize
        serialized = serializer.serialize(event)
        deserialized = serializer.deserialize(serialized)

        # Verify all enum types are preserved
        assert deserialized.data.interval == KlineInterval.HOUR_1
        assert deserialized.data.asset_class == AssetClass.DIGITAL
        assert deserialized.priority == EventPriority.CRITICAL
