# ABOUTME: Unit tests for BaseEvent generic model with comprehensive validation testing
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for generic event base class

import pytest
from datetime import datetime, UTC, timezone, timedelta
from uuid import UUID
from pydantic import ValidationError

from core.models.data.base import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class TestBaseEvent:
    """Test suite for BaseEvent generic model validation and behavior."""

    def test_base_event_creation_with_valid_data(self):
        """Test normal case: creating BaseEvent with valid data."""
        # Arrange
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        test_data = {"message": "test event"}
        
        # Act
        event = BaseEvent[dict](
            event_type=EventType.SYSTEM,
            source="test_service",
            data=test_data,
            timestamp=timestamp
        )
        
        # Assert
        assert event.event_type == EventType.SYSTEM
        assert event.source == "test_service"
        assert event.data == test_data
        assert event.timestamp == timestamp
        assert event.priority == EventPriority.NORMAL  # Default value
        assert event.symbol is None  # Default value
        assert event.correlation_id is None  # Default value
        assert event.metadata == {}  # Default value
        assert UUID(event.event_id)  # Should be valid UUID

    def test_base_event_generic_typing(self):
        """Test generic typing with different data types."""
        # Arrange & Act - string data
        string_event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test string"
        )
        assert isinstance(string_event.data, str)
        assert string_event.data == "test string"
        
        # Arrange & Act - integer data
        int_event = BaseEvent[int](
            event_type=EventType.SYSTEM,
            source="test",
            data=42
        )
        assert isinstance(int_event.data, int)
        assert int_event.data == 42
        
        # Arrange & Act - list data
        list_event = BaseEvent[list](
            event_type=EventType.SYSTEM,
            source="test",
            data=[1, 2, 3]
        )
        assert isinstance(list_event.data, list)
        assert list_event.data == [1, 2, 3]

    def test_base_event_auto_generated_fields(self):
        """Test auto-generated fields like event_id and timestamp."""
        # Arrange & Act
        event1 = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test"
        )
        event2 = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test"
        )
        
        # Assert - event_id should be unique
        assert event1.event_id != event2.event_id
        assert UUID(event1.event_id)
        assert UUID(event2.event_id)
        
        # Assert - timestamp should be auto-generated and recent
        now = datetime.now(UTC)
        assert abs((event1.timestamp - now).total_seconds()) < 1
        assert abs((event2.timestamp - now).total_seconds()) < 1

    def test_base_event_timestamp_validation_timezone_aware(self):
        """Test timestamp validation and timezone conversion."""
        # Arrange
        naive_time = datetime(2024, 1, 1, 12, 0, 0)  # No timezone
        utc_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        est_time = datetime(2024, 1, 1, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
        
        # Act & Assert - naive datetime converted to UTC
        event1 = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            timestamp=naive_time
        )
        assert event1.timestamp.tzinfo == UTC
        assert event1.timestamp == naive_time.replace(tzinfo=UTC)
        
        # Act & Assert - UTC datetime preserved
        event2 = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            timestamp=utc_time
        )
        assert event2.timestamp == utc_time
        
        # Act & Assert - EST datetime converted to UTC
        event3 = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            timestamp=est_time
        )
        assert event3.timestamp.tzinfo == UTC
        assert event3.timestamp == est_time.astimezone(UTC)

    def test_base_event_timestamp_validation_errors(self):
        """Test exception cases for timestamp validation."""
        # Note: Pydantic automatically converts many string formats to datetime
        # So we test with a clearly invalid format
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                event_type=EventType.SYSTEM,
                source="test",
                data="test",
                timestamp="not-a-date"  # Invalid datetime string
            )
        # Pydantic provides its own datetime validation error message
        assert "datetime" in str(exc_info.value).lower() or "input should be a valid datetime" in str(exc_info.value).lower()

    def test_base_event_symbol_validation_normalization(self):
        """Test symbol validation and normalization."""
        # Act & Assert - symbol normalization
        event = BaseEvent[str](
            event_type=EventType.TRADE,
            source="test",
            data="test",
            symbol="  btcusdt  "
        )
        assert event.symbol == "BTCUSDT"
        
        # Act & Assert - None symbol (valid)
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            symbol=None
        )
        assert event.symbol is None

    def test_base_event_symbol_validation_errors(self):
        """Test exception cases for symbol validation."""
        # Act & Assert - empty symbol
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                event_type=EventType.TRADE,
                source="test",
                data="test",
                symbol=""
            )
        assert "Symbol cannot be empty" in str(exc_info.value)
        
        # Act & Assert - whitespace only symbol
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                event_type=EventType.TRADE,
                source="test",
                data="test",
                symbol="   "
            )
        assert "Symbol cannot be empty" in str(exc_info.value)
        
        # Act & Assert - non-string symbol
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                event_type=EventType.TRADE,
                source="test",
                data="test",
                symbol=123
            )
        # Pydantic provides its own string validation error message
        assert "string" in str(exc_info.value).lower()

    def test_base_event_priority_validation(self):
        """Test priority validation with different EventPriority values."""
        # Act & Assert - default priority
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test"
        )
        assert event.priority == EventPriority.NORMAL
        
        # Act & Assert - custom priority
        event = BaseEvent[str](
            event_type=EventType.ERROR,
            source="test",
            data="test",
            priority=EventPriority.CRITICAL
        )
        assert event.priority == EventPriority.CRITICAL
        
        # Act & Assert - custom priority value
        custom_priority = EventPriority.custom(150)
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            priority=custom_priority
        )
        assert event.priority == custom_priority

    def test_base_event_required_fields_validation(self):
        """Test validation of required fields."""
        # Act & Assert - missing event_type
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                source="test",
                data="test"
            )
        assert "Field required" in str(exc_info.value)
        
        # Act & Assert - missing source
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                event_type=EventType.SYSTEM,
                data="test"
            )
        assert "Field required" in str(exc_info.value)
        
        # Act & Assert - missing data
        with pytest.raises(ValidationError) as exc_info:
            BaseEvent[str](
                event_type=EventType.SYSTEM,
                source="test"
            )
        assert "Field required" in str(exc_info.value)

    def test_base_event_optional_fields_with_values(self):
        """Test optional fields when provided with values."""
        # Arrange
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        metadata = {"key": "value", "number": 42}
        
        # Act
        event = BaseEvent[dict](
            event_type=EventType.TRADE,
            source="exchange",
            data={"price": 50000},
            symbol="BTCUSDT",
            priority=EventPriority.HIGH,
            correlation_id="corr-123",
            metadata=metadata,
            timestamp=timestamp
        )
        
        # Assert
        assert event.symbol == "BTCUSDT"
        assert event.priority == EventPriority.HIGH
        assert event.correlation_id == "corr-123"
        assert event.metadata == metadata
        assert event.timestamp == timestamp

    def test_base_event_to_dict_method(self):
        """Test to_dict method for serialization."""
        # Arrange
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        event = BaseEvent[dict](
            event_type=EventType.TRADE,
            source="test",
            data={"price": 50000},
            timestamp=timestamp
        )
        
        # Act
        event_dict = event.to_dict()
        
        # Assert
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == EventType.TRADE
        assert event_dict["source"] == "test"
        assert event_dict["data"] == {"price": 50000}
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert "event_id" in event_dict

    def test_base_event_string_representation(self):
        """Test string representation of BaseEvent."""
        # Arrange
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test_service",
            data="test"
        )
        
        # Act
        event_str = str(event)
        
        # Assert
        assert "BaseEvent" in event_str
        assert event.event_id in event_str
        assert "system" in event_str  # EventType.SYSTEM value
        assert "test_service" in event_str

    def test_base_event_model_serialization(self):
        """Test Pydantic model serialization."""
        # Arrange
        event = BaseEvent[dict](
            event_type=EventType.KLINE,
            source="binance",
            data={"symbol": "BTCUSDT", "price": 50000}
        )
        
        # Act
        serialized = event.model_dump()
        
        # Assert
        assert isinstance(serialized, dict)
        assert serialized["event_type"] == "kline"  # Enum value
        assert serialized["source"] == "binance"
        assert serialized["data"] == {"symbol": "BTCUSDT", "price": 50000}
        assert "event_id" in serialized
        assert "timestamp" in serialized

    def test_base_event_boundary_cases(self):
        """Test boundary cases and edge conditions."""
        # Act & Assert - very long source string
        long_source = "a" * 1000
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source=long_source,
            data="test"
        )
        assert event.source == long_source
        
        # Act & Assert - complex nested data
        complex_data = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2", {"nested": True}]
                }
            },
            "array": [1, 2, 3, {"key": "value"}],
            "null_value": None
        }
        event = BaseEvent[dict](
            event_type=EventType.SYSTEM,
            source="test",
            data=complex_data
        )
        assert event.data == complex_data

    def test_base_event_correlation_id_validation(self):
        """Test correlation_id field validation."""
        # Act & Assert - valid correlation_id
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            correlation_id="correlation-123-abc"
        )
        assert event.correlation_id == "correlation-123-abc"
        
        # Act & Assert - None correlation_id (valid)
        event = BaseEvent[str](
            event_type=EventType.SYSTEM,
            source="test",
            data="test",
            correlation_id=None
        )
        assert event.correlation_id is None