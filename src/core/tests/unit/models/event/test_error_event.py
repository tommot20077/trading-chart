# ABOUTME: Unit tests for ErrorEvent model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for ErrorEvent

import pytest
from datetime import datetime, UTC

from core.models.event.error_event import ErrorEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


class TestErrorEvent:
    """
    Comprehensive unit tests for ErrorEvent model.

    Tests cover:
    - Normal case: Valid ErrorEvent creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    @pytest.mark.unit
    @pytest.mark.external
    def test_error_event_creation_normal_case(self):
        """
        Test normal ErrorEvent creation with error message.

        Verifies:
        - Event type is automatically set to ERROR
        - Priority is automatically set to HIGH
        - Error message is properly stored in data
        - All BaseEvent fields are properly initialized
        """
        # Act
        event = ErrorEvent(error="Database connection failed", source="database_client", symbol="BTCUSDT")

        # Assert
        assert event.event_type == EventType.ERROR
        assert event.priority == EventPriority.HIGH
        assert event.source == "database_client"
        assert event.symbol == "BTCUSDT"
        assert event.data["error"] == "Database connection failed"
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == UTC
        assert len(event.event_id) > 0

    @pytest.mark.unit
    @pytest.mark.external
    def test_error_event_with_error_code(self):
        """
        Test ErrorEvent creation with both error message and error code.

        Verifies:
        - Both error message and error code are stored
        - Event structure remains correct
        """
        # Act
        event = ErrorEvent(error="Invalid API key", error_code="AUTH_001", source="api_client")

        # Assert
        assert event.data["error"] == "Invalid API key"
        assert event.data["error_code"] == "AUTH_001"
        assert event.event_type == EventType.ERROR
        assert event.priority == EventPriority.HIGH

    @pytest.mark.unit
    def test_error_event_without_error_code(self):
        """
        Test ErrorEvent creation without error code.

        Verifies:
        - Error code is optional
        - Event works correctly without error code
        - Only error message is stored when error code is None
        """
        # Act
        event = ErrorEvent(error="Network timeout", source="network_client")

        # Assert
        assert event.data["error"] == "Network timeout"
        assert "error_code" not in event.data
        assert event.event_type == EventType.ERROR

    @pytest.mark.unit
    def test_error_event_forced_event_type(self):
        """
        Test that event_type is always forced to ERROR regardless of input.

        Verifies:
        - Even if different event_type is provided, it's overridden to ERROR
        - Event type field is frozen and cannot be changed
        """
        # Act - Try to set different event_type
        event = ErrorEvent(
            error="Test error",
            source="test_client",
            event_type=EventType.TRADE,  # This should be ignored
        )

        # Assert
        assert event.event_type == EventType.ERROR  # Should be ERROR, not TRADE

    @pytest.mark.unit
    def test_error_event_forced_high_priority(self):
        """
        Test that priority is automatically set to HIGH unless overridden.

        Verifies:
        - Default priority is HIGH for error events
        - Priority can be overridden if explicitly provided
        """
        # Test default HIGH priority
        event1 = ErrorEvent(error="Default priority error", source="test_client")
        assert event1.priority == EventPriority.HIGH

        # Test custom priority override
        event2 = ErrorEvent(
            error="Custom priority error", source="test_client", priority=EventPriority(EventPriority.CRITICAL)
        )
        assert event2.priority == EventPriority.CRITICAL

    @pytest.mark.unit
    def test_error_event_with_additional_data(self):
        """
        Test ErrorEvent with additional data fields.

        Verifies:
        - Additional data is merged with error information
        - Error message and code are preserved
        - Data structure is correct
        """
        # Arrange
        additional_data = {"stack_trace": "line 1\nline 2\nline 3", "user_id": "user_123", "request_id": "req_456"}

        # Act
        event = ErrorEvent(error="Processing failed", error_code="PROC_001", source="processor", data=additional_data)

        # Assert
        assert event.data["error"] == "Processing failed"
        assert event.data["error_code"] == "PROC_001"
        assert event.data["stack_trace"] == "line 1\nline 2\nline 3"
        assert event.data["user_id"] == "user_123"
        assert event.data["request_id"] == "req_456"

    @pytest.mark.unit
    def test_error_event_with_metadata_and_correlation_id(self):
        """
        Test ErrorEvent with additional metadata and correlation ID.

        Verifies:
        - Metadata is properly stored
        - Correlation ID is correctly set
        - All other fields remain valid
        """
        # Arrange
        metadata = {"service": "trading_engine", "version": "1.2.3"}
        correlation_id = "error_trace_001"

        # Act
        event = ErrorEvent(
            error="Order validation failed",
            error_code="ORD_VAL_001",
            source="order_validator",
            metadata=metadata,
            correlation_id=correlation_id,
        )

        # Assert
        assert event.metadata == metadata
        assert event.correlation_id == correlation_id
        assert event.event_type == EventType.ERROR

    @pytest.mark.unit
    def test_error_event_to_dict_method(self):
        """
        Test ErrorEvent to_dict method.

        Verifies:
        - Dictionary contains all expected fields
        - Timestamp is properly formatted as ISO string
        - Data structure is correct
        """
        # Act
        event = ErrorEvent(error="Serialization test error", error_code="SER_001", source="serializer")

        event_dict = event.to_dict()

        # Assert
        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert "source" in event_dict
        assert "data" in event_dict
        assert "priority" in event_dict
        assert event_dict["event_type"] == EventType.ERROR
        assert isinstance(event_dict["timestamp"], str)  # Should be ISO format
        assert event_dict["data"]["error"] == "Serialization test error"
        assert event_dict["data"]["error_code"] == "SER_001"

    @pytest.mark.unit
    def test_error_event_str_representation(self):
        """
        Test ErrorEvent string representation.

        Verifies:
        - String format includes class name, ID, type, and source
        - Format is consistent and readable
        """
        # Act
        event = ErrorEvent(error="String representation test", source="test_client")

        str_repr = str(event)

        # Assert
        assert "ErrorEvent" in str_repr
        assert event.event_id in str_repr
        assert "error" in str_repr
        assert "test_client" in str_repr

    @pytest.mark.unit
    def test_error_event_missing_error_message(self):
        """
        Test ErrorEvent creation with missing error message.

        Verifies:
        - Proper validation error when error message is not provided
        - Error message is descriptive
        """
        # Act & Assert
        with pytest.raises(TypeError):  # Missing required positional argument
            ErrorEvent(
                source="test_client"
                # Missing error parameter
            )

    @pytest.mark.unit
    def test_error_event_empty_error_message(self):
        """
        Test ErrorEvent with empty error message.

        Verifies:
        - Empty string error message is allowed (might be valid in some cases)
        - Event structure remains valid
        """
        # Act
        event = ErrorEvent(
            error="",  # Empty string
            source="test_client",
        )

        # Assert
        assert event.data["error"] == ""
        assert event.event_type == EventType.ERROR

    @pytest.mark.unit
    def test_error_event_missing_required_fields(self):
        """
        Test ErrorEvent creation with missing required fields.

        Verifies:
        - Proper validation errors for missing source
        """
        # Test missing source
        with pytest.raises(Exception):  # Pydantic validation error
            ErrorEvent(
                error="Test error"
                # Missing source
            )

    @pytest.mark.unit
    def test_error_event_symbol_validation_inheritance(self):
        """
        Test that ErrorEvent inherits symbol validation from BaseEvent.

        Verifies:
        - Symbol is normalized (uppercase, stripped)
        - Invalid symbols raise appropriate errors
        - None symbol is allowed
        """
        # Test with valid symbol
        event = ErrorEvent(
            error="Symbol validation test",
            source="test_client",
            symbol="  btcusdt  ",  # With whitespace
        )

        assert event.symbol == "BTCUSDT"  # Should be normalized

        # Test with None symbol (should be allowed)
        event_no_symbol = ErrorEvent(error="No symbol test", source="test_client")

        assert event_no_symbol.symbol is None

        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            ErrorEvent(
                error="Empty symbol test",
                source="test_client",
                symbol="   ",  # Only whitespace
            )

    @pytest.mark.unit
    def test_error_event_data_override_behavior(self):
        """
        Test ErrorEvent data override behavior.

        Verifies:
        - Error message is always set in data regardless of input
        - Error code is set when provided
        - Additional data fields are preserved
        - Error fields cannot be overridden in data
        """
        # Arrange
        conflicting_data = {
            "error": "wrong_error",  # This should be overridden
            "error_code": "wrong_code",  # This should be overridden
            "additional_field": "preserved",
        }

        # Act
        event = ErrorEvent(
            error="Correct error message", error_code="CORRECT_CODE", source="test_client", data=conflicting_data
        )

        # Assert
        assert event.data["error"] == "Correct error message"  # Should be overridden
        assert event.data["error_code"] == "CORRECT_CODE"  # Should be overridden
        assert event.data["additional_field"] == "preserved"  # Should be preserved

    @pytest.mark.unit
    def test_error_event_boundary_values(self):
        """
        Test ErrorEvent with boundary values.

        Verifies:
        - Handles very long error messages
        - Handles special characters in error messages
        - Handles various error code formats
        """
        # Test very long error message
        long_error = "A" * 10000  # Very long error message
        event1 = ErrorEvent(error=long_error, source="test_client")
        assert event1.data["error"] == long_error

        # Test special characters
        special_error = "Error with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        event2 = ErrorEvent(error=special_error, error_code="SPECIAL_001", source="test_client")
        assert event2.data["error"] == special_error
        assert event2.data["error_code"] == "SPECIAL_001"

        # Test unicode characters
        unicode_error = "Unicode error: 测试错误 🚨"
        event3 = ErrorEvent(error=unicode_error, source="test_client")
        assert event3.data["error"] == unicode_error

    @pytest.mark.unit
    def test_error_event_different_priority_levels(self):
        """
        Test ErrorEvent with different priority levels.

        Verifies:
        - All priority levels work correctly
        - Default is HIGH but can be overridden
        """
        priorities = [
            EventPriority(EventPriority.CRITICAL),
            EventPriority(EventPriority.HIGH),
            EventPriority(EventPriority.NORMAL),
            EventPriority(EventPriority.LOW),
        ]

        for priority in priorities:
            event = ErrorEvent(error=f"Error with {priority} priority", source="test_client", priority=priority)
            assert event.priority == priority
            assert event.event_type == EventType.ERROR
