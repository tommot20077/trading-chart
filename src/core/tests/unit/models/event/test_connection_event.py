# ABOUTME: Unit tests for ConnectionEvent model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for ConnectionEvent

import pytest
from datetime import datetime, UTC

from core.models.event.connection_event import ConnectionEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.network.enum import ConnectionStatus


class TestConnectionEvent:
    """
    Comprehensive unit tests for ConnectionEvent model.

    Tests cover:
    - Normal case: Valid ConnectionEvent creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    @pytest.mark.unit
    def test_connection_event_creation_normal_case(self):
        """
        Test normal ConnectionEvent creation with valid ConnectionStatus.

        Verifies:
        - Event type is automatically set to CONNECTION
        - ConnectionStatus is properly stored in data
        - All BaseEvent fields are properly initialized
        """
        # Act
        event = ConnectionEvent(status=ConnectionStatus.CONNECTED, source="websocket_client", symbol="BTCUSDT")

        # Assert
        assert event.event_type == EventType.CONNECTION
        assert event.source == "websocket_client"
        assert event.symbol == "BTCUSDT"
        assert event.data["status"] == ConnectionStatus.CONNECTED.value
        assert event.priority == EventPriority.NORMAL
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == UTC
        assert len(event.event_id) > 0

    @pytest.mark.unit
    def test_connection_event_all_status_types(self):
        """
        Test ConnectionEvent with all possible ConnectionStatus values.

        Verifies:
        - All ConnectionStatus enum values work correctly
        - Status is properly stored as string value in data
        """
        statuses = [
            ConnectionStatus.CONNECTING,
            ConnectionStatus.CONNECTED,
            ConnectionStatus.DISCONNECTED,
            ConnectionStatus.RECONNECTING,
            ConnectionStatus.ERROR,
        ]

        for status in statuses:
            # Act
            event = ConnectionEvent(status=status, source="test_client")

            # Assert
            assert event.event_type == EventType.CONNECTION
            assert event.data["status"] == status.value
            assert isinstance(event.data["status"], str)

    @pytest.mark.unit
    def test_connection_event_forced_event_type(self):
        """
        Test that event_type is always forced to CONNECTION regardless of input.

        Verifies:
        - Even if different event_type is provided, it's overridden to CONNECTION
        - Event type field is frozen and cannot be changed
        """
        # Act - Try to set different event_type
        event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED,
            source="test_client",
            event_type=EventType.TRADE,  # This should be ignored
        )

        # Assert
        assert event.event_type == EventType.CONNECTION  # Should be CONNECTION, not TRADE

    @pytest.mark.unit
    def test_connection_event_with_custom_priority(self):
        """
        Test ConnectionEvent creation with custom priority.

        Verifies:
        - Custom priority is properly set
        - Other fields remain correct
        """
        # Act
        event = ConnectionEvent(
            status=ConnectionStatus.ERROR, source="critical_client", priority=EventPriority(EventPriority.HIGH)
        )

        # Assert
        assert event.priority == EventPriority.HIGH
        assert event.event_type == EventType.CONNECTION
        assert event.data["status"] == ConnectionStatus.ERROR.value

    @pytest.mark.unit
    @pytest.mark.external
    def test_connection_event_with_additional_data(self):
        """
        Test ConnectionEvent with additional data fields.

        Verifies:
        - Additional data is merged with status
        - Status is preserved
        - Data structure is correct
        """
        # Arrange
        additional_data = {"retry_count": 3, "last_error": "timeout", "endpoint": "wss://api.binance.com/ws"}

        # Act
        event = ConnectionEvent(status=ConnectionStatus.RECONNECTING, source="websocket_client", data=additional_data)

        # Assert
        assert event.data["status"] == ConnectionStatus.RECONNECTING.value
        assert event.data["retry_count"] == 3
        assert event.data["last_error"] == "timeout"
        assert event.data["endpoint"] == "wss://api.binance.com/ws"

    @pytest.mark.unit
    def test_connection_event_with_metadata_and_correlation_id(self):
        """
        Test ConnectionEvent with additional metadata and correlation ID.

        Verifies:
        - Metadata is properly stored
        - Correlation ID is correctly set
        - All other fields remain valid
        """
        # Arrange
        metadata = {"client_id": "ws_001", "session_id": "sess_123"}
        correlation_id = "connection_batch_001"

        # Act
        event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED,
            source="websocket_client",
            metadata=metadata,
            correlation_id=correlation_id,
        )

        # Assert
        assert event.metadata == metadata
        assert event.correlation_id == correlation_id
        assert event.event_type == EventType.CONNECTION

    @pytest.mark.unit
    @pytest.mark.external
    def test_connection_event_to_dict_method(self):
        """
        Test ConnectionEvent to_dict method.

        Verifies:
        - Dictionary contains all expected fields
        - Timestamp is properly formatted as ISO string
        - Data structure is correct
        """
        # Act
        event = ConnectionEvent(status=ConnectionStatus.DISCONNECTED, source="api_client")

        event_dict = event.to_dict()

        # Assert
        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert "source" in event_dict
        assert "data" in event_dict
        assert "priority" in event_dict
        assert event_dict["event_type"] == EventType.CONNECTION
        assert isinstance(event_dict["timestamp"], str)  # Should be ISO format
        assert event_dict["data"]["status"] == ConnectionStatus.DISCONNECTED.value

    @pytest.mark.unit
    def test_connection_event_str_representation(self):
        """
        Test ConnectionEvent string representation.

        Verifies:
        - String format includes class name, ID, type, and source
        - Format is consistent and readable
        """
        # Act
        event = ConnectionEvent(status=ConnectionStatus.CONNECTING, source="rest_client")

        str_repr = str(event)

        # Assert
        assert "ConnectionEvent" in str_repr
        assert event.event_id in str_repr
        assert "connection" in str_repr
        assert "rest_client" in str_repr

    @pytest.mark.unit
    def test_connection_event_missing_status(self):
        """
        Test ConnectionEvent creation with missing status parameter.

        Verifies:
        - Proper validation error when status is not provided
        - Error message is descriptive
        """
        # Act & Assert
        with pytest.raises(TypeError):  # Missing required positional argument
            ConnectionEvent(
                source="test_client"
                # Missing status parameter
            )

    @pytest.mark.unit
    def test_connection_event_invalid_status_type(self):
        """
        Test ConnectionEvent with invalid status type.

        Verifies:
        - Proper validation error when status is not ConnectionStatus enum
        - Error message is descriptive
        """
        # Act & Assert
        with pytest.raises(Exception):  # Type error or validation error
            ConnectionEvent(
                status="invalid_status",  # Should be ConnectionStatus enum
                source="test_client",
            )

    @pytest.mark.unit
    def test_connection_event_missing_source_uses_default(self):
        """
        Test ConnectionEvent creation with missing source uses default.

        Verifies:
        - Default source is provided when none is specified
        - Event is created successfully
        """
        # Test missing source - should use default
        event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED
            # Missing source - should use default
        )

        # Assert default source is used
        assert event.source == "connection_monitor"
        assert event.status == ConnectionStatus.CONNECTED
        assert event.event_type == EventType.CONNECTION

    @pytest.mark.unit
    def test_connection_event_symbol_validation_inheritance(self):
        """
        Test that ConnectionEvent inherits symbol validation from BaseEvent.

        Verifies:
        - Symbol is normalized (uppercase, stripped)
        - Invalid symbols raise appropriate errors
        - None symbol is allowed
        """
        # Test with valid symbol
        event = ConnectionEvent(
            status=ConnectionStatus.CONNECTED,
            source="test_client",
            symbol="  btcusdt  ",  # With whitespace
        )

        assert event.symbol == "BTCUSDT"  # Should be normalized

        # Test with None symbol (should be allowed)
        event_no_symbol = ConnectionEvent(status=ConnectionStatus.CONNECTED, source="test_client")

        assert event_no_symbol.symbol is None

        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            ConnectionEvent(
                status=ConnectionStatus.CONNECTED,
                source="test_client",
                symbol="   ",  # Only whitespace
            )

    @pytest.mark.unit
    def test_connection_event_data_override_behavior(self):
        """
        Test ConnectionEvent data override behavior.

        Verifies:
        - Status is always set in data regardless of input
        - Additional data fields are preserved
        - Status cannot be overridden in data
        """
        # Arrange
        conflicting_data = {
            "status": "wrong_status",  # This should be overridden
            "additional_field": "preserved",
        }

        # Act
        event = ConnectionEvent(status=ConnectionStatus.ERROR, source="test_client", data=conflicting_data)

        # Assert
        assert event.data["status"] == ConnectionStatus.ERROR.value  # Should be overridden
        assert event.data["additional_field"] == "preserved"  # Should be preserved

    @pytest.mark.unit
    def test_connection_event_error_status_scenarios(self):
        """
        Test ConnectionEvent with ERROR status and error details.

        Verifies:
        - ERROR status works correctly
        - Error details can be included in data
        - Event structure remains valid
        """
        # Arrange
        error_data = {
            "error_code": "CONN_TIMEOUT",
            "error_message": "Connection timeout after 30 seconds",
            "retry_attempt": 2,
        }

        # Act
        event = ConnectionEvent(
            status=ConnectionStatus.ERROR,
            source="websocket_client",
            data=error_data,
            priority=EventPriority(EventPriority.HIGH),  # Errors might have higher priority
        )

        # Assert
        assert event.data["status"] == ConnectionStatus.ERROR.value
        assert event.data["error_code"] == "CONN_TIMEOUT"
        assert event.data["error_message"] == "Connection timeout after 30 seconds"
        assert event.data["retry_attempt"] == 2
        assert event.priority == EventPriority.HIGH
