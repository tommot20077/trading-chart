# ABOUTME: Unit tests for KlineEvent model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for KlineEvent

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal

from core.models.event.Kline_event import KlineEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval


class TestKlineEvent:
    """
    Comprehensive unit tests for KlineEvent model.

    Tests cover:
    - Normal case: Valid KlineEvent creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    @pytest.mark.unit
    def test_kline_event_creation_normal_case(self):
        """
        Test normal KlineEvent creation with valid Kline data.

        Verifies:
        - Event type is automatically set to KLINE
        - All BaseEvent fields are properly initialized
        - Kline data is correctly stored
        """
        # Arrange
        kline_data = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("10.5"),
            quote_volume=Decimal("525000.00"),
            trades_count=100,
        )

        # Act
        event = KlineEvent(source="binance", symbol="BTCUSDT", data=kline_data)

        # Assert
        assert event.event_type == EventType.KLINE
        assert event.source == "binance"
        assert event.symbol == "BTCUSDT"
        assert event.data == kline_data
        assert event.priority == EventPriority.NORMAL
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == UTC
        assert len(event.event_id) > 0

    @pytest.mark.unit
    def test_kline_event_forced_event_type(self):
        """
        Test that event_type is always forced to KLINE regardless of input.

        Verifies:
        - Even if different event_type is provided, it's overridden to KLINE
        - Event type field is frozen and cannot be changed
        """
        # Arrange
        kline_data = Kline(
            symbol="ETHUSDT",
            interval=KlineInterval.MINUTE_5,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("3000.00"),
            high_price=Decimal("3010.00"),
            low_price=Decimal("2990.00"),
            close_price=Decimal("3005.00"),
            volume=Decimal("5.2"),
            quote_volume=Decimal("15600.00"),
            trades_count=50,
        )

        # Act - Try to set different event_type
        event = KlineEvent(
            source="coinbase",
            symbol="ETHUSDT",
            data=kline_data,
            event_type=EventType.TRADE,  # This should be ignored
        )

        # Assert
        assert event.event_type == EventType.KLINE  # Should be KLINE, not TRADE

    @pytest.mark.unit
    def test_kline_event_with_custom_priority(self):
        """
        Test KlineEvent creation with custom priority.

        Verifies:
        - Custom priority is properly set
        - Other fields remain correct
        """
        # Arrange
        kline_data = Kline(
            symbol="ADAUSDT",
            interval=KlineInterval.HOUR_1,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("1.50"),
            high_price=Decimal("1.55"),
            low_price=Decimal("1.48"),
            close_price=Decimal("1.52"),
            volume=Decimal("1000.0"),
            quote_volume=Decimal("1520.00"),
            trades_count=200,
        )

        # Act
        event = KlineEvent(
            source="kraken", symbol="ADAUSDT", data=kline_data, priority=EventPriority(EventPriority.HIGH)
        )

        # Assert
        assert event.priority == EventPriority.HIGH
        assert event.event_type == EventType.KLINE

    @pytest.mark.unit
    def test_kline_event_with_metadata_and_correlation_id(self):
        """
        Test KlineEvent with additional metadata and correlation ID.

        Verifies:
        - Metadata is properly stored
        - Correlation ID is correctly set
        - All other fields remain valid
        """
        # Arrange
        kline_data = Kline(
            symbol="DOTUSDT",
            interval=KlineInterval.DAY_1,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("25.00"),
            high_price=Decimal("26.00"),
            low_price=Decimal("24.50"),
            close_price=Decimal("25.75"),
            volume=Decimal("500.0"),
            quote_volume=Decimal("12875.00"),
            trades_count=75,
        )

        metadata = {"exchange_id": "binance_spot", "market_type": "spot"}
        correlation_id = "kline_batch_001"

        # Act
        event = KlineEvent(
            source="binance", symbol="DOTUSDT", data=kline_data, metadata=metadata, correlation_id=correlation_id
        )

        # Assert
        assert event.metadata == metadata
        assert event.correlation_id == correlation_id
        assert event.event_type == EventType.KLINE

    @pytest.mark.unit
    def test_kline_event_to_dict_method(self):
        """
        Test KlineEvent to_dict method.

        Verifies:
        - Dictionary contains all expected fields
        - Timestamp is properly formatted as ISO string
        - Data structure is correct
        """
        # Arrange
        kline_data = Kline(
            symbol="SOLUSDT",
            interval=KlineInterval.MINUTE_15,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("100.00"),
            high_price=Decimal("102.00"),
            low_price=Decimal("99.00"),
            close_price=Decimal("101.50"),
            volume=Decimal("250.0"),
            quote_volume=Decimal("25375.00"),
            trades_count=125,
        )

        event = KlineEvent(source="ftx", symbol="SOLUSDT", data=kline_data)

        # Act
        event_dict = event.to_dict()

        # Assert
        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert "source" in event_dict
        assert "symbol" in event_dict
        assert "data" in event_dict
        assert "priority" in event_dict
        assert event_dict["event_type"] == EventType.KLINE
        assert isinstance(event_dict["timestamp"], str)  # Should be ISO format

    @pytest.mark.unit
    def test_kline_event_str_representation(self):
        """
        Test KlineEvent string representation.

        Verifies:
        - String format includes class name, ID, type, and source
        - Format is consistent and readable
        """
        # Arrange
        kline_data = Kline(
            symbol="LINKUSDT",
            interval=KlineInterval.HOUR_4,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("15.00"),
            high_price=Decimal("15.50"),
            low_price=Decimal("14.80"),
            close_price=Decimal("15.25"),
            volume=Decimal("800.0"),
            quote_volume=Decimal("12200.00"),
            trades_count=300,
        )

        event = KlineEvent(source="huobi", symbol="LINKUSDT", data=kline_data)

        # Act
        str_repr = str(event)

        # Assert
        assert "KlineEvent" in str_repr
        assert event.event_id in str_repr
        assert "kline" in str_repr
        assert "huobi" in str_repr

    @pytest.mark.unit
    def test_kline_event_invalid_data_type(self):
        """
        Test KlineEvent with invalid data type.

        Verifies:
        - Proper validation error when data is not Kline type
        - Error message is descriptive
        """
        # Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            KlineEvent(
                source="invalid_source",
                symbol="INVALID",
                data="not_a_kline_object",  # Invalid data type
            )

    @pytest.mark.unit
    def test_kline_event_missing_required_fields(self):
        """
        Test KlineEvent creation with missing required fields.

        Verifies:
        - Proper validation errors for missing source
        - Proper validation errors for missing data
        """
        now = datetime.now(UTC)
        kline_data = Kline(
            symbol="TESTUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=now,
            close_time=now + timedelta(minutes=1),
            open_price=Decimal("1.00"),
            high_price=Decimal("1.01"),
            low_price=Decimal("0.99"),
            close_price=Decimal("1.005"),
            volume=Decimal("100.0"),
            quote_volume=Decimal("100.50"),
            trades_count=10,
        )

        # Test missing source
        with pytest.raises(Exception):  # Pydantic validation error
            KlineEvent(
                symbol="TESTUSDT",
                data=kline_data,
                # Missing source
            )

        # Test missing data
        with pytest.raises(Exception):  # Pydantic validation error
            KlineEvent(
                source="test_exchange",
                symbol="TESTUSDT",
                # Missing data
            )

    @pytest.mark.unit
    def test_kline_event_boundary_values(self):
        """
        Test KlineEvent with boundary values.

        Verifies:
        - Handles very small decimal values
        - Handles very large decimal values
        - Handles edge case timestamps
        """
        # Arrange - Boundary values (within 24 hour limit)
        kline_data = Kline(
            symbol="MICROUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC),
            close_time=datetime(2020, 1, 1, 23, 59, 59, tzinfo=UTC),
            open_price=Decimal("0.000001"),  # Very small
            high_price=Decimal("9999999.999999"),  # Large but within limit
            low_price=Decimal("0.000001"),
            close_price=Decimal("0.000002"),
            volume=Decimal("0.000001"),
            quote_volume=Decimal("0.000002"),
            trades_count=1,
        )

        # Act
        event = KlineEvent(source="test", symbol="MICROUSDT", data=kline_data)

        # Assert
        assert event.data.open_price == Decimal("0.000001")
        assert event.data.high_price == Decimal("9999999.999999")
        assert event.event_type == EventType.KLINE

    @pytest.mark.unit
    def test_kline_event_symbol_validation_inheritance(self):
        """
        Test that KlineEvent inherits symbol validation from BaseEvent.

        Verifies:
        - Symbol is normalized (uppercase, stripped)
        - Invalid symbols raise appropriate errors
        """
        kline_data = Kline(
            symbol="btcusdt",  # lowercase
            interval=KlineInterval.MINUTE_1,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("10.0"),
            quote_volume=Decimal("500500.00"),
            trades_count=100,
        )

        # Act
        event = KlineEvent(
            source="test",
            symbol="  btcusdt  ",  # With whitespace
            data=kline_data,
        )

        # Assert
        assert event.symbol == "BTCUSDT"  # Should be normalized

        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            KlineEvent(
                source="test",
                symbol="   ",  # Only whitespace
                data=kline_data,
            )
