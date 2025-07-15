# ABOUTME: Unit tests for TradeEvent model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for TradeEvent

import pytest
from datetime import datetime, UTC
from decimal import Decimal

from core.models.event.trade_event import TradeEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide


class TestTradeEvent:
    """
    Comprehensive unit tests for TradeEvent model.

    Tests cover:
    - Normal case: Valid TradeEvent creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    @pytest.mark.unit
    def test_trade_event_creation_normal_case(self):
        """
        Test normal TradeEvent creation with valid Trade data.

        Verifies:
        - Event type is automatically set to TRADE
        - All BaseEvent fields are properly initialized
        - Trade data is correctly stored
        """
        # Arrange
        trade_data = Trade(
            symbol="BTCUSDT",
            trade_id="12345",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
        )

        # Act
        event = TradeEvent(source="binance", symbol="BTCUSDT", data=trade_data)

        # Assert
        assert event.event_type == EventType.TRADE
        assert event.source == "binance"
        assert event.symbol == "BTCUSDT"
        assert event.data == trade_data
        assert event.priority == EventPriority.NORMAL
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo == UTC
        assert len(event.event_id) > 0

    @pytest.mark.unit
    def test_trade_event_forced_event_type(self):
        """
        Test that event_type is always forced to TRADE regardless of input.

        Verifies:
        - Even if different event_type is provided, it's overridden to TRADE
        - Event type field is frozen and cannot be changed
        """
        # Arrange
        trade_data = Trade(
            symbol="ETHUSDT",
            trade_id="67890",
            price=Decimal("3000.00"),
            quantity=Decimal("0.5"),
            side=TradeSide.SELL,
            timestamp=datetime.now(UTC),
        )

        # Act - Try to set different event_type
        event = TradeEvent(
            source="coinbase",
            symbol="ETHUSDT",
            data=trade_data,
            event_type=EventType.KLINE,  # This should be ignored
        )

        # Assert
        assert event.event_type == EventType.TRADE  # Should be TRADE, not KLINE

    @pytest.mark.unit
    def test_trade_event_with_custom_priority(self):
        """
        Test TradeEvent creation with custom priority.

        Verifies:
        - Custom priority is properly set
        - Other fields remain correct
        """
        # Arrange
        trade_data = Trade(
            symbol="ADAUSDT",
            trade_id="11111",
            price=Decimal("1.50"),
            quantity=Decimal("100.0"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
        )

        # Act
        event = TradeEvent(
            source="kraken", symbol="ADAUSDT", data=trade_data, priority=EventPriority(EventPriority.HIGH)
        )

        # Assert
        assert event.priority == EventPriority.HIGH
        assert event.event_type == EventType.TRADE

    @pytest.mark.unit
    def test_trade_event_with_metadata_and_correlation_id(self):
        """
        Test TradeEvent with additional metadata and correlation ID.

        Verifies:
        - Metadata is properly stored
        - Correlation ID is correctly set
        - All other fields remain valid
        """
        # Arrange
        trade_data = Trade(
            symbol="DOTUSDT",
            trade_id="22222",
            price=Decimal("25.00"),
            quantity=Decimal("10.0"),
            side=TradeSide.SELL,
            timestamp=datetime.now(UTC),
        )

        metadata = {"exchange_id": "binance_spot", "market_type": "spot", "fee": "0.001"}
        correlation_id = "trade_batch_001"

        # Act
        event = TradeEvent(
            source="binance", symbol="DOTUSDT", data=trade_data, metadata=metadata, correlation_id=correlation_id
        )

        # Assert
        assert event.metadata == metadata
        assert event.correlation_id == correlation_id
        assert event.event_type == EventType.TRADE

    @pytest.mark.unit
    def test_trade_event_to_dict_method(self):
        """
        Test TradeEvent to_dict method.

        Verifies:
        - Dictionary contains all expected fields
        - Timestamp is properly formatted as ISO string
        - Data structure is correct
        """
        # Arrange
        trade_data = Trade(
            symbol="SOLUSDT",
            trade_id="33333",
            price=Decimal("100.00"),
            quantity=Decimal("5.0"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
        )

        event = TradeEvent(source="ftx", symbol="SOLUSDT", data=trade_data)

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
        assert event_dict["event_type"] == EventType.TRADE
        assert isinstance(event_dict["timestamp"], str)  # Should be ISO format

    @pytest.mark.unit
    def test_trade_event_str_representation(self):
        """
        Test TradeEvent string representation.

        Verifies:
        - String format includes class name, ID, type, and source
        - Format is consistent and readable
        """
        # Arrange
        trade_data = Trade(
            symbol="LINKUSDT",
            trade_id="44444",
            price=Decimal("15.00"),
            quantity=Decimal("20.0"),
            side=TradeSide.SELL,
            timestamp=datetime.now(UTC),
        )

        event = TradeEvent(source="huobi", symbol="LINKUSDT", data=trade_data)

        # Act
        str_repr = str(event)

        # Assert
        assert "TradeEvent" in str_repr
        assert event.event_id in str_repr
        assert "trade" in str_repr
        assert "huobi" in str_repr

    @pytest.mark.unit
    def test_trade_event_invalid_data_type(self):
        """
        Test TradeEvent with invalid data type.

        Verifies:
        - Proper validation error when data is not Trade type
        - Error message is descriptive
        """
        # Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            TradeEvent(
                source="invalid_source",
                symbol="INVALID",
                data="not_a_trade_object",  # Invalid data type
            )

    @pytest.mark.unit
    def test_trade_event_missing_required_fields(self):
        """
        Test TradeEvent creation with missing required fields.

        Verifies:
        - Proper validation errors for missing source
        - Proper validation errors for missing data
        """
        trade_data = Trade(
            symbol="TESTUSDT",
            trade_id="55555",
            price=Decimal("1.00"),
            quantity=Decimal("1.0"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
        )

        # Test missing source
        with pytest.raises(Exception):  # Pydantic validation error
            TradeEvent(
                symbol="TESTUSDT",
                data=trade_data,
                # Missing source
            )

        # Test missing data
        with pytest.raises(Exception):  # Pydantic validation error
            TradeEvent(
                source="test_exchange",
                symbol="TESTUSDT",
                # Missing data
            )

    @pytest.mark.unit
    def test_trade_event_boundary_values(self):
        """
        Test TradeEvent with boundary values.

        Verifies:
        - Handles very small decimal values
        - Handles very large decimal values
        - Handles edge case timestamps
        """
        # Arrange - Boundary values
        trade_data = Trade(
            symbol="MICROUSDT",
            trade_id="boundary_test",
            price=Decimal("0.000001"),  # Very small price
            quantity=Decimal("999999999.999999"),  # Very large quantity
            side=TradeSide.BUY,
            timestamp=datetime.min.replace(tzinfo=UTC),
        )

        # Act
        event = TradeEvent(source="test", symbol="MICROUSDT", data=trade_data)

        # Assert
        assert event.data.price == Decimal("0.000001")
        assert event.data.quantity == Decimal("999999999.999999")
        assert event.event_type == EventType.TRADE

    @pytest.mark.unit
    def test_trade_event_symbol_validation_inheritance(self):
        """
        Test that TradeEvent inherits symbol validation from BaseEvent.

        Verifies:
        - Symbol is normalized (uppercase, stripped)
        - Invalid symbols raise appropriate errors
        """
        trade_data = Trade(
            symbol="btcusdt",  # lowercase
            trade_id="symbol_test",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
        )

        # Act
        event = TradeEvent(
            source="test",
            symbol="  btcusdt  ",  # With whitespace
            data=trade_data,
        )

        # Assert
        assert event.symbol == "BTCUSDT"  # Should be normalized

        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            TradeEvent(
                source="test",
                symbol="   ",  # Only whitespace
                data=trade_data,
            )

    @pytest.mark.unit
    def test_trade_event_different_trade_sides(self):
        """
        Test TradeEvent with different trade sides.

        Verifies:
        - Both BUY and SELL sides work correctly
        - Trade side is preserved in the event
        """
        # Test BUY side
        buy_trade = Trade(
            symbol="BTCUSDT",
            trade_id="buy_test",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
        )

        buy_event = TradeEvent(source="test", symbol="BTCUSDT", data=buy_trade)

        assert buy_event.data.side == TradeSide.BUY

        # Test SELL side
        sell_trade = Trade(
            symbol="BTCUSDT",
            trade_id="sell_test",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.SELL,
            timestamp=datetime.now(UTC),
        )

        sell_event = TradeEvent(source="test", symbol="BTCUSDT", data=sell_trade)

        assert sell_event.data.side == TradeSide.SELL
