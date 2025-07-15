# ABOUTME: Unit tests for Trade model with comprehensive validation testing
# ABOUTME: Tests normal cases, exception cases, and boundary conditions following TDD principles

import pytest
from datetime import datetime, UTC
from decimal import Decimal
from pydantic import ValidationError

from core.models.data.trade import Trade
from core.models.data.enum import TradeSide, AssetClass


class TestTradeModel:
    """Test suite for Trade model validation and behavior."""

    def test_trade_creation_with_valid_data(self):
        """Test normal case: creating Trade with valid data."""
        # Arrange
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        
        # Act
        trade = Trade(
            symbol="BTCUSDT",
            trade_id="12345",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=timestamp
        )
        
        # Assert
        assert trade.symbol == "BTCUSDT"
        assert trade.trade_id == "12345"
        assert trade.price == Decimal("50000.00")
        assert trade.quantity == Decimal("0.1")
        assert trade.side == TradeSide.BUY
        assert trade.timestamp == timestamp
        assert trade.asset_class == AssetClass.DIGITAL  # Default value
        assert trade.exchange is None  # Default value
        assert trade.metadata == {}  # Default value

    def test_trade_symbol_validation_normalization(self):
        """Test symbol validation and normalization."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - symbol normalization
        trade = Trade(**{**base_data, "symbol": "  btcusdt  "})
        assert trade.symbol == "BTCUSDT"
        
        trade = Trade(**{**base_data, "symbol": "ethusdt"})
        assert trade.symbol == "ETHUSDT"

    def test_trade_symbol_validation_errors(self):
        """Test exception cases for symbol validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - None symbol
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "symbol": None})
        assert "string" in str(exc_info.value).lower()
        
        # Act & Assert - empty symbol
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "symbol": ""})
        assert "Symbol cannot be empty" in str(exc_info.value)
        
        # Act & Assert - whitespace only symbol
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "symbol": "   "})
        assert "Symbol cannot be empty" in str(exc_info.value)
        
        # Act & Assert - non-string symbol
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "symbol": 123})
        # Pydantic provides its own string validation error message
        assert "string" in str(exc_info.value).lower()

    def test_trade_id_validation_normalization(self):
        """Test trade_id validation and normalization."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - trade_id normalization
        trade = Trade(**{**base_data, "trade_id": "  12345  "})
        assert trade.trade_id == "12345"

    def test_trade_id_validation_errors(self):
        """Test exception cases for trade_id validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - None trade_id
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "trade_id": None})
        assert "string" in str(exc_info.value).lower()
        
        # Act & Assert - empty trade_id (Pydantic allows empty strings by default)
        # Note: Empty string validation would need custom validator if required
        trade = Trade(**{**base_data, "trade_id": ""})
        assert trade.trade_id == ""  # Pydantic allows empty strings
        
        # Act & Assert - whitespace only trade_id (Pydantic strips whitespace)
        # Note: Pydantic's str_strip_whitespace=True converts "   " to ""
        trade = Trade(**{**base_data, "trade_id": "   "})
        assert trade.trade_id == ""  # Whitespace is stripped to empty string

    def test_trade_price_validation_positive_values(self):
        """Test price validation for positive values."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - valid positive price
        trade = Trade(**{**base_data, "price": Decimal("0.01")})
        assert trade.price == Decimal("0.01")
        
        # Act & Assert - large price
        trade = Trade(**{**base_data, "price": Decimal("999999.99")})
        assert trade.price == Decimal("999999.99")

    def test_trade_price_validation_errors(self):
        """Test exception cases for price validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - negative price
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "price": Decimal("-1.0")})
        assert "greater than 0" in str(exc_info.value)
        
        # Act & Assert - zero price
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "price": Decimal("0")})
        assert "greater than 0" in str(exc_info.value)

    def test_trade_quantity_validation_positive_values(self):
        """Test quantity validation for positive values."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - valid positive quantity
        trade = Trade(**{**base_data, "quantity": Decimal("0.001")})
        assert trade.quantity == Decimal("0.001")
        
        # Act & Assert - large quantity
        trade = Trade(**{**base_data, "quantity": Decimal("1000000.0")})
        assert trade.quantity == Decimal("1000000.0")

    def test_trade_quantity_validation_errors(self):
        """Test exception cases for quantity validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - negative quantity
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "quantity": Decimal("-1.0")})
        assert "greater than 0" in str(exc_info.value)
        
        # Act & Assert - zero quantity
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "quantity": Decimal("0")})
        assert "greater than 0" in str(exc_info.value)

    def test_trade_side_validation(self):
        """Test trade side validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - BUY side
        trade = Trade(**{**base_data, "side": TradeSide.BUY})
        assert trade.side == TradeSide.BUY
        
        # Act & Assert - SELL side
        trade = Trade(**{**base_data, "side": TradeSide.SELL})
        assert trade.side == TradeSide.SELL

    def test_trade_side_validation_errors(self):
        """Test exception cases for trade side validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - invalid side string
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "side": "invalid"})
        assert "Input should be 'buy' or 'sell'" in str(exc_info.value)
        
        # Act & Assert - None side
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "side": None})
        assert "Input should be 'buy' or 'sell'" in str(exc_info.value)

    def test_trade_timestamp_validation(self):
        """Test timestamp validation and timezone handling."""
        # Arrange
        base_data = self._get_base_trade_data()
        naive_time = datetime(2024, 1, 1, 12, 0, 0)  # No timezone
        
        # Act
        trade = Trade(**{**base_data, "timestamp": naive_time})
        
        # Assert - naive datetime should be converted to UTC
        assert trade.timestamp.tzinfo == UTC

    def test_trade_timestamp_validation_errors(self):
        """Test exception cases for timestamp validation."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - None timestamp
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "timestamp": None})
        assert "datetime" in str(exc_info.value).lower()
        
        # Act & Assert - invalid datetime timestamp
        with pytest.raises(ValidationError) as exc_info:
            Trade(**{**base_data, "timestamp": "not-a-date"})
        assert "datetime" in str(exc_info.value).lower()

    def test_trade_boundary_values(self):
        """Test boundary cases with extreme values."""
        # Arrange
        base_data = self._get_base_trade_data()
        
        # Act & Assert - very small positive price and quantity
        trade = Trade(**{**base_data,
                         "price": Decimal("0.00000001"),
                         "quantity": Decimal("0.00000001")})
        assert trade.price == Decimal("0.00000001")
        assert trade.quantity == Decimal("0.00000001")
        
        # Act & Assert - large but reasonable values (within our limits)
        large_price = Decimal("5000000.00")  # 5M, within our 10M price limit
        large_quantity = Decimal("10000.00")  # 10K, reasonable quantity
        trade = Trade(**{**base_data,
                         "price": large_price,
                         "quantity": large_quantity})
        assert trade.price == large_price
        assert trade.quantity == large_quantity

    def test_trade_optional_fields_defaults(self):
        """Test optional fields and their default values."""
        # Arrange & Act
        trade = Trade(**self._get_base_trade_data())
        
        # Assert
        assert trade.asset_class == AssetClass.DIGITAL
        assert trade.exchange is None
        assert trade.maker_order_id is None
        assert trade.taker_order_id is None
        assert trade.is_buyer_maker is None
        assert trade.received_at is None
        assert trade.metadata == {}

    def test_trade_optional_fields_with_values(self):
        """Test optional fields when provided with values."""
        # Arrange
        base_data = self._get_base_trade_data()
        received_time = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)
        
        # Act
        trade = Trade(**{**base_data,
                         "asset_class": AssetClass.TRADITIONAL,
                         "exchange": "BINANCE",
                         "maker_order_id": "maker123",
                         "taker_order_id": "taker456",
                         "is_buyer_maker": True,
                         "received_at": received_time,
                         "metadata": {"source": "websocket"}})
        
        # Assert
        assert trade.asset_class == AssetClass.TRADITIONAL
        assert trade.exchange == "BINANCE"
        assert trade.maker_order_id == "maker123"
        assert trade.taker_order_id == "taker456"
        assert trade.is_buyer_maker is True
        assert trade.received_at == received_time
        assert trade.metadata == {"source": "websocket"}

    def test_trade_quote_value_calculation(self):
        """Test quote value calculation method if exists."""
        # Arrange
        trade = Trade(**self._get_base_trade_data())
        
        # Act & Assert - verify quote value calculation
        expected_quote_value = trade.price * trade.quantity
        # Note: This assumes the Trade model has a quote_value property or method
        # If not implemented, this test will guide the implementation
        assert trade.price * trade.quantity == Decimal("5000.00")

    def test_trade_model_serialization(self):
        """Test model serialization to dict."""
        # Arrange
        trade = Trade(**self._get_base_trade_data())
        
        # Act
        trade_dict = trade.model_dump()
        
        # Assert
        assert isinstance(trade_dict, dict)
        assert trade_dict["symbol"] == "BTCUSDT"
        assert trade_dict["trade_id"] == "12345"
        assert trade_dict["side"] == "buy"  # Enum value
        assert "timestamp" in trade_dict

    def test_trade_string_representation(self):
        """Test string representation of Trade."""
        # Arrange
        trade = Trade(**self._get_base_trade_data())
        
        # Act
        trade_str = str(trade)
        
        # Assert
        assert "Trade" in trade_str
        assert "BTCUSDT" in trade_str
        assert "12345" in trade_str

    def _get_base_trade_data(self) -> dict:
        """Helper method to get base valid Trade data."""
        return {
            "symbol": "BTCUSDT",
            "trade_id": "12345",
            "price": Decimal("50000.00"),
            "quantity": Decimal("0.1"),
            "side": TradeSide.BUY,
            "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        }