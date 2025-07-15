# ABOUTME: Unit tests for Kline model with comprehensive validation testing
# ABOUTME: Tests normal cases, exception cases, and boundary conditions following TDD principles

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from pydantic import ValidationError

from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval, AssetClass


class TestKlineModel:
    """Test suite for Kline model validation and behavior."""

    def test_kline_creation_with_valid_data(self):
        """Test normal case: creating Kline with valid data."""
        # Arrange
        open_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        close_time = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)
        
        # Act
        kline = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=open_time,
            close_time=close_time,
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("10.5"),
            quote_volume=Decimal("525000.00"),
            trades_count=100
        )
        
        # Assert
        assert kline.symbol == "BTCUSDT"
        assert kline.interval == KlineInterval.MINUTE_1
        assert kline.open_time == open_time
        assert kline.close_time == close_time
        assert kline.open_price == Decimal("50000.00")
        assert kline.high_price == Decimal("50100.00")
        assert kline.low_price == Decimal("49900.00")
        assert kline.close_price == Decimal("50050.00")
        assert kline.volume == Decimal("10.5")
        assert kline.quote_volume == Decimal("525000.00")
        assert kline.trades_count == 100
        assert kline.asset_class == AssetClass.DIGITAL  # Default value
        assert kline.is_closed is True  # Default value
        assert kline.metadata == {}  # Default value

    def test_kline_symbol_validation_normalization(self):
        """Test symbol validation and normalization."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - symbol normalization
        kline = Kline(**{**base_data, "symbol": "  btcusdt  "})
        assert kline.symbol == "BTCUSDT"
        
        kline = Kline(**{**base_data, "symbol": "ethusdt"})
        assert kline.symbol == "ETHUSDT"

    def test_kline_symbol_validation_errors(self):
        """Test exception cases for symbol validation."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - None symbol
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "symbol": None})
        assert "string" in str(exc_info.value).lower()
        
        # Act & Assert - empty symbol
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "symbol": ""})
        assert "Symbol cannot be empty" in str(exc_info.value)
        
        # Act & Assert - whitespace only symbol
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "symbol": "   "})
        assert "Symbol cannot be empty" in str(exc_info.value)
        
        # Act & Assert - non-string symbol
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "symbol": 123})
        # Pydantic provides its own string validation error message
        assert "string" in str(exc_info.value).lower()

    def test_kline_price_validation_positive_values(self):
        """Test price validation for positive values."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - valid positive prices
        kline = Kline(**{**base_data, 
                         "open_price": Decimal("0.01"),
                         "high_price": Decimal("0.02"),
                         "low_price": Decimal("0.005"),
                         "close_price": Decimal("0.015")})
        assert kline.open_price == Decimal("0.01")

    def test_kline_price_validation_errors(self):
        """Test exception cases for price validation."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - negative open price
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "open_price": Decimal("-1.0")})
        assert "greater than 0" in str(exc_info.value)
        
        # Act & Assert - zero high price
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "high_price": Decimal("0")})
        assert "greater than 0" in str(exc_info.value)

    def test_kline_volume_validation_non_negative(self):
        """Test volume validation for non-negative values."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - zero volume (valid)
        kline = Kline(**{**base_data, "volume": Decimal("0")})
        assert kline.volume == Decimal("0")
        
        # Act & Assert - positive volume
        kline = Kline(**{**base_data, "volume": Decimal("100.5")})
        assert kline.volume == Decimal("100.5")

    def test_kline_volume_validation_errors(self):
        """Test exception cases for volume validation."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - negative volume
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "volume": Decimal("-1.0")})
        assert "greater than or equal to 0" in str(exc_info.value)
        
        # Act & Assert - negative quote volume
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "quote_volume": Decimal("-100.0")})
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_kline_trades_count_validation(self):
        """Test trades count validation."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - zero trades count (valid)
        kline = Kline(**{**base_data, "trades_count": 0})
        assert kline.trades_count == 0
        
        # Act & Assert - positive trades count
        kline = Kline(**{**base_data, "trades_count": 1000})
        assert kline.trades_count == 1000

    def test_kline_trades_count_validation_errors(self):
        """Test exception cases for trades count validation."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - negative trades count
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "trades_count": -1})
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_kline_timezone_validation(self):
        """Test timezone validation and conversion."""
        # Arrange
        base_data = self._get_base_kline_data()
        naive_open_time = datetime(2024, 1, 1, 12, 0, 0)  # No timezone
        naive_close_time = datetime(2024, 1, 1, 12, 1, 0)  # No timezone, 1 minute later
        
        # Act
        kline = Kline(**{**base_data, "open_time": naive_open_time, "close_time": naive_close_time})
        
        # Assert - naive datetime should be converted to UTC
        assert kline.open_time.tzinfo == UTC
        assert kline.close_time.tzinfo == UTC

    def test_kline_time_consistency_validation(self):
        """Test time consistency validation."""
        # Arrange
        open_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        close_time = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)
        base_data = self._get_base_kline_data()
        
        # Act & Assert - valid time order
        kline = Kline(**{**base_data, "open_time": open_time, "close_time": close_time})
        assert kline.open_time < kline.close_time

    def test_kline_time_consistency_validation_errors(self):
        """Test exception cases for time consistency."""
        # Arrange
        open_time = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)
        close_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)  # Before open_time
        base_data = self._get_base_kline_data()
        
        # Act & Assert - close_time before open_time
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data, "open_time": open_time, "close_time": close_time})
        assert "Close time must be after open time" in str(exc_info.value)

    def test_kline_price_consistency_validation(self):
        """Test price consistency validation (OHLC logic)."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - valid OHLC prices
        kline = Kline(**{**base_data,
                         "open_price": Decimal("50000"),
                         "high_price": Decimal("51000"),  # Highest
                         "low_price": Decimal("49000"),   # Lowest
                         "close_price": Decimal("50500")})
        assert kline.high_price >= max(kline.open_price, kline.close_price)
        assert kline.low_price <= min(kline.open_price, kline.close_price)

    def test_kline_price_consistency_validation_errors(self):
        """Test exception cases for price consistency."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - high price lower than open price
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data,
                     "open_price": Decimal("50000"),
                     "high_price": Decimal("49000"),  # Lower than open
                     "low_price": Decimal("48000"),
                     "close_price": Decimal("49500")})
        assert "high_price must be >= max(open_price, close_price)" in str(exc_info.value)
        
        # Act & Assert - low price higher than close price
        with pytest.raises(ValidationError) as exc_info:
            Kline(**{**base_data,
                     "open_price": Decimal("50000"),
                     "high_price": Decimal("51000"),
                     "low_price": Decimal("50500"),  # Higher than close
                     "close_price": Decimal("50200")})
        assert "low_price must be <= min(open_price, close_price)" in str(exc_info.value)

    def test_kline_boundary_values(self):
        """Test boundary cases with extreme values."""
        # Arrange
        base_data = self._get_base_kline_data()
        
        # Act & Assert - very small positive prices
        kline = Kline(**{**base_data,
                         "open_price": Decimal("0.00000001"),
                         "high_price": Decimal("0.00000002"),
                         "low_price": Decimal("0.00000001"),
                         "close_price": Decimal("0.00000001")})
        assert kline.open_price == Decimal("0.00000001")
        
        # Act & Assert - large but reasonable prices (within our limits)
        large_price = Decimal("5000000.00")  # 5M, within our 10M limit
        kline = Kline(**{**base_data,
                         "open_price": large_price,
                         "high_price": large_price,
                         "low_price": large_price,
                         "close_price": large_price})
        assert kline.open_price == large_price

    def test_kline_optional_fields_defaults(self):
        """Test optional fields and their default values."""
        # Arrange & Act
        kline = Kline(**self._get_base_kline_data())
        
        # Assert
        assert kline.exchange is None
        assert kline.taker_buy_volume is None
        assert kline.taker_buy_quote_volume is None
        assert kline.received_at is None
        assert kline.asset_class == AssetClass.DIGITAL
        assert kline.is_closed is True
        assert kline.metadata == {}

    def test_kline_optional_fields_with_values(self):
        """Test optional fields when provided with values."""
        # Arrange
        base_data = self._get_base_kline_data()
        received_time = datetime(2024, 1, 1, 12, 2, 0, tzinfo=UTC)
        
        # Act
        kline = Kline(**{**base_data,
                         "exchange": "BINANCE",
                         "taker_buy_volume": Decimal("5.0"),
                         "taker_buy_quote_volume": Decimal("250000.0"),
                         "received_at": received_time,
                         "asset_class": AssetClass.TRADITIONAL,
                         "is_closed": False,
                         "metadata": {"source": "websocket"}})
        
        # Assert
        assert kline.exchange == "BINANCE"
        assert kline.taker_buy_volume == Decimal("5.0")
        assert kline.taker_buy_quote_volume == Decimal("250000.0")
        assert kline.received_at == received_time
        assert kline.asset_class == AssetClass.TRADITIONAL
        assert kline.is_closed is False
        assert kline.metadata == {"source": "websocket"}

    def test_kline_model_serialization(self):
        """Test model serialization to dict."""
        # Arrange
        kline = Kline(**self._get_base_kline_data())
        
        # Act
        kline_dict = kline.model_dump()
        
        # Assert
        assert isinstance(kline_dict, dict)
        assert kline_dict["symbol"] == "BTCUSDT"
        assert kline_dict["interval"] == "1m"  # Enum value
        assert "open_time" in kline_dict
        assert "close_time" in kline_dict

    def _get_base_kline_data(self) -> dict:
        """Helper method to get base valid Kline data."""
        return {
            "symbol": "BTCUSDT",
            "interval": KlineInterval.MINUTE_1,
            "open_time": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            "close_time": datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC),
            "open_price": Decimal("50000.00"),
            "high_price": Decimal("50100.00"),
            "low_price": Decimal("49900.00"),
            "close_price": Decimal("50050.00"),
            "volume": Decimal("10.5"),
            "quote_volume": Decimal("525000.00"),
            "trades_count": 100
        }