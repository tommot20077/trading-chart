# ABOUTME: Unit tests for InMemoryDataConverter implementation
# ABOUTME: Tests data conversion functionality with various input formats

import pytest
from datetime import datetime, UTC
from decimal import Decimal

from core.implementations.memory.data.data_converter import InMemoryDataConverter
from core.models.data.enum import TradeSide, AssetClass, KlineInterval


class TestInMemoryDataConverter:
    """Test suite for InMemoryDataConverter."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance for testing."""
        return InMemoryDataConverter()

    @pytest.fixture
    def custom_converter(self):
        """Create a converter with custom defaults."""
        return InMemoryDataConverter(default_exchange="test_exchange", default_asset_class=AssetClass.TRADITIONAL)

    # Trade conversion tests

    @pytest.mark.unit
    def test_convert_trade_basic(self, converter):
        """Test basic trade conversion with minimal required fields."""
        raw_trade = {
            "id": "12345",
            "price": "50000.00",
            "quantity": "0.1",
            "side": "buy",
            "timestamp": 1640995200000,  # 2022-01-01 00:00:00 UTC in milliseconds
        }

        trade = converter.convert_trade(raw_trade, "BTCUSDT")

        assert trade.symbol == "BTCUSDT"
        assert trade.trade_id == "12345"
        assert trade.price == Decimal("50000.00")
        assert trade.quantity == Decimal("0.1")
        assert trade.side == TradeSide.BUY
        assert trade.timestamp == datetime.fromtimestamp(1640995200, tz=UTC)
        assert trade.asset_class == AssetClass.DIGITAL
        assert trade.exchange == "memory"

    @pytest.mark.unit
    def test_convert_trade_alternative_field_names(self, converter):
        """Test trade conversion with alternative field names."""
        raw_trade = {
            "trade_id": "67890",
            "p": 45000.5,
            "qty": 0.25,
            "s": "SELL",
            "ts": 1640995200,  # Unix seconds
        }

        trade = converter.convert_trade(raw_trade, "ETHUSDT")

        assert trade.trade_id == "67890"
        assert trade.price == Decimal("45000.5")
        assert trade.quantity == Decimal("0.25")
        assert trade.side == TradeSide.SELL
        assert trade.symbol == "ETHUSDT"

    @pytest.mark.unit
    def test_convert_trade_with_optional_fields(self, converter):
        """Test trade conversion with optional fields."""
        raw_trade = {
            "id": "11111",
            "price": "30000",
            "quantity": "1.0",
            "side": "buy",
            "timestamp": "2022-01-01T00:00:00Z",
            "exchange": "binance",
            "maker_order_id": "maker123",
            "taker_order_id": "taker456",
            "is_buyer_maker": True,
        }

        trade = converter.convert_trade(raw_trade, "BTCUSDT")

        assert trade.exchange == "binance"
        assert trade.maker_order_id == "maker123"
        assert trade.taker_order_id == "taker456"
        assert trade.is_buyer_maker is True

    @pytest.mark.unit
    def test_convert_trade_missing_required_field(self, converter):
        """Test trade conversion fails with missing required field."""
        raw_trade = {
            "id": "12345",
            "price": "50000.00",
            # Missing quantity
            "side": "buy",
            "timestamp": 1640995200000,
        }

        with pytest.raises(ValueError, match="Required field not found"):
            converter.convert_trade(raw_trade, "BTCUSDT")

    @pytest.mark.unit
    def test_convert_multiple_trades(self, converter):
        """Test conversion of multiple trades."""
        raw_trades = [
            {"id": "1", "price": "50000", "quantity": "0.1", "side": "buy", "timestamp": 1640995200000},
            {"id": "2", "price": "50100", "quantity": "0.2", "side": "sell", "timestamp": 1640995260000},
        ]

        trades = converter.convert_multiple_trades(raw_trades, "BTCUSDT")

        assert len(trades) == 2
        assert trades[0].trade_id == "1"
        assert trades[1].trade_id == "2"
        assert trades[0].side == TradeSide.BUY
        assert trades[1].side == TradeSide.SELL

    # Kline conversion tests

    @pytest.mark.unit
    def test_convert_kline_basic(self, converter):
        """Test basic kline conversion."""
        raw_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "50000.00",
            "high": "50500.00",
            "low": "49500.00",
            "close": "50200.00",
            "volume": "10.5",
            "quote_volume": "525000.00",
            "trades_count": 100,
        }

        kline = converter.convert_kline(raw_kline, "BTCUSDT")

        assert kline.symbol == "BTCUSDT"
        assert kline.open_price == Decimal("50000.00")
        assert kline.high_price == Decimal("50500.00")
        assert kline.low_price == Decimal("49500.00")
        assert kline.close_price == Decimal("50200.00")
        assert kline.volume == Decimal("10.5")
        assert kline.quote_volume == Decimal("525000.00")
        assert kline.trades_count == 100
        assert kline.interval == KlineInterval.MINUTE_1  # Default

    @pytest.mark.unit
    def test_convert_kline_alternative_field_names(self, converter):
        """Test kline conversion with alternative field names."""
        raw_kline = {
            "t": 1640995200000,
            "T": 1640995260000,
            "o": 50000,
            "h": 50500,
            "l": 49500,
            "c": 50200,
            "v": 10.5,
            "qv": 525000,
            "n": 100,
            "interval": "5m",
        }

        kline = converter.convert_kline(raw_kline, "BTCUSDT")

        assert kline.interval == KlineInterval.MINUTE_5
        assert kline.open_price == Decimal("50000")

    @pytest.mark.unit
    def test_convert_kline_with_optional_fields(self, converter):
        """Test kline conversion with optional fields."""
        raw_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "50000",
            "high": "50500",
            "low": "49500",
            "close": "50200",
            "volume": "10.5",
            "quote_volume": "525000",
            "trades_count": 100,
            "exchange": "binance",
            "taker_buy_volume": "5.0",
            "taker_buy_quote_volume": "250000",
            "is_closed": False,
        }

        kline = converter.convert_kline(raw_kline, "BTCUSDT")

        assert kline.exchange == "binance"
        assert kline.taker_buy_volume == Decimal("5.0")
        assert kline.taker_buy_quote_volume == Decimal("250000")
        assert kline.is_closed is False

    @pytest.mark.unit
    def test_convert_multiple_klines(self, converter):
        """Test conversion of multiple klines."""
        raw_klines = [
            {
                "open_time": 1640995200000,
                "close_time": 1640995260000,
                "open": "50000",
                "high": "50500",
                "low": "49500",
                "close": "50200",
                "volume": "10.5",
                "quote_volume": "525000",
                "trades_count": 100,
            },
            {
                "open_time": 1640995260000,
                "close_time": 1640995320000,
                "open": "50200",
                "high": "50800",
                "low": "50000",
                "close": "50600",
                "volume": "8.2",
                "quote_volume": "415000",
                "trades_count": 85,
            },
        ]

        klines = converter.convert_multiple_klines(raw_klines, "BTCUSDT")

        assert len(klines) == 2
        assert klines[0].open_price == Decimal("50000")
        assert klines[1].open_price == Decimal("50200")

    # Timestamp conversion tests

    @pytest.mark.unit
    def test_to_internal_timestamp_unix_seconds(self, converter):
        """Test timestamp conversion from Unix seconds."""
        timestamp = converter.to_internal_timestamp(1640995200)
        assert timestamp == 1640995200000

    @pytest.mark.unit
    def test_to_internal_timestamp_unix_milliseconds(self, converter):
        """Test timestamp conversion from Unix milliseconds."""
        timestamp = converter.to_internal_timestamp(1640995200000)
        assert timestamp == 1640995200000

    @pytest.mark.unit
    def test_to_internal_timestamp_iso_string(self, converter):
        """Test timestamp conversion from ISO string."""
        timestamp = converter.to_internal_timestamp("2022-01-01T00:00:00Z")
        assert timestamp == 1640995200000

    @pytest.mark.unit
    def test_to_internal_timestamp_datetime(self, converter):
        """Test timestamp conversion from datetime object."""
        dt = datetime(2022, 1, 1, 0, 0, 0, tzinfo=UTC)
        timestamp = converter.to_internal_timestamp(dt)
        assert timestamp == 1640995200000

    @pytest.mark.unit
    def test_to_internal_timestamp_invalid(self, converter):
        """Test timestamp conversion with invalid input."""
        with pytest.raises(ValueError):
            converter.to_internal_timestamp("invalid")

        with pytest.raises(ValueError):
            converter.to_internal_timestamp(None)

    # Data validation tests

    @pytest.mark.unit
    def test_validate_raw_data_valid_dict(self, converter):
        """Test validation of valid dictionary data."""
        data = {"price": "50000", "timestamp": 1640995200000}

        is_valid, error_msg = converter.validate_raw_data(data)
        assert is_valid is True
        assert error_msg == ""

    @pytest.mark.unit
    def test_validate_raw_data_valid_list(self, converter):
        """Test validation of valid list data."""
        data = [{"price": "50000", "timestamp": 1640995200000}, {"price": "50100", "timestamp": 1640995260000}]

        is_valid, error_msg = converter.validate_raw_data(data)
        assert is_valid is True
        assert error_msg == ""

    @pytest.mark.unit
    def test_validate_raw_data_invalid_empty(self, converter):
        """Test validation of empty data."""
        is_valid, error_msg = converter.validate_raw_data({})
        assert is_valid is False
        assert "cannot be empty" in error_msg

    @pytest.mark.unit
    def test_validate_raw_data_invalid_no_price(self, converter):
        """Test validation of data without price fields."""
        data = {"timestamp": 1640995200000}

        is_valid, error_msg = converter.validate_raw_data(data)
        assert is_valid is False
        assert "price information" in error_msg

    @pytest.mark.unit
    def test_validate_raw_data_invalid_no_timestamp(self, converter):
        """Test validation of data without timestamp fields."""
        data = {"price": "50000"}

        is_valid, error_msg = converter.validate_raw_data(data)
        assert is_valid is False
        assert "timestamp information" in error_msg

    @pytest.mark.unit
    def test_validate_raw_data_none(self, converter):
        """Test validation of None data."""
        is_valid, error_msg = converter.validate_raw_data(None)
        assert is_valid is False
        assert "cannot be None" in error_msg

    # Custom converter tests

    @pytest.mark.unit
    def test_custom_converter_defaults(self, custom_converter):
        """Test converter with custom defaults."""
        raw_trade = {"id": "12345", "price": "50000.00", "quantity": "0.1", "side": "buy", "timestamp": 1640995200000}

        trade = custom_converter.convert_trade(raw_trade, "BTCUSDT")

        assert trade.exchange == "test_exchange"
        assert trade.asset_class == AssetClass.TRADITIONAL

    # Error handling tests

    @pytest.mark.unit
    def test_convert_trade_error_handling(self, converter):
        """Test error handling in trade conversion."""
        raw_trade = {
            "id": "12345",
            "price": "invalid_price",
            "quantity": "0.1",
            "side": "buy",
            "timestamp": 1640995200000,
        }

        with pytest.raises(ValueError, match="Failed to convert trade data"):
            converter.convert_trade(raw_trade, "BTCUSDT")

    @pytest.mark.unit
    def test_convert_multiple_trades_error_handling(self, converter):
        """Test error handling in multiple trades conversion."""
        raw_trades = [
            {"id": "1", "price": "50000", "quantity": "0.1", "side": "buy", "timestamp": 1640995200000},
            {
                "id": "2",
                "price": "invalid_price",  # This will cause an error
                "quantity": "0.2",
                "side": "sell",
                "timestamp": 1640995260000,
            },
        ]

        with pytest.raises(ValueError, match="Failed to convert trade at index 1"):
            converter.convert_multiple_trades(raw_trades, "BTCUSDT")


@pytest.mark.unit
class TestInMemoryDataConverterEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def converter(self):
        return InMemoryDataConverter()

    @pytest.mark.unit
    def test_trade_side_variations(self, converter):
        """Test various trade side formats."""
        test_cases = [
            ("buy", TradeSide.BUY),
            ("BUY", TradeSide.BUY),
            ("Buy", TradeSide.BUY),
            ("b", TradeSide.BUY),
            ("sell", TradeSide.SELL),
            ("SELL", TradeSide.SELL),
            ("Sell", TradeSide.SELL),
            ("s", TradeSide.SELL),
        ]

        for side_input, expected_side in test_cases:
            raw_trade = {"id": "test", "price": "100", "quantity": "1", "side": side_input, "timestamp": 1640995200000}

            trade = converter.convert_trade(raw_trade, "TEST")
            assert trade.side == expected_side

    @pytest.mark.unit
    def test_kline_interval_variations(self, converter):
        """Test various kline interval formats."""
        test_cases = [
            ("1m", KlineInterval.MINUTE_1),
            ("5m", KlineInterval.MINUTE_5),
            ("1h", KlineInterval.HOUR_1),
            ("1d", KlineInterval.DAY_1),
            ("invalid", KlineInterval.MINUTE_1),  # Should default to 1m
        ]

        for interval_input, expected_interval in test_cases:
            raw_kline = {
                "open_time": 1640995200000,
                "close_time": 1640995260000,
                "open": "100",
                "high": "110",
                "low": "90",
                "close": "105",
                "volume": "1000",
                "quote_volume": "105000",
                "trades_count": 50,
                "interval": interval_input,
            }

            kline = converter.convert_kline(raw_kline, "TEST")
            assert kline.interval == expected_interval
