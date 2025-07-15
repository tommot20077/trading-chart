# ABOUTME: Contract tests for AbstractDataConverter interface
# ABOUTME: Verifies all data converter implementations comply with the interface contract

import pytest
from typing import Type, List, Any, Dict
from datetime import datetime, UTC
from decimal import Decimal

from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval, TradeSide
from ..base_contract_test import ContractTestBase


class MockDataConverter(AbstractDataConverter):
    """Mock implementation of AbstractDataConverter for contract testing."""

    def convert_trade(self, raw_trade: Dict[str, Any], symbol: str) -> Trade:
        """Convert raw trade data to Trade model."""
        if not isinstance(raw_trade, dict):
            raise TypeError("raw_trade must be a dictionary")

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        # Validate required fields
        required_fields = ["price", "quantity", "timestamp"]
        for field in required_fields:
            if field not in raw_trade:
                raise ValueError(f"Missing required field: {field}")

        # Convert and validate data types
        try:
            price = Decimal(str(raw_trade["price"]))
            quantity = Decimal(str(raw_trade["quantity"]))
            timestamp_ms = self.to_internal_timestamp(raw_trade["timestamp"])

            if price <= 0:
                raise ValueError("Price must be positive")
            if quantity <= 0:
                raise ValueError("Quantity must be positive")

            return Trade(
                symbol=symbol.upper().strip(),
                price=price,
                quantity=quantity,
                side=TradeSide(raw_trade.get("side", "buy")),
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
                trade_id=raw_trade.get("trade_id", f"mock_{timestamp_ms}"),
                is_buyer_maker=raw_trade.get("is_buyer_maker", False),
            )
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ValueError(f"Invalid trade data: {e}")

    def convert_multiple_trades(self, raw_trades: list[Dict[str, Any]], symbol: str) -> list[Trade]:
        """Convert multiple raw trade records to Trade models."""
        if not isinstance(raw_trades, list):
            raise TypeError("raw_trades must be a list")

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        trades = []
        for i, raw_trade in enumerate(raw_trades):
            try:
                trade = self.convert_trade(raw_trade, symbol)
                trades.append(trade)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error converting trade at index {i}: {e}")

        return trades

    def convert_kline(self, raw_kline: Dict[str, Any], symbol: str) -> Kline:
        """Convert raw kline data to Kline model."""
        if not isinstance(raw_kline, dict):
            raise TypeError("raw_kline must be a dictionary")

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        # Validate required fields
        required_fields = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
        for field in required_fields:
            if field not in raw_kline:
                raise ValueError(f"Missing required field: {field}")

        try:
            open_time_ms = self.to_internal_timestamp(raw_kline["open_time"])
            close_time_ms = self.to_internal_timestamp(raw_kline["close_time"])

            open_price = Decimal(str(raw_kline["open"]))
            high_price = Decimal(str(raw_kline["high"]))
            low_price = Decimal(str(raw_kline["low"]))
            close_price = Decimal(str(raw_kline["close"]))
            volume = Decimal(str(raw_kline["volume"]))

            # Validate price relationships
            if not (low_price <= open_price <= high_price):
                raise ValueError("Invalid price relationship: low <= open <= high")
            if not (low_price <= close_price <= high_price):
                raise ValueError("Invalid price relationship: low <= close <= high")
            if volume < 0:
                raise ValueError("Volume cannot be negative")

            return Kline(
                symbol=symbol.upper().strip(),
                interval=raw_kline.get("interval", KlineInterval.MINUTE_1),
                open_time=datetime.fromtimestamp(open_time_ms / 1000, tz=UTC),
                close_time=datetime.fromtimestamp(close_time_ms / 1000, tz=UTC),
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                quote_volume=Decimal(str(raw_kline.get("quote_volume", volume * close_price))),
                trades_count=raw_kline.get("trades_count", 1),
            )
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ValueError(f"Invalid kline data: {e}")

    def convert_multiple_klines(self, raw_klines: list[Dict[str, Any]], symbol: str) -> list[Kline]:
        """Convert multiple raw kline records to Kline models."""
        if not isinstance(raw_klines, list):
            raise TypeError("raw_klines must be a list")

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        klines = []
        for i, raw_kline in enumerate(raw_klines):
            try:
                kline = self.convert_kline(raw_kline, symbol)
                klines.append(kline)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error converting kline at index {i}: {e}")

        return klines

    def validate_raw_data(self, data: Any) -> tuple[bool, str]:
        """Validate raw data structure and content."""
        if data is None:
            return False, "Data cannot be None"

        if isinstance(data, dict):
            # Validate dictionary structure
            if not data:
                return False, "Data dictionary cannot be empty"

            # Check for basic required structure
            if "timestamp" not in data:
                return False, "Missing timestamp field"

            # Check if it looks like trade data
            if "price" in data and "quantity" in data:
                try:
                    price = float(data["price"])
                    quantity = float(data["quantity"])
                    if price <= 0 or quantity <= 0:
                        return False, "Price and quantity must be positive"
                except (ValueError, TypeError):
                    return False, "Invalid price or quantity format"

            # Check if it looks like kline data
            elif "open" in data and "high" in data and "low" in data and "close" in data:
                try:
                    prices = [float(data[p]) for p in ["open", "high", "low", "close"]]
                    if any(p <= 0 for p in prices):
                        return False, "All prices must be positive"
                    if not (prices[2] <= prices[0] <= prices[1] and prices[2] <= prices[3] <= prices[1]):
                        return False, "Invalid price relationships in kline data"
                except (ValueError, TypeError):
                    return False, "Invalid price format in kline data"

            return True, ""

        elif isinstance(data, list):
            if not data:
                return False, "Data list cannot be empty"

            # Validate each item in the list
            for i, item in enumerate(data):
                is_valid, error_msg = self.validate_raw_data(item)
                if not is_valid:
                    return False, f"Invalid item at index {i}: {error_msg}"

            return True, ""

        else:
            return False, f"Unsupported data type: {type(data).__name__}"

    def to_internal_timestamp(self, timestamp: Any) -> int:
        """Convert timestamp to internal format (milliseconds since epoch)."""
        if timestamp is None:
            raise ValueError("Timestamp cannot be None")

        try:
            if isinstance(timestamp, (int, float)):
                # Assume seconds if less than a reasonable millisecond threshold
                if timestamp < 1e10:  # Less than year 2001 in milliseconds
                    return int(timestamp * 1000)
                else:
                    return int(timestamp)

            elif isinstance(timestamp, str):
                # Try to parse as ISO format first
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    return int(dt.timestamp() * 1000)
                except ValueError:
                    # Try to parse as float/int string
                    num_timestamp = float(timestamp)
                    return self.to_internal_timestamp(num_timestamp)

            elif isinstance(timestamp, datetime):
                return int(timestamp.timestamp() * 1000)

            else:
                raise TypeError(f"Unsupported timestamp type: {type(timestamp).__name__}")

        except (ValueError, TypeError, OverflowError) as e:
            raise ValueError(f"Cannot convert timestamp: {e}")


class TestDataConverterContract(ContractTestBase[AbstractDataConverter]):
    """Contract tests for AbstractDataConverter interface."""

    @property
    def interface_class(self) -> Type[AbstractDataConverter]:
        return AbstractDataConverter

    @property
    def implementations(self) -> List[Type[AbstractDataConverter]]:
        return [
            MockDataConverter,
            # TODO: Add actual implementations when they exist
            # BinanceDataConverter,
            # CoinbaseDataConverter,
        ]

    @pytest.mark.contract
    def test_convert_trade_method_signature(self):
        """Test convert_trade method has correct signature and behavior."""
        method = getattr(self.interface_class, "convert_trade")
        assert hasattr(method, "__isabstractmethod__")

        # Test with mock implementation
        converter = MockDataConverter()

        # Valid trade data should return Trade object
        raw_trade = {
            "price": "50000.50",
            "quantity": "1.5",
            "timestamp": 1640995200,
            "trade_id": "12345",
            "side": "buy",
            "is_buyer_maker": True,
        }
        trade = converter.convert_trade(raw_trade, "BTCUSDT")
        assert isinstance(trade, Trade)
        assert trade.symbol == "BTCUSDT"
        assert trade.price == Decimal("50000.50")
        assert trade.quantity == Decimal("1.5")

    @pytest.mark.contract
    def test_convert_trade_contract_validation(self):
        """Test convert_trade input validation contract."""
        converter = MockDataConverter()

        # Test with invalid input types
        with pytest.raises(TypeError):
            converter.convert_trade("not_a_dict", "BTCUSDT")

        with pytest.raises(TypeError):
            converter.convert_trade(None, "BTCUSDT")

        # Test with invalid symbol
        raw_trade = {"price": "100", "quantity": "1", "timestamp": 1640995200, "side": "buy"}

        with pytest.raises(ValueError):
            converter.convert_trade(raw_trade, "")

        with pytest.raises(ValueError):
            converter.convert_trade(raw_trade, "   ")

        # Test with missing required fields
        with pytest.raises(ValueError):
            converter.convert_trade({}, "BTCUSDT")

        with pytest.raises(ValueError):
            converter.convert_trade({"price": "100"}, "BTCUSDT")  # Missing quantity, timestamp

    @pytest.mark.contract
    def test_convert_multiple_trades_method_signature(self):
        """Test convert_multiple_trades method has correct signature and behavior."""
        method = getattr(self.interface_class, "convert_multiple_trades")
        assert hasattr(method, "__isabstractmethod__")

        converter = MockDataConverter()

        # Valid trades data should return list of Trade objects
        raw_trades = [
            {"price": "50000", "quantity": "1", "timestamp": 1640995200, "side": "buy"},
            {"price": "50100", "quantity": "0.5", "timestamp": 1640995260, "side": "sell"},
        ]
        trades = converter.convert_multiple_trades(raw_trades, "BTCUSDT")
        assert isinstance(trades, list)
        assert len(trades) == 2
        assert all(isinstance(trade, Trade) for trade in trades)

    @pytest.mark.contract
    def test_convert_multiple_trades_contract_validation(self):
        """Test convert_multiple_trades input validation contract."""
        converter = MockDataConverter()

        # Test with invalid input types
        with pytest.raises(TypeError):
            converter.convert_multiple_trades("not_a_list", "BTCUSDT")

        with pytest.raises(TypeError):
            converter.convert_multiple_trades(None, "BTCUSDT")

        # Test with invalid symbol
        with pytest.raises(ValueError):
            converter.convert_multiple_trades([], "")

        # Test with invalid trade in list
        raw_trades = [
            {"price": "50000", "quantity": "1", "timestamp": 1640995200, "side": "buy"},
            {"invalid": "data"},  # Missing required fields
        ]
        with pytest.raises(ValueError):
            converter.convert_multiple_trades(raw_trades, "BTCUSDT")

    @pytest.mark.contract
    def test_convert_kline_method_signature(self):
        """Test convert_kline method has correct signature and behavior."""
        method = getattr(self.interface_class, "convert_kline")
        assert hasattr(method, "__isabstractmethod__")

        converter = MockDataConverter()

        # Valid kline data should return Kline object
        raw_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "50000",
            "high": "50100",
            "low": "49900",
            "close": "50050",
            "volume": "10.5",
            "quote_volume": "525000",
            "trades_count": 100,
        }
        kline = converter.convert_kline(raw_kline, "BTCUSDT")
        assert isinstance(kline, Kline)
        assert kline.symbol == "BTCUSDT"
        assert kline.open_price == Decimal("50000")
        assert kline.high_price == Decimal("50100")

    @pytest.mark.contract
    def test_convert_kline_contract_validation(self):
        """Test convert_kline input validation contract."""
        converter = MockDataConverter()

        # Test with invalid input types
        with pytest.raises(TypeError):
            converter.convert_kline("not_a_dict", "BTCUSDT")

        # Test with invalid symbol
        raw_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "50000",
            "high": "50100",
            "low": "49900",
            "close": "50050",
            "volume": "10",
        }

        with pytest.raises(ValueError):
            converter.convert_kline(raw_kline, "")

        # Test with missing required fields
        with pytest.raises(ValueError):
            converter.convert_kline({}, "BTCUSDT")

        # Test with invalid price relationships
        invalid_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "50000",
            "high": "49000",  # High less than open - invalid
            "low": "49900",
            "close": "50050",
            "volume": "10",
        }
        with pytest.raises(ValueError):
            converter.convert_kline(invalid_kline, "BTCUSDT")

    @pytest.mark.contract
    def test_convert_multiple_klines_method_signature(self):
        """Test convert_multiple_klines method has correct signature and behavior."""
        method = getattr(self.interface_class, "convert_multiple_klines")
        assert hasattr(method, "__isabstractmethod__")

        converter = MockDataConverter()

        # Valid klines data should return list of Kline objects
        raw_klines = [
            {
                "open_time": 1640995200000,
                "close_time": 1640995260000,
                "open": "50000",
                "high": "50100",
                "low": "49900",
                "close": "50050",
                "volume": "10",
            },
            {
                "open_time": 1640995260000,
                "close_time": 1640995320000,
                "open": "50050",
                "high": "50200",
                "low": "50000",
                "close": "50150",
                "volume": "8",
            },
        ]
        klines = converter.convert_multiple_klines(raw_klines, "BTCUSDT")
        assert isinstance(klines, list)
        assert len(klines) == 2
        assert all(isinstance(kline, Kline) for kline in klines)

    @pytest.mark.contract
    def test_validate_raw_data_method_signature(self):
        """Test validate_raw_data method has correct signature and behavior."""
        method = getattr(self.interface_class, "validate_raw_data")
        assert hasattr(method, "__isabstractmethod__")

        converter = MockDataConverter()

        # Valid data should return (True, "")
        valid_trade_data = {"price": "100", "quantity": "1", "timestamp": 1640995200}
        is_valid, error_msg = converter.validate_raw_data(valid_trade_data)
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)
        assert is_valid is True
        assert error_msg == ""

        # Invalid data should return (False, error_message)
        invalid_data = None
        is_valid, error_msg = converter.validate_raw_data(invalid_data)
        assert is_valid is False
        assert len(error_msg) > 0

    @pytest.mark.contract
    def test_validate_raw_data_contract_behavior(self):
        """Test validate_raw_data contract behavior with various inputs."""
        converter = MockDataConverter()

        # Test with valid trade data
        valid_trade = {"price": "100", "quantity": "1", "timestamp": 1640995200}
        is_valid, error_msg = converter.validate_raw_data(valid_trade)
        assert is_valid is True

        # Test with valid kline data
        valid_kline = {"open": "100", "high": "110", "low": "90", "close": "105", "timestamp": 1640995200}
        is_valid, error_msg = converter.validate_raw_data(valid_kline)
        assert is_valid is True

        # Test with invalid data types
        is_valid, error_msg = converter.validate_raw_data("invalid")
        assert is_valid is False
        assert "Unsupported data type" in error_msg

        # Test with empty data
        is_valid, error_msg = converter.validate_raw_data({})
        assert is_valid is False
        assert "empty" in error_msg.lower()

    @pytest.mark.contract
    def test_to_internal_timestamp_method_signature(self):
        """Test to_internal_timestamp method has correct signature and behavior."""
        method = getattr(self.interface_class, "to_internal_timestamp")
        assert hasattr(method, "__isabstractmethod__")

        converter = MockDataConverter()

        # Should return integer timestamp in milliseconds
        timestamp_ms = converter.to_internal_timestamp(1640995200)  # Unix seconds
        assert isinstance(timestamp_ms, int)
        assert timestamp_ms == 1640995200000  # Should be converted to milliseconds

        # Test with milliseconds input
        timestamp_ms = converter.to_internal_timestamp(1640995200000)
        assert timestamp_ms == 1640995200000  # Should remain the same

    @pytest.mark.contract
    def test_to_internal_timestamp_contract_validation(self):
        """Test to_internal_timestamp input validation and conversion contract."""
        converter = MockDataConverter()

        # Test with various valid formats
        test_cases = [
            (1640995200, 1640995200000),  # Unix seconds
            (1640995200000, 1640995200000),  # Unix milliseconds
            (1640995200.5, 1640995200500),  # Unix seconds with fraction
            ("1640995200", 1640995200000),  # String seconds
            ("1640995200000", 1640995200000),  # String milliseconds
        ]

        for input_ts, expected_ms in test_cases:
            result = converter.to_internal_timestamp(input_ts)
            assert result == expected_ms, f"Failed for input {input_ts}"

        # Test with invalid inputs
        invalid_inputs = [None, "invalid", [], {}, object()]
        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                converter.to_internal_timestamp(invalid_input)

    @pytest.mark.contract
    def test_data_conversion_consistency_contract(self):
        """Test consistency between single and batch conversion methods."""
        converter = MockDataConverter()

        # Test trade conversion consistency
        raw_trade = {"price": "100", "quantity": "1", "timestamp": 1640995200, "side": "buy"}
        single_trade = converter.convert_trade(raw_trade, "BTCUSDT")
        batch_trades = converter.convert_multiple_trades([raw_trade], "BTCUSDT")

        assert len(batch_trades) == 1
        assert batch_trades[0].symbol == single_trade.symbol
        assert batch_trades[0].price == single_trade.price
        assert batch_trades[0].quantity == single_trade.quantity

        # Test kline conversion consistency
        raw_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "100",
            "high": "110",
            "low": "90",
            "close": "105",
            "volume": "10",
        }
        single_kline = converter.convert_kline(raw_kline, "BTCUSDT")
        batch_klines = converter.convert_multiple_klines([raw_kline], "BTCUSDT")

        assert len(batch_klines) == 1
        assert batch_klines[0].symbol == single_kline.symbol
        assert batch_klines[0].open_price == single_kline.open_price
        assert batch_klines[0].close_price == single_kline.close_price

    @pytest.mark.contract
    def test_error_handling_contract(self):
        """Test error handling contract across all methods."""
        converter = MockDataConverter()

        # All methods should raise appropriate exceptions for invalid inputs
        methods_to_test = [
            ("convert_trade", [None, "BTCUSDT"]),
            ("convert_multiple_trades", [None, "BTCUSDT"]),
            ("convert_kline", [None, "BTCUSDT"]),
            ("convert_multiple_klines", [None, "BTCUSDT"]),
            ("to_internal_timestamp", [None]),
        ]

        for method_name, args in methods_to_test:
            method = getattr(converter, method_name)
            with pytest.raises((ValueError, TypeError)):
                method(*args)

    @pytest.mark.contract
    def test_symbol_normalization_contract(self):
        """Test symbol normalization contract across conversion methods."""
        converter = MockDataConverter()

        # Test trade conversion with various symbol formats
        raw_trade = {"price": "100", "quantity": "1", "timestamp": 1640995200, "side": "buy"}

        test_symbols = [
            ("btcusdt", "BTCUSDT"),
            ("  ETHUSDT  ", "ETHUSDT"),
            ("BTC-USD", "BTC-USD"),
        ]

        for input_symbol, expected_symbol in test_symbols:
            trade = converter.convert_trade(raw_trade, input_symbol)
            assert trade.symbol == expected_symbol

        # Test kline conversion with symbol normalization
        raw_kline = {
            "open_time": 1640995200000,
            "close_time": 1640995260000,
            "open": "100",
            "high": "110",
            "low": "90",
            "close": "105",
            "volume": "10",
        }

        for input_symbol, expected_symbol in test_symbols:
            kline = converter.convert_kline(raw_kline, input_symbol)
            assert kline.symbol == expected_symbol
