# ABOUTME: Integration tests for DataProvider + DataConverter collaboration
# ABOUTME: Tests the complete data flow from provider acquisition to converter transformation

import pytest
from typing import List, Dict, Any
from decimal import Decimal

from core.interfaces.data.provider import AbstractDataProvider
from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.kline import Kline
from core.models.data.trade import Trade


@pytest.mark.integration
@pytest.mark.asyncio
class TestProviderConverterIntegration:
    """Integration tests for DataProvider + DataConverter workflow"""

    async def test_provider_converter_kline_flow(
        self,
        test_provider: AbstractDataProvider,
        test_converter: AbstractDataConverter,
        sample_kline_data: Dict[str, List[Dict[str, Any]]],
    ):
        """Test complete K-line data flow: Provider → Converter → Kline models"""
        # Arrange
        symbol = "BTCUSDT"
        raw_klines = sample_kline_data["binance"]

        # Simulate provider returning raw data
        # Note: In real integration, provider would fetch from external source

        # Act: Convert raw data using converter
        converted_klines = test_converter.convert_multiple_klines(raw_klines, symbol)

        # Assert: Verify conversion results
        assert len(converted_klines) == len(raw_klines)
        assert all(isinstance(kline, Kline) for kline in converted_klines)

        # Verify first kline data integrity
        first_kline = converted_klines[0]
        first_raw = raw_klines[0]

        assert first_kline.symbol == symbol
        assert first_kline.open_price == Decimal(first_raw["o"])
        assert first_kline.close_price == Decimal(first_raw["c"])
        assert first_kline.high_price == Decimal(first_raw["h"])
        assert first_kline.low_price == Decimal(first_raw["l"])
        assert first_kline.volume == Decimal(first_raw["v"])

    async def test_provider_converter_trade_flow(
        self,
        test_provider: AbstractDataProvider,
        test_converter: AbstractDataConverter,
        sample_trade_data: Dict[str, List[Dict[str, Any]]],
    ):
        """Test complete trade data flow: Provider → Converter → Trade models"""
        # Arrange
        symbol = "BTCUSDT"
        raw_trades = sample_trade_data["binance"]

        # Act: Convert raw data using converter
        converted_trades = test_converter.convert_multiple_trades(raw_trades, symbol)

        # Assert: Verify conversion results
        assert len(converted_trades) == len(raw_trades)
        assert all(isinstance(trade, Trade) for trade in converted_trades)

        # Verify first trade data integrity
        first_trade = converted_trades[0]
        first_raw = raw_trades[0]

        assert first_trade.symbol == symbol
        assert first_trade.price == Decimal(first_raw["price"])
        assert first_trade.quantity == Decimal(first_raw["qty"])
        assert first_trade.trade_id == str(first_raw["id"])

    async def test_data_validation_integration(
        self, test_converter: AbstractDataConverter, sample_kline_data: Dict[str, List[Dict[str, Any]]]
    ):
        """Test data validation before conversion"""
        # Arrange
        valid_data = sample_kline_data["binance"][0]
        invalid_data = {"invalid": "data"}

        # Act & Assert: Valid data should pass validation
        is_valid, error_msg = test_converter.validate_raw_data(valid_data)
        assert is_valid is True
        assert error_msg == ""

        # Act & Assert: Invalid data should fail validation
        is_valid, error_msg = test_converter.validate_raw_data(invalid_data)
        assert is_valid is False
        assert error_msg != ""

    async def test_timestamp_conversion_integration(self, test_converter: AbstractDataConverter):
        """Test timestamp conversion consistency"""
        # Arrange
        test_timestamps = [
            1640995200000,  # Unix milliseconds
            1640995200,  # Unix seconds
            "1640995200000",  # String milliseconds
            "1640995200",  # String seconds
        ]

        # Act & Assert
        for timestamp in test_timestamps:
            converted = test_converter.to_internal_timestamp(timestamp)
            assert isinstance(converted, int)
            assert converted > 0
            # Should be in milliseconds (13 digits for recent timestamps)
            assert len(str(converted)) >= 13

    async def test_multiple_exchange_format_support(
        self, test_converter: AbstractDataConverter, sample_kline_data: Dict[str, List[Dict[str, Any]]]
    ):
        """Test converter handles different exchange formats"""
        # Test Binance format
        binance_data = sample_kline_data["binance"]
        binance_klines = test_converter.convert_multiple_klines(binance_data, "BTCUSDT")
        assert len(binance_klines) > 0

        # Note: OKX format would need specific converter implementation
        # This test demonstrates the pattern for multi-exchange support

    async def test_error_handling_integration(self, test_converter: AbstractDataConverter):
        """Test error handling in provider-converter integration"""
        # Test malformed data handling
        malformed_kline = {"t": "invalid_timestamp", "o": "not_a_number", "h": None, "l": "", "c": "50000", "v": "1.0"}

        with pytest.raises((ValueError, TypeError)):
            test_converter.convert_kline(malformed_kline, "BTCUSDT")

    async def test_batch_processing_performance(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test batch processing performance with larger datasets"""
        # Arrange: Generate larger dataset
        large_dataset = mock_data_generator.generate_binance_kline_data("BTCUSDT", 1000)

        # Act: Process batch
        converted_klines = test_converter.convert_multiple_klines(large_dataset, "BTCUSDT")

        # Assert: Verify batch processing
        assert len(converted_klines) == 1000
        assert all(isinstance(kline, Kline) for kline in converted_klines)

        # Verify data integrity for random samples
        for i in [0, 500, 999]:
            assert converted_klines[i].symbol == "BTCUSDT"
            assert converted_klines[i].open_price > 0
            assert converted_klines[i].volume >= 0
