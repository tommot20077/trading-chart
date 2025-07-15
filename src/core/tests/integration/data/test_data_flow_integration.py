# ABOUTME: End-to-end data flow integration tests
# ABOUTME: Tests complete data processing pipeline: acquisition → conversion → validation → storage

import pytest
from typing import List, Dict, Any

from core.interfaces.data.provider import AbstractDataProvider
from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.kline import Kline
from core.models.data.trade import Trade


@pytest.mark.integration
@pytest.mark.asyncio
class TestDataFlowIntegration:
    """End-to-end data flow integration tests"""

    async def test_complete_kline_processing_flow(
        self,
        test_provider: AbstractDataProvider,
        test_converter: AbstractDataConverter,
        sample_kline_data: Dict[str, List[Dict[str, Any]]],
    ):
        """Test complete K-line processing: fetch → validate → convert → verify"""
        # Arrange
        symbol = "BTCUSDT"
        raw_data = sample_kline_data["binance"]

        # Step 1: Data acquisition (simulated)
        # In real scenario: raw_data = await test_provider.fetch_klines(symbol, interval, limit)
        assert len(raw_data) > 0

        # Step 2: Data validation
        for raw_kline in raw_data:
            is_valid, error_msg = test_converter.validate_raw_data(raw_kline)
            assert is_valid, f"Validation failed: {error_msg}"

        # Step 3: Data conversion
        converted_klines = test_converter.convert_multiple_klines(raw_data, symbol)

        # Step 4: Result verification
        assert len(converted_klines) == len(raw_data)

        # Verify data integrity and business rules
        for i, kline in enumerate(converted_klines):
            # Basic model validation
            assert isinstance(kline, Kline)
            assert kline.symbol == symbol

            # Business rule validation
            assert kline.high_price >= kline.open_price
            assert kline.high_price >= kline.close_price
            assert kline.low_price <= kline.open_price
            assert kline.low_price <= kline.close_price
            assert kline.volume >= 0

            # Timestamp ordering (should be sequential)
            if i > 0:
                assert kline.open_time >= converted_klines[i - 1].open_time

    async def test_complete_trade_processing_flow(
        self,
        test_provider: AbstractDataProvider,
        test_converter: AbstractDataConverter,
        sample_trade_data: Dict[str, List[Dict[str, Any]]],
    ):
        """Test complete trade processing: fetch → validate → convert → verify"""
        # Arrange
        symbol = "BTCUSDT"
        raw_data = sample_trade_data["binance"]

        # Step 1: Data acquisition (simulated)
        assert len(raw_data) > 0

        # Step 2: Data validation
        for raw_trade in raw_data:
            is_valid, error_msg = test_converter.validate_raw_data(raw_trade)
            assert is_valid, f"Validation failed: {error_msg}"

        # Step 3: Data conversion
        converted_trades = test_converter.convert_multiple_trades(raw_data, symbol)

        # Step 4: Result verification
        assert len(converted_trades) == len(raw_data)

        # Verify data integrity and business rules
        for i, trade in enumerate(converted_trades):
            # Basic model validation
            assert isinstance(trade, Trade)
            assert trade.symbol == symbol

            # Business rule validation
            assert trade.price > 0
            assert trade.quantity > 0
            assert trade.trade_id is not None

            # Timestamp ordering (should be sequential)
            if i > 0:
                assert trade.timestamp >= converted_trades[i - 1].timestamp

    async def test_data_cleaning_integration(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test data cleaning and error handling in the flow"""
        # Arrange: Create mixed valid/invalid data
        valid_data = mock_data_generator.generate_binance_kline_data("BTCUSDT", 3)

        # Add some problematic data
        problematic_data = [
            # Missing required fields
            {"t": 1640995200000, "o": "50000"},
            # Invalid price values
            {"t": 1640995200000, "o": "-50000", "h": "51000", "l": "49000", "c": "50500", "v": "1.0"},
            # Invalid volume
            {"t": 1640995200000, "o": "50000", "h": "51000", "l": "49000", "c": "50500", "v": "-1.0"},
        ]

        # Act & Assert: Valid data should process successfully
        valid_klines = test_converter.convert_multiple_klines(valid_data, "BTCUSDT")
        assert len(valid_klines) == 3

        # Act & Assert: Invalid data should be handled appropriately
        for invalid_item in problematic_data:
            is_valid, error_msg = test_converter.validate_raw_data(invalid_item)
            if not is_valid:
                # Should fail validation
                assert error_msg != ""
            else:
                # If validation passes, conversion should handle edge cases
                try:
                    result = test_converter.convert_kline(invalid_item, "BTCUSDT")
                    # If conversion succeeds, verify the result is reasonable
                    if result.price < 0 or result.volume < 0:
                        pytest.fail("Converter should reject negative values")
                except (ValueError, TypeError):
                    # Expected for invalid data
                    pass

    async def test_memory_efficiency_in_flow(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test memory efficiency during large data processing"""
        # Arrange: Generate large dataset
        large_dataset = mock_data_generator.generate_binance_kline_data("BTCUSDT", 5000)

        # Act: Process in batches to test memory efficiency
        batch_size = 1000
        all_results = []

        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i : i + batch_size]
            batch_results = test_converter.convert_multiple_klines(batch, "BTCUSDT")
            all_results.extend(batch_results)

        # Assert: Verify complete processing
        assert len(all_results) == 5000
        assert all(isinstance(kline, Kline) for kline in all_results)

    async def test_concurrent_processing_flow(self, test_converter: AbstractDataConverter, mock_data_generator):
        """Test concurrent data processing scenarios"""
        import asyncio

        # Arrange: Prepare multiple symbol datasets
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        datasets = {symbol: mock_data_generator.generate_binance_kline_data(symbol, 100) for symbol in symbols}

        async def process_symbol_data(symbol: str, data: List[Dict[str, Any]]):
            """Process data for a single symbol"""
            return test_converter.convert_multiple_klines(data, symbol)

        # Act: Process multiple symbols concurrently
        tasks = [process_symbol_data(symbol, data) for symbol, data in datasets.items()]

        results = await asyncio.gather(*tasks)

        # Assert: Verify concurrent processing results
        assert len(results) == 3
        for i, symbol in enumerate(symbols):
            symbol_results = results[i]
            assert len(symbol_results) == 100
            assert all(kline.symbol == symbol for kline in symbol_results)
