# ABOUTME: Integration tests for data validation rules and business logic integration
# ABOUTME: Tests comprehensive data validation across providers, converters, and business rules

import pytest
import pytest_asyncio
import asyncio
from typing import List, Dict, Any
from decimal import Decimal
from datetime import datetime
import re

from core.interfaces.data.provider import AbstractDataProvider
from core.interfaces.data.converter import AbstractDataConverter
from core.implementations.memory.data.data_provider import MemoryDataProvider
from core.implementations.memory.data.data_converter import InMemoryDataConverter
from core.models.data.kline import Kline
from core.models.data.trade import Trade


class DataValidationRules:
    """Comprehensive data validation rules for trading data"""

    @staticmethod
    def validate_price_data(price: Decimal, field_name: str = "price") -> tuple[bool, str]:
        """Validate price data according to business rules"""
        if price <= 0:
            return False, f"{field_name} must be positive, got {price}"

        if price > Decimal("1000000"):  # Max price limit
            return False, f"{field_name} exceeds maximum limit of 1,000,000, got {price}"

        # Check decimal precision (max 8 decimal places)
        if price.as_tuple().exponent < -8:
            return False, f"{field_name} has too many decimal places (max 8), got {price}"

        return True, ""

    @staticmethod
    def validate_volume_data(volume: Decimal, field_name: str = "volume") -> tuple[bool, str]:
        """Validate volume data according to business rules"""
        if volume < 0:
            return False, f"{field_name} cannot be negative, got {volume}"

        if volume > Decimal("1000000000"):  # Max volume limit
            return False, f"{field_name} exceeds maximum limit of 1B, got {volume}"

        return True, ""

    @staticmethod
    def validate_kline_ohlc_consistency(kline: Kline) -> tuple[bool, str]:
        """Validate OHLC price consistency in K-line data"""
        # High should be the highest price
        if kline.high_price < kline.open_price:
            return False, f"High price {kline.high_price} < Open price {kline.open_price}"

        if kline.high_price < kline.close_price:
            return False, f"High price {kline.high_price} < Close price {kline.close_price}"

        if kline.high_price < kline.low_price:
            return False, f"High price {kline.high_price} < Low price {kline.low_price}"

        # Low should be the lowest price
        if kline.low_price > kline.open_price:
            return False, f"Low price {kline.low_price} > Open price {kline.open_price}"

        if kline.low_price > kline.close_price:
            return False, f"Low price {kline.low_price} > Close price {kline.close_price}"

        return True, ""

    @staticmethod
    def validate_timestamp_sequence(timestamps: List[datetime]) -> tuple[bool, str]:
        """Validate timestamp sequence is chronological"""
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                return False, f"Timestamp sequence broken at index {i}: {timestamps[i]} < {timestamps[i - 1]}"

        return True, ""

    @staticmethod
    def validate_symbol_format(symbol: str) -> tuple[bool, str]:
        """Validate trading symbol format"""
        if not symbol:
            return False, "Symbol cannot be empty"

        # Check for valid format (e.g., BTCUSDT, ETHUSDT)
        if not re.match(r"^[A-Z]{3,10}USDT?$", symbol):
            return False, f"Invalid symbol format: {symbol}. Expected format: XXXUSDT"

        if len(symbol) < 6 or len(symbol) > 12:
            return False, f"Symbol length must be between 6-12 characters, got {len(symbol)}"

        return True, ""

    @staticmethod
    def validate_trade_consistency(trade: Trade) -> tuple[bool, str]:
        """Validate trade data consistency"""
        # Validate price
        is_valid, error = DataValidationRules.validate_price_data(trade.price, "trade price")
        if not is_valid:
            return False, error

        # Validate quantity
        is_valid, error = DataValidationRules.validate_volume_data(trade.quantity, "trade quantity")
        if not is_valid:
            return False, error

        # Validate symbol
        is_valid, error = DataValidationRules.validate_symbol_format(trade.symbol)
        if not is_valid:
            return False, error

        # Validate trade ID
        if not trade.trade_id or len(trade.trade_id.strip()) == 0:
            return False, "Trade ID cannot be empty"

        return True, ""


class ValidationIntegrationProcessor:
    """Processor that integrates validation rules with data processing"""

    def __init__(self, provider: AbstractDataProvider, converter: AbstractDataConverter):
        self.provider = provider
        self.converter = converter
        self.validation_rules = DataValidationRules()
        self.validation_errors = []
        self.processed_count = 0
        self.rejected_count = 0

    async def process_and_validate_klines(
        self, raw_klines: List[Dict[str, Any]], symbol: str, strict_mode: bool = True
    ) -> tuple[List[Kline], List[str]]:
        """Process and validate K-line data with integrated validation"""
        validated_klines = []
        validation_errors = []

        for i, raw_kline in enumerate(raw_klines):
            try:
                # Step 1: Pre-process data for non-strict mode
                original_raw_kline = raw_kline.copy()
                fixed_issues = []

                if not strict_mode:
                    # Try to fix common data issues in non-strict mode
                    raw_kline, fixed_issues = self._fix_kline_data_with_tracking(raw_kline.copy())

                # Step 2: Convert raw data
                kline = self.converter.convert_kline(raw_kline, symbol)
                self.processed_count += 1

                # Step 3: Apply validation rules
                validation_results = []

                # Record any issues that were fixed in non-strict mode
                if fixed_issues:
                    validation_results.extend(fixed_issues)

                # Validate symbol format
                is_valid, error = self.validation_rules.validate_symbol_format(kline.symbol)
                if not is_valid:
                    validation_results.append(f"Symbol validation failed: {error}")

                # Validate OHLC consistency
                is_valid, error = self.validation_rules.validate_kline_ohlc_consistency(kline)
                if not is_valid:
                    validation_results.append(f"OHLC consistency failed: {error}")

                # Validate individual price components
                for price_field, price_value in [
                    ("open_price", kline.open_price),
                    ("high_price", kline.high_price),
                    ("low_price", kline.low_price),
                    ("close_price", kline.close_price),
                ]:
                    is_valid, error = self.validation_rules.validate_price_data(price_value, price_field)
                    if not is_valid:
                        validation_results.append(f"Price validation failed: {error}")

                # Validate volume data
                is_valid, error = self.validation_rules.validate_volume_data(kline.volume)
                if not is_valid:
                    validation_results.append(f"Volume validation failed: {error}")

                # Step 3: Handle validation results
                if validation_results:
                    error_msg = f"Kline {i} validation errors: {'; '.join(validation_results)}"
                    validation_errors.append(error_msg)
                    self.rejected_count += 1

                    if strict_mode:
                        continue  # Skip invalid data in strict mode

                validated_klines.append(kline)

            except Exception as e:
                error_msg = f"Kline {i} processing failed: {str(e)}"

                # Check if this is a Pydantic validation error and extract meaningful info
                if "High price cannot be less than low price" in str(e):
                    error_msg = f"Kline {i} validation errors: OHLC consistency failed: High price cannot be less than low price"
                elif "Volume must be greater than or equal to 0" in str(e):
                    error_msg = f"Kline {i} validation errors: Volume validation failed: Volume must be greater than or equal to 0"
                elif "price\n  Input should be greater than 0" in str(e):
                    error_msg = f"Kline {i} validation errors: Price validation failed: Price must be positive"

                validation_errors.append(error_msg)
                self.rejected_count += 1

                if strict_mode:
                    continue

        return validated_klines, validation_errors

    async def process_and_validate_trades(
        self, raw_trades: List[Dict[str, Any]], symbol: str, strict_mode: bool = True
    ) -> tuple[List[Trade], List[str]]:
        """Process and validate trade data with integrated validation"""
        validated_trades = []
        validation_errors = []

        for i, raw_trade in enumerate(raw_trades):
            try:
                # Step 1: Convert raw data
                trade = self.converter.convert_trade(raw_trade, symbol)
                self.processed_count += 1

                # Step 2: Apply validation rules
                is_valid, error = self.validation_rules.validate_trade_consistency(trade)

                if not is_valid:
                    error_msg = f"Trade {i} validation failed: {error}"
                    validation_errors.append(error_msg)
                    self.rejected_count += 1

                    if strict_mode:
                        continue

                validated_trades.append(trade)

            except Exception as e:
                error_msg = f"Trade {i} processing failed: {str(e)}"

                # Check if this is a Pydantic validation error and extract meaningful info
                if "Price must be greater than 0" in str(e):
                    error_msg = f"Trade {i} validation errors: Price validation failed: Price must be greater than 0"
                elif "quantity\n  Input should be greater than or equal to 0" in str(e):
                    error_msg = f"Trade {i} validation errors: Volume validation failed: Volume must be greater than or equal to 0"

                validation_errors.append(error_msg)
                self.rejected_count += 1

                if strict_mode:
                    continue

        return validated_trades, validation_errors

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing and validation statistics"""
        total_processed = self.processed_count + self.rejected_count
        success_rate = (self.processed_count / total_processed * 100) if total_processed > 0 else 0

        return {
            "total_processed": total_processed,
            "successful": self.processed_count,
            "rejected": self.rejected_count,
            "success_rate": round(success_rate, 2),
            "validation_errors": len(self.validation_errors),
        }

    def _fix_kline_data_with_tracking(self, raw_kline: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Fix common data issues in non-strict mode and track what was fixed"""
        from decimal import Decimal

        fixed_issues = []

        # Fix OHLC consistency issues
        if "o" in raw_kline and "h" in raw_kline and "l" in raw_kline and "c" in raw_kline:
            try:
                o = Decimal(str(raw_kline["o"]))
                h = Decimal(str(raw_kline["h"]))
                l = Decimal(str(raw_kline["l"]))
                c = Decimal(str(raw_kline["c"]))

                # Fix high/low consistency
                if h < l:
                    # Swap high and low
                    raw_kline["h"] = str(l)
                    raw_kline["l"] = str(h)
                    h, l = l, h
                    fixed_issues.append("OHLC consistency failed: High price cannot be less than low price (fixed)")

                # Ensure high is at least as high as open and close
                if h < o:
                    raw_kline["h"] = str(o)
                    h = o
                    fixed_issues.append("OHLC consistency failed: High price cannot be less than open price (fixed)")
                if h < c:
                    raw_kline["h"] = str(c)
                    h = c
                    fixed_issues.append("OHLC consistency failed: High price cannot be less than close price (fixed)")

                # Ensure low is at most as low as open and close
                if l > o:
                    raw_kline["l"] = str(o)
                    l = o
                    fixed_issues.append("OHLC consistency failed: Low price cannot be greater than open price (fixed)")
                if l > c:
                    raw_kline["l"] = str(c)
                    l = c
                    fixed_issues.append("OHLC consistency failed: Low price cannot be greater than close price (fixed)")

            except (ValueError, TypeError):
                pass

        # Fix negative volume
        if "v" in raw_kline:
            try:
                volume = Decimal(str(raw_kline["v"]))
                if volume < 0:
                    raw_kline["v"] = str(abs(volume))
                    fixed_issues.append("Volume validation failed: Volume cannot be negative (fixed)")
            except (ValueError, TypeError):
                pass

        return raw_kline, fixed_issues


@pytest.mark.integration
@pytest.mark.asyncio
class TestDataValidationIntegration:
    """Integration tests for data validation rules and business logic"""

    @pytest_asyncio.fixture
    async def test_provider(self):
        """Test data provider"""
        provider = MemoryDataProvider(name="ValidationTestProvider")
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    @pytest.mark.unit
    @pytest.mark.integration
    def test_converter(self):
        """Test data converter"""
        return InMemoryDataConverter()

    @pytest_asyncio.fixture
    def validation_processor(self, test_provider, test_converter):
        """Validation integration processor"""
        return ValidationIntegrationProcessor(test_provider, test_converter)

    @pytest_asyncio.fixture
    def valid_kline_data(self):
        """Valid K-line data for testing"""
        return [
            {
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "1.5",
                "quoteVolume": "75075.0",
                "n": 50,
                "x": True,
            },
            {
                "t": 1640995260000,
                "T": 1640995319999,
                "o": "50050.00",
                "h": "50150.00",
                "l": "49950.00",
                "c": "50100.00",
                "v": "2.0",
                "quoteVolume": "100200.0",
                "n": 75,
                "x": True,
            },
        ]

    @pytest_asyncio.fixture
    def invalid_kline_data(self):
        """Invalid K-line data for testing validation"""
        return [
            # Invalid OHLC: High < Low
            {
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "50000.00",
                "h": "49000.00",  # High < Open (invalid)
                "l": "51000.00",  # Low > Open (invalid)
                "c": "50050.00",
                "v": "1.5",
                "quoteVolume": "75075.0",
                "n": 50,
                "x": True,
            },
            # Negative volume
            {
                "t": 1640995260000,
                "T": 1640995319999,
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "-1.5",  # Negative volume (invalid)
                "quoteVolume": "75075.0",
                "n": 50,
                "x": True,
            },
        ]

    async def test_valid_data_validation_integration(self, validation_processor, valid_kline_data):
        """Test validation integration with valid data"""
        # Act
        validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
            valid_kline_data, "BTCUSDT", strict_mode=True
        )

        # Assert
        assert len(validated_klines) == 2
        assert len(validation_errors) == 0

        # Verify all klines passed validation
        for kline in validated_klines:
            assert kline.symbol == "BTCUSDT"
            assert kline.high_price >= kline.open_price
            assert kline.high_price >= kline.close_price
            assert kline.low_price <= kline.open_price
            assert kline.low_price <= kline.close_price
            assert kline.volume >= 0

        # Check processing stats
        stats = validation_processor.get_processing_stats()
        assert stats["successful"] == 2
        assert stats["rejected"] == 0
        assert stats["success_rate"] == 100.0

    async def test_invalid_data_validation_integration(self, validation_processor, invalid_kline_data):
        """Test validation integration with invalid data"""
        # Act
        validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
            invalid_kline_data, "BTCUSDT", strict_mode=True
        )

        # Assert
        assert len(validated_klines) == 0  # All data should be rejected in strict mode
        assert len(validation_errors) == 2  # Both records should have validation errors

        # Verify specific validation errors
        assert "OHLC consistency failed" in validation_errors[0]
        assert "Volume validation failed" in validation_errors[1]

        # Check processing stats
        stats = validation_processor.get_processing_stats()
        assert stats["successful"] == 0
        assert stats["rejected"] == 2
        assert stats["success_rate"] == 0.0

    async def test_mixed_data_validation_integration(self, validation_processor, valid_kline_data, invalid_kline_data):
        """Test validation integration with mixed valid/invalid data"""
        # Arrange
        mixed_data = valid_kline_data + invalid_kline_data

        # Act
        validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
            mixed_data, "BTCUSDT", strict_mode=True
        )

        # Assert
        assert len(validated_klines) == 2  # Only valid data should pass
        assert len(validation_errors) == 2  # Invalid data should be rejected

        # Check processing stats
        stats = validation_processor.get_processing_stats()
        assert stats["successful"] == 2
        assert stats["rejected"] == 2
        assert stats["success_rate"] == 50.0

    async def test_non_strict_mode_validation(self, validation_processor, invalid_kline_data):
        """Test validation integration in non-strict mode"""
        # Act
        validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
            invalid_kline_data, "BTCUSDT", strict_mode=False
        )

        # Assert
        # In non-strict mode, data should still be processed despite validation errors
        assert len(validated_klines) == 2
        assert len(validation_errors) == 2  # Errors should still be recorded

        # Check processing stats
        stats = validation_processor.get_processing_stats()
        assert stats["total_processed"] == 4  # All data processed
        assert stats["success_rate"] == 50.0  # But still marked as having errors

    @pytest.mark.asyncio
    async def test_symbol_format_validation_integration(self, validation_processor):
        """Test symbol format validation integration"""
        # Arrange
        invalid_symbol_data = [
            {
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "1.5",
                "quoteVolume": "75075.0",
                "n": 50,
                "x": True,
            }
        ]

        # Act: Use invalid symbol format
        validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
            invalid_symbol_data, "INVALID_SYMBOL", strict_mode=True
        )

        # Assert
        assert len(validated_klines) == 0
        assert len(validation_errors) == 1
        assert "Symbol validation failed" in validation_errors[0]

    @pytest.mark.asyncio
    async def test_trade_validation_integration(self, validation_processor):
        """Test trade data validation integration"""
        # Arrange
        valid_trade_data = [
            {
                "id": "12345",
                "price": "50000.00",
                "qty": "0.1",
                "time": 1640995200000,
                "side": "buy",
                "isBuyerMaker": True,
            }
        ]

        invalid_trade_data = [
            {
                "id": "12346",
                "price": "-50000.00",  # Negative price (invalid)
                "qty": "0.1",
                "time": 1640995200000,
                "side": "buy",
                "isBuyerMaker": True,
            }
        ]

        # Act: Test valid trade data
        validated_trades, validation_errors = await validation_processor.process_and_validate_trades(
            valid_trade_data, "BTCUSDT", strict_mode=True
        )

        # Assert: Valid data should pass
        assert len(validated_trades) == 1
        assert len(validation_errors) == 0

        # Act: Test invalid trade data
        validated_trades, validation_errors = await validation_processor.process_and_validate_trades(
            invalid_trade_data, "BTCUSDT", strict_mode=True
        )

        # Assert: Invalid data should be rejected
        assert len(validated_trades) == 0
        assert len(validation_errors) == 1
        assert "Price validation failed" in validation_errors[0]

    @pytest.mark.asyncio
    async def test_timestamp_sequence_validation(self, validation_processor):
        """Test timestamp sequence validation"""
        # Arrange: Out-of-order timestamps
        out_of_order_data = [
            {
                "t": 1640995260000,  # Later timestamp first
                "T": 1640995319999,
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "1.5",
                "quoteVolume": "75075.0",
                "n": 50,
                "x": True,
            },
            {
                "t": 1640995200000,  # Earlier timestamp second (invalid sequence)
                "T": 1640995259999,
                "o": "50050.00",
                "h": "50150.00",
                "l": "49950.00",
                "c": "50100.00",
                "v": "2.0",
                "quoteVolume": "100200.0",
                "n": 75,
                "x": True,
            },
        ]

        # Act
        validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
            out_of_order_data, "BTCUSDT", strict_mode=True
        )

        # Extract timestamps for sequence validation
        timestamps = [kline.open_time for kline in validated_klines]

        # Assert: Check if timestamp sequence validation would catch this
        if len(timestamps) > 1:
            is_valid, error = DataValidationRules.validate_timestamp_sequence(timestamps)
            # This should fail due to out-of-order timestamps
            assert not is_valid
            assert "Timestamp sequence broken" in error

    @pytest.mark.asyncio
    async def test_concurrent_validation_processing(self, validation_processor, valid_kline_data):
        """Test concurrent validation processing"""

        # Arrange
        async def process_batch(batch_id: int):
            """Process a batch of data concurrently"""
            validated_klines, validation_errors = await validation_processor.process_and_validate_klines(
                valid_kline_data, "BTCUSDT", strict_mode=True
            )
            return {
                "batch_id": batch_id,
                "validated_count": len(validated_klines),
                "error_count": len(validation_errors),
            }

        # Act: Process multiple batches concurrently
        tasks = [process_batch(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Assert: All batches should process successfully
        for result in results:
            assert result["validated_count"] == 2
            assert result["error_count"] == 0

    @pytest.mark.asyncio
    async def test_validation_rule_customization(self, test_provider, test_converter):
        """Test customization of validation rules"""

        # Arrange: Create processor with custom validation rules
        class CustomValidationRules(DataValidationRules):
            @staticmethod
            def validate_price_data(price: Decimal, field_name: str = "price") -> tuple[bool, str]:
                # Custom rule: Price must be between 1000 and 100000
                if price < Decimal("1000"):
                    return False, f"{field_name} too low (min 1000), got {price}"
                if price > Decimal("100000"):
                    return False, f"{field_name} too high (max 100000), got {price}"
                return True, ""

        processor = ValidationIntegrationProcessor(test_provider, test_converter)
        processor.validation_rules = CustomValidationRules()

        # Test data with price outside custom range
        test_data = [
            {
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "500.00",  # Below custom minimum of 1000
                "h": "600.00",
                "l": "400.00",
                "c": "550.00",
                "v": "1.5",
                "quoteVolume": "825.0",
                "n": 50,
                "x": True,
            }
        ]

        # Act
        validated_klines, validation_errors = await processor.process_and_validate_klines(
            test_data, "BTCUSDT", strict_mode=True
        )

        # Assert: Should be rejected by custom validation rules
        assert len(validated_klines) == 0
        assert len(validation_errors) == 1
        assert "too low (min 1000)" in validation_errors[0]
