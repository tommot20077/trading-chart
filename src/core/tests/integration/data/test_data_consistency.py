# ABOUTME: Integration tests for data consistency guarantees across different sources
# ABOUTME: Tests data integrity, validation, and consistency when switching between providers

import pytest
import pytest_asyncio
import asyncio
from typing import List, Dict, Any
from decimal import Decimal
import hashlib
import json
from datetime import datetime, timezone

from core.interfaces.data.provider import AbstractDataProvider
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.exceptions.base import ValidationException


class ConsistentDataProvider(AbstractDataProvider):
    """Mock provider that provides consistent data across calls"""

    def __init__(self, name: str, base_price: Decimal = Decimal("50000")):
        self._name = name
        self._connected = False
        self._base_price = base_price
        self._data_version = 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def close(self) -> None:
        await self.disconnect()

    async def ping(self) -> float:
        return 50.0

    async def get_exchange_info(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "status": "normal",
            "serverTime": datetime.now(timezone.utc).isoformat(),
            "dataVersion": self._data_version,
        }

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "status": "TRADING",
            "baseAsset": symbol.split("/")[0] if "/" in symbol else symbol[:3],
            "quoteAsset": symbol.split("/")[1] if "/" in symbol else symbol[3:],
            "provider": self._name,
            "basePrice": str(self._base_price),
        }

    def get_consistent_kline_data(self, symbol: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generate consistent K-line data"""
        base_time = 1640995200000  # Fixed timestamp for consistency
        data = []

        for i in range(count):
            timestamp = base_time + (i * 60000)
            # Use deterministic price calculation
            price_offset = Decimal(str(i * 10))
            open_price = self._base_price + price_offset
            close_price = open_price + Decimal("20")
            high_price = open_price + Decimal("100")
            low_price = open_price - Decimal("50")
            volume = Decimal("1.5")

            data.append(
                {
                    "t": timestamp,
                    "T": timestamp + 59999,
                    "o": str(open_price),
                    "c": str(close_price),
                    "h": str(high_price),
                    "l": str(low_price),
                    "v": str(volume),
                    "quoteVolume": str(volume * close_price),
                    "n": 50,
                    "x": True,
                }
            )

        return data

    def get_consistent_trade_data(self, symbol: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generate consistent trade data"""
        base_time = 1640995200000
        data = []

        for i in range(count):
            timestamp = base_time + (i * 1000)
            price = self._base_price + Decimal(str(i * 5))
            quantity = Decimal("0.1")

            data.append(
                {
                    "id": 1000 + i,
                    "price": str(price),
                    "qty": str(quantity),
                    "time": timestamp,
                    "side": "buy" if i % 2 == 0 else "sell",
                    "isBuyerMaker": i % 2 == 0,
                }
            )

        return data

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        return True, ""

    # Required abstract method implementations
    async def stream_trades(self, symbol: str, *, start_from=None):
        """Mock implementation - not used in consistency tests"""
        raise NotImplementedError("Consistent provider - stream_trades not implemented")

    async def stream_klines(self, symbol: str, interval, *, start_from=None):
        """Mock implementation - not used in consistency tests"""
        raise NotImplementedError("Consistent provider - stream_klines not implemented")

    async def fetch_historical_trades(self, symbol: str, start_time, end_time, *, limit=None):
        """Mock implementation - not used in consistency tests"""
        raise NotImplementedError("Consistent provider - fetch_historical_trades not implemented")

    async def fetch_historical_klines(self, symbol: str, interval, start_time, end_time, *, limit=None):
        """Mock implementation - not used in consistency tests"""
        raise NotImplementedError("Consistent provider - fetch_historical_klines not implemented")

    async def convert_multiple_trades(self, raw_trades, symbol: str):
        """Mock implementation - not used in consistency tests"""
        raise NotImplementedError("Consistent provider - convert_multiple_trades not implemented")

    async def convert_multiple_klines(self, raw_klines, symbol: str):
        """Mock implementation - not used in consistency tests"""
        raise NotImplementedError("Consistent provider - convert_multiple_klines not implemented")


class DataConsistencyValidator:
    """Validator for ensuring data consistency across providers"""

    @staticmethod
    def calculate_data_hash(data: Any) -> str:
        """Calculate hash of data for consistency checking"""
        if isinstance(data, (list, dict)):
            json_str = json.dumps(data, sort_keys=True, default=str)
        else:
            json_str = str(data)
        return hashlib.md5(json_str.encode()).hexdigest()

    @staticmethod
    def validate_kline_consistency(klines1: List[Kline], klines2: List[Kline]) -> tuple[bool, str]:
        """Validate consistency between two sets of K-line data"""
        if len(klines1) != len(klines2):
            return False, f"Length mismatch: {len(klines1)} vs {len(klines2)}"

        for i, (k1, k2) in enumerate(zip(klines1, klines2)):
            if k1.symbol != k2.symbol:
                return False, f"Symbol mismatch at index {i}: {k1.symbol} vs {k2.symbol}"

            if k1.open_time != k2.open_time:
                return False, f"Open time mismatch at index {i}: {k1.open_time} vs {k2.open_time}"

            if k1.open_price != k2.open_price:
                return False, f"Open price mismatch at index {i}: {k1.open_price} vs {k2.open_price}"

            if k1.close_price != k2.close_price:
                return False, f"Close price mismatch at index {i}: {k1.close_price} vs {k2.close_price}"

            if k1.high_price != k2.high_price:
                return False, f"High price mismatch at index {i}: {k1.high_price} vs {k2.high_price}"

            if k1.low_price != k2.low_price:
                return False, f"Low price mismatch at index {i}: {k1.low_price} vs {k2.low_price}"

            if k1.volume != k2.volume:
                return False, f"Volume mismatch at index {i}: {k1.volume} vs {k2.volume}"

        return True, ""

    @staticmethod
    def validate_trade_consistency(trades1: List[Trade], trades2: List[Trade]) -> tuple[bool, str]:
        """Validate consistency between two sets of trade data"""
        if len(trades1) != len(trades2):
            return False, f"Length mismatch: {len(trades1)} vs {len(trades2)}"

        for i, (t1, t2) in enumerate(zip(trades1, trades2)):
            if t1.symbol != t2.symbol:
                return False, f"Symbol mismatch at index {i}: {t1.symbol} vs {t2.symbol}"

            if t1.trade_id != t2.trade_id:
                return False, f"Trade ID mismatch at index {i}: {t1.trade_id} vs {t2.trade_id}"

            if t1.price != t2.price:
                return False, f"Price mismatch at index {i}: {t1.price} vs {t2.price}"

            if t1.quantity != t2.quantity:
                return False, f"Quantity mismatch at index {i}: {t1.quantity} vs {t2.quantity}"

            if t1.side != t2.side:
                return False, f"Side mismatch at index {i}: {t1.side} vs {t2.side}"

        return True, ""

    @staticmethod
    def validate_business_rules(klines: List[Kline]) -> tuple[bool, str]:
        """Validate business rules for K-line data"""
        for i, kline in enumerate(klines):
            # High should be >= open and close
            if kline.high_price < kline.open_price:
                return False, f"High < Open at index {i}: {kline.high_price} < {kline.open_price}"

            if kline.high_price < kline.close_price:
                return False, f"High < Close at index {i}: {kline.high_price} < {kline.close_price}"

            # Low should be <= open and close
            if kline.low_price > kline.open_price:
                return False, f"Low > Open at index {i}: {kline.low_price} > {kline.open_price}"

            if kline.low_price > kline.close_price:
                return False, f"Low > Close at index {i}: {kline.low_price} > {kline.close_price}"

            # Volume should be non-negative
            if kline.volume < 0:
                return False, f"Negative volume at index {i}: {kline.volume}"

            # Prices should be positive
            if any(price <= 0 for price in [kline.open_price, kline.high_price, kline.low_price, kline.close_price]):
                return False, f"Non-positive price at index {i}"

        return True, ""


@pytest.mark.integration
@pytest.mark.asyncio
class TestDataConsistency:
    """Integration tests for data consistency guarantees"""

    @pytest_asyncio.fixture
    async def provider_a(self):
        """First consistent data provider"""
        provider = ConsistentDataProvider("ProviderA", Decimal("50000"))
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    async def provider_b(self):
        """Second consistent data provider with same base price"""
        provider = ConsistentDataProvider("ProviderB", Decimal("50000"))
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    async def provider_c_different_price(self):
        """Third provider with different base price for testing inconsistency"""
        provider = ConsistentDataProvider("ProviderC", Decimal("51000"))
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    def validator(self):
        """Data consistency validator"""
        return DataConsistencyValidator()

    @pytest.mark.asyncio
    async def test_identical_data_from_same_source(self, provider_a, test_converter, validator):
        """Test that same provider returns identical data across multiple calls"""
        # Arrange
        symbol = "BTCUSDT"

        # Act: Get data multiple times from same provider
        raw_data_1 = provider_a.get_consistent_kline_data(symbol, 5)
        raw_data_2 = provider_a.get_consistent_kline_data(symbol, 5)

        converted_data_1 = test_converter.convert_multiple_klines(raw_data_1, symbol)
        converted_data_2 = test_converter.convert_multiple_klines(raw_data_2, symbol)

        # Assert: Data should be identical
        is_consistent, error_msg = validator.validate_kline_consistency(converted_data_1, converted_data_2)
        assert is_consistent, f"Data inconsistency: {error_msg}"

        # Hash should be identical (excluding dynamic fields like received_at)
        def get_comparable_data(klines):
            return [
                {
                    "symbol": k.symbol,
                    "open_time": k.open_time.isoformat(),
                    "close_time": k.close_time.isoformat(),
                    "open_price": str(k.open_price),
                    "high_price": str(k.high_price),
                    "low_price": str(k.low_price),
                    "close_price": str(k.close_price),
                    "volume": str(k.volume),
                    "quote_volume": str(k.quote_volume),
                    "trades_count": k.trades_count,
                    "interval": k.interval if isinstance(k.interval, str) else k.interval.value,
                }
                for k in klines
            ]

        hash_1 = validator.calculate_data_hash(get_comparable_data(converted_data_1))
        hash_2 = validator.calculate_data_hash(get_comparable_data(converted_data_2))
        assert hash_1 == hash_2

    async def test_consistent_data_across_equivalent_providers(self, provider_a, provider_b, test_converter, validator):
        """Test data consistency across providers with same configuration"""
        # Arrange
        symbol = "BTCUSDT"

        # Act: Get data from both providers
        raw_data_a = provider_a.get_consistent_kline_data(symbol, 5)
        raw_data_b = provider_b.get_consistent_kline_data(symbol, 5)

        converted_data_a = test_converter.convert_multiple_klines(raw_data_a, symbol)
        converted_data_b = test_converter.convert_multiple_klines(raw_data_b, symbol)

        # Assert: Data should be consistent
        is_consistent, error_msg = validator.validate_kline_consistency(converted_data_a, converted_data_b)
        assert is_consistent, f"Cross-provider inconsistency: {error_msg}"

    @pytest.mark.asyncio
    async def test_business_rule_validation_consistency(self, provider_a, test_converter, validator):
        """Test that business rules are consistently applied"""
        # Arrange
        symbol = "BTCUSDT"

        # Act: Get and convert data
        raw_data = provider_a.get_consistent_kline_data(symbol, 10)
        converted_data = test_converter.convert_multiple_klines(raw_data, symbol)

        # Assert: Business rules should be satisfied
        is_valid, error_msg = validator.validate_business_rules(converted_data)
        assert is_valid, f"Business rule violation: {error_msg}"

    async def test_data_integrity_during_concurrent_access(self, provider_a, test_converter, validator):
        """Test data integrity during concurrent access"""
        # Arrange
        symbol = "BTCUSDT"

        async def get_and_convert_data(provider_id: int):
            """Get and convert data with provider ID for tracking"""
            raw_data = provider_a.get_consistent_kline_data(symbol, 5)
            converted_data = test_converter.convert_multiple_klines(raw_data, symbol)

            # Use consistent hash calculation (excluding dynamic fields)
            def get_comparable_data(klines):
                return [
                    {
                        "symbol": k.symbol,
                        "open_time": k.open_time.isoformat(),
                        "close_time": k.close_time.isoformat(),
                        "open_price": str(k.open_price),
                        "high_price": str(k.high_price),
                        "low_price": str(k.low_price),
                        "close_price": str(k.close_price),
                        "volume": str(k.volume),
                        "quote_volume": str(k.quote_volume),
                        "trades_count": k.trades_count,
                        "interval": k.interval if isinstance(k.interval, str) else k.interval.value,
                    }
                    for k in klines
                ]

            return {
                "provider_id": provider_id,
                "data": converted_data,
                "hash": validator.calculate_data_hash(get_comparable_data(converted_data)),
            }

        # Act: Concurrent access
        tasks = [get_and_convert_data(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Assert: All results should be identical
        reference_hash = results[0]["hash"]
        for result in results[1:]:
            assert result["hash"] == reference_hash, "Concurrent access inconsistency"

        # Validate consistency between all results
        reference_data = results[0]["data"]
        for i, result in enumerate(results[1:], 1):
            is_consistent, error_msg = validator.validate_kline_consistency(reference_data, result["data"])
            assert is_consistent, f"Concurrent result {i} inconsistent: {error_msg}"

    async def test_cross_provider_data_validation(
        self, provider_a, provider_c_different_price, test_converter, validator
    ):
        """Test validation when providers have different data"""
        # Arrange
        symbol = "BTCUSDT"

        # Act: Get data from providers with different configurations
        raw_data_a = provider_a.get_consistent_kline_data(symbol, 5)
        raw_data_c = provider_c_different_price.get_consistent_kline_data(symbol, 5)

        converted_data_a = test_converter.convert_multiple_klines(raw_data_a, symbol)
        converted_data_c = test_converter.convert_multiple_klines(raw_data_c, symbol)

        # Assert: Data should be different (inconsistent)
        is_consistent, error_msg = validator.validate_kline_consistency(converted_data_a, converted_data_c)
        assert not is_consistent, "Expected inconsistency between different providers"

        # But both should satisfy business rules
        is_valid_a, _ = validator.validate_business_rules(converted_data_a)
        is_valid_c, _ = validator.validate_business_rules(converted_data_c)
        assert is_valid_a and is_valid_c, "Both datasets should satisfy business rules"

    @pytest.mark.asyncio
    async def test_trade_data_consistency(self, provider_a, test_converter, validator):
        """Test consistency for trade data"""
        # Arrange
        symbol = "BTCUSDT"

        # Act: Get trade data multiple times
        raw_trades_1 = provider_a.get_consistent_trade_data(symbol, 5)
        raw_trades_2 = provider_a.get_consistent_trade_data(symbol, 5)

        converted_trades_1 = test_converter.convert_multiple_trades(raw_trades_1, symbol)
        converted_trades_2 = test_converter.convert_multiple_trades(raw_trades_2, symbol)

        # Assert: Trade data should be consistent
        is_consistent, error_msg = validator.validate_trade_consistency(converted_trades_1, converted_trades_2)
        assert is_consistent, f"Trade data inconsistency: {error_msg}"

    async def test_data_consistency_with_provider_switching(self, provider_a, provider_b, test_converter, validator):
        """Test data consistency when switching between providers"""
        # Arrange
        symbol = "BTCUSDT"
        providers = [provider_a, provider_b]
        all_results = []

        # Act: Switch between providers and collect data
        for i in range(6):  # 3 switches
            provider = providers[i % 2]
            raw_data = provider.get_consistent_kline_data(symbol, 3)
            converted_data = test_converter.convert_multiple_klines(raw_data, symbol)
            all_results.append({"provider": provider.name, "data": converted_data, "iteration": i})

        # Assert: Data from same provider should be consistent
        provider_a_results = [r for r in all_results if r["provider"] == "ProviderA"]
        provider_b_results = [r for r in all_results if r["provider"] == "ProviderB"]

        # Check consistency within each provider
        for i in range(1, len(provider_a_results)):
            is_consistent, error_msg = validator.validate_kline_consistency(
                provider_a_results[0]["data"], provider_a_results[i]["data"]
            )
            assert is_consistent, f"ProviderA inconsistency: {error_msg}"

        for i in range(1, len(provider_b_results)):
            is_consistent, error_msg = validator.validate_kline_consistency(
                provider_b_results[0]["data"], provider_b_results[i]["data"]
            )
            assert is_consistent, f"ProviderB inconsistency: {error_msg}"

    @pytest.mark.asyncio
    async def test_data_validation_with_corrupted_input(self, test_converter, validator):
        """Test data validation with intentionally corrupted input"""
        # Arrange: Create corrupted data
        corrupted_kline_data = [
            {
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "50000",
                "c": "49000",  # Close < Open
                "h": "48000",  # High < Open and Close (invalid)
                "l": "51000",  # Low > Open and Close (invalid)
                "v": "-1.0",  # Negative volume (invalid)
                "quoteVolume": "50000",
                "n": 50,
                "x": True,
            }
        ]

        # Act: Convert corrupted data
        try:
            converted_data = test_converter.convert_multiple_klines(corrupted_kline_data, "BTCUSDT")

            # Assert: Business rule validation should fail
            is_valid, error_msg = validator.validate_business_rules(converted_data)
            assert not is_valid, f"Expected business rule violation, but got: {error_msg}"

        except (ValueError, ValidationException):
            # Also acceptable - converter should reject invalid data
            pass

    async def test_timestamp_consistency_across_providers(self, provider_a, provider_b, test_converter):
        """Test timestamp consistency across providers"""
        # Arrange
        symbol = "BTCUSDT"

        # Act: Get data from both providers
        raw_data_a = provider_a.get_consistent_kline_data(symbol, 5)
        raw_data_b = provider_b.get_consistent_kline_data(symbol, 5)

        converted_data_a = test_converter.convert_multiple_klines(raw_data_a, symbol)
        converted_data_b = test_converter.convert_multiple_klines(raw_data_b, symbol)

        # Assert: Timestamps should be identical
        for i, (kline_a, kline_b) in enumerate(zip(converted_data_a, converted_data_b)):
            assert kline_a.open_time == kline_b.open_time, f"Open time mismatch at index {i}"
            assert kline_a.close_time == kline_b.close_time, f"Close time mismatch at index {i}"

    @pytest.mark.asyncio
    async def test_data_consistency_under_load(self, provider_a, test_converter, validator):
        """Test data consistency under high load"""
        # Arrange
        symbol = "BTCUSDT"

        async def process_batch(batch_id: int):
            """Process a batch of data"""
            raw_data = provider_a.get_consistent_kline_data(symbol, 10)
            converted_data = test_converter.convert_multiple_klines(raw_data, symbol)

            # Validate business rules
            is_valid, error_msg = validator.validate_business_rules(converted_data)
            if not is_valid:
                raise ValueError(f"Batch {batch_id} validation failed: {error_msg}")

            # Use consistent hash calculation (excluding dynamic fields)
            def get_comparable_data(klines):
                return [
                    {
                        "symbol": k.symbol,
                        "open_time": k.open_time.isoformat(),
                        "close_time": k.close_time.isoformat(),
                        "open_price": str(k.open_price),
                        "high_price": str(k.high_price),
                        "low_price": str(k.low_price),
                        "close_price": str(k.close_price),
                        "volume": str(k.volume),
                        "quote_volume": str(k.quote_volume),
                        "trades_count": k.trades_count,
                        "interval": k.interval if isinstance(k.interval, str) else k.interval.value,
                    }
                    for k in klines
                ]

            return {
                "batch_id": batch_id,
                "data_hash": validator.calculate_data_hash(get_comparable_data(converted_data)),
                "count": len(converted_data),
            }

        # Act: Process multiple batches concurrently
        tasks = [process_batch(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # Assert: All batches should have same hash (consistent data)
        reference_hash = results[0]["data_hash"]
        for result in results:
            assert result["data_hash"] == reference_hash, f"Batch {result['batch_id']} inconsistent"
            assert result["count"] == 10, f"Batch {result['batch_id']} wrong count"
