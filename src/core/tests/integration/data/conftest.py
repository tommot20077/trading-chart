# ABOUTME: Test fixtures and configuration for data processing integration tests
# ABOUTME: Provides mock data generators, test providers, converters, and shared test utilities

import pytest
from datetime import datetime, timezone
from typing import Dict, List, Any

from core.interfaces.data.provider import AbstractDataProvider
from core.interfaces.data.converter import AbstractDataConverter
from core.implementations.memory.data.data_provider import MemoryDataProvider
from core.implementations.memory.data.data_converter import InMemoryDataConverter


class MockExchangeDataGenerator:
    """Mock data generator for different exchange formats"""

    @staticmethod
    def generate_binance_kline_data(symbol: str = "BTCUSDT", count: int = 10) -> List[Dict[str, Any]]:
        """Generate mock Binance K-line data format"""
        base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        data = []

        for i in range(count):
            timestamp = base_time + (i * 60000)  # 1 minute intervals
            open_price = 50000 + (i * 10)
            high_price = open_price + 100
            low_price = open_price - 50
            close_price = open_price + 20
            volume = 1.5 + (i * 0.1)

            data.append(
                {
                    "t": timestamp,  # Open time
                    "T": timestamp + 59999,  # Close time
                    "s": symbol,  # Symbol
                    "i": "1m",  # Interval
                    "f": 100 + i,  # First trade ID
                    "L": 200 + i,  # Last trade ID
                    "o": str(open_price),  # Open price
                    "c": str(close_price),  # Close price
                    "h": str(high_price),  # High price
                    "l": str(low_price),  # Low price
                    "v": str(volume),  # Base asset volume
                    "n": 50 + i,  # Number of trades
                    "x": True,  # Is this kline closed?
                    "quoteVolume": str(volume * close_price),  # Quote asset volume
                    "V": str(volume * 0.8),  # Taker buy base asset volume
                    "Q": str(volume * 0.8 * close_price),  # Taker buy quote asset volume
                }
            )

        return data

    @staticmethod
    def generate_okx_kline_data(symbol: str = "BTC-USDT", count: int = 10) -> List[List[str]]:
        """Generate mock OKX K-line data format (array of arrays)"""
        base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        data = []

        for i in range(count):
            timestamp = base_time + (i * 60000)
            open_price = 50000 + (i * 10)
            high_price = open_price + 100
            low_price = open_price - 50
            close_price = open_price + 20
            volume = 1.5 + (i * 0.1)

            # OKX format: [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
            data.append(
                [
                    str(timestamp),
                    str(open_price),
                    str(high_price),
                    str(low_price),
                    str(close_price),
                    str(volume),
                    str(volume * close_price),
                    "0",
                    "1",
                ]
            )

        return data

    @staticmethod
    def generate_binance_trade_data(symbol: str = "BTCUSDT", count: int = 10) -> List[Dict[str, Any]]:
        """Generate mock Binance trade data format"""
        base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        data = []

        for i in range(count):
            timestamp = base_time + (i * 1000)
            price = 50000 + (i * 5)
            quantity = 0.1 + (i * 0.01)

            data.append(
                {
                    "id": 1000 + i,
                    "price": str(price),
                    "qty": str(quantity),
                    "quoteQty": str(price * quantity),
                    "time": timestamp,
                    "side": "buy" if i % 2 == 0 else "sell",
                    "isBuyerMaker": i % 2 == 0,
                    "isBestMatch": True,
                }
            )

        return data


@pytest.fixture
def mock_data_generator():
    """Fixture providing mock exchange data generator"""
    return MockExchangeDataGenerator()


@pytest.fixture
def test_provider() -> AbstractDataProvider:
    """Fixture providing test data provider instance"""
    return MemoryDataProvider()


@pytest.fixture
def test_converter() -> AbstractDataConverter:
    """Fixture providing test data converter instance"""
    return InMemoryDataConverter()


@pytest.fixture
def sample_kline_data(mock_data_generator):
    """Fixture providing sample K-line data for testing"""
    return {
        "binance": mock_data_generator.generate_binance_kline_data("BTCUSDT", 5),
        "okx": mock_data_generator.generate_okx_kline_data("BTC-USDT", 5),
    }


@pytest.fixture
def sample_trade_data(mock_data_generator):
    """Fixture providing sample trade data for testing"""
    return {"binance": mock_data_generator.generate_binance_trade_data("BTCUSDT", 5)}


@pytest.fixture
def test_symbols():
    """Fixture providing test trading symbols"""
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]


@pytest.fixture
def cleanup_test_data():
    """Fixture for cleaning up test data after tests"""
    yield
    # Cleanup logic here if needed
    pass
