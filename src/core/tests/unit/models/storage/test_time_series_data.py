# ABOUTME: Unit tests for TimeSeriesData protocol
# ABOUTME: Tests cover protocol compliance, timestamp handling, and various data implementations

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal


class MockTradeData:
    """Mock implementation of TimeSeriesData protocol for Trade-like data."""

    def __init__(self, symbol: str, timestamp: datetime, price: Decimal = None, volume: Decimal = None):
        self.symbol = symbol
        self.timestamp = timestamp
        self.price = price or Decimal("100.0")
        self.volume = volume or Decimal("1.0")

    @property
    def primary_timestamp(self) -> datetime:
        """Return the primary timestamp for this data point."""
        return self.timestamp


class MockKlineData:
    """Mock implementation of TimeSeriesData protocol for Kline-like data."""

    def __init__(self, symbol: str, timestamp: datetime, open_time: datetime = None):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open_time = open_time or timestamp
        self.close_time = timestamp + timedelta(minutes=1)
        self.open_price = Decimal("100.0")
        self.close_price = Decimal("101.0")

    @property
    def primary_timestamp(self) -> datetime:
        """Return the primary timestamp for this data point."""
        # For Kline data, we might want to use open_time as primary
        return getattr(self, "open_time", self.timestamp)


class MockCustomTimeSeriesData:
    """Mock implementation with custom primary_timestamp logic."""

    def __init__(self, symbol: str, timestamp: datetime, created_at: datetime = None):
        self.symbol = symbol
        self.timestamp = timestamp
        self.created_at = created_at or timestamp

    @property
    def primary_timestamp(self) -> datetime:
        """Return the primary timestamp for this data point."""
        # Custom logic: prefer created_at over timestamp
        return self.created_at


class TestTimeSeriesDataProtocol:
    """Test cases for TimeSeriesData protocol implementation."""

    @pytest.mark.unit
    def test_trade_data_protocol_compliance(self):
        """Test that MockTradeData implements TimeSeriesData protocol."""
        now = datetime.now(UTC)
        trade_data = MockTradeData("BTCUSDT", now)

        # Test protocol requirements
        assert hasattr(trade_data, "symbol")
        assert hasattr(trade_data, "timestamp")
        assert hasattr(trade_data, "primary_timestamp")

        # Test values
        assert trade_data.symbol == "BTCUSDT"
        assert trade_data.timestamp == now
        assert trade_data.primary_timestamp == now

    @pytest.mark.unit
    def test_kline_data_protocol_compliance(self):
        """Test that MockKlineData implements TimeSeriesData protocol."""
        now = datetime.now(UTC)
        open_time = now - timedelta(minutes=1)
        kline_data = MockKlineData("ETHUSDT", now, open_time)

        # Test protocol requirements
        assert hasattr(kline_data, "symbol")
        assert hasattr(kline_data, "timestamp")
        assert hasattr(kline_data, "primary_timestamp")

        # Test values
        assert kline_data.symbol == "ETHUSDT"
        assert kline_data.timestamp == now
        assert kline_data.primary_timestamp == open_time

    @pytest.mark.unit
    def test_custom_data_protocol_compliance(self):
        """Test that MockCustomTimeSeriesData implements TimeSeriesData protocol."""
        now = datetime.now(UTC)
        created_at = now - timedelta(seconds=30)
        custom_data = MockCustomTimeSeriesData("ADAUSDT", now, created_at)

        # Test protocol requirements
        assert hasattr(custom_data, "symbol")
        assert hasattr(custom_data, "timestamp")
        assert hasattr(custom_data, "primary_timestamp")

        # Test values
        assert custom_data.symbol == "ADAUSDT"
        assert custom_data.timestamp == now
        assert custom_data.primary_timestamp == created_at

    @pytest.mark.unit
    def test_primary_timestamp_default_implementation(self):
        """Test the default primary_timestamp implementation for Trade-like data."""
        now = datetime.now(UTC)
        trade_data = MockTradeData("BTCUSDT", now)

        # Default implementation should return timestamp
        assert trade_data.primary_timestamp == trade_data.timestamp
        assert trade_data.primary_timestamp == now

    @pytest.mark.unit
    def test_primary_timestamp_custom_implementation(self):
        """Test custom primary_timestamp implementation."""
        now = datetime.now(UTC)
        open_time = now - timedelta(minutes=1)
        kline_data = MockKlineData("ETHUSDT", now, open_time)

        # Custom implementation should return open_time
        assert kline_data.primary_timestamp == open_time
        assert kline_data.primary_timestamp != kline_data.timestamp

    @pytest.mark.unit
    def test_symbol_variations(self):
        """Test TimeSeriesData with various symbol formats."""
        now = datetime.now(UTC)
        symbols = [
            "BTCUSDT",
            "ETH-USD",
            "BTC_USDT",
            "SPOT_BTCUSDT",
            "btcusdt",  # lowercase
            "BTC/USDT",
            "XRP-EUR",
        ]

        for symbol in symbols:
            trade_data = MockTradeData(symbol, now)
            assert trade_data.symbol == symbol

    @pytest.mark.unit
    def test_timestamp_variations(self):
        """Test TimeSeriesData with various timestamp formats."""
        timestamps = [
            datetime.now(UTC),
            datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2030, 12, 31, 23, 59, 59, tzinfo=UTC),
            datetime.now(UTC) - timedelta(days=365),
            datetime.now(UTC) + timedelta(days=365),
        ]

        for timestamp in timestamps:
            trade_data = MockTradeData("TESTUSDT", timestamp)
            assert trade_data.timestamp == timestamp
            assert trade_data.primary_timestamp == timestamp

    @pytest.mark.unit
    def test_timestamp_timezone_handling(self):
        """Test TimeSeriesData with different timezone timestamps."""
        base_time = datetime(2023, 1, 1, 12, 0, 0)

        # UTC timestamp
        utc_time = base_time.replace(tzinfo=UTC)
        trade_data = MockTradeData("BTCUSDT", utc_time)
        assert trade_data.timestamp.tzinfo == UTC
        assert trade_data.primary_timestamp.tzinfo == UTC

    @pytest.mark.unit
    def test_multiple_data_instances(self):
        """Test creating multiple TimeSeriesData instances."""
        now = datetime.now(UTC)

        # Create multiple instances
        trade1 = MockTradeData("BTCUSDT", now)
        trade2 = MockTradeData("ETHUSDT", now + timedelta(seconds=1))
        kline1 = MockKlineData("ADAUSDT", now + timedelta(seconds=2))

        # Test independence
        assert trade1.symbol != trade2.symbol
        assert trade1.timestamp != trade2.timestamp
        assert trade2.symbol != kline1.symbol

        # Test protocol compliance for all
        for data in [trade1, trade2, kline1]:
            assert hasattr(data, "symbol")
            assert hasattr(data, "timestamp")
            assert hasattr(data, "primary_timestamp")

    @pytest.mark.unit
    def test_primary_timestamp_consistency(self):
        """Test that primary_timestamp returns consistent values."""
        now = datetime.now(UTC)
        trade_data = MockTradeData("BTCUSDT", now)

        # Multiple calls should return the same value
        first_call = trade_data.primary_timestamp
        second_call = trade_data.primary_timestamp
        third_call = trade_data.primary_timestamp

        assert first_call == second_call == third_call
        assert first_call == now

    @pytest.mark.unit
    def test_primary_timestamp_with_microseconds(self):
        """Test primary_timestamp with microsecond precision."""
        now = datetime.now(UTC)
        microsecond_time = now.replace(microsecond=123456)

        trade_data = MockTradeData("BTCUSDT", microsecond_time)
        assert trade_data.primary_timestamp.microsecond == 123456
        assert trade_data.primary_timestamp == microsecond_time

    @pytest.mark.unit
    def test_boundary_timestamps(self):
        """Test TimeSeriesData with boundary timestamp values."""
        # Minimum datetime
        min_time = datetime.min.replace(tzinfo=UTC)
        trade_min = MockTradeData("TESTUSDT", min_time)
        assert trade_min.primary_timestamp == min_time

        # Maximum datetime (within reasonable bounds)
        max_time = datetime(2100, 12, 31, 23, 59, 59, tzinfo=UTC)
        trade_max = MockTradeData("TESTUSDT", max_time)
        assert trade_max.primary_timestamp == max_time

    @pytest.mark.unit
    def test_symbol_edge_cases(self):
        """Test TimeSeriesData with edge case symbols."""
        now = datetime.now(UTC)
        edge_symbols = [
            "A",  # Single character
            "VERYLONGSYMBOLNAMETHATMIGHTEXIST",  # Very long
            "123SYMBOL",  # Starting with numbers
            "SYMBOL123",  # Ending with numbers
            "SYM-BOL_123",  # Mixed characters
        ]

        for symbol in edge_symbols:
            trade_data = MockTradeData(symbol, now)
            assert trade_data.symbol == symbol

    @pytest.mark.unit
    def test_additional_attributes_preservation(self):
        """Test that additional attributes beyond protocol are preserved."""
        now = datetime.now(UTC)
        price = Decimal("50000.50")
        volume = Decimal("1.25")

        trade_data = MockTradeData("BTCUSDT", now, price, volume)

        # Protocol requirements
        assert trade_data.symbol == "BTCUSDT"
        assert trade_data.timestamp == now
        assert trade_data.primary_timestamp == now

        # Additional attributes
        assert trade_data.price == price
        assert trade_data.volume == volume

    @pytest.mark.unit
    def test_inheritance_behavior(self):
        """Test that TimeSeriesData works with inheritance."""

        class ExtendedTradeData(MockTradeData):
            def __init__(self, symbol: str, timestamp: datetime, exchange: str):
                super().__init__(symbol, timestamp)
                self.exchange = exchange

            @property
            def primary_timestamp(self) -> datetime:
                # Override with custom logic
                return self.timestamp

        now = datetime.now(UTC)
        extended_data = ExtendedTradeData("BTCUSDT", now, "binance")

        # Test protocol compliance
        assert extended_data.symbol == "BTCUSDT"
        assert extended_data.timestamp == now
        assert extended_data.primary_timestamp == now
        assert extended_data.exchange == "binance"
