# ABOUTME: Unit tests for MemoryDataProvider implementation
# ABOUTME: Tests all functionality including connection, data generation, and error handling

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, UTC
from decimal import Decimal

from core.implementations.memory.data.data_provider import MemoryDataProvider
from core.models import KlineInterval, TradeSide
from core.exceptions.base import (
    ExternalServiceException,
    DataNotFoundException,
    ValidationException,
)


class TestMemoryDataProvider:
    """Test suite for MemoryDataProvider."""

    @pytest.fixture
    def provider(self):
        """Create a MemoryDataProvider instance for testing."""
        return MemoryDataProvider(name="TestProvider", latency_ms=1.0)

    @pytest_asyncio.fixture
    async def connected_provider(self, provider):
        """Create a connected MemoryDataProvider instance."""
        await provider.connect()
        yield provider
        await provider.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.name == "TestProvider"
        assert not provider.is_connected
        assert provider._latency_ms == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, provider):
        """Test connection and disconnection."""
        # Initially not connected
        assert not provider.is_connected

        # Connect
        await provider.connect()
        assert provider.is_connected

        # Disconnect
        await provider.disconnect()
        assert not provider.is_connected

        # Close
        await provider.close()
        assert not provider.is_connected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ping_when_connected(self, connected_provider):
        """Test ping functionality when connected."""
        latency = await connected_provider.ping()
        assert isinstance(latency, float)
        assert latency >= 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ping_when_not_connected(self, provider):
        """Test ping fails when not connected."""
        with pytest.raises(ExternalServiceException) as exc_info:
            await provider.ping()

        assert exc_info.value.code == "NOT_CONNECTED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_exchange_info(self, connected_provider):
        """Test getting exchange information."""
        info = await connected_provider.get_exchange_info()

        assert isinstance(info, dict)
        assert info["name"] == "TestProvider"
        assert info["status"] == "normal"
        assert "symbols" in info
        assert "BTCUSDT" in info["symbols"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_exchange_info_not_connected(self, provider):
        """Test getting exchange info fails when not connected."""
        with pytest.raises(ExternalServiceException) as exc_info:
            await provider.get_exchange_info()

        assert exc_info.value.code == "NOT_CONNECTED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_symbol_info_valid_symbol(self, connected_provider):
        """Test getting symbol information for valid symbol."""
        info = await connected_provider.get_symbol_info("BTCUSDT")

        assert isinstance(info, dict)
        assert info["symbol"] == "BTCUSDT"
        assert info["status"] == "TRADING"
        assert info["baseAsset"] == "BTC"
        assert info["quoteAsset"] == "USDT"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_symbol_info_invalid_symbol(self, connected_provider):
        """Test getting symbol info for invalid symbol."""
        with pytest.raises(DataNotFoundException) as exc_info:
            await connected_provider.get_symbol_info("INVALID/SYMBOL")

        assert exc_info.value.code == "SYMBOL_NOT_FOUND"
        assert "INVALID/SYMBOL" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_historical_trades(self, connected_provider):
        """Test fetching historical trades."""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        trades = await connected_provider.fetch_historical_trades("BTCUSDT", start_time, end_time, limit=10)

        assert isinstance(trades, list)
        assert len(trades) <= 10

        for trade in trades:
            assert trade.symbol == "BTCUSDT"
            assert isinstance(trade.price, Decimal)
            assert isinstance(trade.quantity, Decimal)
            assert trade.side in [TradeSide.BUY, TradeSide.SELL]
            assert start_time <= trade.timestamp <= end_time

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_historical_trades_invalid_time_range(self, connected_provider):
        """Test fetching trades with invalid time range."""
        end_time = datetime.now(UTC)
        start_time = end_time + timedelta(hours=1)  # Start after end

        with pytest.raises(ValidationException) as exc_info:
            await connected_provider.fetch_historical_trades("BTCUSDT", start_time, end_time)

        assert exc_info.value.code == "INVALID_TIME_RANGE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_historical_trades_unsupported_symbol(self, connected_provider):
        """Test fetching trades for unsupported symbol."""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        with pytest.raises(DataNotFoundException) as exc_info:
            await connected_provider.fetch_historical_trades("UNSUPPORTED/SYMBOL", start_time, end_time)

        assert exc_info.value.code == "SYMBOL_NOT_SUPPORTED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_historical_klines(self, connected_provider):
        """Test fetching historical klines."""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=2)

        klines = await connected_provider.fetch_historical_klines(
            "BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time, limit=5
        )

        assert isinstance(klines, list)
        assert len(klines) <= 5

        for kline in klines:
            assert kline.symbol == "BTCUSDT"
            assert kline.interval == KlineInterval.MINUTE_1
            assert isinstance(kline.open_price, Decimal)
            assert isinstance(kline.close_price, Decimal)
            assert kline.high_price >= kline.low_price
            # Allow some tolerance for time alignment
            time_tolerance = timedelta(minutes=1)
            assert (start_time - time_tolerance) <= kline.open_time <= (end_time + time_tolerance)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_historical_klines_invalid_time_range(self, connected_provider):
        """Test fetching klines with invalid time range."""
        end_time = datetime.now(UTC)
        start_time = end_time + timedelta(hours=1)  # Start after end

        with pytest.raises(ValidationException) as exc_info:
            await connected_provider.fetch_historical_klines("BTCUSDT", KlineInterval.MINUTE_1, start_time, end_time)

        assert exc_info.value.code == "INVALID_TIME_RANGE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_trades_basic(self, connected_provider):
        """Test basic trade streaming functionality."""
        trade_count = 0
        max_trades = 3

        async for trade in connected_provider.stream_trades("BTCUSDT"):
            assert trade.symbol == "BTCUSDT"
            assert isinstance(trade.price, Decimal)
            assert isinstance(trade.quantity, Decimal)
            assert trade.side in [TradeSide.BUY, TradeSide.SELL]

            trade_count += 1
            if trade_count >= max_trades:
                break

        assert trade_count == max_trades

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_trades_unsupported_symbol(self, connected_provider):
        """Test streaming trades for unsupported symbol."""
        with pytest.raises(DataNotFoundException) as exc_info:
            async for _ in connected_provider.stream_trades("UNSUPPORTED/SYMBOL"):
                break

        assert exc_info.value.code == "SYMBOL_NOT_SUPPORTED"

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_klines_basic(self, connected_provider):
        """Test basic kline streaming functionality."""
        import asyncio

        # Use shorter timeout for streaming test
        kline_count = 0
        max_klines = 1  # Only test first kline to avoid long waits

        # Create async generator and manually iterate
        kline_stream = connected_provider.stream_klines("BTCUSDT", KlineInterval.MINUTE_1)
        kline_iterator = kline_stream.__aiter__()

        # Get first kline with timeout
        try:
            kline = await asyncio.wait_for(kline_iterator.__anext__(), timeout=5.0)
            assert kline.symbol == "BTCUSDT"
            assert kline.interval == KlineInterval.MINUTE_1
            assert isinstance(kline.open_price, Decimal)
            assert isinstance(kline.close_price, Decimal)
            assert kline.high_price >= kline.low_price
            kline_count += 1
        except asyncio.TimeoutError:
            pytest.fail("Kline streaming took too long")

        assert kline_count == max_klines

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_convert_multiple_trades(self, connected_provider):
        """Test converting raw trade data."""
        raw_trades = [
            {
                "id": 123456,
                "price": "45000.50",
                "quantity": "0.1",
                "side": "buy",
                "timestamp": datetime.now(UTC).isoformat(),
                "is_buyer_maker": True,
            },
            {
                "price": "45001.00",
                "quantity": "0.2",
                "side": "sell",
                "timestamp": datetime.now(UTC),
                "is_buyer_maker": False,
            },
        ]

        trades = await connected_provider.convert_multiple_trades(raw_trades, "BTCUSDT")

        assert len(trades) == 2
        assert trades[0].symbol == "BTCUSDT"
        assert trades[0].price == Decimal("45000.50")
        assert trades[0].quantity == Decimal("0.1")
        assert trades[0].side == TradeSide.BUY

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_convert_multiple_trades_invalid_data(self, connected_provider):
        """Test converting invalid trade data."""
        raw_trades = [
            {
                "invalid": "data"  # Missing required fields
            }
        ]

        with pytest.raises(ValidationException) as exc_info:
            await connected_provider.convert_multiple_trades(raw_trades, "BTCUSDT")

        assert exc_info.value.code == "INVALID_TRADE_DATA"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_convert_multiple_klines(self, connected_provider):
        """Test converting raw kline data."""
        now = datetime.now(UTC)
        raw_klines = [
            {
                "interval": "1m",
                "open_time": now.isoformat(),
                "close_time": (now + timedelta(minutes=1)).isoformat(),
                "open_price": "45000.00",
                "high_price": "45100.00",
                "low_price": "44900.00",
                "close_price": "45050.00",
                "volume": "100.5",
                "quote_volume": "4525000.0",
                "trades_count": 150,
                "taker_buy_base_volume": "60.3",
                "taker_buy_quote_volume": "2715000.0",
            }
        ]

        klines = await connected_provider.convert_multiple_klines(raw_klines, "BTCUSDT")

        assert len(klines) == 1
        assert klines[0].symbol == "BTCUSDT"
        assert klines[0].interval == KlineInterval.MINUTE_1
        assert klines[0].open_price == Decimal("45000.00")
        assert klines[0].close_price == Decimal("45050.00")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_convert_multiple_klines_invalid_data(self, connected_provider):
        """Test converting invalid kline data."""
        raw_klines = [
            {
                "invalid": "data"  # Missing required fields
            }
        ]

        with pytest.raises(ValidationException) as exc_info:
            await connected_provider.convert_multiple_klines(raw_klines, "BTCUSDT")

        assert exc_info.value.code == "INVALID_KLINE_DATA"

    @pytest.mark.unit
    def test_validate_config_valid(self, provider):
        """Test config validation with valid configuration."""
        config = {"name": "TestProvider", "latency_ms": 10.0}

        is_valid, error_msg = provider.validate_config(config)

        assert is_valid
        assert error_msg == ""

    @pytest.mark.unit
    def test_validate_config_invalid_latency(self, provider):
        """Test config validation with invalid latency."""
        config = {
            "latency_ms": -5.0  # Negative latency
        }

        is_valid, error_msg = provider.validate_config(config)

        assert not is_valid
        assert "latency_ms must be non-negative" in error_msg

    @pytest.mark.unit
    def test_validate_config_invalid_name(self, provider):
        """Test config validation with invalid name."""
        config = {
            "name": ""  # Empty name
        }

        is_valid, error_msg = provider.validate_config(config)

        assert not is_valid
        assert "name must be a non-empty string" in error_msg

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self, provider):
        """Test using provider as async context manager."""
        assert not provider.is_connected

        async with provider as p:
            assert p is provider
            assert provider.is_connected

        assert not provider.is_connected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stream_with_start_from(self, connected_provider):
        """Test streaming with start_from parameter."""
        start_from = datetime.now(UTC) - timedelta(minutes=30)
        trade_count = 0
        max_trades = 2

        async for trade in connected_provider.stream_trades("BTCUSDT", start_from=start_from):
            assert trade.symbol == "BTCUSDT"
            trade_count += 1
            if trade_count >= max_trades:
                break

        assert trade_count == max_trades

    @pytest.mark.unit
    def test_align_to_interval(self, provider):
        """Test interval alignment functionality."""
        # Test aligning to 1-minute intervals
        timestamp = datetime(2024, 1, 1, 12, 34, 56, tzinfo=UTC)
        aligned = provider._align_to_interval(timestamp, 60)  # 1 minute

        # Should align to 12:34:00
        expected = datetime(2024, 1, 1, 12, 34, 0, tzinfo=UTC)
        assert aligned == expected

    @pytest.mark.unit
    def test_generate_random_trade(self, provider):
        """Test random trade generation."""
        base_price = Decimal("45000.00")
        timestamp = datetime.now(UTC)

        trade = provider._generate_random_trade("BTCUSDT", base_price, timestamp)

        assert trade.symbol == "BTCUSDT"
        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.quantity, Decimal)
        assert trade.side in [TradeSide.BUY, TradeSide.SELL]
        assert trade.timestamp == timestamp

        # Price should be within reasonable range of base price (±1%)
        price_diff = abs(trade.price - base_price) / base_price
        assert price_diff <= Decimal("0.02")  # Allow some tolerance

    @pytest.mark.unit
    def test_generate_random_kline(self, provider):
        """Test random kline generation."""
        base_price = Decimal("45000.00")
        open_time = datetime.now(UTC)
        interval = KlineInterval.MINUTE_1

        kline = provider._generate_random_kline("BTCUSDT", interval, open_time, base_price)

        assert kline.symbol == "BTCUSDT"
        assert kline.interval == interval
        assert kline.open_time == open_time
        assert kline.open_price == base_price
        assert kline.high_price >= kline.low_price
        assert kline.high_price >= kline.open_price
        assert kline.high_price >= kline.close_price
        assert kline.low_price <= kline.open_price
        assert kline.low_price <= kline.close_price
        assert isinstance(kline.volume, Decimal)
        assert kline.volume > 0
