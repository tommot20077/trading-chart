# ABOUTME: Unit tests for NoOpDataConverter
# ABOUTME: Tests for no-operation data conversion implementation

import pytest
from decimal import Decimal
from datetime import datetime

from core.implementations.noop.data.converter import NoOpDataConverter
from core.models.data.enum import KlineInterval


class TestNoOpDataConverter:
    """Test cases for NoOpDataConverter."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test converter initialization."""
        converter = NoOpDataConverter()
        assert converter is not None

    @pytest.mark.unit
    def test_convert_trade(self):
        """Test trade conversion returns fake trade."""
        converter = NoOpDataConverter()
        raw_trade = {"price": 50000, "quantity": 1.5}
        symbol = "BTC/USDT"

        trade = converter.convert_trade(raw_trade, symbol)

        assert trade.symbol == symbol
        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.quantity, Decimal)
        assert isinstance(trade.timestamp, datetime)
        assert trade.trade_id == "noop-trade-id"
        assert isinstance(trade.is_buyer_maker, bool)

    @pytest.mark.unit
    def test_convert_multiple_trades(self):
        """Test converting multiple trades."""
        converter = NoOpDataConverter()
        raw_trades = [{"price": 50000}, {"price": 51000}, {"price": 49000}]
        symbol = "ETH/USDT"

        trades = converter.convert_multiple_trades(raw_trades, symbol)

        assert len(trades) == 3
        assert all(trade.symbol == symbol for trade in trades)
        assert all(isinstance(trade.price, Decimal) for trade in trades)

    @pytest.mark.unit
    def test_convert_kline(self):
        """Test kline conversion returns fake kline."""
        converter = NoOpDataConverter()
        raw_kline = {"open": 50000, "high": 51000, "low": 49000, "close": 50500}
        symbol = "BTC/USDT"

        kline = converter.convert_kline(raw_kline, symbol)

        assert kline.symbol == symbol
        assert kline.interval == KlineInterval.MINUTE_1
        assert isinstance(kline.open_price, Decimal)
        assert isinstance(kline.high_price, Decimal)
        assert isinstance(kline.low_price, Decimal)
        assert isinstance(kline.close_price, Decimal)
        assert isinstance(kline.volume, Decimal)

    @pytest.mark.unit
    def test_convert_multiple_klines(self):
        """Test converting multiple klines."""
        converter = NoOpDataConverter()
        raw_klines = [{"open": 50000}, {"open": 51000}]
        symbol = "ETH/USDT"

        klines = converter.convert_multiple_klines(raw_klines, symbol)

        assert len(klines) == 2
        assert all(kline.symbol == symbol for kline in klines)
        assert all(isinstance(kline.open_price, Decimal) for kline in klines)

    @pytest.mark.unit
    def test_validate_raw_data_always_succeeds(self):
        """Test that data validation always succeeds."""
        converter = NoOpDataConverter()

        # Should always return True regardless of input
        assert converter.validate_raw_data({}) == (True, "")
        assert converter.validate_raw_data({"invalid": "data"}) == (True, "")
        assert converter.validate_raw_data(None) == (True, "")
        assert converter.validate_raw_data([1, 2, 3]) == (True, "")

    @pytest.mark.unit
    def test_to_internal_timestamp(self):
        """Test timestamp conversion returns current time."""
        converter = NoOpDataConverter()

        # Should return current timestamp regardless of input
        ts1 = converter.to_internal_timestamp("2023-01-01")
        ts2 = converter.to_internal_timestamp(1672531200)
        ts3 = converter.to_internal_timestamp(None)

        assert isinstance(ts1, int)
        assert isinstance(ts2, int)
        assert isinstance(ts3, int)
        assert ts1 > 0
        assert ts2 > 0
        assert ts3 > 0
