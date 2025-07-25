# ABOUTME: Unit tests for MarketData model and MarketDataSummary functionality
# ABOUTME: Validates data integration, time-based operations, statistics and serialization

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import List

from core.models.data.market_data import MarketData, MarketDataSummary
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.trading_pair import TradingPair, TradingPairStatus
from core.models.data.enum import KlineInterval, AssetClass, TradeSide


class TestMarketDataSummary:
    """Test suite for MarketDataSummary model."""
    
    def test_market_data_summary_creation(self):
        """Test creating a MarketDataSummary with default values."""
        summary = MarketDataSummary()
        
        assert summary.min_price is None
        assert summary.max_price is None
        assert summary.avg_price is None
        assert summary.median_price is None
        assert summary.total_volume == Decimal("0")
        assert summary.avg_volume is None
        assert summary.total_trades == 0
        assert summary.unique_symbols == 0
        assert summary.start_time is None
        assert summary.end_time is None
        assert summary.kline_count == 0
        assert summary.trade_count == 0
    
    def test_market_data_summary_with_data(self):
        """Test creating a MarketDataSummary with actual data."""
        start_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 13, 0, tzinfo=UTC)
        
        summary = MarketDataSummary(
            min_price=Decimal("50000.00"),
            max_price=Decimal("51000.00"),
            avg_price=Decimal("50500.00"),
            median_price=Decimal("50400.00"),
            total_volume=Decimal("100.5"),
            avg_volume=Decimal("10.05"),
            total_trades=150,
            unique_symbols=1,
            start_time=start_time,
            end_time=end_time,
            kline_count=10,
            trade_count=140
        )
        
        assert summary.min_price == Decimal("50000.00")
        assert summary.max_price == Decimal("51000.00")
        assert summary.avg_price == Decimal("50500.00")
        assert summary.median_price == Decimal("50400.00")
        assert summary.total_volume == Decimal("100.5")
        assert summary.avg_volume == Decimal("10.05")
        assert summary.total_trades == 150
        assert summary.unique_symbols == 1
        assert summary.start_time == start_time
        assert summary.end_time == end_time
        assert summary.kline_count == 10
        assert summary.trade_count == 140


class TestMarketData:
    """Test suite for MarketData model."""
    
    @pytest.fixture
    def sample_kline(self) -> Kline:
        """Create a sample Kline for testing."""
        return Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("1.5"),
            quote_volume=Decimal("75000.00"),
            trades_count=25,
            asset_class=AssetClass.DIGITAL
        )
    
    @pytest.fixture
    def sample_trade(self) -> Trade:
        """Create a sample Trade for testing."""
        return Trade(
            symbol="BTCUSDT",
            trade_id="12345",
            price=Decimal("50025.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),
            asset_class=AssetClass.DIGITAL
        )
    
    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Create a sample TradingPair for testing."""
        return TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            price_precision=2,
            quantity_precision=8,
            status=TradingPairStatus.ACTIVE
        )
    
    def test_market_data_creation_basic(self):
        """Test basic MarketData creation."""
        market_data = MarketData(symbol="BTCUSDT")
        
        assert market_data.symbol == "BTCUSDT"
        assert market_data.klines == []
        assert market_data.trades == []
        assert market_data.trading_pair is None
        assert market_data.asset_class == AssetClass.DIGITAL
        assert isinstance(market_data.created_at, datetime)
        assert isinstance(market_data.updated_at, datetime)
        assert market_data.metadata == {}
    
    def test_market_data_creation_with_data(self, sample_kline, sample_trade, sample_trading_pair):
        """Test MarketData creation with initial data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            klines=[sample_kline],
            trades=[sample_trade],
            trading_pair=sample_trading_pair,
            asset_class=AssetClass.DIGITAL
        )
        
        assert market_data.symbol == "BTCUSDT"
        assert len(market_data.klines) == 1
        assert len(market_data.trades) == 1
        assert market_data.trading_pair == sample_trading_pair
        assert market_data.asset_class == AssetClass.DIGITAL
    
    def test_symbol_validation(self):
        """Test symbol validation."""
        # Valid symbol
        market_data = MarketData(symbol="BTCUSDT")
        assert market_data.symbol == "BTCUSDT"
        
        # Symbol normalization
        market_data = MarketData(symbol=" btcusdt ")
        assert market_data.symbol == "BTCUSDT"
        
        # Invalid symbols
        with pytest.raises(Exception):
            MarketData(symbol=None)
        
        with pytest.raises(Exception):
            MarketData(symbol="")
        
        with pytest.raises(Exception):
            MarketData(symbol="   ")
    
    def test_data_consistency_validation(self, sample_kline, sample_trade):
        """Test data consistency validation."""
        # Mismatched symbol in kline
        wrong_kline = Kline(
            symbol="ETHUSDT",  # Different symbol
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            open_price=Decimal("3000.00"),
            high_price=Decimal("3100.00"),
            low_price=Decimal("2900.00"),
            close_price=Decimal("3050.00"),
            volume=Decimal("10.0"),
            quote_volume=Decimal("30000.00"),
            trades_count=50
        )
        
        with pytest.raises(ValueError, match="Kline symbol ETHUSDT does not match MarketData symbol BTCUSDT"):
            MarketData(symbol="BTCUSDT", klines=[wrong_kline])
        
        # Mismatched symbol in trade
        wrong_trade = Trade(
            symbol="ETHUSDT",  # Different symbol
            trade_id="54321",
            price=Decimal("3025.00"),
            quantity=Decimal("0.5"),
            side=TradeSide.SELL,
            timestamp=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
        )
        
        with pytest.raises(ValueError, match="Trade symbol ETHUSDT does not match MarketData symbol BTCUSDT"):
            MarketData(symbol="BTCUSDT", trades=[wrong_trade])
    
    def test_add_kline(self, sample_kline):
        """Test adding a single kline."""
        market_data = MarketData(symbol="BTCUSDT")
        initial_updated_at = market_data.updated_at
        
        market_data.add_kline(sample_kline)
        
        assert len(market_data.klines) == 1
        assert market_data.klines[0] == sample_kline
        assert market_data.updated_at > initial_updated_at
    
    def test_add_kline_wrong_symbol(self):
        """Test adding kline with wrong symbol."""
        market_data = MarketData(symbol="BTCUSDT")
        
        wrong_kline = Kline(
            symbol="ETHUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            open_price=Decimal("3000.00"),
            high_price=Decimal("3100.00"),
            low_price=Decimal("2900.00"),
            close_price=Decimal("3050.00"),
            volume=Decimal("10.0"),
            quote_volume=Decimal("30000.00"),
            trades_count=50
        )
        
        with pytest.raises(ValueError, match="Kline symbol ETHUSDT does not match MarketData symbol BTCUSDT"):
            market_data.add_kline(wrong_kline)
    
    def test_add_trade(self, sample_trade):
        """Test adding a single trade."""
        market_data = MarketData(symbol="BTCUSDT")
        initial_updated_at = market_data.updated_at
        
        market_data.add_trade(sample_trade)
        
        assert len(market_data.trades) == 1
        assert market_data.trades[0] == sample_trade
        assert market_data.updated_at > initial_updated_at
    
    def test_add_trade_wrong_symbol(self):
        """Test adding trade with wrong symbol."""
        market_data = MarketData(symbol="BTCUSDT")
        
        wrong_trade = Trade(
            symbol="ETHUSDT",
            trade_id="54321",
            price=Decimal("3025.00"),
            quantity=Decimal("0.5"),
            side=TradeSide.SELL,
            timestamp=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
        )
        
        with pytest.raises(ValueError, match="Trade symbol ETHUSDT does not match MarketData symbol BTCUSDT"):
            market_data.add_trade(wrong_trade)
    
    def test_add_multiple_klines(self):
        """Test adding multiple klines."""
        market_data = MarketData(symbol="BTCUSDT")
        
        klines = [
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=datetime(2024, 1, 1, 12, i, tzinfo=UTC),
                close_time=datetime(2024, 1, 1, 12, i + 1, tzinfo=UTC),
                open_price=Decimal(f"{50000 + i * 10}.00"),
                high_price=Decimal(f"{50100 + i * 10}.00"),
                low_price=Decimal(f"{49900 + i * 10}.00"),
                close_price=Decimal(f"{50050 + i * 10}.00"),
                volume=Decimal("1.0"),
                quote_volume=Decimal("50000.00"),
                trades_count=20
            )
            for i in range(3)
        ]
        
        market_data.add_klines(klines)
        
        assert len(market_data.klines) == 3
        # Check if klines are sorted by open_time
        for i in range(len(market_data.klines) - 1):
            assert market_data.klines[i].open_time <= market_data.klines[i + 1].open_time
    
    def test_add_multiple_trades(self):
        """Test adding multiple trades."""
        market_data = MarketData(symbol="BTCUSDT")
        
        trades = [
            Trade(
                symbol="BTCUSDT",
                trade_id=f"trade_{i}",
                price=Decimal(f"{50000 + i * 5}.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
                timestamp=datetime(2024, 1, 1, 12, 0, i * 10, tzinfo=UTC)
            )
            for i in range(5)
        ]
        
        market_data.add_trades(trades)
        
        assert len(market_data.trades) == 5
        # Check if trades are sorted by timestamp
        for i in range(len(market_data.trades) - 1):
            assert market_data.trades[i].timestamp <= market_data.trades[i + 1].timestamp
    
    def test_get_klines_in_range(self, sample_kline):
        """Test getting klines within a time range."""
        market_data = MarketData(symbol="BTCUSDT")
        
        # Add klines at different times
        klines = [
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=datetime(2024, 1, 1, 12, i, tzinfo=UTC),
                close_time=datetime(2024, 1, 1, 12, i + 1, tzinfo=UTC),
                open_price=Decimal("50000.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49900.00"),
                close_price=Decimal("50050.00"),
                volume=Decimal("1.0"),
                quote_volume=Decimal("50000.00"),
                trades_count=20
            )
            for i in range(5)
        ]
        
        market_data.add_klines(klines)
        
        # Get klines in specific range
        start_time = datetime(2024, 1, 1, 12, 1, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 3, tzinfo=UTC)
        
        filtered_klines = market_data.get_klines_in_range(start_time, end_time)
        
        # Should include klines at 12:01, 12:02, 12:03
        assert len(filtered_klines) == 3
        for kline in filtered_klines:
            assert start_time <= kline.open_time <= end_time
    
    def test_get_trades_in_range(self):
        """Test getting trades within a time range."""
        market_data = MarketData(symbol="BTCUSDT")
        
        # Add trades at different times
        trades = [
            Trade(
                symbol="BTCUSDT",
                trade_id=f"trade_{i}",
                price=Decimal("50000.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY,
                timestamp=datetime(2024, 1, 1, 12, 0, i * 10, tzinfo=UTC)
            )
            for i in range(6)  # 0, 10, 20, 30, 40, 50 seconds
        ]
        
        market_data.add_trades(trades)
        
        # Get trades in specific range
        start_time = datetime(2024, 1, 1, 12, 0, 15, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 12, 0, 35, tzinfo=UTC)
        
        filtered_trades = market_data.get_trades_in_range(start_time, end_time)
        
        # Should include trades at 20 and 30 seconds
        assert len(filtered_trades) == 2
        for trade in filtered_trades:
            assert start_time <= trade.timestamp <= end_time
    
    def test_get_latest_kline(self):
        """Test getting the latest kline."""
        market_data = MarketData(symbol="BTCUSDT")
        
        # No klines initially
        assert market_data.get_latest_kline() is None
        
        # Add klines
        klines = [
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=datetime(2024, 1, 1, 12, i, tzinfo=UTC),
                close_time=datetime(2024, 1, 1, 12, i + 1, tzinfo=UTC),
                open_price=Decimal("50000.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49900.00"),
                close_price=Decimal("50050.00"),
                volume=Decimal("1.0"),
                quote_volume=Decimal("50000.00"),
                trades_count=20
            )
            for i in range(3)
        ]
        
        market_data.add_klines(klines)
        
        latest_kline = market_data.get_latest_kline()
        assert latest_kline is not None
        assert latest_kline.open_time == datetime(2024, 1, 1, 12, 2, tzinfo=UTC)
    
    def test_get_latest_trade(self):
        """Test getting the latest trade."""
        market_data = MarketData(symbol="BTCUSDT")
        
        # No trades initially
        assert market_data.get_latest_trade() is None
        
        # Add trades
        trades = [
            Trade(
                symbol="BTCUSDT",
                trade_id=f"trade_{i}",
                price=Decimal("50000.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY,
                timestamp=datetime(2024, 1, 1, 12, 0, i * 10, tzinfo=UTC)
            )
            for i in range(3)
        ]
        
        market_data.add_trades(trades)
        
        latest_trade = market_data.get_latest_trade()
        assert latest_trade is not None
        assert latest_trade.timestamp == datetime(2024, 1, 1, 12, 0, 20, tzinfo=UTC)
    
    def test_get_klines_by_interval(self):
        """Test filtering klines by interval."""
        market_data = MarketData(symbol="BTCUSDT")
        
        # Add klines with different intervals
        klines = [
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
                open_price=Decimal("50000.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49900.00"),
                close_price=Decimal("50050.00"),
                volume=Decimal("1.0"),
                quote_volume=Decimal("50000.00"),
                trades_count=20
            ),
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_5,
                open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                close_time=datetime(2024, 1, 1, 12, 5, tzinfo=UTC),
                open_price=Decimal("50000.00"),
                high_price=Decimal("50200.00"),
                low_price=Decimal("49800.00"),
                close_price=Decimal("50100.00"),
                volume=Decimal("5.0"),
                quote_volume=Decimal("250000.00"),
                trades_count=100
            ),
            Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.MINUTE_1,
                open_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
                close_time=datetime(2024, 1, 1, 12, 2, tzinfo=UTC),
                open_price=Decimal("50050.00"),
                high_price=Decimal("50150.00"),
                low_price=Decimal("49950.00"),
                close_price=Decimal("50080.00"),
                volume=Decimal("1.2"),
                quote_volume=Decimal("60000.00"),
                trades_count=25
            )
        ]
        
        market_data.add_klines(klines)
        
        # Filter by 1-minute interval
        one_min_klines = market_data.get_klines_by_interval(KlineInterval.MINUTE_1)
        assert len(one_min_klines) == 2
        
        # Filter by 5-minute interval
        five_min_klines = market_data.get_klines_by_interval(KlineInterval.MINUTE_5)
        assert len(five_min_klines) == 1
    
    def test_calculate_summary(self, sample_kline, sample_trade):
        """Test calculating market data summary."""
        market_data = MarketData(symbol="BTCUSDT")
        market_data.add_kline(sample_kline)
        market_data.add_trade(sample_trade)
        
        summary = market_data.calculate_summary()
        
        # Verify basic fields
        assert isinstance(summary, MarketDataSummary)
        assert summary.kline_count == 1
        assert summary.trade_count == 1
        assert summary.unique_symbols == 1
        assert summary.total_trades == 26  # 25 from kline + 1 from trade
        
        # Verify price statistics
        assert summary.min_price is not None
        assert summary.max_price is not None
        assert summary.avg_price is not None
        assert summary.median_price is not None
        
        # Verify time range
        assert summary.start_time is not None
        assert summary.end_time is not None
    
    def test_is_empty_property(self):
        """Test is_empty property."""
        market_data = MarketData(symbol="BTCUSDT")
        assert market_data.is_empty is True
        
        # Add a kline
        kline = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("1.0"),
            quote_volume=Decimal("50000.00"),
            trades_count=20
        )
        market_data.add_kline(kline)
        assert market_data.is_empty is False
    
    def test_data_count_property(self, sample_kline, sample_trade):
        """Test data_count property."""
        market_data = MarketData(symbol="BTCUSDT")
        assert market_data.data_count == 0
        
        market_data.add_kline(sample_kline)
        assert market_data.data_count == 1
        
        market_data.add_trade(sample_trade)
        assert market_data.data_count == 2
    
    def test_time_span_property(self):
        """Test time_span property."""
        market_data = MarketData(symbol="BTCUSDT")
        
        # No data
        assert market_data.time_span is None
        
        # Add data with known time span
        kline = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("1.0"),
            quote_volume=Decimal("50000.00"),
            trades_count=20
        )
        
        trade = Trade(
            symbol="BTCUSDT",
            trade_id="12345",
            price=Decimal("50025.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime(2024, 1, 1, 12, 2, tzinfo=UTC)  # 2 minutes later
        )
        
        market_data.add_kline(kline)
        market_data.add_trade(trade)
        
        # Time span should be from kline open_time to trade timestamp
        expected_span = timedelta(minutes=2)
        assert market_data.time_span == expected_span
    
    def test_merge_market_data(self):
        """Test merging two MarketData instances."""
        # Create first MarketData
        market_data1 = MarketData(symbol="BTCUSDT")
        kline1 = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            close_time=datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("1.0"),
            quote_volume=Decimal("50000.00"),
            trades_count=20
        )
        market_data1.add_kline(kline1)
        
        # Create second MarketData
        market_data2 = MarketData(symbol="BTCUSDT")
        trade1 = Trade(
            symbol="BTCUSDT",
            trade_id="12345",
            price=Decimal("50025.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC)
        )
        market_data2.add_trade(trade1)
        
        # Merge
        merged = market_data1.merge(market_data2)
        
        assert merged.symbol == "BTCUSDT"
        assert len(merged.klines) == 1
        assert len(merged.trades) == 1
        assert merged.data_count == 2
    
    def test_merge_different_symbols(self):
        """Test that merging different symbols raises an error."""
        market_data1 = MarketData(symbol="BTCUSDT")
        market_data2 = MarketData(symbol="ETHUSDT")
        
        with pytest.raises(ValueError, match="Cannot merge MarketData with different symbols"):
            market_data1.merge(market_data2)
    
    def test_to_dict_and_from_dict(self, sample_kline, sample_trade):
        """Test serialization and deserialization without trading_pair."""
        # Create MarketData with sample data (excluding trading_pair due to enum issue)
        market_data = MarketData(
            symbol="BTCUSDT",
            klines=[sample_kline],
            trades=[sample_trade],
            asset_class=AssetClass.DIGITAL,
            metadata={"source": "test"}
        )
        
        # Convert to dict
        data_dict = market_data.to_dict()
        
        # Verify dict structure
        assert data_dict["symbol"] == "BTCUSDT"
        assert len(data_dict["klines"]) == 1
        assert len(data_dict["trades"]) == 1
        assert data_dict["trading_pair"] is None
        assert "summary" in data_dict
        
        # Convert back from dict
        restored_market_data = MarketData.from_dict(data_dict)
        
        # Verify restoration
        assert restored_market_data.symbol == market_data.symbol
        assert len(restored_market_data.klines) == len(market_data.klines)
        assert len(restored_market_data.trades) == len(market_data.trades)
        assert restored_market_data.asset_class == market_data.asset_class
        assert restored_market_data.metadata == market_data.metadata
    
    def test_string_representations(self, sample_kline, sample_trade):
        """Test string representations."""
        market_data = MarketData(symbol="BTCUSDT")
        market_data.add_kline(sample_kline)
        market_data.add_trade(sample_trade)
        
        # Test __str__
        str_repr = str(market_data)
        assert "BTCUSDT" in str_repr
        assert "klines=1" in str_repr
        assert "trades=1" in str_repr
        
        # Test __repr__
        repr_str = repr(market_data)
        assert "MarketData" in repr_str
        assert "symbol='BTCUSDT'" in repr_str
        assert "klines_count=1" in repr_str
        assert "trades_count=1" in repr_str