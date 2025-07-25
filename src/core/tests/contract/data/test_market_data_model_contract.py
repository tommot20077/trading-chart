# ABOUTME: Contract tests for MarketData model ensuring compliance with expected interfaces
# ABOUTME: Verifies serialization, validation, and business rule contracts for MarketData data model

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Dict, Any, List

from core.models.data.market_data import MarketData, MarketDataSummary
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.trading_pair import TradingPair, TradingPairStatus
from core.models.data.enum import AssetClass, KlineInterval, TradeSide


class TestMarketDataModelContract:
    """
    Contract tests for MarketData model.
    
    These tests verify that the MarketData model conforms to expected contracts
    for serialization, validation, and business logic behavior.
    """

    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Provides a sample trading pair for testing."""
        return TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
        )

    @pytest.fixture
    def sample_klines(self) -> List[Kline]:
        """Provides sample kline data for testing."""
        now = datetime.now(UTC)
        return [
            Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=now - timedelta(minutes=2),
                close_time=now - timedelta(minutes=1),
                open_price=Decimal("50000"),
                high_price=Decimal("50100"),
                low_price=Decimal("49900"),
                close_price=Decimal("50050"),
                volume=Decimal("1.5"),
                quote_volume=Decimal("75075"),
                trades_count=10,
            ),
            Kline(
                symbol="BTC/USDT",
                interval=KlineInterval.MINUTE_1,
                open_time=now - timedelta(minutes=1),
                close_time=now,
                open_price=Decimal("50050"),
                high_price=Decimal("50200"),
                low_price=Decimal("50000"),
                close_price=Decimal("50150"),
                volume=Decimal("2.0"),
                quote_volume=Decimal("100300"),
                trades_count=15,
            ),
        ]

    @pytest.fixture
    def sample_trades(self) -> List[Trade]:
        """Provides sample trade data for testing."""
        now = datetime.now(UTC)
        return [
            Trade(
                symbol="BTC/USDT",
                trade_id="trade_1",
                price=Decimal("50000"),
                quantity=Decimal("0.5"),
                side=TradeSide.BUY,
                timestamp=now - timedelta(minutes=1),
            ),
            Trade(
                symbol="BTC/USDT",
                trade_id="trade_2",
                price=Decimal("50100"),
                quantity=Decimal("0.3"),
                side=TradeSide.SELL,
                timestamp=now - timedelta(seconds=30),
            ),
        ]

    @pytest.fixture
    def market_data_instance(self, sample_klines, sample_trades) -> MarketData:
        """Provides a valid MarketData instance for testing."""
        return MarketData(
            symbol="BTC/USDT",
            klines=sample_klines,
            trades=sample_trades,
        )

    # Serialization Contract Tests

    @pytest.mark.contract
    def test_model_serialization_contract(self, market_data_instance):
        """Test that MarketData model can be serialized to dict consistently."""
        serialized = market_data_instance.to_dict()
        
        # Contract: Must return a dictionary
        assert isinstance(serialized, dict)
        
        # Contract: Must include all required fields
        required_fields = ["symbol", "klines", "trades", "created_at"]
        for field in required_fields:
            assert field in serialized
            
        # Contract: Nested objects should be serialized properly
        assert isinstance(serialized["klines"], list)
        assert isinstance(serialized["trades"], list)

    @pytest.mark.contract
    def test_model_deserialization_contract(self, sample_klines, sample_trades):
        """Test that MarketData model can be deserialized from dict consistently."""
        market_data = MarketData(
            symbol="BTC/USDT",
            klines=sample_klines,
            trades=sample_trades,
        )
        serialized = market_data.to_dict()
        
        # Contract: Must be able to recreate from serialized data
        recreated = MarketData.from_dict(serialized)  # Use from_dict method
        
        # Contract: Recreated instance must have same core data
        assert recreated.symbol == market_data.symbol
        assert len(recreated.klines) == len(market_data.klines)
        assert len(recreated.trades) == len(market_data.trades)

    @pytest.mark.contract
    def test_json_serialization_contract(self, market_data_instance):
        """Test that MarketData model maintains JSON serialization contract."""
        # Contract: Must be JSON serializable
        json_data = market_data_instance.model_dump(mode="json")
        assert isinstance(json_data, dict)
        
        # Contract: All datetime fields must be ISO format strings in JSON mode
        assert isinstance(json_data["created_at"], str)
        # Should be parseable back to datetime
        datetime.fromisoformat(json_data["created_at"].replace('Z', '+00:00'))

    # Validation Contract Tests

    @pytest.mark.contract
    def test_required_field_validation_contract(self, sample_klines, sample_trades):
        """Test that MarketData model enforces required field validation contract."""
        # Contract: symbol is required
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            MarketData(klines=sample_klines, trades=sample_trades)

    @pytest.mark.contract
    def test_data_type_validation_contract(self):
        """Test that MarketData model enforces data type validation contract."""
        # Contract: klines must be list of Kline objects
        with pytest.raises(Exception):
            MarketData(
                symbol="BTC/USDT",
                klines=["invalid"],  # Should be list of Kline objects
                trades=[],
            )
        
        # Contract: trades must be list of Trade objects
        with pytest.raises(Exception):
            MarketData(
                symbol="BTC/USDT",
                klines=[],
                trades=["invalid"],  # Should be list of Trade objects
            )

    # Business Logic Contract Tests

    @pytest.mark.contract
    def test_data_consistency_contract(self, market_data_instance):
        """Test that MarketData model maintains data consistency contract."""
        # Contract: All klines should have the same symbol as MarketData
        for kline in market_data_instance.klines:
            assert kline.symbol == market_data_instance.symbol
        
        # Contract: All trades should have the same symbol as MarketData
        for trade in market_data_instance.trades:
            assert trade.symbol == market_data_instance.symbol

    @pytest.mark.contract
    def test_time_range_contract(self, market_data_instance):
        """Test that MarketData model maintains time range contract."""
        earliest = market_data_instance.get_earliest_timestamp()
        latest = market_data_instance.get_latest_timestamp()
        
        # Contract: Should return datetime objects or None
        assert isinstance(earliest, datetime) or earliest is None
        assert isinstance(latest, datetime) or latest is None
        
        # Contract: If both times exist, earliest should be <= latest
        if earliest and latest:
            assert earliest <= latest

    @pytest.mark.contract
    def test_filtering_contract(self, market_data_instance):
        """Test that MarketData model maintains filtering contract."""
        # Contract: Should be able to get data in range
        now = datetime.now(UTC)
        start_time = now - timedelta(hours=1)
        end_time = now
        
        klines_in_range = market_data_instance.get_klines_in_range(start_time, end_time)
        trades_in_range = market_data_instance.get_trades_in_range(start_time, end_time)
        
        # Contract: Should return lists
        assert isinstance(klines_in_range, list)
        assert isinstance(trades_in_range, list)
        
        # Contract: All data should be within time range
        for kline in klines_in_range:
            assert start_time <= kline.open_time <= end_time
        
        for trade in trades_in_range:
            assert start_time <= trade.timestamp <= end_time

    @pytest.mark.contract
    def test_aggregation_contract(self, market_data_instance):
        """Test that MarketData model maintains aggregation contract."""
        # Contract: Should be able to get klines by interval
        klines_by_interval = market_data_instance.get_klines_by_interval(KlineInterval.MINUTE_1)
        
        # Contract: Should return list of klines
        assert isinstance(klines_by_interval, list)
        
        # Contract: All klines should have the requested interval
        for kline in klines_by_interval:
            assert kline.interval == KlineInterval.MINUTE_1

    @pytest.mark.contract
    def test_statistics_contract(self, market_data_instance):
        """Test that MarketData model maintains statistics contract."""
        summary = market_data_instance.calculate_summary()
        
        # Contract: Should return MarketDataSummary instance
        assert isinstance(summary, MarketDataSummary)
        
        # Contract: Should have valid statistics
        assert isinstance(summary.total_volume, Decimal)
        assert isinstance(summary.total_trades, int)
        assert isinstance(summary.kline_count, int)
        assert isinstance(summary.trade_count, int)
        
        # Contract: Counts should match actual data
        assert summary.kline_count == len(market_data_instance.klines)
        assert summary.trade_count == len(market_data_instance.trades)

    @pytest.mark.contract
    def test_data_validation_contract(self, market_data_instance):
        """Test that MarketData model maintains data validation contract."""
        # Contract: Should be able to validate data integrity
        is_valid = market_data_instance.validate_data_integrity()
        
        # Contract: Should return boolean
        assert isinstance(is_valid, bool)

    @pytest.mark.contract
    def test_empty_data_contract(self):
        """Test that MarketData model handles empty data contract."""
        # Contract: Should handle empty klines and trades
        empty_market_data = MarketData(symbol="BTC/USDT")
        
        # Contract: Should still be valid
        assert empty_market_data.validate_data_integrity()
        
        # Contract: Summary should handle empty data
        summary = empty_market_data.calculate_summary()
        assert summary.kline_count == 0
        assert summary.trade_count == 0
        assert summary.total_volume == Decimal("0")

    @pytest.mark.contract
    def test_immutability_contract(self, market_data_instance):
        """Test that MarketData model maintains immutability contract."""
        original_kline_count = len(market_data_instance.klines)
        original_trade_count = len(market_data_instance.trades)
        
        # Contract: Operations should not modify original data
        now = datetime.now(UTC)
        filtered_klines = market_data_instance.get_klines_in_range(
            now - timedelta(hours=1), now
        )
        
        # Original data should remain unchanged
        assert len(market_data_instance.klines) == original_kline_count
        assert len(market_data_instance.trades) == original_trade_count

    @pytest.mark.contract
    def test_string_representation_contract(self, market_data_instance):
        """Test that MarketData model maintains string representation contract."""
        # Contract: Should have meaningful string representation
        str_repr = str(market_data_instance)
        assert "MarketData(" in str_repr
        assert market_data_instance.symbol in str_repr
        
        # Contract: Should include data counts
        assert "klines=" in str_repr
        assert "trades=" in str_repr