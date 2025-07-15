# ABOUTME: Unit tests for data enumeration classes including KlineInterval, AssetClass, and TradeSide
# ABOUTME: Tests cover normal cases, exception cases, and boundary cases with comprehensive validation

import pytest
from datetime import timedelta

from core.models.data.enum import KlineInterval, AssetClass, TradeSide


class TestKlineInterval:
    """Test cases for KlineInterval enum."""

    def test_kline_interval_values_normal_case(self):
        """Test that KlineInterval enum has correct string values."""
        assert KlineInterval.MINUTE_1 == "1m"
        assert KlineInterval.MINUTE_3 == "3m"
        assert KlineInterval.MINUTE_5 == "5m"
        assert KlineInterval.MINUTE_15 == "15m"
        assert KlineInterval.MINUTE_30 == "30m"
        assert KlineInterval.HOUR_1 == "1h"
        assert KlineInterval.HOUR_2 == "2h"
        assert KlineInterval.HOUR_4 == "4h"
        assert KlineInterval.HOUR_6 == "6h"
        assert KlineInterval.HOUR_8 == "8h"
        assert KlineInterval.HOUR_12 == "12h"
        assert KlineInterval.DAY_1 == "1d"
        assert KlineInterval.DAY_3 == "3d"
        assert KlineInterval.WEEK_1 == "1w"
        assert KlineInterval.MONTH_1 == "1M"

    def test_to_seconds_normal_case(self):
        """Test to_seconds method with valid intervals."""
        assert KlineInterval.to_seconds(KlineInterval.MINUTE_1) == 60
        assert KlineInterval.to_seconds(KlineInterval.MINUTE_3) == 180
        assert KlineInterval.to_seconds(KlineInterval.MINUTE_5) == 300
        assert KlineInterval.to_seconds(KlineInterval.MINUTE_15) == 900
        assert KlineInterval.to_seconds(KlineInterval.MINUTE_30) == 1800
        assert KlineInterval.to_seconds(KlineInterval.HOUR_1) == 3600
        assert KlineInterval.to_seconds(KlineInterval.HOUR_2) == 7200
        assert KlineInterval.to_seconds(KlineInterval.HOUR_4) == 14400
        assert KlineInterval.to_seconds(KlineInterval.HOUR_6) == 21600
        assert KlineInterval.to_seconds(KlineInterval.HOUR_8) == 28800
        assert KlineInterval.to_seconds(KlineInterval.HOUR_12) == 43200
        assert KlineInterval.to_seconds(KlineInterval.DAY_1) == 86400
        assert KlineInterval.to_seconds(KlineInterval.DAY_3) == 259200
        assert KlineInterval.to_seconds(KlineInterval.WEEK_1) == 604800
        assert KlineInterval.to_seconds(KlineInterval.MONTH_1) == 2592000

    def test_to_timedelta_normal_case(self):
        """Test to_timedelta method with valid intervals."""
        assert KlineInterval.to_timedelta(KlineInterval.MINUTE_1) == timedelta(seconds=60)
        assert KlineInterval.to_timedelta(KlineInterval.HOUR_1) == timedelta(seconds=3600)
        assert KlineInterval.to_timedelta(KlineInterval.DAY_1) == timedelta(seconds=86400)
        assert KlineInterval.to_timedelta(KlineInterval.WEEK_1) == timedelta(seconds=604800)

    def test_to_seconds_consistency_with_timedelta(self):
        """Test that to_seconds and to_timedelta are consistent."""
        for interval in KlineInterval:
            seconds = KlineInterval.to_seconds(interval)
            td = KlineInterval.to_timedelta(interval)
            assert td.total_seconds() == seconds

    def test_kline_interval_iteration(self):
        """Test iteration over all KlineInterval members."""
        intervals = list(KlineInterval)
        assert len(intervals) == 15  # Total number of intervals
        
        # Check that all expected intervals are present
        expected_intervals = [
            KlineInterval.MINUTE_1, KlineInterval.MINUTE_3, KlineInterval.MINUTE_5,
            KlineInterval.MINUTE_15, KlineInterval.MINUTE_30, KlineInterval.HOUR_1,
            KlineInterval.HOUR_2, KlineInterval.HOUR_4, KlineInterval.HOUR_6,
            KlineInterval.HOUR_8, KlineInterval.HOUR_12, KlineInterval.DAY_1,
            KlineInterval.DAY_3, KlineInterval.WEEK_1, KlineInterval.MONTH_1
        ]
        
        for expected in expected_intervals:
            assert expected in intervals

    def test_kline_interval_from_string(self):
        """Test creating KlineInterval from string value."""
        assert KlineInterval("1m") == KlineInterval.MINUTE_1
        assert KlineInterval("1h") == KlineInterval.HOUR_1
        assert KlineInterval("1d") == KlineInterval.DAY_1
        assert KlineInterval("1w") == KlineInterval.WEEK_1
        assert KlineInterval("1M") == KlineInterval.MONTH_1

    def test_kline_interval_invalid_value_raises_exception(self):
        """Test that invalid interval value raises ValueError."""
        with pytest.raises(ValueError):
            KlineInterval("invalid_interval")
        
        with pytest.raises(ValueError):
            KlineInterval("2m")  # Not a valid interval
        
        with pytest.raises(ValueError):
            KlineInterval("")

    def test_to_seconds_mathematical_relationships(self):
        """Test mathematical relationships between different intervals."""
        # 1 hour = 60 minutes
        assert KlineInterval.to_seconds(KlineInterval.HOUR_1) == 60 * KlineInterval.to_seconds(KlineInterval.MINUTE_1)
        
        # 1 day = 24 hours
        assert KlineInterval.to_seconds(KlineInterval.DAY_1) == 24 * KlineInterval.to_seconds(KlineInterval.HOUR_1)
        
        # 3 minutes = 3 * 1 minute
        assert KlineInterval.to_seconds(KlineInterval.MINUTE_3) == 3 * KlineInterval.to_seconds(KlineInterval.MINUTE_1)
        
        # 3 days = 3 * 1 day
        assert KlineInterval.to_seconds(KlineInterval.DAY_3) == 3 * KlineInterval.to_seconds(KlineInterval.DAY_1)


class TestAssetClass:
    """Test cases for AssetClass enum."""

    def test_asset_class_values_normal_case(self):
        """Test that AssetClass enum has correct string values."""
        assert AssetClass.DIGITAL == "digital"
        assert AssetClass.TRADITIONAL == "traditional"
        assert AssetClass.DERIVATIVE == "derivative"
        assert AssetClass.SYNTHETIC == "synthetic"

    def test_asset_class_string_representation(self):
        """Test __str__ method returns correct string representation."""
        assert str(AssetClass.DIGITAL) == "digital"
        assert str(AssetClass.TRADITIONAL) == "traditional"
        assert str(AssetClass.DERIVATIVE) == "derivative"
        assert str(AssetClass.SYNTHETIC) == "synthetic"

    def test_asset_class_repr_representation(self):
        """Test __repr__ method returns correct representation."""
        assert repr(AssetClass.DIGITAL) == "AssetClass.DIGITAL"
        assert repr(AssetClass.TRADITIONAL) == "AssetClass.TRADITIONAL"
        assert repr(AssetClass.DERIVATIVE) == "AssetClass.DERIVATIVE"
        assert repr(AssetClass.SYNTHETIC) == "AssetClass.SYNTHETIC"

    def test_asset_class_equality(self):
        """Test equality comparison of AssetClass enum members."""
        assert AssetClass.DIGITAL == AssetClass.DIGITAL
        assert AssetClass.TRADITIONAL == AssetClass.TRADITIONAL
        assert AssetClass.DIGITAL != AssetClass.TRADITIONAL
        assert AssetClass.DERIVATIVE != AssetClass.SYNTHETIC

    def test_asset_class_membership(self):
        """Test membership checking for AssetClass enum."""
        asset_values = [asset.value for asset in AssetClass]
        assert "digital" in asset_values
        assert "traditional" in asset_values
        assert "derivative" in asset_values
        assert "synthetic" in asset_values
        assert "invalid_asset" not in asset_values

    def test_asset_class_iteration(self):
        """Test iteration over AssetClass enum members."""
        assets = list(AssetClass)
        assert len(assets) == 4
        assert AssetClass.DIGITAL in assets
        assert AssetClass.TRADITIONAL in assets
        assert AssetClass.DERIVATIVE in assets
        assert AssetClass.SYNTHETIC in assets

    def test_asset_class_from_string(self):
        """Test creating AssetClass from string value."""
        assert AssetClass("digital") == AssetClass.DIGITAL
        assert AssetClass("traditional") == AssetClass.TRADITIONAL
        assert AssetClass("derivative") == AssetClass.DERIVATIVE
        assert AssetClass("synthetic") == AssetClass.SYNTHETIC

    def test_asset_class_invalid_value_raises_exception(self):
        """Test that invalid asset class value raises ValueError."""
        with pytest.raises(ValueError):
            AssetClass("invalid_asset")
        
        with pytest.raises(ValueError):
            AssetClass("")
        
        with pytest.raises(ValueError):
            AssetClass("DIGITAL")  # Case sensitive


class TestTradeSide:
    """Test cases for TradeSide enum."""

    def test_trade_side_values_normal_case(self):
        """Test that TradeSide enum has correct string values."""
        assert TradeSide.BUY == "buy"
        assert TradeSide.SELL == "sell"

    def test_trade_side_string_representation(self):
        """Test string representation of TradeSide enum members."""
        assert str(TradeSide.BUY) == "TradeSide.BUY"
        assert str(TradeSide.SELL) == "TradeSide.SELL"

    def test_trade_side_equality(self):
        """Test equality comparison of TradeSide enum members."""
        assert TradeSide.BUY == TradeSide.BUY
        assert TradeSide.SELL == TradeSide.SELL
        assert TradeSide.BUY != TradeSide.SELL

    def test_trade_side_membership(self):
        """Test membership checking for TradeSide enum."""
        side_values = [side.value for side in TradeSide]
        assert "buy" in side_values
        assert "sell" in side_values
        assert "invalid_side" not in side_values

    def test_trade_side_iteration(self):
        """Test iteration over TradeSide enum members."""
        sides = list(TradeSide)
        assert len(sides) == 2
        assert TradeSide.BUY in sides
        assert TradeSide.SELL in sides

    def test_trade_side_from_string(self):
        """Test creating TradeSide from string value."""
        assert TradeSide("buy") == TradeSide.BUY
        assert TradeSide("sell") == TradeSide.SELL

    def test_trade_side_invalid_value_raises_exception(self):
        """Test that invalid trade side value raises ValueError."""
        with pytest.raises(ValueError):
            TradeSide("invalid_side")
        
        with pytest.raises(ValueError):
            TradeSide("BUY")  # Case sensitive
        
        with pytest.raises(ValueError):
            TradeSide("")

    def test_trade_side_opposite_logic(self):
        """Test logical relationships between buy and sell sides."""
        # This tests business logic understanding
        all_sides = list(TradeSide)
        assert len(all_sides) == 2
        
        # Ensure we have exactly buy and sell
        assert TradeSide.BUY in all_sides
        assert TradeSide.SELL in all_sides
        
        # Test that they are different
        assert TradeSide.BUY != TradeSide.SELL