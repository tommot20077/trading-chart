# ABOUTME: Unit tests for common type definitions
# ABOUTME: Tests TypedDict classes and type aliases for type safety validation

import pytest
from datetime import datetime, UTC
from typing import get_type_hints

from core.models.types import (
    UserData,
    AlertData,
    ExchangeInfo,
    SymbolInfo,
    RawTradeData,
    RawKlineData,
    StatisticsResult,
    BackfillStatus,
    ProviderConfig,
    TimestampType,
    RawDataType,
    MetadataValue,
)


class TestTypeDefinitions:
    """Test suite for common type definitions."""

    def test_user_data_type(self):
        """Test UserData TypedDict functionality."""
        # Should accept optional fields
        user_data: UserData = {
            "user_id": "123",
            "username": "testuser",
            "roles": ["admin", "trader"],
        }
        
        assert user_data["user_id"] == "123"
        assert user_data["username"] == "testuser"
        assert user_data["roles"] == ["admin", "trader"]

    def test_alert_data_type(self):
        """Test AlertData TypedDict functionality."""
        now = datetime.now(UTC)
        
        alert_data: AlertData = {
            "id": "alert_123",
            "rule_name": "price_alert",
            "severity": "high",
            "status": "active",
            "message": "Price threshold exceeded",
            "labels": {"symbol": "BTCUSDT", "exchange": "binance"},
            "annotations": {"description": "Alert triggered"},
            "fired_at": now,
            "trace_id": "trace_456",
        }
        
        assert alert_data["id"] == "alert_123"
        assert alert_data["rule_name"] == "price_alert"
        assert alert_data["fired_at"] == now

    def test_exchange_info_type(self):
        """Test ExchangeInfo TypedDict functionality."""
        now = datetime.now(UTC)
        
        exchange_info: ExchangeInfo = {
            "name": "Binance",
            "timezone": "UTC",
            "server_time": now,
            "symbols": ["BTCUSDT", "ETHUSDT"],
        }
        
        assert exchange_info["name"] == "Binance"
        assert exchange_info["server_time"] == now

    def test_symbol_info_type(self):
        """Test SymbolInfo TypedDict functionality."""
        symbol_info: SymbolInfo = {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "base_precision": 8,
            "quote_precision": 8,
            "order_types": ["LIMIT", "MARKET"],
            "is_spot_trading_allowed": True,
            "is_margin_trading_allowed": False,
        }
        
        assert symbol_info["symbol"] == "BTCUSDT"
        assert symbol_info["base_asset"] == "BTC"
        assert symbol_info["is_spot_trading_allowed"] is True

    def test_raw_trade_data_type(self):
        """Test RawTradeData TypedDict functionality."""
        now = datetime.now(UTC)
        
        raw_trade: RawTradeData = {
            "id": "12345",
            "price": "50000.00",
            "quantity": "0.1",
            "time": now,
            "is_buyer_maker": True,
            "symbol": "BTCUSDT",
        }
        
        assert raw_trade["id"] == "12345"
        assert raw_trade["price"] == "50000.00"
        assert raw_trade["time"] == now

    def test_raw_kline_data_type(self):
        """Test RawKlineData TypedDict functionality."""
        now = datetime.now(UTC)
        
        raw_kline: RawKlineData = {
            "open_time": now,
            "close_time": now,
            "open_price": "49000.00",
            "high_price": "51000.00",
            "low_price": "48000.00",
            "close_price": "50000.00",
            "volume": "100.5",
            "number_of_trades": "150",
            "taker_buy_base_volume": "60.2",
            "taker_buy_quote_volume": "3010000.00",
            "symbol": "BTCUSDT",
            "interval": "1h",
        }
        
        assert raw_kline["symbol"] == "BTCUSDT"
        assert raw_kline["interval"] == "1h"
        assert raw_kline["open_price"] == "49000.00"

    def test_statistics_result_type(self):
        """Test StatisticsResult TypedDict functionality."""
        now = datetime.now(UTC)
        
        stats: StatisticsResult = {
            "count": 1000,
            "min_timestamp": now,
            "max_timestamp": now,
            "storage_size_bytes": 50000,
            "avg_size_bytes": 50.0,
            "price_statistics": {"min": 48000.0, "max": 52000.0, "avg": 50000.0},
            "volume_statistics": {"min": 0.1, "max": 10.0, "avg": 1.5},
        }
        
        assert stats["count"] == 1000
        assert stats["storage_size_bytes"] == 50000
        assert stats["price_statistics"]["avg"] == 50000.0

    def test_backfill_status_type(self):
        """Test BackfillStatus TypedDict functionality."""
        now = datetime.now(UTC)
        
        backfill_status: BackfillStatus = {
            "symbol": "BTCUSDT",
            "start_time": now,
            "end_time": now,
            "status": "completed",
            "progress": 100.0,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
        
        assert backfill_status["symbol"] == "BTCUSDT"
        assert backfill_status["status"] == "completed"
        assert backfill_status["progress"] == 100.0

    def test_provider_config_type(self):
        """Test ProviderConfig TypedDict functionality."""
        config: ProviderConfig = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "base_url": "https://api.binance.com",
            "timeout": 30,
            "rate_limit": 1200,
            "retry_count": 3,
            "retry_delay": 1.0,
            "use_testnet": False,
            "custom_headers": {"User-Agent": "Trading-Bot/1.0"},
        }
        
        assert config["api_key"] == "test_key"
        assert config["timeout"] == 30
        assert config["use_testnet"] is False

    def test_timestamp_type_alias(self):
        """Test TimestampType alias accepts various timestamp formats."""
        # Should accept different timestamp types
        timestamp_int: TimestampType = 1640995200
        timestamp_float: TimestampType = 1640995200.123
        timestamp_str: TimestampType = "2022-01-01T00:00:00Z"
        timestamp_datetime: TimestampType = datetime.now(UTC)
        
        assert isinstance(timestamp_int, int)
        assert isinstance(timestamp_float, float)
        assert isinstance(timestamp_str, str)
        assert isinstance(timestamp_datetime, datetime)

    def test_metadata_value_type_alias(self):
        """Test MetadataValue alias accepts various metadata types."""
        # Should accept different metadata value types
        str_value: MetadataValue = "test_string"
        int_value: MetadataValue = 42
        float_value: MetadataValue = 3.14
        bool_value: MetadataValue = True
        list_value: MetadataValue = [1, 2, 3]
        dict_value: MetadataValue = {"key": "value"}
        none_value: MetadataValue = None
        
        assert str_value == "test_string"
        assert int_value == 42
        assert float_value == 3.14
        assert bool_value is True
        assert list_value == [1, 2, 3]
        assert dict_value == {"key": "value"}
        assert none_value is None

    def test_type_annotations_exist(self):
        """Test that all TypedDict classes have proper type annotations."""
        # Check that type hints are properly defined
        user_data_hints = get_type_hints(UserData)
        alert_data_hints = get_type_hints(AlertData)
        
        # UserData should have optional fields
        assert "user_id" in user_data_hints
        assert "username" in user_data_hints
        assert "roles" in user_data_hints
        
        # AlertData should have required fields
        assert "id" in alert_data_hints
        assert "rule_name" in alert_data_hints
        assert "fired_at" in alert_data_hints

    def test_type_definitions_are_importable(self):
        """Test that all type definitions can be imported successfully."""
        # This test passes if the imports at the top of the file work
        # and verifies that the type system doesn't have circular dependencies
        assert UserData is not None
        assert AlertData is not None
        assert ExchangeInfo is not None
        assert SymbolInfo is not None
        assert RawTradeData is not None
        assert RawKlineData is not None
        assert StatisticsResult is not None
        assert BackfillStatus is not None
        assert ProviderConfig is not None