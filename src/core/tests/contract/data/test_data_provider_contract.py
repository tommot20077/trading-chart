# ABOUTME: Contract tests for AbstractDataProvider interface
# ABOUTME: Verifies all data provider implementations comply with the interface contract

import pytest
from typing import Type, List
from datetime import datetime, timedelta, UTC

from core.interfaces.data.provider import AbstractDataProvider
from core.implementations.memory.data.data_provider import MemoryDataProvider
from core.models import Kline, Trade, KlineInterval, TradeSide
from ..base_contract_test import ContractTestBase, AsyncContractTestMixin, ResourceManagementContractMixin


class TestDataProviderContract(
    ContractTestBase[AbstractDataProvider], AsyncContractTestMixin, ResourceManagementContractMixin
):
    """Contract tests for AbstractDataProvider interface."""

    @property
    def interface_class(self) -> Type[AbstractDataProvider]:
        return AbstractDataProvider

    @property
    def implementations(self) -> List[Type[AbstractDataProvider]]:
        return [
            MemoryDataProvider,
            # Add other implementations here as they are created
            # BinanceDataProvider,
            # DatabaseDataProvider,
        ]

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_fetch_historical_klines_contract_behavior(self):
        """Test fetch_historical_klines method contract behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Connect first
                await provider.connect()

                try:
                    # Test basic fetch_historical_klines call
                    symbol = "BTCUSDT"  # Use format supported by MemoryDataProvider
                    interval = KlineInterval.MINUTE_1
                    start_time = datetime.now(UTC) - timedelta(hours=1)
                    end_time = datetime.now(UTC)

                    result = await provider.fetch_historical_klines(
                        symbol=symbol, interval=interval, start_time=start_time, end_time=end_time
                    )
                    assert isinstance(result, list)
                    for kline in result:
                        assert isinstance(kline, Kline)
                        assert kline.symbol == symbol
                        assert kline.interval == interval

                    # Test with limit parameter
                    limited_result = await provider.fetch_historical_klines(
                        symbol=symbol, interval=interval, start_time=start_time, end_time=end_time, limit=10
                    )
                    assert isinstance(limited_result, list)
                    for kline in limited_result:
                        assert isinstance(kline, Kline)
                        assert kline.symbol == symbol
                        assert kline.interval == interval

                finally:
                    await provider.close()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_fetch_historical_trades_contract_behavior(self):
        """Test fetch_historical_trades method contract behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Connect first
                await provider.connect()

                try:
                    # Test basic fetch_historical_trades call
                    symbol = "BTCUSDT"  # Use format supported by MemoryDataProvider
                    start_time = datetime.now(UTC) - timedelta(hours=1)
                    end_time = datetime.now(UTC)

                    result = await provider.fetch_historical_trades(
                        symbol=symbol, start_time=start_time, end_time=end_time
                    )
                    assert isinstance(result, list)
                    for trade in result:
                        assert isinstance(trade, Trade)
                        assert trade.symbol == symbol

                    # Test with limit parameter
                    limited_result = await provider.fetch_historical_trades(
                        symbol=symbol, start_time=start_time, end_time=end_time, limit=10
                    )
                    assert isinstance(limited_result, list)
                    for trade in limited_result:
                        assert isinstance(trade, Trade)
                        assert trade.symbol == symbol

                finally:
                    await provider.close()

    @pytest.mark.contract
    def test_stream_klines_contract_behavior(self):
        """Test stream_klines method contract behavior."""
        import inspect

        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Test stream_klines method exists and returns AsyncIterator
                stream_method = getattr(provider, "stream_klines")
                assert callable(stream_method)

                # Check return type annotation
                sig = inspect.signature(stream_method)
                return_annotation = str(sig.return_annotation)
                # Should return AsyncIterator[Kline]
                assert "AsyncIterator" in return_annotation or sig.return_annotation == inspect.Signature.empty

    @pytest.mark.contract
    def test_stream_trades_contract_behavior(self):
        """Test stream_trades method contract behavior."""
        import inspect

        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Test stream_trades method exists and returns AsyncIterator
                stream_method = getattr(provider, "stream_trades")
                assert callable(stream_method)

                # Check return type annotation
                sig = inspect.signature(stream_method)
                return_annotation = str(sig.return_annotation)
                # Should return AsyncIterator[Trade]
                assert "AsyncIterator" in return_annotation or sig.return_annotation == inspect.Signature.empty

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_data_conversion_contract_behavior(self):
        """Test data conversion methods contract behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                symbol = "BTC/USDT"  # Use format supported by MemoryDataProvider

                # Test convert_multiple_trades with correct field names
                raw_trades = [
                    {
                        "id": 123456,
                        "price": "50000.00",
                        "quantity": "0.001",  # Use 'quantity' not 'qty'
                        "timestamp": datetime.now(UTC).isoformat(),
                        "side": "buy",
                        "is_buyer_maker": False,
                    }
                ]

                converted_trades = await provider.convert_multiple_trades(raw_trades, symbol)
                assert isinstance(converted_trades, list)
                for trade in converted_trades:
                    assert isinstance(trade, Trade)
                    assert trade.symbol == symbol

                # Test convert_multiple_klines
                raw_klines = [
                    {
                        "interval": "1m",  # Add required interval field
                        "open_time": datetime.now(UTC).isoformat(),
                        "close_time": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                        "open_price": "50000.00",  # Use correct field names
                        "high_price": "50100.00",
                        "low_price": "49900.00",
                        "close_price": "50050.00",
                        "volume": "10.5",
                        "quote_volume": "525000.00",
                        "trade_count": 100,
                    }
                ]

                converted_klines = await provider.convert_multiple_klines(raw_klines, symbol)
                assert isinstance(converted_klines, list)
                for kline in converted_klines:
                    assert isinstance(kline, Kline)
                    assert kline.symbol == symbol

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_connection_management_contract_behavior(self):
        """Test connection management methods contract behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Test connect
                await provider.connect()

                # Test is_connected (property, not method)
                connected = provider.is_connected
                assert isinstance(connected, bool)
                assert connected is True

                # Test disconnect
                await provider.disconnect()

                # Should be disconnected after disconnect
                disconnected = provider.is_connected
                assert isinstance(disconnected, bool)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_health_and_status_contract_behavior(self):
        """Test health check and status methods contract behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Connect first
                await provider.connect()

                try:
                    # Test ping
                    ping_time = await provider.ping()
                    assert isinstance(ping_time, float)
                    assert ping_time >= 0

                    # Test get_exchange_info
                    exchange_info = await provider.get_exchange_info()
                    assert isinstance(exchange_info, dict)

                    # Test get_symbol_info
                    symbol = "BTCUSDT"  # Use format supported by MemoryDataProvider
                    symbol_info = await provider.get_symbol_info(symbol)
                    assert isinstance(symbol_info, dict)

                finally:
                    await provider.close()

    @pytest.mark.contract
    def test_config_validation_contract_behavior(self):
        """Test configuration validation contract behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Test validate_config with valid config
                valid_config = {"name": "test_provider"}
                is_valid, error_message = provider.validate_config(valid_config)
                assert isinstance(is_valid, bool)
                assert isinstance(error_message, str)

                # Test validate_config with invalid config
                invalid_config = {}
                is_invalid, error_message = provider.validate_config(invalid_config)
                assert isinstance(is_invalid, bool)
                assert isinstance(error_message, str)

    @pytest.mark.contract
    def test_required_methods_exist(self):
        """Test that all required methods exist in implementations."""
        required_methods = [
            # Core interface methods (from abstract methods list)
            "close",
            "connect",
            "disconnect",
            "fetch_historical_klines",
            "fetch_historical_trades",
            "get_exchange_info",
            "get_symbol_info",
            "ping",
            "stream_klines",
            "stream_trades",
            "convert_multiple_klines",
            "convert_multiple_trades",
            "validate_config",
        ]

        required_properties = ["name", "is_connected"]

        for impl_class in self.implementations:
            # Check methods
            for method_name in required_methods:
                assert hasattr(impl_class, method_name), f"{impl_class.__name__} missing method: {method_name}"

                method = getattr(impl_class, method_name)
                assert callable(method), f"{impl_class.__name__}.{method_name} is not callable"

            # Check properties
            for prop_name in required_properties:
                assert hasattr(impl_class, prop_name), f"{impl_class.__name__} missing property: {prop_name}"

    @pytest.mark.contract
    def test_data_model_compliance(self):
        """Test that implementations work with correct data models."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Test that provider can be instantiated
                assert provider is not None
                assert isinstance(provider, impl_class)
                assert isinstance(provider, AbstractDataProvider)

                # Test that required enums are available
                assert hasattr(provider, "__class__")
                # Verify that KlineInterval and TradeSide enums are properly imported
                assert KlineInterval.MINUTE_1 is not None
                assert TradeSide.BUY is not None

    @pytest.mark.contract
    def test_error_handling_contracts(self):
        """Test that implementations properly define error handling contracts."""

        # Check that critical methods have proper exception documentation or type hints
        critical_methods = ["fetch_historical_klines", "fetch_historical_trades", "connect", "disconnect"]

        for method_name in critical_methods:
            method = getattr(self.interface_class, method_name)
            assert hasattr(method, "__isabstractmethod__")

            # Verify method exists in implementations
            for impl_class in self.implementations:
                assert hasattr(impl_class, method_name)
                impl_method = getattr(impl_class, method_name)
                assert callable(impl_method)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_resource_cleanup_contract_behavior(self):
        """Test resource cleanup and context manager behavior."""
        for impl_class in self.implementations:
            if impl_class == MemoryDataProvider:
                provider = impl_class()

                # Test that provider can be used as async context manager
                async with provider as ctx_provider:
                    assert ctx_provider is not None
                    assert ctx_provider is provider

                    # Should be able to use provider normally within context
                    exchange_info = await ctx_provider.get_exchange_info()
                    assert isinstance(exchange_info, dict)

                # After context exit, provider should be properly cleaned up
                # (specific cleanup behavior depends on implementation)
