# ABOUTME: Integration tests for multi-data source switching and management
# ABOUTME: Tests seamless switching between different data sources and failure handling

import pytest
import pytest_asyncio
import asyncio
from typing import List, Any

from core.interfaces.data.provider import AbstractDataProvider
from core.implementations.memory.data.data_provider import MemoryDataProvider
from core.exceptions.base import ExternalServiceException


class MockFailingDataProvider(AbstractDataProvider):
    """Mock data provider that simulates failures for testing"""

    def __init__(self, name: str = "FailingProvider", fail_after: int = 3):
        self._name = name
        self._connected = False
        self._call_count = 0
        self._fail_after = fail_after

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
        if not self._connected:
            raise ExternalServiceException("Provider not connected")
        return 50.0

    async def get_exchange_info(self) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count > self._fail_after:
            raise ExternalServiceException("Simulated provider failure")
        return {"name": self._name, "status": "normal"}

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count > self._fail_after:
            raise ExternalServiceException("Simulated provider failure")
        return {"symbol": symbol, "status": "TRADING"}

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        return True, ""

    # Required abstract method implementations
    async def stream_trades(self, symbol: str, *, start_from=None):
        """Mock implementation - not used in switching tests"""
        raise NotImplementedError("Mock provider - stream_trades not implemented")

    async def stream_klines(self, symbol: str, interval, *, start_from=None):
        """Mock implementation - not used in switching tests"""
        raise NotImplementedError("Mock provider - stream_klines not implemented")

    async def fetch_historical_trades(self, symbol: str, start_time, end_time, *, limit=None):
        """Mock implementation - not used in switching tests"""
        raise NotImplementedError("Mock provider - fetch_historical_trades not implemented")

    async def fetch_historical_klines(self, symbol: str, interval, start_time, end_time, *, limit=None):
        """Mock implementation - not used in switching tests"""
        raise NotImplementedError("Mock provider - fetch_historical_klines not implemented")

    async def convert_multiple_trades(self, raw_trades, symbol: str):
        """Mock implementation - not used in switching tests"""
        raise NotImplementedError("Mock provider - convert_multiple_trades not implemented")

    async def convert_multiple_klines(self, raw_klines, symbol: str):
        """Mock implementation - not used in switching tests"""
        raise NotImplementedError("Mock provider - convert_multiple_klines not implemented")


class DataSourceManager:
    """Simple data source manager for testing multi-source scenarios"""

    def __init__(self, providers: List[AbstractDataProvider]):
        self.providers = providers
        self.current_provider_index = 0
        self.fallback_enabled = True

    @property
    def current_provider(self) -> AbstractDataProvider:
        return self.providers[self.current_provider_index]

    async def get_data_with_fallback(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation with automatic fallback to next provider on failure"""
        last_exception = None

        for attempt in range(len(self.providers)):
            provider = self.providers[self.current_provider_index]

            try:
                if operation == "get_exchange_info":
                    return await provider.get_exchange_info()
                elif operation == "get_symbol_info":
                    return await provider.get_symbol_info(args[0])
                else:
                    raise ValueError(f"Unknown operation: {operation}")

            except ExternalServiceException as e:
                last_exception = e
                print(f"Provider {provider.name} failed: {e}")

                if self.fallback_enabled and attempt < len(self.providers) - 1:
                    self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
                    print(f"Switching to provider: {self.current_provider.name}")
                    continue
                else:
                    break

        if last_exception:
            raise last_exception

    async def switch_to_provider(self, provider_name: str) -> bool:
        """Manually switch to a specific provider"""
        for i, provider in enumerate(self.providers):
            if provider.name == provider_name:
                self.current_provider_index = i
                return True
        return False


@pytest.mark.integration
@pytest.mark.asyncio
class TestDataSourceSwitching:
    """Integration tests for data source switching and management"""

    @pytest_asyncio.fixture
    async def primary_provider(self):
        """Primary data provider (reliable)"""
        provider = MemoryDataProvider(name="PrimaryProvider")
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    async def secondary_provider(self):
        """Secondary data provider (backup)"""
        provider = MemoryDataProvider(name="SecondaryProvider")
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    async def failing_provider(self):
        """Failing data provider for testing failure scenarios"""
        provider = MockFailingDataProvider(name="FailingProvider", fail_after=2)
        await provider.connect()
        yield provider
        await provider.close()

    @pytest_asyncio.fixture
    def data_source_manager(self, primary_provider, secondary_provider):
        """Data source manager with multiple providers"""
        return DataSourceManager([primary_provider, secondary_provider])

    @pytest.mark.asyncio
    async def test_manual_provider_switching(self, data_source_manager):
        """Test manual switching between data providers"""
        # Arrange
        manager = data_source_manager

        # Act & Assert: Start with first provider
        assert manager.current_provider.name == "PrimaryProvider"

        # Act: Switch to secondary provider
        success = await manager.switch_to_provider("SecondaryProvider")

        # Assert: Switch successful
        assert success is True
        assert manager.current_provider.name == "SecondaryProvider"

        # Act: Switch back to primary
        success = await manager.switch_to_provider("PrimaryProvider")

        # Assert: Switch successful
        assert success is True
        assert manager.current_provider.name == "PrimaryProvider"

        # Act: Try to switch to non-existent provider
        success = await manager.switch_to_provider("NonExistentProvider")

        # Assert: Switch failed, still on primary
        assert success is False
        assert manager.current_provider.name == "PrimaryProvider"

    @pytest.mark.asyncio
    async def test_automatic_failover_on_provider_failure(self, failing_provider, secondary_provider):
        """Test automatic failover when primary provider fails"""
        # Arrange
        manager = DataSourceManager([failing_provider, secondary_provider])

        # Act & Assert: First few calls should succeed
        result1 = await manager.get_data_with_fallback("get_exchange_info")
        assert result1["name"] == "FailingProvider"

        result2 = await manager.get_data_with_fallback("get_exchange_info")
        assert result2["name"] == "FailingProvider"

        # Act: Next call should trigger failure and fallback
        result3 = await manager.get_data_with_fallback("get_exchange_info")

        # Assert: Should have switched to secondary provider
        assert result3["name"] == "SecondaryProvider"
        assert manager.current_provider.name == "SecondaryProvider"

    @pytest.mark.asyncio
    async def test_data_consistency_across_providers(self, primary_provider, secondary_provider):
        """Test data consistency when switching between providers"""
        # Arrange
        manager = DataSourceManager([primary_provider, secondary_provider])
        symbol = "BTCUSDT"

        # Act: Get symbol info from primary provider
        primary_info = await manager.get_data_with_fallback("get_symbol_info", symbol)

        # Act: Switch to secondary provider
        await manager.switch_to_provider("SecondaryProvider")
        secondary_info = await manager.get_data_with_fallback("get_symbol_info", symbol)

        # Assert: Both providers should return consistent symbol information
        assert primary_info["symbol"] == secondary_info["symbol"]
        assert primary_info["symbol"] == symbol

        # Both should indicate trading status (though exact format may differ)
        assert "status" in primary_info
        assert "status" in secondary_info

    @pytest.mark.asyncio
    async def test_provider_health_monitoring(self, primary_provider, failing_provider):
        """Test provider health monitoring and status checking"""
        # Arrange
        providers = [primary_provider, failing_provider]

        # Act & Assert: Check initial health
        for provider in providers:
            assert provider.is_connected is True
            ping_time = await provider.ping()
            assert ping_time > 0

        # Act: Simulate some operations on failing provider
        manager = DataSourceManager([failing_provider, primary_provider])

        # First few operations should succeed
        await manager.get_data_with_fallback("get_exchange_info")
        await manager.get_data_with_fallback("get_exchange_info")

        # Next operation should fail and trigger fallback
        result = await manager.get_data_with_fallback("get_exchange_info")

        # Assert: Should have fallen back to healthy provider
        assert result["name"] == "PrimaryProvider"

    @pytest.mark.asyncio
    async def test_concurrent_provider_access(self, primary_provider, secondary_provider):
        """Test concurrent access to different providers"""
        # Arrange
        manager1 = DataSourceManager([primary_provider])
        manager2 = DataSourceManager([secondary_provider])

        async def get_exchange_info(manager, provider_name):
            """Helper function for concurrent access"""
            result = await manager.get_data_with_fallback("get_exchange_info")
            return result["name"]

        # Act: Concurrent access to different providers
        tasks = [
            get_exchange_info(manager1, "PrimaryProvider"),
            get_exchange_info(manager2, "SecondaryProvider"),
            get_exchange_info(manager1, "PrimaryProvider"),
            get_exchange_info(manager2, "SecondaryProvider"),
        ]

        results = await asyncio.gather(*tasks)

        # Assert: All operations should succeed with correct providers
        assert results[0] == "PrimaryProvider"
        assert results[1] == "SecondaryProvider"
        assert results[2] == "PrimaryProvider"
        assert results[3] == "SecondaryProvider"

    async def test_provider_switching_with_data_conversion(
        self, primary_provider, secondary_provider, test_converter, sample_kline_data
    ):
        """Test provider switching while maintaining data conversion consistency"""
        # Arrange
        manager = DataSourceManager([primary_provider, secondary_provider])
        raw_klines = sample_kline_data["binance"]
        symbol = "BTCUSDT"

        # Act: Convert data using primary provider context
        converted_klines_1 = test_converter.convert_multiple_klines(raw_klines, symbol)

        # Act: Switch provider and convert same data
        await manager.switch_to_provider("SecondaryProvider")
        converted_klines_2 = test_converter.convert_multiple_klines(raw_klines, symbol)

        # Assert: Conversion results should be identical regardless of provider
        assert len(converted_klines_1) == len(converted_klines_2)

        for i in range(len(converted_klines_1)):
            kline1, kline2 = converted_klines_1[i], converted_klines_2[i]
            assert kline1.symbol == kline2.symbol
            assert kline1.open_price == kline2.open_price
            assert kline1.close_price == kline2.close_price
            assert kline1.high_price == kline2.high_price
            assert kline1.low_price == kline2.low_price
            assert kline1.volume == kline2.volume

    @pytest.mark.asyncio
    async def test_provider_fallback_chain(self):
        """Test fallback chain with multiple failing providers"""
        # Arrange: Create a chain of providers with different failure points
        failing_provider_1 = MockFailingDataProvider("FailingProvider1", fail_after=1)
        failing_provider_2 = MockFailingDataProvider("FailingProvider2", fail_after=1)
        reliable_provider = MemoryDataProvider("ReliableProvider")

        await failing_provider_1.connect()
        await failing_provider_2.connect()
        await reliable_provider.connect()

        try:
            manager = DataSourceManager([failing_provider_1, failing_provider_2, reliable_provider])

            # Act: First call should succeed on first provider
            result1 = await manager.get_data_with_fallback("get_exchange_info")
            assert result1["name"] == "FailingProvider1"

            # Act: Second call should fail on first, succeed on second
            result2 = await manager.get_data_with_fallback("get_exchange_info")
            assert result2["name"] == "FailingProvider2"

            # Act: Third call should fail on both, succeed on reliable
            result3 = await manager.get_data_with_fallback("get_exchange_info")
            assert result3["name"] == "ReliableProvider"

        finally:
            await failing_provider_1.close()
            await failing_provider_2.close()
            await reliable_provider.close()

    @pytest.mark.asyncio
    async def test_provider_switching_performance(self, primary_provider, secondary_provider):
        """Test performance impact of provider switching"""
        # Arrange
        manager = DataSourceManager([primary_provider, secondary_provider])

        # Act: Measure time for operations without switching
        import time

        start_time = time.time()

        for _ in range(10):
            await manager.get_data_with_fallback("get_exchange_info")

        no_switch_time = time.time() - start_time

        # Act: Measure time with switching between each operation
        start_time = time.time()

        for i in range(10):
            provider_name = "PrimaryProvider" if i % 2 == 0 else "SecondaryProvider"
            await manager.switch_to_provider(provider_name)
            await manager.get_data_with_fallback("get_exchange_info")

        with_switch_time = time.time() - start_time

        # Assert: Switching should not add significant overhead (less than 2x)
        assert with_switch_time < no_switch_time * 2

        # Both should complete in reasonable time (less than 1 second for 10 operations)
        assert no_switch_time < 1.0
        assert with_switch_time < 2.0
