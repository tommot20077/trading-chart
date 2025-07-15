# ABOUTME: pytest configuration and fixtures for event system integration tests
# ABOUTME: Provides comprehensive fixtures for event bus, serializer, storage, and test data

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Dict, Any
from uuid import uuid4

from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.memory.event.event_middleware_bus import EventMiddlewareBus
from core.implementations.memory.event.event_serializer import MemoryEventSerializer
from core.implementations.memory.storage.event_storage import InMemoryEventStorage
from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository

# from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.components.event.handler_registry import EventHandlerRegistry

from core.models.data.event import BaseEvent
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.enum import KlineInterval, TradeSide
from core.models.network.enum import ConnectionStatus


# ============================================================================
# Core Event System Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def event_bus():
    """Create a clean InMemoryEventBus for testing."""
    bus = InMemoryEventBus()
    yield bus
    await bus.close()


@pytest_asyncio.fixture
async def event_middleware_bus(event_bus):
    """Create an EventMiddlewareBus wrapping the base event bus."""
    middleware_bus = EventMiddlewareBus(event_bus)
    yield middleware_bus
    await middleware_bus.close()


@pytest.fixture
def event_serializer():
    """Create a MemoryEventSerializer for testing."""
    return MemoryEventSerializer()


@pytest_asyncio.fixture
async def metadata_repository():
    """Create a clean InMemoryMetadataRepository for testing."""
    repo = InMemoryMetadataRepository()
    yield repo
    await repo.close()


@pytest_asyncio.fixture
async def event_storage(event_serializer, metadata_repository):
    """Create an InMemoryEventStorage with real dependencies."""
    storage = InMemoryEventStorage(event_serializer, metadata_repository)
    yield storage
    await storage.close()


@pytest.fixture
def event_handler_registry():
    """Create a clean EventHandlerRegistry for testing."""
    return EventHandlerRegistry()


# @pytest_asyncio.fixture
# async def middleware_pipeline():
#     """Create a clean InMemoryMiddlewarePipeline for testing."""
#     pipeline = InMemoryMiddlewarePipeline()
#     yield pipeline
#     await pipeline.close()


# ============================================================================
# Test Event Data Fixtures
# ============================================================================


@pytest.fixture
def sample_trade_event():
    """Create a sample TradeEvent for testing."""
    return TradeEvent(
        source="binance",
        symbol="BTC/USDT",
        data=Trade(
            symbol="BTC/USDT",
            trade_id="12345",
            price=Decimal("45000.00"),
            quantity=Decimal("0.1"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        ),
        priority=EventPriority.NORMAL,
    )


@pytest.fixture
def sample_kline_event():
    """Create a sample KlineEvent for testing."""
    return KlineEvent(
        source="binance",
        symbol="BTC/USDT",
        data=Kline(
            symbol="BTC/USDT",
            interval=KlineInterval.MINUTE_1,
            open_time=datetime.now(UTC),
            close_time=datetime.now(UTC) + timedelta(minutes=1),
            open_price=Decimal("44900.00"),
            high_price=Decimal("45100.00"),
            low_price=Decimal("44800.00"),
            close_price=Decimal("45000.00"),
            volume=Decimal("10.5"),
            quote_volume=Decimal("472500.00"),
            trades_count=150,
            taker_buy_base_volume=Decimal("6.2"),
            taker_buy_quote_volume=Decimal("279000.00"),
        ),
        priority=EventPriority.NORMAL,
    )


@pytest.fixture
def sample_connection_event():
    """Create a sample ConnectionEvent for testing."""
    return ConnectionEvent(
        status=ConnectionStatus.CONNECTED,
        source="binance",
        symbol=None,
        data={"endpoint": "wss://stream.binance.com:9443/ws/btcusdt@trade"},
        priority=EventPriority.HIGH,
    )


@pytest.fixture
def sample_error_event():
    """Create a sample ErrorEvent for testing."""
    return ErrorEvent(
        source="binance",
        symbol="BTC/USDT",
        data={"context": "trade_processing"},
        priority=EventPriority.CRITICAL,
        error="Connection timeout",
        error_code="TIMEOUT_001",
    )


@pytest.fixture
def mixed_events(sample_trade_event, sample_kline_event, sample_connection_event, sample_error_event):
    """Create a list of mixed event types for testing."""
    return [sample_trade_event, sample_kline_event, sample_connection_event, sample_error_event]


# ============================================================================
# Test Handler Fixtures
# ============================================================================


@pytest.fixture
def sync_event_handlers():
    """Create synchronous event handlers for testing."""
    handlers = {}

    def create_handler(event_type: EventType, name: str):
        def handler(event: BaseEvent):
            handlers[f"{name}_calls"] = handlers.get(f"{name}_calls", 0) + 1
            handlers[f"{name}_last_event"] = event

        handler.__name__ = name
        return handler

    handlers["trade_handler"] = create_handler(EventType.TRADE, "trade_handler")
    handlers["kline_handler"] = create_handler(EventType.KLINE, "kline_handler")
    handlers["connection_handler"] = create_handler(EventType.CONNECTION, "connection_handler")
    handlers["error_handler"] = create_handler(EventType.ERROR, "error_handler")
    handlers["universal_handler"] = create_handler(None, "universal_handler")

    return handlers


@pytest.fixture
def async_event_handlers():
    """Create asynchronous event handlers for testing."""
    handlers = {}

    def create_async_handler(event_type: EventType, name: str):
        async def handler(event: BaseEvent):
            await asyncio.sleep(0.01)  # Simulate async work
            handlers[f"{name}_calls"] = handlers.get(f"{name}_calls", 0) + 1
            handlers[f"{name}_last_event"] = event

        handler.__name__ = name
        return handler

    handlers["async_trade_handler"] = create_async_handler(EventType.TRADE, "async_trade_handler")
    handlers["async_kline_handler"] = create_async_handler(EventType.KLINE, "async_kline_handler")
    handlers["async_connection_handler"] = create_async_handler(EventType.CONNECTION, "async_connection_handler")
    handlers["async_error_handler"] = create_async_handler(EventType.ERROR, "async_error_handler")

    return handlers


# ============================================================================
# Test Configuration Fixtures
# ============================================================================


@pytest.fixture
def event_test_config():
    """Configuration for event integration tests."""
    return {
        "timeout": 5.0,
        "max_events": 1000,
        "batch_size": 100,
        "test_symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
        "test_sources": ["binance", "okx", "coinbase"],
        "performance_thresholds": {
            "publish_latency_ms": 200,
            "handler_execution_ms": 50,
            "serialization_ms": 5,
            "storage_ms": 20,
        },
    }


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def event_factory():
    """Factory for creating test events with customizable parameters."""

    def create_event(
        event_type: EventType,
        source: str = "test_source",
        symbol: str = "BTC/USDT",
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs,
    ) -> BaseEvent:
        base_data = {"source": source, "symbol": symbol, "priority": priority, **kwargs}

        if event_type == EventType.TRADE:
            base_data["data"] = Trade(
                symbol=symbol,
                trade_id=str(uuid4()),
                price=Decimal("45000.00"),
                quantity=Decimal("0.1"),
                side=TradeSide.BUY,
                timestamp=datetime.now(UTC),
                is_buyer_maker=True,
            )
            return TradeEvent(**base_data)

        elif event_type == EventType.KLINE:
            base_data["data"] = Kline(
                symbol=symbol,
                interval=KlineInterval.MINUTE_1,
                open_time=datetime.now(UTC),
                close_time=datetime.now(UTC) + timedelta(minutes=1),
                open_price=Decimal("44900.00"),
                high_price=Decimal("45100.00"),
                low_price=Decimal("44800.00"),
                close_price=Decimal("45000.00"),
                volume=Decimal("10.5"),
                quote_volume=Decimal("472500.00"),
                trades_count=150,
                taker_buy_base_volume=Decimal("6.2"),
                taker_buy_quote_volume=Decimal("279000.00"),
            )
            return KlineEvent(**base_data)

        elif event_type == EventType.CONNECTION:
            base_data["data"] = {"endpoint": f"wss://stream.{source}.com/ws"}
            return ConnectionEvent(status=ConnectionStatus.CONNECTED, **base_data)

        elif event_type == EventType.ERROR:
            base_data["data"] = {"context": "test_context"}
            base_data["error"] = "Test error"
            base_data["error_code"] = "TEST_001"
            return ErrorEvent(**base_data)

        else:
            # Generic BaseEvent
            base_data["data"] = {"test": "data"}
            base_data["event_type"] = event_type
            return BaseEvent(**base_data)

    return create_event


@pytest.fixture
def performance_monitor():
    """Monitor for tracking performance metrics during tests."""
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}

        def start_timer(self, name: str):
            self.metrics[f"{name}_start"] = time.perf_counter()

        def end_timer(self, name: str) -> float:
            if f"{name}_start" not in self.metrics:
                raise ValueError(f"Timer {name} was not started")

            duration = time.perf_counter() - self.metrics[f"{name}_start"]
            self.metrics[f"{name}_duration"] = duration
            return duration

        def get_duration(self, name: str) -> float:
            return self.metrics.get(f"{name}_duration", 0.0)

        def get_all_metrics(self) -> Dict[str, Any]:
            return self.metrics.copy()

    return PerformanceMonitor()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

# @pytest.fixture(autouse=True)
# async def cleanup_event_system():
#     """Automatic cleanup for event system components."""
#     # Setup phase
#     yield
#
#     # Cleanup phase - ensure all async resources are properly closed
#     # This runs after each test automatically
#     await asyncio.sleep(0.01)  # Allow pending tasks to complete
