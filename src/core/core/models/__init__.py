# ABOUTME: Models package initialization
# ABOUTME: Exports all core data models and related classes

# Authentication models
from .auth import AuthRequest, AuthToken, Role, Permission

# Data models
from .data import (
    BaseEvent,
    KlineInterval,
    AssetClass,
    TradeSide,
    Kline,
    Trade,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderExecutionType,
    TradingPair,
    TradingPairStatus,
    MarketData,
    MarketDataSummary,
)

# Event models
from .event import EventPriority, EventType

# Middleware models
from .middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus, PipelineResult

# Type definitions
from .types import UserData, AlertData, StatisticsResult, BackfillStatus, ProviderConfig

# Network models
from .network import ConnectionStatus

__all__ = [
    # Authentication
    "AuthRequest",
    "AuthToken",
    "Role",
    "Permission",
    # Data
    "BaseEvent",
    "KlineInterval",
    "AssetClass",
    "TradeSide",
    "Kline",
    "Trade",
    "Order",
    "OrderStatus",
    "OrderType",
    "OrderSide",
    "TimeInForce",
    "OrderExecutionType",
    "TradingPair",
    "TradingPairStatus",
    "MarketData",
    "MarketDataSummary",
    # Event
    "EventPriority",
    "EventType",
    # Middleware
    "MiddlewareContext",
    "MiddlewareResult",
    "MiddlewareStatus",
    "PipelineResult",
    # Types
    "UserData",
    "AlertData",
    "StatisticsResult",
    "BackfillStatus",
    "ProviderConfig",
    # Network
    "ConnectionStatus",
]
