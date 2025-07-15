# ABOUTME: Models package initialization
# ABOUTME: Exports all core data models and related classes

# Authentication models
from .auth import AuthRequest, AuthToken, Role, Permission

# Data models
from .data import BaseEvent, KlineInterval, AssetClass, TradeSide, Kline, Trade

# Event models
from .event import EventPriority, EventType

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
    # Event
    "EventPriority",
    "EventType",
    # Network
    "ConnectionStatus",
]
