# ABOUTME: Core interfaces package exports
# ABOUTME: Exports all abstract interfaces for authentication, data, events, storage, and observability

# Authentication interfaces
from .auth import AbstractAuthenticator, AbstractAuthorizer, AbstractTokenManager

# Common interfaces
from .common import AbstractRateLimiter

# Data interfaces
from .data import AbstractDataConverter, AbstractDataProvider

# Event interfaces
from .event import (
    AbstractEventBus,
    EventHandler,
    AsyncEventHandler,
    AbstractEventSerializer,
    AbstractEventStorage,
)

# Storage interfaces
from .storage import (
    AbstractKlineRepository,
    AbstractMetadataRepository,
    AbstractTimeSeriesRepository,
)

# Observability interfaces
from .observability import AbstractNotificationHandler

__all__ = [
    # Authentication
    "AbstractAuthenticator",
    "AbstractAuthorizer",
    "AbstractTokenManager",
    # Common
    "AbstractRateLimiter",
    # Data
    "AbstractDataConverter",
    "AbstractDataProvider",
    # Event
    "AbstractEventBus",
    "EventHandler",
    "AsyncEventHandler",
    "AbstractEventSerializer",
    "AbstractEventStorage",
    # Storage
    "AbstractKlineRepository",
    "AbstractMetadataRepository",
    "AbstractTimeSeriesRepository",
    # Observability
    "AbstractNotificationHandler",
]
