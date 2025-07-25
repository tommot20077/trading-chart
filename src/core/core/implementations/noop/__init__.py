# ABOUTME: NoOp implementations package
# ABOUTME: Contains no-operation implementations for testing and benchmarking

# Auth implementations
from .auth.authenticator import NoOpAuthenticator
from .auth.authorizer import NoOpAuthorizer
from .auth.token_manager import NoOpTokenManager

# Common implementations
from .common.rate_limiter import NoOpRateLimiter

# Data implementations
from .data.converter import NoOpDataConverter
from .data.provider import NoOpDataProvider

# Event implementations
from .event.event_bus import NoOpEventBus
from .event.event_serializer import NoOpEventSerializer
from .storage.event_storage import NoOpEventStorage

# Observability implementations
from .observability.notification_handler import NoOpNotificationHandler

# Middleware implementations
from .middleware.pipeline import NoOpMiddlewarePipeline

# Storage implementations
from .storage.metadata_repository import NoOpMetadataRepository
from .storage.time_series_repository import NoOpTimeSeriesRepository
from .storage.kline_repository import NoOpKlineRepository

__all__ = [
    # Auth
    "NoOpAuthenticator",
    "NoOpAuthorizer",
    "NoOpTokenManager",
    # Common
    "NoOpRateLimiter",
    # Data
    "NoOpDataConverter",
    "NoOpDataProvider",
    # Event
    "NoOpEventBus",
    "NoOpEventSerializer",
    "NoOpEventStorage",
    # Middleware
    "NoOpMiddlewarePipeline",
    # Observability
    "NoOpNotificationHandler",
    # Storage
    "NoOpMetadataRepository",
    "NoOpTimeSeriesRepository",
    "NoOpKlineRepository",
]
