# ABOUTME: No-operation implementations for observability interfaces
# ABOUTME: Provides minimal notification handling for testing and performance benchmarks

from .notification_handler import NoOpNotificationHandler

__all__ = [
    "NoOpNotificationHandler",
]
