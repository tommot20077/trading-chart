# ABOUTME: Configuration package initialization
# ABOUTME: Exports configuration classes and utilities for the core library

# Main settings aggregator and convenience imports
from core.config.settings import CoreSettings, get_settings
from core.config.logging import (
    LoggerConfig,
    LoggingSettings,
    setup_logging,
    setup_otel_logging,
    get_logger,
    configure_for_testing,
    configure_for_production,
    configure_for_development,
)

__all__ = [
    "CoreSettings",
    "get_settings",
    "LoggerConfig",
    "LoggingSettings",
    "setup_logging",
    "setup_otel_logging",
    "get_logger",
    "configure_for_testing",
    "configure_for_production",
    "configure_for_development",
]
