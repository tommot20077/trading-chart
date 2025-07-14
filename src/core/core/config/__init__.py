# ABOUTME: Configuration package initialization
# ABOUTME: Exports configuration classes and utilities for the core library

# Main settings aggregator and convenience imports
from core.config.settings import CoreSettings, get_settings

__all__ = [
    "CoreSettings",
    "get_settings",
]
