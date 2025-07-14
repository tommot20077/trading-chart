# ABOUTME: Main configuration composition for the application.
# ABOUTME: Assembles all configuration classes into a single, accessible object.

from functools import lru_cache

from ._base import BaseCoreSettings


class CoreSettings(BaseCoreSettings):
    """Represents the complete, composed configuration for the application.

    This class acts as the final aggregator for all configuration settings.
    It inherits from `BaseCoreSettings` to include foundational settings and is
    designed to be extended with other specific settings classes (e.g., for
    database, cache, APIs) through inheritance.

    This composition pattern allows for a clean separation of concerns, where
    each configuration module is self-contained, but the final application
    has a single, unified `Settings` object.

    Example of future extension:
        class DatabaseSettings(BaseSettings):
            DB_URL: str = "sqlite:///./test.db"

        class Settings(BaseCoreSettings, DatabaseSettings):
            pass

    The `get_settings` function provides a singleton instance of this class.
    """

    pass


@lru_cache
def get_settings() -> CoreSettings:
    """Provides a singleton instance of the application settings.

        This function uses a cache (`lru_cache`) to ensure that the `Settings`
        object is instantiated only once. This prevents the performance overhead of
        repeatedly reading environment variables and files, and guarantees a
    is    consistent configuration state across the application.

        Returns:
            A single, cached instance of the Settings class.
    """
    return CoreSettings()


# A globally accessible, singleton instance of the application settings.
# Import this instance into other modules to access configuration values.
settings = get_settings()
