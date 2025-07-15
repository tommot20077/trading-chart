# ABOUTME: Base configuration classes for the core library
# ABOUTME: Provides fundamental configuration settings and validation logic

from typing import Literal
import zoneinfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseCoreSettings(BaseSettings):
    """Defines the foundational configuration for the application ecosystem.

    This class acts as the central repository for essential, non-domain-specific
    parameters. It leverages `pydantic-settings` to load configurations from
    environment variables or `.env` files, ensuring a flexible and secure setup
    across various deployment environments (e.g., development, production).

    The settings here are considered universal and are intended to be inherited
    by more specific configuration classes, forming a composable and extensible
    configuration system.

    Attributes:
        APP_NAME: The name of the application, used for identification in logs and monitoring systems.
        ENV: The runtime environment, which controls environment-specific behaviors like debugging and logging levels.
        DEBUG: A flag to enable or disable debug mode, which may increase log verbosity or enable diagnostic endpoints.
        LOG_LEVEL: The minimum level for log messages to be processed.
        LOG_FORMAT: The format for log output, supporting structured (JSON) and human-readable (txt) formats.
        TIMEZONE: The canonical timezone for the application.
        model_config: Pydantic's configuration dictionary, specifying how settings are loaded.
    """

    # Application Identity
    APP_NAME: str = Field(
        default="TradingChart",
        description="The name of the application, used for identification in logs and monitoring.",
    )

    # Environment Configuration
    ENV: Literal["development", "staging", "production"] = Field(
        default="development",
        description="The application's runtime environment. Controls features like debugging and logging verbosity.",
    )
    DEBUG: bool = Field(
        default=False,
        description="Flag to enable or disable debug mode. Should be False in production.",
    )

    # Logging Configuration
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="The minimum level for log messages to be processed.",
    )
    LOG_FORMAT: Literal["json", "txt"] = Field(
        default="txt",
        description="The output format for logs. Use 'json' for production environments.",
    )

    # Globalization
    TIMEZONE: str = Field(
        default="UTC",
        description="The canonical timezone for the application. All datetime objects should be handled in this timezone.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("ENV", mode="before")
    @classmethod
    def validate_env_case_insensitive(cls, v: str) -> str:
        """Validate ENV field with case-insensitive mapping.

        Accepts common environment aliases and normalizes them:
        - dev, develop -> development
        - prod -> production
        - stage -> staging
        """
        if isinstance(v, str):
            v_lower = v.lower().strip()
            env_mapping = {
                "dev": "development",
                "develop": "development",
                "development": "development",
                "stage": "staging",
                "staging": "staging",
                "prod": "production",
                "production": "production",
            }
            return env_mapping.get(v_lower, v_lower)
        return v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level_case_insensitive(cls, v: str) -> str:
        """Validate LOG_LEVEL field with case-insensitive normalization."""
        if isinstance(v, str):
            return v.upper().strip()
        return v

    @field_validator("LOG_FORMAT", mode="before")
    @classmethod
    def validate_log_format_case_insensitive(cls, v: str) -> str:
        """Validate LOG_FORMAT field with case-insensitive normalization."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            format_mapping = {
                "json": "json",
                "structured": "json",
                "txt": "txt",
                "text": "txt",
            }
            return format_mapping.get(v_lower, v_lower)
        return v

    @field_validator("TIMEZONE", mode="before")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate TIMEZONE field to ensure it's a valid IANA timezone identifier.

        Uses Python's zoneinfo module to validate timezone names.
        Accepts standard IANA timezone identifiers like:
        - UTC
        - America/New_York
        - Asia/Shanghai
        - Europe/London

        Raises:
            ValueError: If the timezone is not a valid IANA timezone identifier.
        """
        if not isinstance(v, str):
            return v

        v_stripped = v.strip()

        # Handle empty string
        if not v_stripped:
            raise ValueError(
                "Invalid timezone ''. Must be a valid IANA timezone identifier "
                "(e.g., 'UTC', 'America/New_York', 'Asia/Shanghai')."
            )

        # Validate the timezone using zoneinfo
        try:
            # Test if the timezone is valid by trying to create a ZoneInfo object
            zoneinfo.ZoneInfo(v_stripped)
            return v_stripped
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            # If not found, try with proper case for IANA names
            if "/" in v_stripped:
                # Try with proper case (e.g., "america/new_york" -> "America/New_York")
                parts = v_stripped.split("/")
                if len(parts) >= 2:
                    # Capitalize first part (continent/region)
                    parts[0] = parts[0].capitalize()
                    # Capitalize and format city names
                    for i in range(1, len(parts)):
                        parts[i] = parts[i].replace("_", " ").title().replace(" ", "_")

                    proper_case_tz = "/".join(parts)

                    try:
                        zoneinfo.ZoneInfo(proper_case_tz)
                        return proper_case_tz
                    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
                        pass

            # If all attempts fail, raise a validation error
            raise ValueError(
                f"Invalid timezone '{v_stripped}'. Must be a valid IANA timezone identifier "
                f"(e.g., 'UTC', 'America/New_York', 'Asia/Shanghai')."
            )
