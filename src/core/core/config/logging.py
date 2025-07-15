# ABOUTME: Loguru configuration for the core library
# ABOUTME: Provides unified logging setup with console colorization and file output

import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LoggerConfig(BaseModel):
    """Configuration for loguru logger."""

    # Console output configuration
    console_enabled: bool = True
    console_level: str = "INFO"
    console_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    console_colorize: bool = True
    console_backtrace: bool = True
    console_diagnose: bool = True

    # File output configuration
    file_enabled: bool = True
    file_level: str = "DEBUG"
    file_path: Union[str, Path] = "logs/trading-chart.log"
    file_format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    file_rotation: str = "100 MB"
    file_retention: str = "30 days"
    file_compression: str = "gz"
    file_serialize: bool = False

    # Structured logging for file output
    structured_enabled: bool = True
    structured_level: str = "DEBUG"
    structured_path: Union[str, Path] = "logs/trading-chart-structured.jsonl"
    structured_format: str = "{message}"
    structured_serialize: bool = True

    # Error file output
    error_file_enabled: bool = True
    error_file_level: str = "ERROR"
    error_file_path: Union[str, Path] = "logs/trading-chart-errors.log"

    # OpenTelemetry integration
    otel_enabled: bool = False
    otel_trace_id_key: str = "trace_id"
    otel_span_id_key: str = "span_id"

    # Performance settings
    enqueue: bool = True  # Async logging
    catch: bool = True  # Catch exceptions in logging


class LoggingSettings(BaseSettings):
    """Logging settings that can be configured via environment variables."""

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file_enabled: bool = Field(default=True, validation_alias="LOG_FILE_ENABLED")
    log_file_path: str = Field(default="logs/trading-chart.log", validation_alias="LOG_FILE_PATH")
    log_structured_enabled: bool = Field(default=True, validation_alias="LOG_STRUCTURED_ENABLED")
    log_console_colorize: bool = Field(default=True, validation_alias="LOG_CONSOLE_COLORIZE")
    log_otel_enabled: bool = Field(default=False, validation_alias="LOG_OTEL_ENABLED")

    model_config = {"env_prefix": "TRADING_CHART_"}


def setup_logging(config: Optional[LoggerConfig] = None) -> None:
    """
    Setup loguru logger with the specified configuration.

    Args:
        config: Logger configuration. If None, uses default configuration.
    """
    if config is None:
        # Load from environment variables
        settings = LoggingSettings()
        config = LoggerConfig(
            console_level=settings.log_level,
            file_enabled=settings.log_file_enabled,
            file_path=settings.log_file_path,
            structured_enabled=settings.log_structured_enabled,
            console_colorize=settings.log_console_colorize,
            otel_enabled=settings.log_otel_enabled,
            file_level=settings.log_level,
            structured_level=settings.log_level,
        )

    # Remove default handler
    logger.remove()

    # Add console handler
    if config.console_enabled:
        logger.add(
            sys.stdout,
            level=config.console_level,
            format=config.console_format,
            colorize=config.console_colorize,
            backtrace=config.console_backtrace,
            diagnose=config.console_diagnose,
            enqueue=config.enqueue,
            catch=config.catch,
        )

    # Add file handler
    if config.file_enabled:
        # Ensure log directory exists
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            config.file_path,
            level=config.file_level,
            format=config.file_format,
            rotation=config.file_rotation,
            retention=config.file_retention,
            compression=config.file_compression,
            serialize=config.file_serialize,
            enqueue=config.enqueue,
            catch=config.catch,
        )

    # Add structured JSON log handler
    if config.structured_enabled:
        structured_path = Path(config.structured_path)
        structured_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            config.structured_path,
            level=config.structured_level,
            format=config.structured_format,
            serialize=config.structured_serialize,
            rotation=config.file_rotation,
            retention=config.file_retention,
            compression=config.file_compression,
            enqueue=config.enqueue,
            catch=config.catch,
        )

    # Add error file handler
    if config.error_file_enabled:
        error_path = Path(config.error_file_path)
        error_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            config.error_file_path,
            level=config.error_file_level,
            format=config.file_format,
            rotation=config.file_rotation,
            retention=config.file_retention,
            compression=config.file_compression,
            enqueue=config.enqueue,
            catch=config.catch,
        )


def setup_otel_logging(trace_id_key: str = "trace_id", span_id_key: str = "span_id") -> None:
    """
    Setup OpenTelemetry integration for logging.

    Args:
        trace_id_key: Key name for trace ID in log records
        span_id_key: Key name for span ID in log records
    """
    try:
        from opentelemetry import trace

        def add_trace_info(record) -> None:
            """Add trace information to log records."""
            span = trace.get_current_span()
            if span != trace.INVALID_SPAN:
                span_context = span.get_span_context()
                if span_context.is_valid:
                    record["extra"][trace_id_key] = f"{span_context.trace_id:032x}"
                    record["extra"][span_id_key] = f"{span_context.span_id:016x}"

        # Configure loguru to add trace information
        logger.configure(patcher=add_trace_info)

    except ImportError:
        logger.warning("OpenTelemetry not available, skipping OTel logging setup")


def get_logger(name: str):
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance bound to the specified name
    """
    return logger.bind(name=name)


def configure_for_testing() -> None:
    """Configure logging for testing environment."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        catch=False,
    )


def configure_for_production() -> None:
    """Configure logging for production environment."""
    config = LoggerConfig(
        console_level="INFO",
        console_colorize=False,
        console_backtrace=False,
        console_diagnose=False,
        file_level="DEBUG",
        structured_enabled=True,
        error_file_enabled=True,
        otel_enabled=True,
    )
    setup_logging(config)
    setup_otel_logging()


def configure_for_development() -> None:
    """Configure logging for development environment."""
    config = LoggerConfig(
        console_level="DEBUG",
        console_colorize=True,
        console_backtrace=True,
        console_diagnose=True,
        file_level="DEBUG",
        structured_enabled=True,
        error_file_enabled=True,
        otel_enabled=False,
    )
    setup_logging(config)


# Default setup - can be overridden by applications
setup_logging()
