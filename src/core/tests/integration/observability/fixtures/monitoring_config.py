# ABOUTME: Monitoring configuration for observability integration tests
# ABOUTME: Provides specialized configurations for different testing scenarios including performance thresholds and logging

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import tempfile


@dataclass
class LoggingConfig:
    """Configuration for test logging."""

    level: str = "DEBUG"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    colorize: bool = True
    serialize: bool = False
    rotation: Optional[str] = None
    retention: Optional[str] = None
    compression: Optional[str] = None

    # Test-specific settings
    capture_stderr: bool = True
    capture_stdout: bool = False
    temp_file_suffix: str = ".log"
    temp_file_prefix: str = "observability_test_"


@dataclass
class NotificationConfig:
    """Configuration for notification handler testing."""

    max_queue_size: int = 1000
    max_history_size: int = 5000
    history_retention_hours: float = 2.0
    cleanup_interval_seconds: float = 60.0
    max_retry_attempts: int = 3

    # Test-specific settings
    simulate_failure_rate: float = 0.0
    processing_delay_seconds: float = 0.0
    enable_background_cleanup: bool = True

    # Performance testing variants
    high_volume_max_queue_size: int = 10000
    high_volume_max_history_size: int = 50000
    stress_test_failure_rate: float = 0.5
    performance_test_delay: float = 0.1


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring testing."""

    sample_interval: float = 0.1
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0
    queue_size_threshold: int = 100
    error_rate_threshold: float = 5.0

    # Test-specific settings
    fast_sample_interval: float = 0.05
    slow_sample_interval: float = 0.5
    strict_cpu_threshold: float = 60.0
    strict_memory_threshold: float = 60.0
    lenient_cpu_threshold: float = 95.0
    lenient_memory_threshold: float = 95.0

    # Benchmark settings
    benchmark_duration: float = 10.0
    benchmark_event_count: int = 100
    benchmark_handler_count: int = 50


@dataclass
class AlertConfig:
    """Configuration for alert testing."""

    enable_console_alerts: bool = True
    enable_file_alerts: bool = False
    alert_cooldown_seconds: float = 1.0
    max_alerts_per_minute: int = 60

    # Test scenarios
    alert_types: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "high_cpu": {"threshold": 80.0, "severity": "warning", "description": "CPU usage exceeded threshold"},
            "high_memory": {"threshold": 80.0, "severity": "warning", "description": "Memory usage exceeded threshold"},
            "high_queue_size": {
                "threshold": 100,
                "severity": "error",
                "description": "Event queue size exceeded threshold",
            },
            "high_error_rate": {
                "threshold": 5.0,
                "severity": "critical",
                "description": "Error rate exceeded threshold",
            },
        }
    )


@dataclass
class TestScenarioConfig:
    """Configuration for different test scenarios."""

    # Basic functionality testing
    basic: Dict[str, Any] = field(
        default_factory=lambda: {
            "notification": NotificationConfig(),
            "performance": PerformanceConfig(sample_interval=0.1, cpu_threshold=90.0, memory_threshold=90.0),
            "logging": LoggingConfig(level="INFO"),
            "alerts": AlertConfig(),
        }
    )

    # High performance testing
    performance: Dict[str, Any] = field(
        default_factory=lambda: {
            "notification": NotificationConfig(
                max_queue_size=10000, max_history_size=50000, processing_delay_seconds=0.0
            ),
            "performance": PerformanceConfig(sample_interval=0.05, cpu_threshold=95.0, memory_threshold=95.0),
            "logging": LoggingConfig(level="WARNING"),
            "alerts": AlertConfig(enable_console_alerts=False),
        }
    )

    # Stress testing with failures
    stress: Dict[str, Any] = field(
        default_factory=lambda: {
            "notification": NotificationConfig(
                max_queue_size=500, max_history_size=1000, simulate_failure_rate=0.3, processing_delay_seconds=0.1
            ),
            "performance": PerformanceConfig(
                sample_interval=0.05, cpu_threshold=70.0, memory_threshold=70.0, error_rate_threshold=10.0
            ),
            "logging": LoggingConfig(level="DEBUG"),
            "alerts": AlertConfig(max_alerts_per_minute=120, alert_cooldown_seconds=0.5),
        }
    )

    # Development/debugging configuration
    debug: Dict[str, Any] = field(
        default_factory=lambda: {
            "notification": NotificationConfig(
                max_queue_size=100, max_history_size=500, cleanup_interval_seconds=10.0, processing_delay_seconds=0.0
            ),
            "performance": PerformanceConfig(sample_interval=0.2, cpu_threshold=99.0, memory_threshold=99.0),
            "logging": LoggingConfig(level="DEBUG", capture_stderr=True, capture_stdout=True),
            "alerts": AlertConfig(enable_console_alerts=True),
        }
    )


class ObservabilityTestConfigFactory:
    """Factory for creating observability test configurations."""

    @staticmethod
    def create_config(scenario: str = "basic") -> TestScenarioConfig:
        """
        Create configuration for specified test scenario.

        Args:
            scenario: Test scenario name ("basic", "performance", "stress", "debug")

        Returns:
            TestScenarioConfig instance for the scenario
        """
        config = TestScenarioConfig()

        if scenario not in ["basic", "performance", "stress", "debug"]:
            raise ValueError(f"Unknown scenario: {scenario}. Available: basic, performance, stress, debug")

        return config

    @staticmethod
    def create_notification_config(scenario: str = "basic", **overrides) -> NotificationConfig:
        """Create notification configuration with optional overrides."""
        config = TestScenarioConfig()
        base_config = getattr(config, scenario)["notification"]

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

        return base_config

    @staticmethod
    def create_performance_config(scenario: str = "basic", **overrides) -> PerformanceConfig:
        """Create performance configuration with optional overrides."""
        config = TestScenarioConfig()
        base_config = getattr(config, scenario)["performance"]

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

        return base_config

    @staticmethod
    def create_logging_config(scenario: str = "basic", temp_dir: Optional[Path] = None, **overrides) -> LoggingConfig:
        """Create logging configuration with optional overrides."""
        config = TestScenarioConfig()
        base_config = getattr(config, scenario)["logging"]

        # Create temp file if temp_dir provided
        if temp_dir:
            temp_file = tempfile.NamedTemporaryFile(
                dir=temp_dir, prefix=base_config.temp_file_prefix, suffix=base_config.temp_file_suffix, delete=False
            )
            temp_file.close()
            overrides["log_file"] = temp_file.name

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

        return base_config

    @staticmethod
    def create_alert_config(scenario: str = "basic", **overrides) -> AlertConfig:
        """Create alert configuration with optional overrides."""
        config = TestScenarioConfig()
        base_config = getattr(config, scenario)["alerts"]

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

        return base_config


# Predefined configurations for common test scenarios

DEFAULT_TEST_CONFIG = ObservabilityTestConfigFactory.create_config("basic")

PERFORMANCE_TEST_CONFIG = ObservabilityTestConfigFactory.create_config("performance")

STRESS_TEST_CONFIG = ObservabilityTestConfigFactory.create_config("stress")

DEBUG_TEST_CONFIG = ObservabilityTestConfigFactory.create_config("debug")


# Configuration validation


def validate_notification_config(config: NotificationConfig) -> bool:
    """Validate notification configuration parameters."""
    if config.max_queue_size <= 0:
        return False
    if config.max_history_size <= 0:
        return False
    if config.history_retention_hours <= 0:
        return False
    if config.cleanup_interval_seconds <= 0:
        return False
    if config.max_retry_attempts < 0:
        return False
    if not (0.0 <= config.simulate_failure_rate <= 1.0):
        return False
    if config.processing_delay_seconds < 0:
        return False
    return True


def validate_performance_config(config: PerformanceConfig) -> bool:
    """Validate performance configuration parameters."""
    if config.sample_interval <= 0:
        return False
    if not (0.0 <= config.cpu_threshold <= 100.0):
        return False
    if not (0.0 <= config.memory_threshold <= 100.0):
        return False
    if config.queue_size_threshold < 0:
        return False
    if config.error_rate_threshold < 0:
        return False
    return True


def validate_alert_config(config: AlertConfig) -> bool:
    """Validate alert configuration parameters."""
    if config.alert_cooldown_seconds < 0:
        return False
    if config.max_alerts_per_minute <= 0:
        return False

    # Validate alert types
    for alert_type, alert_data in config.alert_types.items():
        if "threshold" not in alert_data:
            return False
        if "severity" not in alert_data:
            return False
        if "description" not in alert_data:
            return False

    return True


def validate_test_config(config: TestScenarioConfig) -> Dict[str, bool]:
    """
    Validate complete test configuration.

    Returns:
        Dictionary mapping component names to validation results
    """
    return {
        "notification_basic": validate_notification_config(config.basic["notification"]),
        "performance_basic": validate_performance_config(config.basic["performance"]),
        "alert_basic": validate_alert_config(config.basic["alerts"]),
        "notification_performance": validate_notification_config(config.performance["notification"]),
        "performance_performance": validate_performance_config(config.performance["performance"]),
        "alert_performance": validate_alert_config(config.performance["alerts"]),
        "notification_stress": validate_notification_config(config.stress["notification"]),
        "performance_stress": validate_performance_config(config.stress["performance"]),
        "alert_stress": validate_alert_config(config.stress["alerts"]),
        "notification_debug": validate_notification_config(config.debug["notification"]),
        "performance_debug": validate_performance_config(config.debug["performance"]),
        "alert_debug": validate_alert_config(config.debug["alerts"]),
    }
