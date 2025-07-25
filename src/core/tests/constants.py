# ABOUTME: Test constants and configuration values for core test suite
# ABOUTME: Provides configurable timeouts, limits, and other test parameters for consistent testing

import os
from typing import Final

# Test timeout constants (in seconds)
class TestTimeouts:
    """Configurable timeout constants for different test scenarios."""
    
    # Basic operation timeouts
    QUICK_OPERATION: Final[float] = float(os.getenv("TEST_QUICK_TIMEOUT", "0.1"))
    STANDARD_OPERATION: Final[float] = float(os.getenv("TEST_STANDARD_TIMEOUT", "0.2"))
    SLOW_OPERATION: Final[float] = float(os.getenv("TEST_SLOW_TIMEOUT", "0.5"))
    
    # Event processing timeouts
    EVENT_PROCESSING: Final[float] = float(os.getenv("TEST_EVENT_PROCESSING_TIMEOUT", "1.0"))
    EVENT_FLUSH: Final[float] = float(os.getenv("TEST_EVENT_FLUSH_TIMEOUT", "5.0"))
    
    # Integration test timeouts
    INTEGRATION_SHORT: Final[float] = float(os.getenv("TEST_INTEGRATION_SHORT_TIMEOUT", "5.0"))
    INTEGRATION_MEDIUM: Final[float] = float(os.getenv("TEST_INTEGRATION_MEDIUM_TIMEOUT", "10.0"))
    INTEGRATION_LONG: Final[float] = float(os.getenv("TEST_INTEGRATION_LONG_TIMEOUT", "30.0"))
    
    # Performance test timeouts
    PERFORMANCE_BASELINE: Final[float] = float(os.getenv("TEST_PERFORMANCE_TIMEOUT", "10.0"))
    
    # Recovery and error handling timeouts
    RECOVERY_TIMEOUT: Final[float] = float(os.getenv("TEST_RECOVERY_TIMEOUT", "5.0"))
    ERROR_PROPAGATION: Final[float] = float(os.getenv("TEST_ERROR_PROPAGATION_TIMEOUT", "1.0"))

# Test retry and attempt constants
class TestRetries:
    """Constants for test retry logic and attempt limits."""
    
    MAX_RETRY_ATTEMPTS: Final[int] = int(os.getenv("TEST_MAX_RETRIES", "3"))
    RETRY_DELAY: Final[float] = float(os.getenv("TEST_RETRY_DELAY", "0.1"))
    STABILITY_CHECK_ATTEMPTS: Final[int] = int(os.getenv("TEST_STABILITY_ATTEMPTS", "5"))

# Performance test thresholds
class PerformanceThresholds:
    """Performance test threshold constants."""
    
    # Processing rate thresholds (events/second)
    MIN_PROCESSING_RATE: Final[float] = float(os.getenv("TEST_MIN_PROCESSING_RATE", "100.0"))
    EXPECTED_PROCESSING_RATE: Final[float] = float(os.getenv("TEST_EXPECTED_PROCESSING_RATE", "1000.0"))
    
    # Memory usage thresholds (MB)
    MAX_MEMORY_USAGE: Final[float] = float(os.getenv("TEST_MAX_MEMORY_MB", "100.0"))
    
    # Latency thresholds (milliseconds)
    MAX_LATENCY_MS: Final[float] = float(os.getenv("TEST_MAX_LATENCY_MS", "10.0"))
    ACCEPTABLE_LATENCY_MS: Final[float] = float(os.getenv("TEST_ACCEPTABLE_LATENCY_MS", "5.0"))
    
    # Error rate thresholds (percentage)
    MAX_ERROR_RATE: Final[float] = float(os.getenv("TEST_MAX_ERROR_RATE", "1.0"))
    TIMEOUT_RATE_THRESHOLD: Final[float] = float(os.getenv("TEST_TIMEOUT_RATE_THRESHOLD", "0.1"))

# Test data size constants
class TestDataSizes:
    """Constants for test data generation and limits."""
    
    SMALL_DATASET: Final[int] = int(os.getenv("TEST_SMALL_DATASET_SIZE", "10"))
    MEDIUM_DATASET: Final[int] = int(os.getenv("TEST_MEDIUM_DATASET_SIZE", "100"))
    LARGE_DATASET: Final[int] = int(os.getenv("TEST_LARGE_DATASET_SIZE", "1000"))
    
    # Event count limits for testing
    EVENT_BURST_SIZE: Final[int] = int(os.getenv("TEST_EVENT_BURST_SIZE", "50"))
    EVENT_STRESS_SIZE: Final[int] = int(os.getenv("TEST_EVENT_STRESS_SIZE", "500"))
    
    # String size limits for boundary testing
    MAX_STRING_LENGTH: Final[int] = int(os.getenv("TEST_MAX_STRING_LENGTH", "10000"))
    LARGE_DICT_SIZE: Final[int] = int(os.getenv("TEST_LARGE_DICT_SIZE", "1000"))

# Test environment flags
class TestFlags:
    """Boolean flags for controlling test behavior."""
    
    ENABLE_PERFORMANCE_TESTS: Final[bool] = os.getenv("TEST_ENABLE_PERFORMANCE", "true").lower() == "true"
    ENABLE_STRESS_TESTS: Final[bool] = os.getenv("TEST_ENABLE_STRESS", "false").lower() == "true"
    ENABLE_STABILITY_TESTS: Final[bool] = os.getenv("TEST_ENABLE_STABILITY", "true").lower() == "true"
    VERBOSE_LOGGING: Final[bool] = os.getenv("TEST_VERBOSE_LOGGING", "false").lower() == "true"
    
    # Resource cleanup flags
    AUTO_CLEANUP: Final[bool] = os.getenv("TEST_AUTO_CLEANUP", "true").lower() == "true"
    PRESERVE_TEST_DATA: Final[bool] = os.getenv("TEST_PRESERVE_DATA", "false").lower() == "true"

# Test resource limits
class ResourceLimits:
    """Constants for test resource consumption limits."""
    
    MAX_CONCURRENT_OPERATIONS: Final[int] = int(os.getenv("TEST_MAX_CONCURRENT_OPS", "10"))
    MAX_QUEUE_SIZE: Final[int] = int(os.getenv("TEST_MAX_QUEUE_SIZE", "1000"))
    MAX_EVENT_HANDLERS: Final[int] = int(os.getenv("TEST_MAX_EVENT_HANDLERS", "50"))
    
    # Connection and resource pools
    MAX_CONNECTIONS: Final[int] = int(os.getenv("TEST_MAX_CONNECTIONS", "10"))
    CONNECTION_POOL_SIZE: Final[int] = int(os.getenv("TEST_CONNECTION_POOL_SIZE", "5"))

# Test marker constants
class TestMarkers:
    """Constants for pytest markers and test categorization."""
    
    UNIT: Final[str] = "unit"
    INTEGRATION: Final[str] = "integration"
    CONTRACT: Final[str] = "contract"
    BENCHMARK: Final[str] = "benchmark"
    PERFORMANCE: Final[str] = "performance"
    STABILITY: Final[str] = "stability"
    EXTERNAL: Final[str] = "external"
    CONFIG: Final[str] = "config"

# Convenience aggregations for common use cases
class CommonTimeouts:
    """Commonly used timeout combinations for typical test scenarios."""
    
    @staticmethod
    def get_async_operation_timeout() -> float:
        """Get timeout for standard async operations."""
        return TestTimeouts.STANDARD_OPERATION
    
    @staticmethod
    def get_event_test_timeout() -> float:
        """Get timeout for event-related tests."""
        return TestTimeouts.EVENT_PROCESSING
    
    @staticmethod
    def get_integration_timeout() -> float:
        """Get timeout for integration tests."""
        return TestTimeouts.INTEGRATION_SHORT
    
    @staticmethod
    def get_performance_timeout() -> float:
        """Get timeout for performance tests."""
        return TestTimeouts.PERFORMANCE_BASELINE

# Configuration validation
def validate_test_configuration() -> None:
    """Validate that test configuration values are reasonable."""
    # Ensure timeouts are positive
    timeout_attrs = [attr for attr in dir(TestTimeouts) if not attr.startswith('_')]
    for attr in timeout_attrs:
        value = getattr(TestTimeouts, attr)
        if value <= 0:
            raise ValueError(f"Test timeout {attr} must be positive, got {value}")
    
    # Ensure sizes are reasonable
    if TestDataSizes.SMALL_DATASET >= TestDataSizes.MEDIUM_DATASET:
        raise ValueError("Small dataset size must be less than medium dataset size")
    
    if TestDataSizes.MEDIUM_DATASET >= TestDataSizes.LARGE_DATASET:
        raise ValueError("Medium dataset size must be less than large dataset size")
    
    # Ensure performance thresholds are reasonable
    if PerformanceThresholds.MIN_PROCESSING_RATE >= PerformanceThresholds.EXPECTED_PROCESSING_RATE:
        raise ValueError("Min processing rate must be less than expected processing rate")

# Auto-validate configuration on import
validate_test_configuration()