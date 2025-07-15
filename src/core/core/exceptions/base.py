# ABOUTME: Core exception classes for the trading system
# ABOUTME: Provides structured error handling with context and error codes

from typing import Dict, Any


class CoreException(Exception):
    """Base exception class for the trading system.

    Provides structured error handling with optional error codes and contextual
    details. All custom exceptions in the system should inherit from this class
    to ensure consistent error handling patterns.

    Attributes:
        message: Human-readable error message
        code: Optional error code for programmatic handling
        details: Optional dictionary containing contextual information
    """

    def __init__(self, message: str, code: str | None = None, details: Dict[str, Any] | None = None):
        """Initialize CoreException with message, optional code and details.

        Args:
            message: Human-readable error message
            code: Optional error code for programmatic handling
            details: Optional dictionary containing contextual information
        """
        self.message = message
        self.code = code
        self.details = details.copy() if details else {}
        super().__init__(self.message)


class ValidationException(CoreException):
    """Exception raised for data validation errors.

    Used when input data fails validation checks, such as:
    - Invalid data formats
    - Missing required fields
    - Data outside acceptable ranges
    - Type mismatches

    Should include specific details about what validation failed.
    """

    pass


class BusinessLogicException(CoreException):
    """Exception raised for business logic violations.

    Used when operations violate business rules, such as:
    - Insufficient balance for trades
    - Trading pair not supported
    - Market hours restrictions
    - Risk management limits exceeded

    Should include context about the business rule that was violated.
    """

    pass


class DataNotFoundException(CoreException):
    """Exception raised when requested data is not found.

    Used when queries or lookups fail to find the requested data, such as:
    - Trading pair not found
    - User account not found
    - Historical data not available
    - Configuration not found

    Should include details about what was being searched for.
    """

    pass


class ExternalServiceException(CoreException):
    """Exception raised for external service failures.

    Used when external services (exchanges, databases, APIs) fail, such as:
    - Exchange API errors
    - Database connection failures
    - Network timeouts
    - Service unavailability

    Should include details about the service and failure reason.
    """

    pass


class ConfigurationException(CoreException):
    """Exception raised for configuration errors.

    Used when system configuration is invalid or missing, such as:
    - Missing required configuration values
    - Invalid configuration format
    - Configuration validation failures
    - Environment setup issues

    Should include details about the configuration issue.
    """

    pass


class AuthenticationException(CoreException):
    """Exception raised for authentication errors.

    Used when authentication fails, such as:
    - Invalid credentials
    - Expired tokens
    - Authentication service unavailable

    Should include context about the authentication failure.
    """

    pass


class AuthorizationError(CoreException):
    """Exception raised for authorization errors.

    Used when authorization fails, such as:
    - Insufficient permissions
    - Role-based access denied
    - Resource access forbidden
    - Permission denied

    Should include context about the authorization failure.
    """

    pass


class RateLimitExceededException(CoreException):
    """Exception raised when rate limits are exceeded.

    Used when requests exceed configured rate limits, such as:
    - API rate limits
    - Database connection limits
    - Resource usage limits
    - Request throttling

    Should include details about the limit that was exceeded.
    """

    pass


class DataIntegrityException(CoreException):
    """Exception raised for data integrity violations.

    Used when data integrity constraints are violated, such as:
    - Foreign key constraints
    - Unique constraints
    - Data consistency checks
    - Transaction rollback requirements

    Should include details about the integrity violation.
    """

    pass


class TimeoutException(CoreException):
    """Exception raised for operation timeouts.

    Used when operations exceed their timeout limits, such as:
    - Database query timeouts
    - API request timeouts
    - Lock acquisition timeouts
    - Processing timeouts

    Should include details about the timeout that occurred.
    """

    pass


class EventSerializationError(CoreException):
    """Exception raised when event serialization fails.

    Used when events cannot be converted to bytes, such as:
    - Unsupported data types in event payload
    - Malformed event data
    - Serializer configuration errors
    - Memory or resource limitations

    Should include details about the serialization failure.
    """

    pass


class EventDeserializationError(CoreException):
    """Exception raised when event deserialization fails.

    Used when byte data cannot be converted back to events, such as:
    - Corrupted or malformed byte data
    - Incompatible serialization format
    - Missing required event fields
    - Version compatibility issues

    Should include details about the deserialization failure.
    """

    pass


class StorageError(CoreException):
    """Exception raised for storage operation failures.

    Used when storage operations encounter issues, such as:
    - Database connection failures
    - File I/O errors
    - Storage capacity limits exceeded
    - Data corruption during storage
    - Transaction failures

    Should include details about the storage operation that failed.
    """

    pass


class NotSupportedError(CoreException):
    """Exception raised when a requested operation is not supported.

    Used when a feature or operation is not supported by the current implementation, such as:
    - Unsupported data formats
    - Unsupported query operations
    - Feature not implemented in current backend
    - Platform-specific limitations

    Should include details about what operation was attempted.
    """

    pass
