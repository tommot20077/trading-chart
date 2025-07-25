# ABOUTME: Middleware-specific exception classes for error handling
# ABOUTME: Provides structured error handling for middleware pipeline operations and lifecycle management

from core.exceptions.base import CoreException


class MiddlewareError(CoreException):
    """Base exception class for middleware-related errors.

    This is the base class for all middleware-specific exceptions.
    It provides structured error handling for middleware operations
    including pipeline execution, configuration, and lifecycle management.

    Should be used as a base for more specific middleware exceptions
    rather than being raised directly.
    """

    pass


class MiddlewareExecutionError(MiddlewareError):
    """Exception raised when middleware execution fails.

    Used when middleware encounters errors during execution, such as:
    - Middleware process method throws unhandled exceptions
    - Critical errors in middleware logic
    - Resource allocation failures during execution
    - Dependency injection failures

    Should include details about the middleware that failed and the execution context.
    """

    pass


class MiddlewareConfigurationError(MiddlewareError):
    """Exception raised for middleware configuration errors.

    Used when middleware configuration is invalid or missing, such as:
    - Missing required configuration parameters
    - Invalid configuration values
    - Configuration validation failures
    - Conflicting middleware settings

    Should include details about the configuration issue and affected middleware.
    """

    pass


class MiddlewarePipelineError(MiddlewareError):
    """Exception raised for middleware pipeline operation errors.

    Used when pipeline operations fail, such as:
    - Adding middleware to a closed pipeline
    - Pipeline execution failures
    - Middleware ordering conflicts
    - Pipeline state management errors

    Should include details about the pipeline operation that failed.
    """

    pass


class MiddlewareTimeoutError(MiddlewareError):
    """Exception raised when middleware operations timeout.

    Used when middleware operations exceed configured timeout limits, such as:
    - Individual middleware execution timeout
    - Pipeline execution timeout
    - Middleware initialization timeout
    - Cleanup operation timeout

    Should include details about the timeout that occurred and the middleware involved.
    """

    pass


class MiddlewareCircularDependencyError(MiddlewareError):
    """Exception raised when circular dependencies are detected in middleware.

    Used when middleware dependencies form a circular chain, such as:
    - Middleware A depends on B, B depends on C, C depends on A
    - Self-referential middleware dependencies
    - Complex dependency cycles in middleware graph

    Should include details about the dependency chain that forms the cycle.
    """

    pass


class MiddlewarePriorityConflictError(MiddlewareError):
    """Exception raised when middleware priority conflicts occur.

    Used when middleware priority assignment causes conflicts, such as:
    - Multiple middleware with same priority in strict ordering mode
    - Priority values outside acceptable range
    - Priority conflicts with system middleware

    Should include details about the conflicting priorities and affected middleware.
    """

    pass


class MiddlewareValidationError(MiddlewareError):
    """Exception raised when middleware validation fails.

    Used when middleware or middleware context fails validation, such as:
    - Middleware implementation doesn't match interface contract
    - Context data validation failures
    - Result validation failures
    - Middleware metadata validation errors

    Should include details about the validation failure and expected vs actual values.
    """

    pass
