# ABOUTME: Common models package exports
# ABOUTME: Exports fundamental exceptions and base classes

from core.exceptions.base import (
    CoreException,
    ValidationException,
    BusinessLogicException,
    DataNotFoundException,
    ExternalServiceException,
    ConfigurationException,
    AuthenticationException,
    AuthorizationError,
    RateLimitExceededException,
    DataIntegrityException,
    TimeoutException,
    EventSerializationError,
    EventDeserializationError,
    StorageError,
    NotSupportedError,
)

from core.exceptions.middleware import (
    MiddlewareError,
    MiddlewareExecutionError,
    MiddlewareConfigurationError,
    MiddlewarePipelineError,
    MiddlewareTimeoutError,
    MiddlewareCircularDependencyError,
    MiddlewarePriorityConflictError,
    MiddlewareValidationError,
)

__all__ = [
    "CoreException",
    "ValidationException",
    "BusinessLogicException",
    "DataNotFoundException",
    "ExternalServiceException",
    "ConfigurationException",
    "AuthenticationException",
    "AuthorizationError",
    "RateLimitExceededException",
    "DataIntegrityException",
    "TimeoutException",
    "EventSerializationError",
    "EventDeserializationError",
    "StorageError",
    "NotSupportedError",
    # Middleware exceptions
    "MiddlewareError",
    "MiddlewareExecutionError",
    "MiddlewareConfigurationError",
    "MiddlewarePipelineError",
    "MiddlewareTimeoutError",
    "MiddlewareCircularDependencyError",
    "MiddlewarePriorityConflictError",
    "MiddlewareValidationError",
]
