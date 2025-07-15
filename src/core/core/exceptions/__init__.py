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
]
