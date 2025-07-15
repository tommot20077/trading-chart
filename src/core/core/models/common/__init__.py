# ABOUTME: Common models package exports
# ABOUTME: Exports fundamental exceptions and base classes

from .exceptions import (
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
]
