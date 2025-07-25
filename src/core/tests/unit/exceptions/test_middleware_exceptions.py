# ABOUTME: Unit tests for middleware exception classes
# ABOUTME: Tests exception inheritance, initialization, and error context handling

import pytest
from core.exceptions import (
    MiddlewareError,
    MiddlewareExecutionError,
    MiddlewareConfigurationError,
    MiddlewarePipelineError,
    MiddlewareTimeoutError,
    MiddlewareCircularDependencyError,
    MiddlewarePriorityConflictError,
    MiddlewareValidationError,
    CoreException,
)


class TestMiddlewareExceptions:
    """Test suite for middleware exception classes."""

    def test_middleware_error_inheritance(self):
        """Test that MiddlewareError inherits from CoreException."""
        exception = MiddlewareError("Test middleware error")
        
        assert isinstance(exception, CoreException)
        assert isinstance(exception, MiddlewareError)
        assert str(exception) == "Test middleware error"

    def test_middleware_error_with_code_and_details(self):
        """Test MiddlewareError with error code and contextual details."""
        details = {"middleware_name": "AuthMiddleware", "stage": "pre_execution"}
        exception = MiddlewareError(
            "Middleware execution failed",
            code="MW001",
            details=details
        )
        
        assert exception.message == "Middleware execution failed"
        assert exception.code == "MW001"
        assert exception.details == details
        assert str(exception) == "Middleware execution failed"

    def test_middleware_execution_error(self):
        """Test MiddlewareExecutionError functionality."""
        exception = MiddlewareExecutionError(
            "Middleware process method failed",
            details={"error": "NoneType object has no attribute 'process'"}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert isinstance(exception, CoreException)
        assert exception.message == "Middleware process method failed"

    def test_middleware_configuration_error(self):
        """Test MiddlewareConfigurationError functionality."""
        exception = MiddlewareConfigurationError(
            "Invalid middleware configuration",
            code="CONFIG_INVALID",
            details={"field": "timeout", "value": -1, "expected": "> 0"}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert exception.code == "CONFIG_INVALID"
        assert exception.details["field"] == "timeout"

    def test_middleware_pipeline_error(self):
        """Test MiddlewarePipelineError functionality."""
        exception = MiddlewarePipelineError(
            "Cannot add middleware to closed pipeline",
            details={"pipeline_state": "closed", "middleware": "LoggingMiddleware"}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert "closed pipeline" in exception.message

    def test_middleware_timeout_error(self):
        """Test MiddlewareTimeoutError functionality."""
        exception = MiddlewareTimeoutError(
            "Middleware execution timeout",
            details={"timeout_seconds": 30, "middleware": "DatabaseMiddleware"}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert exception.details["timeout_seconds"] == 30

    def test_middleware_circular_dependency_error(self):
        """Test MiddlewareCircularDependencyError functionality."""
        exception = MiddlewareCircularDependencyError(
            "Circular dependency detected",
            details={"cycle": ["A", "B", "C", "A"]}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert exception.details["cycle"] == ["A", "B", "C", "A"]

    def test_middleware_priority_conflict_error(self):
        """Test MiddlewarePriorityConflictError functionality."""
        exception = MiddlewarePriorityConflictError(
            "Priority conflict detected",
            details={"priority": 100, "conflicting_middleware": ["Auth", "Rate"]}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert exception.details["priority"] == 100

    def test_middleware_validation_error(self):
        """Test MiddlewareValidationError functionality."""
        exception = MiddlewareValidationError(
            "Middleware validation failed",
            details={"expected": "AbstractMiddleware", "actual": "dict"}
        )
        
        assert isinstance(exception, MiddlewareError)
        assert exception.details["expected"] == "AbstractMiddleware"

    def test_all_middleware_exceptions_inherit_from_base(self):
        """Test that all middleware exceptions inherit from MiddlewareError."""
        exception_classes = [
            MiddlewareExecutionError,
            MiddlewareConfigurationError,
            MiddlewarePipelineError,
            MiddlewareTimeoutError,
            MiddlewareCircularDependencyError,
            MiddlewarePriorityConflictError,
            MiddlewareValidationError,
        ]
        
        for exception_class in exception_classes:
            exception = exception_class("Test message")
            assert isinstance(exception, MiddlewareError)
            assert isinstance(exception, CoreException)

    def test_exception_details_immutability(self):
        """Test that exception details don't affect the original dict."""
        original_details = {"key": "value"}
        exception = MiddlewareError("Test", details=original_details)
        
        # Modify the exception's details
        exception.details["new_key"] = "new_value"
        
        # Original dict should be unchanged
        assert "new_key" not in original_details
        assert original_details == {"key": "value"}

    def test_exception_with_none_details(self):
        """Test exception creation with None details."""
        exception = MiddlewareError("Test message", details=None)
        
        assert exception.details == {}
        assert isinstance(exception.details, dict)

    def test_exception_string_representation(self):
        """Test string representation of middleware exceptions."""
        exception = MiddlewareExecutionError("Process failed")
        
        # Should return the message when converted to string
        assert str(exception) == "Process failed"
        
        # Exception message should be accessible
        assert exception.message == "Process failed"