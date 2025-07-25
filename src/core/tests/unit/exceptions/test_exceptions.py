# ABOUTME: Unit tests for common exception classes in the trading system
# ABOUTME: Tests cover normal cases, exception cases, and boundary cases following TDD principles
import pytest

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
)


class TestCoreException:
    """Test cases for CoreException base class."""

    @pytest.mark.unit
    def test_core_exception_with_message_only(self):
        """Test CoreException with message only."""
        message = "Test error message"
        exception = CoreException(message)

        assert exception.message == message
        assert exception.code is None
        assert exception.details == {}
        assert str(exception) == message

    @pytest.mark.unit
    def test_core_exception_with_all_parameters(self):
        """Test CoreException with all parameters."""
        message = "Test error message"
        code = "TEST_ERROR"
        details = {"key": "value", "number": 42}

        exception = CoreException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details == details
        assert str(exception) == message

    @pytest.mark.unit
    def test_core_exception_with_none_details(self):
        """Test CoreException with None details."""
        message = "Test error message"
        code = "TEST_ERROR"

        exception = CoreException(message, code, None)

        assert exception.message == message
        assert exception.code == code
        assert exception.details == {}

    @pytest.mark.unit
    def test_core_exception_inheritance(self):
        """Test that CoreException inherits from Exception."""
        exception = CoreException("test")
        assert isinstance(exception, Exception)

    @pytest.mark.unit
    def test_core_exception_empty_message(self):
        """Test CoreException with empty message."""
        exception = CoreException("")
        assert exception.message == ""
        assert str(exception) == ""

    @pytest.mark.unit
    def test_core_exception_details_immutability(self):
        """Test that details dict is properly handled."""
        original_details = {"key": "value"}
        exception = CoreException("test", details=original_details)

        # Modify original dict
        original_details["new_key"] = "new_value"

        # Exception should have original details only
        assert "new_key" not in exception.details
        assert exception.details == {"key": "value"}


class TestValidationException:
    """Test cases for ValidationException."""

    @pytest.mark.unit
    def test_validation_exception_inheritance(self):
        """Test that ValidationException inherits from CoreException."""
        exception = ValidationException("validation error")
        assert isinstance(exception, CoreException)
        assert isinstance(exception, Exception)

    @pytest.mark.unit
    def test_validation_exception_with_validation_details(self):
        """Test ValidationException with validation-specific details."""
        message = "Field validation failed"
        code = "VALIDATION_ERROR"
        details = {"field": "email", "value": "invalid-email", "constraint": "must be valid email format"}

        exception = ValidationException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["field"] == "email"
        assert exception.details["constraint"] == "must be valid email format"


class TestBusinessLogicException:
    """Test cases for BusinessLogicException."""

    @pytest.mark.unit
    def test_business_logic_exception_inheritance(self):
        """Test that BusinessLogicException inherits from CoreException."""
        exception = BusinessLogicException("business rule violation")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    def test_business_logic_exception_with_business_context(self):
        """Test BusinessLogicException with business-specific details."""
        message = "Insufficient balance for trade"
        code = "INSUFFICIENT_BALANCE"
        details = {"account_balance": 100.0, "required_amount": 150.0, "currency": "USD"}

        exception = BusinessLogicException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["account_balance"] == 100.0
        assert exception.details["required_amount"] == 150.0


class TestDataNotFoundException:
    """Test cases for DataNotFoundException."""

    @pytest.mark.unit
    def test_data_not_found_exception_inheritance(self):
        """Test that DataNotFoundException inherits from CoreException."""
        exception = DataNotFoundException("data not found")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    def test_data_not_found_exception_with_search_context(self):
        """Test DataNotFoundException with search-specific details."""
        message = "Trading pair not found"
        code = "TRADING_PAIR_NOT_FOUND"
        details = {"symbol": "BTCUSD", "exchange": "binance", "search_criteria": {"base": "BTC", "quote": "USD"}}

        exception = DataNotFoundException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["symbol"] == "BTCUSD"
        assert exception.details["exchange"] == "binance"


class TestExternalServiceException:
    """Test cases for ExternalServiceException."""

    @pytest.mark.unit
    def test_external_service_exception_inheritance(self):
        """Test that ExternalServiceException inherits from CoreException."""
        exception = ExternalServiceException("external service error")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    @pytest.mark.external
    def test_external_service_exception_with_service_context(self):
        """Test ExternalServiceException with service-specific details."""
        message = "Exchange API rate limit exceeded"
        code = "API_RATE_LIMIT"
        details = {"service": "binance_api", "endpoint": "/api/v3/ticker/price", "retry_after": 60, "status_code": 429}

        exception = ExternalServiceException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["service"] == "binance_api"
        assert exception.details["retry_after"] == 60


class TestConfigurationException:
    """Test cases for ConfigurationException."""

    @pytest.mark.unit
    def test_configuration_exception_inheritance(self):
        """Test that ConfigurationException inherits from CoreException."""
        exception = ConfigurationException("configuration error")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    @pytest.mark.config
    @pytest.mark.external
    def test_configuration_exception_with_config_context(self):
        """Test ConfigurationException with configuration-specific details."""
        message = "Missing required configuration value"
        code = "MISSING_CONFIG"
        details = {"config_key": "database.host", "config_file": "settings.yaml", "environment": "production"}

        exception = ConfigurationException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["config_key"] == "database.host"
        assert exception.details["environment"] == "production"


class TestAuthenticationException:
    """Test cases for AuthenticationException."""

    @pytest.mark.unit
    def test_authentication_exception_inheritance(self):
        """Test that AuthenticationException inherits from CoreException."""
        exception = AuthenticationException("authentication failed")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    def test_authentication_exception_with_auth_context(self):
        """Test AuthenticationException with authentication-specific details."""
        message = "Invalid JWT token"
        code = "INVALID_TOKEN"
        details = {"token_type": "JWT", "expiry": "2023-12-31T23:59:59Z", "issuer": "auth_service"}

        exception = AuthenticationException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["token_type"] == "JWT"
        assert exception.details["issuer"] == "auth_service"


class TestAuthorizationError:
    """Test cases for AuthorizationError."""

    @pytest.mark.unit
    def test_authorization_error_inheritance(self):
        """Test that AuthorizationError inherits from CoreException."""
        exception = AuthorizationError("authorization failed")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    @pytest.mark.external
    def test_authorization_error_with_permission_context(self):
        """Test AuthorizationError with permission-specific details."""
        message = "Insufficient permissions for resource access"
        code = "INSUFFICIENT_PERMISSIONS"
        details = {
            "required_permission": "admin",
            "user_permissions": ["read", "write"],
            "resource": "/api/admin/users",
            "user_id": "user123",
        }

        exception = AuthorizationError(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["required_permission"] == "admin"
        assert exception.details["user_permissions"] == ["read", "write"]


class TestRateLimitExceededException:
    """Test cases for RateLimitExceededException."""

    @pytest.mark.unit
    def test_rate_limit_exceeded_exception_inheritance(self):
        """Test that RateLimitExceededException inherits from CoreException."""
        exception = RateLimitExceededException("rate limit exceeded")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    @pytest.mark.external
    def test_rate_limit_exceeded_exception_with_limit_context(self):
        """Test RateLimitExceededException with rate limit-specific details."""
        message = "API request rate limit exceeded"
        code = "RATE_LIMIT_EXCEEDED"
        details = {"limit": 100, "window": "1 minute", "current_count": 105, "reset_time": "2023-12-31T12:01:00Z"}

        exception = RateLimitExceededException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["limit"] == 100
        assert exception.details["current_count"] == 105


class TestDataIntegrityException:
    """Test cases for DataIntegrityException."""

    @pytest.mark.unit
    def test_data_integrity_exception_inheritance(self):
        """Test that DataIntegrityException inherits from CoreException."""
        exception = DataIntegrityException("data integrity violation")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    def test_data_integrity_exception_with_constraint_context(self):
        """Test DataIntegrityException with constraint-specific details."""
        message = "Foreign key constraint violation"
        code = "FK_CONSTRAINT_VIOLATION"
        details = {
            "table": "orders",
            "column": "user_id",
            "constraint": "fk_orders_user_id",
            "referenced_table": "users",
        }

        exception = DataIntegrityException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["table"] == "orders"
        assert exception.details["constraint"] == "fk_orders_user_id"


class TestTimeoutException:
    """Test cases for TimeoutException."""

    @pytest.mark.unit
    def test_timeout_exception_inheritance(self):
        """Test that TimeoutException inherits from CoreException."""
        exception = TimeoutException("operation timeout")
        assert isinstance(exception, CoreException)

    @pytest.mark.unit
    @pytest.mark.external
    def test_timeout_exception_with_timeout_context(self):
        """Test TimeoutException with timeout-specific details."""
        message = "Database query timeout"
        code = "QUERY_TIMEOUT"
        details = {"timeout_seconds": 30, "operation": "SELECT * FROM large_table", "connection_id": "conn_123"}

        exception = TimeoutException(message, code, details)

        assert exception.message == message
        assert exception.code == code
        assert exception.details["timeout_seconds"] == 30
        assert exception.details["operation"] == "SELECT * FROM large_table"


class TestExceptionHierarchy:
    """Test cases for exception hierarchy and relationships."""

    @pytest.mark.unit
    def test_all_exceptions_inherit_from_core_exception(self):
        """Test that all custom exceptions inherit from CoreException."""
        exception_classes = [
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
        ]

        for exception_class in exception_classes:
            exception = exception_class("test message")
            assert isinstance(exception, CoreException)
            assert isinstance(exception, Exception)

    @pytest.mark.unit
    def test_exception_class_names(self):
        """Test that exception class names follow naming conventions."""
        exception_classes = [
            ValidationException,
            BusinessLogicException,
            DataNotFoundException,
            ExternalServiceException,
            ConfigurationException,
            AuthenticationException,
            AuthorizationError,  # Note: This one ends with "Error" not "Exception"
            RateLimitExceededException,
            DataIntegrityException,
            TimeoutException,
        ]

        for exception_class in exception_classes:
            class_name = exception_class.__name__
            # Most should end with "Exception", but AuthorizationError is an exception
            if class_name != "AuthorizationError":
                assert class_name.endswith("Exception")

    @pytest.mark.unit
    def test_exception_instantiation_boundary_cases(self):
        """Test boundary cases for exception instantiation."""
        # Test with very long message
        long_message = "x" * 10000
        exception = CoreException(long_message)
        assert len(exception.message) == 10000

        # Test with special characters in message
        special_message = "Error: 特殊字符 🚨 \n\t\r"
        exception = CoreException(special_message)
        assert exception.message == special_message

        # Test with large details dict
        large_details = {f"key_{i}": f"value_{i}" for i in range(1000)}
        exception = CoreException("test", details=large_details)
        assert len(exception.details) == 1000
