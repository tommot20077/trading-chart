# ABOUTME: Unit tests for validation result models, testing ValidationResult, ValidationIssue and related classes
# ABOUTME: Ensures correctness of validation result structure including issue tracking, severity classification and summary generation

import pytest
from core.validators.validation_result import ValidationResult, ValidationIssue, ValidationSeverity


class TestValidationSeverity:
    """Test cases for ValidationSeverity enum."""

    def test_severity_values(self):
        """Test validation severity enum values."""
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.ERROR == "error"
        assert ValidationSeverity.CRITICAL == "critical"


class TestValidationIssue:
    """Test cases for ValidationIssue model."""

    def test_create_basic_issue(self):
        """Test creating basic validation issue."""
        issue = ValidationIssue(
            code="TEST_ERROR",
            severity=ValidationSeverity.ERROR,
            message="Test error message",
        )
        
        assert issue.code == "TEST_ERROR"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error message"
        assert issue.field_path is None
        assert issue.expected_value is None
        assert issue.actual_value is None
        assert issue.suggestion is None
        assert issue.metadata == {}

    def test_create_detailed_issue(self):
        """Test creating detailed validation issue."""
        issue = ValidationIssue(
            code="VALIDATION_FAILED",
            severity=ValidationSeverity.WARNING,
            message="Value validation failed",
            field_path="user.email",
            expected_value="valid email format",
            actual_value="invalid-email",
            suggestion="Please provide a valid email address",
            metadata={"validator": "email", "attempts": 3},
        )
        
        assert issue.code == "VALIDATION_FAILED"
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == "Value validation failed"
        assert issue.field_path == "user.email"
        assert issue.expected_value == "valid email format"
        assert issue.actual_value == "invalid-email"
        assert issue.suggestion == "Please provide a valid email address"
        assert issue.metadata == {"validator": "email", "attempts": 3}

    def test_string_validation(self):
        """Test string validation and cleanup."""
        issue = ValidationIssue(
            code="  TEST_CODE  ",
            severity=ValidationSeverity.INFO,
            message="  Test message  ",
        )
        
        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"


class TestValidationResult:
    """Test cases for ValidationResult model."""

    def test_create_empty_result(self):
        """Test creating empty validation result."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.issues == []
        assert result.validated_models == []
        assert result.validation_context == {}

    def test_create_result_with_data(self):
        """Test creating validation result with data."""
        result = ValidationResult(
            is_valid=False,
            validated_models=["Order", "TradingPair"],
            validation_context={"timestamp": "2024-01-01", "validator_version": "1.0"},
        )
        
        assert result.is_valid is False
        assert result.validated_models == ["Order", "TradingPair"]
        assert result.validation_context == {"timestamp": "2024-01-01", "validator_version": "1.0"}

    def test_add_issue_method(self):
        """Test add issue method."""
        result = ValidationResult(is_valid=True)
        
        result.add_issue(
            code="TEST_WARNING",
            severity=ValidationSeverity.WARNING,
            message="This is a warning",
            field_path="field1",
            expected_value="expected",
            actual_value="actual",
            suggestion="Fix the issue",
            custom_field="custom_value",
        )
        
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.code == "TEST_WARNING"
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == "This is a warning"
        assert issue.field_path == "field1"
        assert issue.expected_value == "expected"
        assert issue.actual_value == "actual"
        assert issue.suggestion == "Fix the issue"
        assert issue.metadata == {"custom_field": "custom_value"}
        
        # Adding WARNING should not change is_valid status
        assert result.is_valid is True

    def test_add_error_changes_validity(self):
        """Test that adding error-level issue changes validity."""
        result = ValidationResult(is_valid=True)
        
        result.add_issue(
            code="TEST_ERROR",
            severity=ValidationSeverity.ERROR,
            message="This is an error",
        )
        
        assert result.is_valid is False

    def test_add_critical_changes_validity(self):
        """Test that adding critical error changes validity."""
        result = ValidationResult(is_valid=True)
        
        result.add_issue(
            code="TEST_CRITICAL",
            severity=ValidationSeverity.CRITICAL,
            message="This is critical",
        )
        
        assert result.is_valid is False

    def test_has_errors_property(self):
        """Test has_errors property."""
        result = ValidationResult(is_valid=True)
        
        # Initial state has no errors
        assert result.has_errors is False
        
        # Add INFO level issue
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info message")
        assert result.has_errors is False
        
        # Add WARNING level issue
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning message")
        assert result.has_errors is False
        
        # Add ERROR level issue
        result.add_issue("ERROR_CODE", ValidationSeverity.ERROR, "Error message")
        assert result.has_errors is True
        
        # Add CRITICAL level issue
        result.add_issue("CRITICAL_CODE", ValidationSeverity.CRITICAL, "Critical message")
        assert result.has_errors is True

    def test_has_warnings_property(self):
        """Test has_warnings property."""
        result = ValidationResult(is_valid=True)
        
        # Initial state has no warnings
        assert result.has_warnings is False
        
        # Add INFO level issue
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info message")
        assert result.has_warnings is False
        
        # Add WARNING level issue
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning message")
        assert result.has_warnings is True

    def test_error_count_property(self):
        """Test error_count property."""
        result = ValidationResult(is_valid=True)
        
        assert result.error_count == 0
        
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info")
        assert result.error_count == 0
        
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning")
        assert result.error_count == 0
        
        result.add_issue("ERROR_CODE", ValidationSeverity.ERROR, "Error")
        assert result.error_count == 1
        
        result.add_issue("CRITICAL_CODE", ValidationSeverity.CRITICAL, "Critical")
        assert result.error_count == 2

    def test_warning_count_property(self):
        """Test warning_count property."""
        result = ValidationResult(is_valid=True)
        
        assert result.warning_count == 0
        
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info")
        assert result.warning_count == 0
        
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning")
        assert result.warning_count == 1
        
        result.add_issue("ERROR_CODE", ValidationSeverity.ERROR, "Error")
        assert result.warning_count == 1
        
        result.add_issue("WARNING_CODE2", ValidationSeverity.WARNING, "Warning 2")
        assert result.warning_count == 2

    def test_get_issues_by_severity(self):
        """Test getting issues by severity level."""
        result = ValidationResult(is_valid=True)
        
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info")
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning")
        result.add_issue("ERROR_CODE", ValidationSeverity.ERROR, "Error")
        result.add_issue("CRITICAL_CODE", ValidationSeverity.CRITICAL, "Critical")
        result.add_issue("WARNING_CODE2", ValidationSeverity.WARNING, "Warning 2")
        
        info_issues = result.get_issues_by_severity(ValidationSeverity.INFO)
        assert len(info_issues) == 1
        assert info_issues[0].code == "INFO_CODE"
        
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert len(warning_issues) == 2
        assert {issue.code for issue in warning_issues} == {"WARNING_CODE", "WARNING_CODE2"}
        
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) == 1
        assert error_issues[0].code == "ERROR_CODE"
        
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        assert len(critical_issues) == 1
        assert critical_issues[0].code == "CRITICAL_CODE"

    def test_get_summary(self):
        """Test getting validation result summary."""
        result = ValidationResult(
            is_valid=False,
            validated_models=["Order", "TradingPair", "Trade"],
        )
        
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info")
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning")
        result.add_issue("ERROR_CODE", ValidationSeverity.ERROR, "Error")
        result.add_issue("CRITICAL_CODE", ValidationSeverity.CRITICAL, "Critical")
        
        summary = result.get_summary()
        
        expected_summary = {
            "is_valid": False,
            "total_issues": 4,
            "error_count": 2,  # ERROR + CRITICAL
            "warning_count": 1,
            "validated_models": ["Order", "TradingPair", "Trade"],
        }
        
        assert summary == expected_summary

    def test_complex_validation_scenario(self):
        """Test complex validation scenario."""
        result = ValidationResult(
            is_valid=True,
            validated_models=["Order", "TradingPair"],
        )
        
        # Add multiple issues at different severity levels
        result.add_issue(
            code="ORDER_INFO",
            severity=ValidationSeverity.INFO,
            message="Order created successfully",
            field_path="order.id",
            metadata={"order_id": "12345"},
        )
        
        result.add_issue(
            code="PRECISION_WARNING",
            severity=ValidationSeverity.WARNING,
            message="Price precision may be too high",
            field_path="order.price",
            suggestion="Consider reducing precision",
        )
        
        result.add_issue(
            code="QUANTITY_ERROR",
            severity=ValidationSeverity.ERROR,
            message="Quantity below minimum",
            field_path="order.quantity",
            expected_value=">=0.001",
            actual_value="0.0005",
        )
        
        # Verify result status
        assert result.is_valid is False  # Because there's an ERROR
        assert result.has_errors is True
        assert result.has_warnings is True
        assert result.error_count == 1
        assert result.warning_count == 1
        assert len(result.issues) == 3
        
        # Verify summary
        summary = result.get_summary()
        assert summary["total_issues"] == 3
        assert summary["error_count"] == 1
        assert summary["warning_count"] == 1