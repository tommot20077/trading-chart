# ABOUTME: 驗證結果模型的單元測試，測試ValidationResult、ValidationIssue等類的功能
# ABOUTME: 確保驗證結果結構的正確性，包括問題追踪、嚴重程度分類和摘要生成

import pytest
from core.validators.validation_result import ValidationResult, ValidationIssue, ValidationSeverity


class TestValidationSeverity:
    """驗證嚴重程度枚舉測試."""

    def test_severity_values(self):
        """測試嚴重程度枚舉值."""
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.ERROR == "error"
        assert ValidationSeverity.CRITICAL == "critical"


class TestValidationIssue:
    """驗證問題模型測試."""

    def test_create_basic_issue(self):
        """測試創建基本驗證問題."""
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
        """測試創建詳細驗證問題."""
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
        """測試字符串驗證和清理."""
        issue = ValidationIssue(
            code="  TEST_CODE  ",
            severity=ValidationSeverity.INFO,
            message="  Test message  ",
        )
        
        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"


class TestValidationResult:
    """驗證結果模型測試."""

    def test_create_empty_result(self):
        """測試創建空驗證結果."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.issues == []
        assert result.validated_models == []
        assert result.validation_context == {}

    def test_create_result_with_data(self):
        """測試創建帶數據的驗證結果."""
        result = ValidationResult(
            is_valid=False,
            validated_models=["Order", "TradingPair"],
            validation_context={"timestamp": "2024-01-01", "validator_version": "1.0"},
        )
        
        assert result.is_valid is False
        assert result.validated_models == ["Order", "TradingPair"]
        assert result.validation_context == {"timestamp": "2024-01-01", "validator_version": "1.0"}

    def test_add_issue_method(self):
        """測試添加驗證問題方法."""
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
        
        # 添加WARNING不應改變is_valid狀態
        assert result.is_valid is True

    def test_add_error_changes_validity(self):
        """測試添加錯誤級別問題會改變有效性."""
        result = ValidationResult(is_valid=True)
        
        result.add_issue(
            code="TEST_ERROR",
            severity=ValidationSeverity.ERROR,
            message="This is an error",
        )
        
        assert result.is_valid is False

    def test_add_critical_changes_validity(self):
        """測試添加嚴重錯誤會改變有效性."""
        result = ValidationResult(is_valid=True)
        
        result.add_issue(
            code="TEST_CRITICAL",
            severity=ValidationSeverity.CRITICAL,
            message="This is critical",
        )
        
        assert result.is_valid is False

    def test_has_errors_property(self):
        """測試has_errors屬性."""
        result = ValidationResult(is_valid=True)
        
        # 初始狀態無錯誤
        assert result.has_errors is False
        
        # 添加INFO級別問題
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info message")
        assert result.has_errors is False
        
        # 添加WARNING級別問題
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning message")
        assert result.has_errors is False
        
        # 添加ERROR級別問題
        result.add_issue("ERROR_CODE", ValidationSeverity.ERROR, "Error message")
        assert result.has_errors is True
        
        # 添加CRITICAL級別問題
        result.add_issue("CRITICAL_CODE", ValidationSeverity.CRITICAL, "Critical message")
        assert result.has_errors is True

    def test_has_warnings_property(self):
        """測試has_warnings屬性."""
        result = ValidationResult(is_valid=True)
        
        # 初始狀態無警告
        assert result.has_warnings is False
        
        # 添加INFO級別問題
        result.add_issue("INFO_CODE", ValidationSeverity.INFO, "Info message")
        assert result.has_warnings is False
        
        # 添加WARNING級別問題
        result.add_issue("WARNING_CODE", ValidationSeverity.WARNING, "Warning message")
        assert result.has_warnings is True

    def test_error_count_property(self):
        """測試error_count屬性."""
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
        """測試warning_count屬性."""
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
        """測試根據嚴重程度獲取問題列表."""
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
        """測試獲取驗證結果摘要."""
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
        """測試複雜驗證場景."""
        result = ValidationResult(
            is_valid=True,
            validated_models=["Order", "TradingPair"],
        )
        
        # 添加多個不同級別的問題
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
        
        # 驗證結果狀態
        assert result.is_valid is False  # 因為有ERROR
        assert result.has_errors is True
        assert result.has_warnings is True
        assert result.error_count == 1
        assert result.warning_count == 1
        assert len(result.issues) == 3
        
        # 驗證摘要
        summary = result.get_summary()
        assert summary["total_issues"] == 3
        assert summary["error_count"] == 1
        assert summary["warning_count"] == 1