# ABOUTME: 驗證結果模型，定義跨模型驗證的結果結構和錯誤處理
# ABOUTME: 提供標準化的驗證結果格式，包含成功/失敗狀態、錯誤信息和修復建議

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class ValidationSeverity(str, Enum):
    """驗證問題嚴重程度."""

    INFO = "info"  # 信息級別，不影響功能
    WARNING = "warning"  # 警告級別，可能影響性能或準確性
    ERROR = "error"  # 錯誤級別，違反業務規則
    CRITICAL = "critical"  # 嚴重錯誤，可能導致系統問題


class ValidationIssue(BaseModel):
    """單個驗證問題的詳細信息."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    code: str = Field(description="驗證問題的唯一標識碼")
    severity: ValidationSeverity = Field(description="問題嚴重程度")
    message: str = Field(description="問題描述信息")
    field_path: Optional[str] = Field(None, description="有問題的字段路徑")
    expected_value: Optional[Any] = Field(None, description="期望的值")
    actual_value: Optional[Any] = Field(None, description="實際的值")
    suggestion: Optional[str] = Field(None, description="修復建議")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元數據")


class ValidationResult(BaseModel):
    """跨模型驗證結果."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    is_valid: bool = Field(description="總體驗證是否通過")
    issues: List[ValidationIssue] = Field(default_factory=list, description="發現的問題列表")
    validated_models: List[str] = Field(default_factory=list, description="參與驗證的模型類型")
    validation_context: Dict[str, Any] = Field(default_factory=dict, description="驗證上下文信息")

    @property
    def has_errors(self) -> bool:
        """檢查是否有錯誤級別的問題."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """檢查是否有警告級別的問題."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    @property
    def error_count(self) -> int:
        """獲取錯誤數量."""
        return sum(
            1 for issue in self.issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        )

    @property
    def warning_count(self) -> int:
        """獲取警告數量."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)

    def add_issue(
        self,
        code: str,
        severity: ValidationSeverity,
        message: str,
        field_path: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        suggestion: Optional[str] = None,
        **metadata,
    ) -> None:
        """添加驗證問題."""
        issue = ValidationIssue(
            code=code,
            severity=severity,
            message=message,
            field_path=field_path,
            expected_value=expected_value,
            actual_value=actual_value,
            suggestion=suggestion,
            metadata=metadata,
        )
        self.issues.append(issue)

        # 如果有錯誤級別的問題，將整體驗證標記為失敗
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """根據嚴重程度獲取問題列表."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_summary(self) -> Dict[str, Any]:
        """獲取驗證結果摘要."""
        return {
            "is_valid": self.is_valid,
            "total_issues": len(self.issues),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validated_models": self.validated_models,
        }
