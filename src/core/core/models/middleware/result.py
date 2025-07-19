# ABOUTME: MiddlewareResult and MiddlewareStatus models for middleware execution results
# ABOUTME: Contains execution status, timing, error information, and result data

from datetime import datetime, UTC
from typing import Any, Dict, Optional, List
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class MiddlewareStatus(str, Enum):
    """
    Middleware execution status enumeration.

    This enum defines the possible states of middleware execution.
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class MiddlewareResult(BaseModel):
    """
    Middleware execution result.

    This model contains all information about the outcome of middleware execution,
    including status, timing, error information, and result data.
    """

    # Basic execution information
    middleware_name: str = Field(description="Name of the middleware that produced this result")
    status: MiddlewareStatus = Field(description="Execution status")

    # Timing information
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When the middleware started execution"
    )
    completed_at: Optional[datetime] = Field(default=None, description="When the middleware completed execution")

    # Result data
    data: Optional[Any] = Field(default=None, description="Result data from middleware execution")

    # Error information
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed error information")

    # Execution statistics
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in megabytes")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Pipeline control
    should_continue: bool = Field(default=True, description="Whether the pipeline should continue execution")

    # Context modifications
    modified_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Context modifications made by this middleware"
    )

    model_config = ConfigDict(
        # Note: json_encoders is deprecated in Pydantic v2
        # datetime serialization is handled automatically via isoformat()
    )

    def mark_completed(self) -> None:
        """
        Mark the middleware execution as completed.

        This method sets the completion timestamp and calculates execution time.
        """
        self.completed_at = datetime.now(UTC)
        if self.started_at and self.execution_time_ms is None:
            self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def mark_failed(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the middleware execution as failed.

        Args:
            error_message: Error message describing the failure.
            error_details: Optional detailed error information.
        """
        self.status = MiddlewareStatus.FAILED
        self.error = error_message
        self.error_details = error_details if error_details is not None else {}
        self.should_continue = False
        self.mark_completed()

    def mark_skipped(self, reason: str) -> None:
        """
        Mark the middleware execution as skipped.

        Args:
            reason: Reason why the middleware was skipped.
        """
        self.status = MiddlewareStatus.SKIPPED
        self.metadata["skip_reason"] = reason
        self.mark_completed()

    def mark_cancelled(self) -> None:
        """Mark the middleware execution as cancelled."""
        self.status = MiddlewareStatus.CANCELLED
        self.should_continue = False
        self.mark_completed()

    def is_successful(self) -> bool:
        """
        Check if the middleware execution was successful.

        Returns:
            bool: True if status is SUCCESS, False otherwise.
        """
        return self.status == MiddlewareStatus.SUCCESS

    def is_failed(self) -> bool:
        """
        Check if the middleware execution failed.

        Returns:
            bool: True if status is FAILED, False otherwise.
        """
        return self.status == MiddlewareStatus.FAILED

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the middleware execution.

        Returns:
            Dict[str, Any]: Summary containing key execution metrics.
        """
        summary = {
            "middleware_name": self.middleware_name,
            "status": self.status.value,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "should_continue": self.should_continue,
            "has_error": self.error is not None,
            "has_data": self.data is not None,
            "data": self.data,
            "metadata": self.metadata,
        }

        # Include error information if present
        if self.error is not None:
            summary["error"] = self.error
        if self.error_details is not None:
            summary["error_details"] = self.error_details

        return summary


class PipelineResult(BaseModel):
    """
    Result of executing an entire middleware pipeline.

    This model aggregates results from all middleware in the pipeline.
    """

    # Basic pipeline information
    pipeline_name: str = Field(default="MiddlewarePipeline", description="Name of the pipeline")
    status: MiddlewareStatus = Field(description="Overall pipeline status")
    status_fixed: bool = Field(default=False, description="Whether the status has been manually fixed")

    # Timing information
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When the pipeline started execution"
    )
    completed_at: Optional[datetime] = Field(default=None, description="When the pipeline completed execution")

    # Execution results
    middleware_results: List[MiddlewareResult] = Field(
        default_factory=list, description="Results from individual middleware executions"
    )

    # Summary statistics
    total_middlewares: int = Field(default=0, description="Total number of middleware in pipeline")
    executed_middlewares: int = Field(default=0, description="Number of middleware that were executed")
    successful_middlewares: int = Field(default=0, description="Number of successful middleware executions")
    failed_middlewares: int = Field(default=0, description="Number of failed middleware executions")

    # Pipeline metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata")

    model_config = ConfigDict(
        # Note: json_encoders is deprecated in Pydantic v2
        # datetime serialization is handled automatically via isoformat()
    )

    def add_middleware_result(self, result: MiddlewareResult) -> None:
        """
        Add a middleware result to the pipeline result.

        Args:
            result: MiddlewareResult to add to the pipeline.
        """
        self.middleware_results.append(result)
        self.executed_middlewares += 1

        if result.is_successful():
            self.successful_middlewares += 1
        elif result.is_failed():
            self.failed_middlewares += 1

    def fix_status(self, status: MiddlewareStatus) -> None:
        """
        Fix the pipeline status to a specific value.

        This prevents mark_completed() from overriding the status based on statistics.
        Used for special cases like cancelled contexts or explicit pipeline success despite middleware failures.

        Args:
            status: The status to fix the pipeline to
        """
        self.status = status
        self.status_fixed = True

    def mark_completed(self) -> None:
        """Mark the pipeline execution as completed."""
        self.completed_at = datetime.now(UTC)

        # Only auto-determine status if it hasn't been explicitly fixed
        if not self.status_fixed:
            if self.failed_middlewares > 0:
                self.status = MiddlewareStatus.FAILED
            elif self.executed_middlewares == 0:
                self.status = MiddlewareStatus.SKIPPED
            else:
                self.status = MiddlewareStatus.SUCCESS

    def get_execution_time_ms(self) -> Optional[float]:
        """
        Get the total pipeline execution time in milliseconds.

        Returns:
            Optional[float]: Execution time in milliseconds, or None if not completed.
        """
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline execution.

        Returns:
            Dict[str, Any]: Summary containing key pipeline metrics.
        """
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "total_middlewares": self.total_middlewares,
            "executed_middlewares": self.executed_middlewares,
            "successful_middlewares": self.successful_middlewares,
            "failed_middlewares": self.failed_middlewares,
            "execution_time_ms": self.get_execution_time_ms(),
            "success_rate": self.successful_middlewares / max(self.executed_middlewares, 1),
        }
