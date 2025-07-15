# ABOUTME: Unit tests for MiddlewareResult and PipelineResult models
# ABOUTME: Tests result creation, status tracking, and execution metrics

import pytest
from datetime import datetime, UTC

from core.models.middleware import MiddlewareResult, MiddlewareStatus, PipelineResult


class TestMiddlewareResult:
    """Unit tests for MiddlewareResult model."""

    @pytest.mark.unit
    def test_result_creation_with_required_fields(self):
        """Test creating result with required fields."""
        result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)

        assert result.middleware_name == "TestMiddleware"
        assert result.status == MiddlewareStatus.SUCCESS
        assert isinstance(result.started_at, datetime)
        assert result.completed_at is None
        assert result.data is None
        assert result.error is None
        assert result.error_details is None
        assert result.execution_time_ms is None
        assert result.memory_usage_mb is None
        assert result.metadata == {}
        assert result.should_continue is True
        assert result.modified_context is None

    @pytest.mark.unit
    def test_result_creation_with_all_fields(self):
        """Test creating result with all fields."""
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)

        result = MiddlewareResult(
            middleware_name="TestMiddleware",
            status=MiddlewareStatus.SUCCESS,
            started_at=started_at,
            completed_at=completed_at,
            data={"key": "value"},
            error="Test error",
            error_details={"code": 500},
            execution_time_ms=150.5,
            memory_usage_mb=2.3,
            metadata={"info": "test"},
            should_continue=False,
            modified_context={"user_id": "123"},
        )

        assert result.middleware_name == "TestMiddleware"
        assert result.status == MiddlewareStatus.SUCCESS
        assert result.started_at == started_at
        assert result.completed_at == completed_at
        assert result.data == {"key": "value"}
        assert result.error == "Test error"
        assert result.error_details == {"code": 500}
        assert result.execution_time_ms == 150.5
        assert result.memory_usage_mb == 2.3
        assert result.metadata == {"info": "test"}
        assert result.should_continue is False
        assert result.modified_context == {"user_id": "123"}

    @pytest.mark.unit
    def test_mark_completed(self):
        """Test marking result as completed."""
        result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)

        # Initially not completed
        assert result.completed_at is None
        assert result.execution_time_ms is None

        # Mark as completed
        result.mark_completed()

        # Should have completion time and execution time
        assert result.completed_at is not None
        assert isinstance(result.completed_at, datetime)
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.unit
    def test_mark_completed_with_existing_execution_time(self):
        """Test marking completed when execution time already set."""
        result = MiddlewareResult(
            middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS, execution_time_ms=100.0
        )

        # Mark as completed
        result.mark_completed()

        # Should not override existing execution time
        assert result.execution_time_ms == 100.0
        assert result.completed_at is not None

    @pytest.mark.unit
    def test_mark_failed(self):
        """Test marking result as failed."""
        result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)

        # Mark as failed
        result.mark_failed("Test error", {"code": 500})

        assert result.status == MiddlewareStatus.FAILED
        assert result.error == "Test error"
        assert result.error_details == {"code": 500}
        assert result.should_continue is False
        assert result.completed_at is not None

    @pytest.mark.unit
    def test_mark_failed_without_details(self):
        """Test marking failed without error details."""
        result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)

        # Mark as failed without details
        result.mark_failed("Test error")

        assert result.status == MiddlewareStatus.FAILED
        assert result.error == "Test error"
        assert result.error_details == {}
        assert result.should_continue is False

    @pytest.mark.unit
    def test_mark_skipped(self):
        """Test marking result as skipped."""
        result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)

        # Mark as skipped
        result.mark_skipped("Not applicable")

        assert result.status == MiddlewareStatus.SKIPPED
        assert result.metadata["skip_reason"] == "Not applicable"
        assert result.completed_at is not None

    @pytest.mark.unit
    def test_mark_cancelled(self):
        """Test marking result as cancelled."""
        result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)

        # Mark as cancelled
        result.mark_cancelled()

        assert result.status == MiddlewareStatus.CANCELLED
        assert result.should_continue is False
        assert result.completed_at is not None

    @pytest.mark.unit
    def test_is_successful(self):
        """Test is_successful method."""
        success_result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)
        assert success_result.is_successful() is True

        failed_result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.FAILED)
        assert failed_result.is_successful() is False

        skipped_result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SKIPPED)
        assert skipped_result.is_successful() is False

    @pytest.mark.unit
    def test_is_failed(self):
        """Test is_failed method."""
        failed_result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.FAILED)
        assert failed_result.is_failed() is True

        success_result = MiddlewareResult(middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS)
        assert success_result.is_failed() is False

    @pytest.mark.unit
    def test_get_execution_summary(self):
        """Test getting execution summary."""
        result = MiddlewareResult(
            middleware_name="TestMiddleware",
            status=MiddlewareStatus.SUCCESS,
            execution_time_ms=150.5,
            memory_usage_mb=2.3,
            data={"key": "value"},
            error="Test error",
        )

        summary = result.get_execution_summary()

        assert summary["middleware_name"] == "TestMiddleware"
        assert summary["status"] == "success"
        assert summary["execution_time_ms"] == 150.5
        assert summary["memory_usage_mb"] == 2.3
        assert summary["should_continue"] is True
        assert summary["has_error"] is True
        assert summary["has_data"] is True

    @pytest.mark.unit
    def test_json_serialization(self):
        """Test JSON serialization."""
        result = MiddlewareResult(
            middleware_name="TestMiddleware", status=MiddlewareStatus.SUCCESS, data={"key": "value"}
        )

        # Should be serializable to dict
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert result_dict["middleware_name"] == "TestMiddleware"
        assert result_dict["status"] == "success"

        # Should be serializable to JSON
        result_json = result.model_dump_json()
        assert isinstance(result_json, str)
        assert '"middleware_name":"TestMiddleware"' in result_json


class TestPipelineResult:
    """Unit tests for PipelineResult model."""

    @pytest.mark.unit
    def test_pipeline_result_creation(self):
        """Test creating pipeline result."""
        result = PipelineResult(pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS, total_middlewares=3)

        assert result.pipeline_name == "TestPipeline"
        assert result.status == MiddlewareStatus.SUCCESS
        assert result.total_middlewares == 3
        assert isinstance(result.started_at, datetime)
        assert result.completed_at is None
        assert result.middleware_results == []
        assert result.executed_middlewares == 0
        assert result.successful_middlewares == 0
        assert result.failed_middlewares == 0
        assert result.metadata == {}

    @pytest.mark.unit
    def test_add_middleware_result(self):
        """Test adding middleware results."""
        pipeline_result = PipelineResult(pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS)

        # Add successful result
        success_result = MiddlewareResult(middleware_name="SuccessMiddleware", status=MiddlewareStatus.SUCCESS)
        pipeline_result.add_middleware_result(success_result)

        assert len(pipeline_result.middleware_results) == 1
        assert pipeline_result.executed_middlewares == 1
        assert pipeline_result.successful_middlewares == 1
        assert pipeline_result.failed_middlewares == 0

        # Add failed result
        failed_result = MiddlewareResult(middleware_name="FailedMiddleware", status=MiddlewareStatus.FAILED)
        pipeline_result.add_middleware_result(failed_result)

        assert len(pipeline_result.middleware_results) == 2
        assert pipeline_result.executed_middlewares == 2
        assert pipeline_result.successful_middlewares == 1
        assert pipeline_result.failed_middlewares == 1

    @pytest.mark.unit
    def test_mark_completed_with_success(self):
        """Test marking pipeline as completed with success."""
        pipeline_result = PipelineResult(pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS)

        # Add successful result
        success_result = MiddlewareResult(middleware_name="SuccessMiddleware", status=MiddlewareStatus.SUCCESS)
        pipeline_result.add_middleware_result(success_result)

        # Mark as completed
        pipeline_result.mark_completed()

        assert pipeline_result.completed_at is not None
        assert pipeline_result.status == MiddlewareStatus.SUCCESS

    @pytest.mark.unit
    def test_mark_completed_with_failures(self):
        """Test marking pipeline as completed with failures."""
        pipeline_result = PipelineResult(pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS)

        # Add failed result
        failed_result = MiddlewareResult(middleware_name="FailedMiddleware", status=MiddlewareStatus.FAILED)
        pipeline_result.add_middleware_result(failed_result)

        # Mark as completed
        pipeline_result.mark_completed()

        assert pipeline_result.completed_at is not None
        assert pipeline_result.status == MiddlewareStatus.FAILED

    @pytest.mark.unit
    def test_mark_completed_with_no_execution(self):
        """Test marking pipeline as completed with no execution."""
        pipeline_result = PipelineResult(pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS)

        # Mark as completed without any middleware results
        pipeline_result.mark_completed()

        assert pipeline_result.completed_at is not None
        assert pipeline_result.status == MiddlewareStatus.SKIPPED

    @pytest.mark.unit
    def test_get_execution_time_ms(self):
        """Test getting execution time."""
        pipeline_result = PipelineResult(pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS)

        # Not completed yet
        assert pipeline_result.get_execution_time_ms() is None

        # Mark as completed
        pipeline_result.mark_completed()

        # Should have execution time
        execution_time = pipeline_result.get_execution_time_ms()
        assert execution_time is not None
        assert execution_time >= 0

    @pytest.mark.unit
    def test_get_summary(self):
        """Test getting pipeline summary."""
        pipeline_result = PipelineResult(
            pipeline_name="TestPipeline", status=MiddlewareStatus.SUCCESS, total_middlewares=2
        )

        # Add results
        success_result = MiddlewareResult(middleware_name="SuccessMiddleware", status=MiddlewareStatus.SUCCESS)
        failed_result = MiddlewareResult(middleware_name="FailedMiddleware", status=MiddlewareStatus.FAILED)

        pipeline_result.add_middleware_result(success_result)
        pipeline_result.add_middleware_result(failed_result)
        pipeline_result.mark_completed()

        summary = pipeline_result.get_summary()

        assert summary["pipeline_name"] == "TestPipeline"
        assert summary["status"] == "failed"  # Because of failed middleware
        assert summary["total_middlewares"] == 2
        assert summary["executed_middlewares"] == 2
        assert summary["successful_middlewares"] == 1
        assert summary["failed_middlewares"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["execution_time_ms"] is not None
