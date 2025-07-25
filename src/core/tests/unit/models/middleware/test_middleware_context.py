# ABOUTME: Unit tests for MiddlewareContext model
# ABOUTME: Tests context creation, modification, and execution tracking

import pytest
from datetime import datetime
from uuid import UUID

from core.models.middleware import MiddlewareContext


class TestMiddlewareContext:
    """Unit tests for MiddlewareContext model."""

    @pytest.mark.unit
    def test_context_creation_with_defaults(self):
        """Test creating context with default values."""
        context = MiddlewareContext()

        # Check auto-generated fields
        assert context.id is not None
        assert UUID(context.id)  # Should be valid UUID
        assert isinstance(context.timestamp, datetime)
        assert context.timestamp.tzinfo is not None

        # Check default values
        assert context.request_id is None
        assert context.event_type is None
        assert context.symbol is None
        assert context.data is None
        assert context.metadata == {}
        assert context.trace_id is None
        assert context.span_id is None
        assert context.user_id is None
        assert context.session_id is None
        assert context.is_cancelled is False
        assert context.execution_path == []
        assert context.enable_logging is True
        assert context.enable_metrics is True
        assert context.enable_tracing is True

    @pytest.mark.unit
    def test_context_creation_with_data(self):
        """Test creating context with specific data."""
        test_data = {"key": "value", "number": 42}
        context = MiddlewareContext(
            event_type="TRADE", symbol="BTCUSD", data=test_data, user_id="user123", session_id="session456"
        )

        assert context.event_type == "TRADE"
        assert context.symbol == "BTCUSD"
        assert context.data == test_data
        assert context.user_id == "user123"
        assert context.session_id == "session456"

    @pytest.mark.unit
    def test_context_with_metadata(self):
        """Test creating context with metadata."""
        metadata = {"authorization": "Bearer token123", "user_agent": "test-client", "ip_address": "127.0.0.1"}
        context = MiddlewareContext(metadata=metadata)

        assert context.metadata == metadata
        assert context.metadata["authorization"] == "Bearer token123"

    @pytest.mark.unit
    def test_context_with_tracing_info(self):
        """Test creating context with tracing information."""
        context = MiddlewareContext(trace_id="trace123", span_id="span456", request_id="req789")

        assert context.trace_id == "trace123"
        assert context.span_id == "span456"
        assert context.request_id == "req789"

    @pytest.mark.unit
    def test_add_execution_step(self):
        """Test adding execution steps to context."""
        context = MiddlewareContext()

        # Initially empty
        assert context.execution_path == []

        # Add steps
        context.add_execution_step("AuthMiddleware")
        assert context.execution_path == ["AuthMiddleware"]

        context.add_execution_step("LoggingMiddleware")
        assert context.execution_path == ["AuthMiddleware", "LoggingMiddleware"]

        context.add_execution_step("RateLimitMiddleware")
        assert context.execution_path == ["AuthMiddleware", "LoggingMiddleware", "RateLimitMiddleware"]

    @pytest.mark.unit
    def test_cancel_context(self):
        """Test cancelling context."""
        context = MiddlewareContext()

        # Initially not cancelled
        assert context.is_cancelled is False

        # Cancel context
        context.cancel()
        assert context.is_cancelled is True

    @pytest.mark.unit
    def test_get_execution_duration(self):
        """Test getting execution duration."""
        context = MiddlewareContext()

        # Should return a positive duration
        duration = context.get_execution_duration()
        assert isinstance(duration, float)
        assert duration >= 0

        # Duration should increase over time
        import time

        time.sleep(0.01)  # Small delay
        duration2 = context.get_execution_duration()
        assert duration2 > duration

    @pytest.mark.unit
    def test_clone_context(self):
        """Test cloning context."""
        original = MiddlewareContext(
            event_type="TRADE", symbol="BTCUSD", data={"price": 50000}, user_id="user123", metadata={"auth": "token"}
        )
        original.add_execution_step("TestMiddleware")

        # Clone context
        cloned = original.clone()

        # Should have different ID and timestamp
        assert cloned.id != original.id
        assert cloned.timestamp != original.timestamp

        # Should have same data but different objects
        assert cloned.event_type == original.event_type
        assert cloned.symbol == original.symbol
        assert cloned.data == original.data
        assert cloned.user_id == original.user_id
        assert cloned.metadata == original.metadata

        # Execution path should be preserved
        assert cloned.execution_path == original.execution_path

        # Should be independent objects
        cloned.add_execution_step("NewMiddleware")
        assert len(cloned.execution_path) == 2
        assert len(original.execution_path) == 1

    @pytest.mark.unit
    def test_context_configuration_flags(self):
        """Test context configuration flags."""
        context = MiddlewareContext(enable_logging=False, enable_metrics=False, enable_tracing=False)

        assert context.enable_logging is False
        assert context.enable_metrics is False
        assert context.enable_tracing is False

    @pytest.mark.unit
    def test_context_with_generic_data_type(self):
        """Test context with generic data type."""
        # Test with string data
        str_context = MiddlewareContext[str](data="test string")
        assert str_context.data == "test string"

        # Test with dict data
        dict_context = MiddlewareContext[dict](data={"key": "value"})
        assert dict_context.data == {"key": "value"}

        # Test with list data
        list_context = MiddlewareContext[list](data=[1, 2, 3])
        assert list_context.data == [1, 2, 3]

    @pytest.mark.unit
    def test_context_json_serialization(self):
        """Test context JSON serialization."""
        context = MiddlewareContext(event_type="TRADE", symbol="BTCUSD", data={"price": 50000}, user_id="user123")

        # Should be serializable to dict
        context_dict = context.model_dump()
        assert isinstance(context_dict, dict)
        assert context_dict["event_type"] == "TRADE"
        assert context_dict["symbol"] == "BTCUSD"
        assert context_dict["data"] == {"price": 50000}
        assert context_dict["user_id"] == "user123"

        # Should be serializable to JSON
        context_json = context.model_dump_json()
        assert isinstance(context_json, str)
        assert '"event_type":"TRADE"' in context_json

    @pytest.mark.unit
    def test_context_validation(self):
        """Test context validation."""
        # Valid context should not raise
        context = MiddlewareContext(event_type="TRADE", symbol="BTCUSD", data={"price": 50000})
        assert context.event_type == "TRADE"

        # Test with invalid data should still work (no strict validation)
        context_with_none = MiddlewareContext(data=None)
        assert context_with_none.data is None
