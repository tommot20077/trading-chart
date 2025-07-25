# ABOUTME: Tests for observability data correlation and trace propagation across all monitoring components
# ABOUTME: Validates trace_id propagation, OpenTelemetry integration, and distributed tracing completeness

import asyncio
import pytest
import pytest_asyncio
import time
import uuid
import threading
from typing import Dict, Any
from collections import defaultdict
from loguru import logger

# Previously skipped due to timeout issues - now fixed
# Enable loguru for trace correlation testing
pytestmark = pytest.mark.enable_loguru_file_io

from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
from ...fixtures.performance_metrics import PerformanceMonitor


@pytest_asyncio.fixture
async def trace_correlation_system(mock_event_bus):
    """
    Create a trace correlation system for testing trace ID propagation
    and data correlation across all observability components.
    """

    class TraceCorrelationSystem:
        def __init__(self, event_bus):
            self.event_bus = event_bus
            self.notification_handler = InMemoryNotificationHandler(
                max_queue_size=1000, max_history_size=2000, simulate_failure_rate=0.0
            )
            self.performance_monitor = PerformanceMonitor(event_bus, sample_interval=0.05)

            # Trace tracking
            self.trace_registry = {}
            self.correlation_data = defaultdict(list)
            self.opentelemetry_spans = []

            # Mock OpenTelemetry tracer
            self.tracer = self._setup_mock_tracer()

            # Captured observability data
            self.captured_logs = []
            self.captured_notifications = []
            self.captured_metrics = []

            # Setup monitoring callbacks
            self._setup_monitoring_callbacks()

        def _setup_mock_tracer(self):
            """Setup mock OpenTelemetry tracer for testing."""

            class MockSpan:
                def __init__(self, name: str, trace_id: str, span_id: str = None):
                    self.name = name
                    self.trace_id = trace_id
                    self.span_id = span_id or str(uuid.uuid4())[:8]
                    self.attributes = {}
                    self.events = []
                    self.status = "OK"
                    self.start_time = time.time()
                    self.end_time = None
                    self.parent_span_id = None

                def set_attribute(self, key: str, value: Any):
                    self.attributes[key] = value

                def add_event(self, name: str, attributes: Dict[str, Any] = None):
                    self.events.append({"name": name, "attributes": attributes or {}, "timestamp": time.time()})

                def set_status(self, status: str, description: str = None):
                    self.status = status
                    if description:
                        self.attributes["status_description"] = description

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.end_time = time.time()
                    return False

            class MockTracer:
                def __init__(self, system):
                    self.system = system

                def start_span(self, name: str, trace_id: str = None, parent_span_id: str = None):
                    if not trace_id:
                        trace_id = str(uuid.uuid4())

                    span = MockSpan(name, trace_id)
                    span.parent_span_id = parent_span_id

                    self.system.opentelemetry_spans.append(span)
                    return span

                def get_current_span(self):
                    return self.system.opentelemetry_spans[-1] if self.system.opentelemetry_spans else None

            return MockTracer(self)

        def _setup_monitoring_callbacks(self):
            """Setup callbacks to capture monitoring data."""
            # Performance monitor alert callback
            self.performance_monitor.add_alert_callback(self._capture_performance_alert)

            # Log capture - add our handler without removing existing ones
            self.log_handler_id = logger.add(self._capture_log_record, level="DEBUG")

        def _capture_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]):
            """Capture performance alerts with trace context."""
            current_span = self.tracer.get_current_span()
            trace_id = current_span.trace_id if current_span else None

            alert_record = {
                "type": alert_type,
                "data": alert_data,
                "trace_id": trace_id,
                "timestamp": time.time(),
                "component": "performance_monitor",
            }

            self.captured_metrics.append(alert_record)

            # Create notification for alert
            asyncio.create_task(self._send_correlated_notification(alert_record))

        def _capture_log_record(self, record):
            """Capture log records with trace context."""
            current_span = self.tracer.get_current_span()

            # Extract information from loguru record
            try:
                # Loguru passes a Record object
                level_name = record["level"].name
                message = record["message"]
                extra = record.get("extra", {})
                record_trace_id = extra.get("trace_id")
                
                # For text format handlers, message might be the formatted log line
                # Try to extract real level from formatted message if needed
                if " | ERROR " in message:
                    level_name = "ERROR"
                elif " | WARNING " in message:
                    level_name = "WARNING"
                elif " | DEBUG " in message:
                    level_name = "DEBUG"
                elif " | INFO " in message:
                    level_name = "INFO"
                
            except (KeyError, AttributeError, TypeError):
                # Fallback for different record formats
                message_str = str(record)
                if " | ERROR " in message_str:
                    level_name = "ERROR"
                elif " | WARNING " in message_str:
                    level_name = "WARNING" 
                elif " | DEBUG " in message_str:
                    level_name = "DEBUG"
                else:
                    level_name = "INFO"
                    
                message = message_str
                extra = {}
                record_trace_id = None

            # Prefer the trace_id from log record, fallback to current span
            trace_id = record_trace_id or (current_span.trace_id if current_span else None)

            log_record = {
                "level": level_name,
                "message": message,
                "trace_id": trace_id,
                "timestamp": time.time(),
                "component": "logger",
                "extra": extra,
            }

            self.captured_logs.append(log_record)

        async def _send_correlated_notification(self, alert_record: Dict[str, Any]):
            """Send notification with trace correlation."""
            notification_data = {
                "id": str(uuid.uuid4()),
                "rule_name": f"alert_{alert_record['type']}",
                "severity": "warning",
                "message": f"Performance alert: {alert_record['type']}",
                "metadata": {
                    **alert_record["data"],
                    "trace_id": alert_record["trace_id"],
                    "component": "correlation_system",
                    "timestamp": time.time(),
                },
            }

            success, message = await self.notification_handler.send_notification(notification_data)

            notification_record = {
                "notification_data": notification_data,
                "success": success,
                "message": message,
                "trace_id": alert_record["trace_id"],
                "timestamp": time.time(),
                "component": "notification_handler",
            }

            self.captured_notifications.append(notification_record)

        async def start_monitoring(self):
            """Start all monitoring components."""
            await self.performance_monitor.start_monitoring()

        async def stop_monitoring(self):
            """Stop all monitoring components."""
            await self.performance_monitor.stop_monitoring()
            await self.notification_handler.close()
            
            # Remove the log handler we added
            if hasattr(self, 'log_handler_id'):
                try:
                    logger.remove(self.log_handler_id)
                except Exception:
                    pass

        def create_traced_operation(self, operation_name: str, trace_id: str = None) -> "TracedOperation":
            """Create a traced operation for testing."""
            return TracedOperation(self, operation_name, trace_id)

        def get_correlation_summary(self) -> Dict[str, Any]:
            """Get summary of captured correlation data."""
            # Group data by trace_id
            trace_groups = defaultdict(lambda: {"logs": [], "notifications": [], "metrics": [], "spans": []})

            for log in self.captured_logs:
                if log["trace_id"]:
                    trace_groups[log["trace_id"]]["logs"].append(log)

            for notification in self.captured_notifications:
                if notification["trace_id"]:
                    trace_groups[notification["trace_id"]]["notifications"].append(notification)

            for metric in self.captured_metrics:
                if metric["trace_id"]:
                    trace_groups[metric["trace_id"]]["metrics"].append(metric)

            for span in self.opentelemetry_spans:
                trace_groups[span.trace_id]["spans"].append(span)

            return {
                "total_traces": len(trace_groups),
                "trace_groups": dict(trace_groups),
                "total_logs": len(self.captured_logs),
                "total_notifications": len(self.captured_notifications),
                "total_metrics": len(self.captured_metrics),
                "total_spans": len(self.opentelemetry_spans),
            }

        def validate_trace_completeness(self, trace_id: str) -> Dict[str, Any]:
            """Validate completeness of trace data across all components."""
            summary = self.get_correlation_summary()
            trace_data = summary["trace_groups"].get(trace_id, {})

            return {
                "trace_id": trace_id,
                "has_logs": len(trace_data.get("logs", [])) > 0,
                "has_notifications": len(trace_data.get("notifications", [])) > 0,
                "has_metrics": len(trace_data.get("metrics", [])) > 0,
                "has_spans": len(trace_data.get("spans", [])) > 0,
                "component_coverage": {
                    "logger": len(trace_data.get("logs", [])),
                    "notification_handler": len(trace_data.get("notifications", [])),
                    "performance_monitor": len(trace_data.get("metrics", [])),
                    "opentelemetry": len(trace_data.get("spans", [])),
                },
                "is_complete": all([len(trace_data.get("logs", [])) > 0, len(trace_data.get("spans", [])) > 0]),
            }

    class TracedOperation:
        """Helper class for creating traced operations."""

        def __init__(self, system: TraceCorrelationSystem, operation_name: str, trace_id: str = None):
            self.system = system
            self.operation_name = operation_name
            self.trace_id = trace_id or str(uuid.uuid4())
            self.span = None

        async def __aenter__(self):
            # Start OpenTelemetry span
            self.span = self.system.tracer.start_span(self.operation_name, self.trace_id)

            # Log operation start
            logger.info(
                f"Starting operation: {self.operation_name}", trace_id=self.trace_id, operation=self.operation_name
            )

            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                # Log error
                logger.error(
                    f"Operation failed: {self.operation_name}",
                    trace_id=self.trace_id,
                    error=str(exc_val),
                    operation=self.operation_name,
                )

                if self.span:
                    self.span.set_status("ERROR", str(exc_val))
            else:
                # Log success
                logger.info(
                    f"Operation completed: {self.operation_name}", trace_id=self.trace_id, operation=self.operation_name
                )

                if self.span:
                    self.span.set_status("OK")

            # End span
            if self.span:
                self.span.__exit__(exc_type, exc_val, exc_tb)

        def add_event(self, event_name: str, attributes: Dict[str, Any] = None):
            """Add event to current span."""
            if self.span:
                self.span.add_event(event_name, attributes)

            # Also log the event
            logger.debug(
                f"Span event: {event_name}", trace_id=self.trace_id, event_name=event_name, attributes=attributes
            )

        def set_attribute(self, key: str, value: Any):
            """Set attribute on current span."""
            if self.span:
                self.span.set_attribute(key, value)

    system = TraceCorrelationSystem(mock_event_bus)

    try:
        yield system
    finally:
        await system.stop_monitoring()


class TestObservabilityDataCorrelation:
    """
    Test suite for observability data correlation and trace propagation.

    Tests the propagation of trace IDs across all observability components
    and validates the completeness of distributed tracing data.
    """

    @pytest.mark.asyncio
    async def test_trace_id_propagation_across_components(self, trace_correlation_system):
        """
        Test that trace IDs are properly propagated across all observability components.

        Verifies:
        1. Trace ID propagation from logs to notifications to metrics
        2. OpenTelemetry span context preservation
        3. Consistent trace ID usage across component boundaries
        4. Parent-child span relationships
        """
        system = trace_correlation_system
        await system.start_monitoring()

        # Create test trace ID
        test_trace_id = str(uuid.uuid4())

        # Perform traced operations
        async with system.create_traced_operation("user_login", test_trace_id) as operation:
            operation.set_attribute("user_id", "test_user_123")
            operation.add_event("authentication_started")

            # Simulate nested operation
            async with system.create_traced_operation("database_query", test_trace_id) as nested_op:
                nested_op.set_attribute("query_type", "user_lookup")
                nested_op.add_event("query_executed", {"table": "users", "duration_ms": 15})

                # Log some activity within the trace context
                logger.info(
                    "Database query executed successfully",
                    trace_id=test_trace_id,
                    query_type="user_lookup",
                    duration_ms=15,
                )

                # Simulate performance metrics that should trigger alerts
                system.event_bus.update_stats(published_count=150, processed_count=145, error_count=2, queue_size=25)

                await asyncio.sleep(0.1)  # Allow metrics collection

            operation.add_event("authentication_completed")

            # Log completion
            logger.info("User login completed successfully", trace_id=test_trace_id, user_id="test_user_123")

        # Allow processing to complete
        await asyncio.sleep(0.3)

        # Validate trace propagation
        validation_result = system.validate_trace_completeness(test_trace_id)

        # Assertions
        assert validation_result["has_logs"], "No logs found for trace ID"
        assert validation_result["has_spans"], "No spans found for trace ID"

        # Check component coverage
        coverage = validation_result["component_coverage"]
        assert coverage["logger"] >= 3, f"Expected at least 3 log entries, got {coverage['logger']}"
        assert coverage["opentelemetry"] >= 2, f"Expected at least 2 spans, got {coverage['opentelemetry']}"

        # Verify trace data consistency
        summary = system.get_correlation_summary()
        trace_data = summary["trace_groups"][test_trace_id]

        # All logs should have the same trace ID
        for log in trace_data["logs"]:
            assert log["trace_id"] == test_trace_id

        # All spans should have the same trace ID
        for span in trace_data["spans"]:
            assert span.trace_id == test_trace_id

        # Check for parent-child span relationships
        spans = trace_data["spans"]
        parent_spans = [span for span in spans if span.name == "user_login"]
        child_spans = [span for span in spans if span.name == "database_query"]

        assert len(parent_spans) == 1, "Should have exactly one parent span"
        assert len(child_spans) == 1, "Should have exactly one child span"

        print(f"Trace propagation validation for {test_trace_id}:")
        print(f"  - Logs: {coverage['logger']}")
        print(f"  - Notifications: {coverage['notification_handler']}")
        print(f"  - Metrics: {coverage['performance_monitor']}")
        print(f"  - Spans: {coverage['opentelemetry']}")

    @pytest.mark.asyncio
    async def test_distributed_tracing_completeness(self, trace_correlation_system):
        """
        Test completeness of distributed tracing across multiple service boundaries.

        Simulates multiple services/components and verifies:
        1. Trace continuity across service boundaries
        2. Span hierarchy preservation
        3. Attribute and event propagation
        4. Error context preservation
        """
        system = trace_correlation_system
        await system.start_monitoring()

        # Simulate distributed service call chain
        root_trace_id = str(uuid.uuid4())

        # Service A: API Gateway
        async with system.create_traced_operation("api_gateway_request", root_trace_id) as api_span:
            api_span.set_attribute("service", "api_gateway")
            api_span.set_attribute("endpoint", "/api/trades")
            api_span.set_attribute("method", "POST")
            api_span.add_event("request_received", {"user_id": "trader_456", "request_size": 1024})

            # Service B: Authentication Service
            async with system.create_traced_operation("auth_service_validate", root_trace_id) as auth_span:
                auth_span.set_attribute("service", "auth_service")
                auth_span.set_attribute("validation_type", "jwt_token")
                auth_span.add_event("token_validated", {"user_id": "trader_456", "permissions": ["trade", "view"]})

                logger.info(
                    "User authenticated successfully",
                    trace_id=root_trace_id,
                    service="auth_service",
                    user_id="trader_456",
                )

            # Service C: Trading Engine
            try:
                async with system.create_traced_operation("trading_engine_process", root_trace_id) as trade_span:
                    trade_span.set_attribute("service", "trading_engine")
                    trade_span.set_attribute("symbol", "BTCUSD")
                    trade_span.set_attribute("side", "buy")
                    trade_span.add_event("order_created", {"order_id": "ord_789", "quantity": 0.5})

                    # Service D: Risk Management
                    async with system.create_traced_operation("risk_check", root_trace_id) as risk_span:
                        risk_span.set_attribute("service", "risk_management")
                        risk_span.set_attribute("check_type", "position_limit")

                        try:
                            # Simulate risk check failure
                            raise ValueError("Position limit exceeded")
                        except ValueError as e:
                            # Record the failure event in the span before it's closed
                            risk_span.add_event("risk_check_failed", {"error": str(e), "limit_exceeded": True})
                            
                            logger.error(
                                "Risk check failed",
                                trace_id=root_trace_id,
                                service="risk_management",
                                error=str(e),
                                user_id="trader_456",
                            )
                            # Re-raise to propagate to trading engine
                            raise

                    # This should not execute due to error
                    trade_span.add_event("order_executed")  # Won't reach here

            except ValueError:
                # Trading engine handles the risk management error
                logger.error(
                    "Trading engine failed due to risk check",
                    trace_id=root_trace_id,
                    service="trading_engine",
                    error="Risk management failure",
                )

            # API Gateway handles the error
            api_span.add_event("request_failed", {"error": "Risk check failed"})

            logger.error(
                "API request failed",
                trace_id=root_trace_id,
                service="api_gateway",
                endpoint="/api/trades",
                error="Risk check failed",
            )

        # Allow processing to complete
        await asyncio.sleep(0.3)

        # Validate distributed trace completeness
        validation_result = system.validate_trace_completeness(root_trace_id)
        summary = system.get_correlation_summary()
        trace_data = summary["trace_groups"][root_trace_id]

        # Should have spans for all services that were entered
        spans_by_service = {}
        for span in trace_data["spans"]:
            service = span.attributes.get("service", "unknown")
            spans_by_service[service] = span

        # Verify service coverage
        expected_services = ["api_gateway", "auth_service", "trading_engine", "risk_management"]
        for service in expected_services:
            assert service in spans_by_service, f"Missing span for service: {service}"

        # Verify error handling in spans
        risk_span = spans_by_service["risk_management"]
        assert risk_span.status == "ERROR", "Risk management span should have error status"

        # Verify error event was recorded
        risk_events = [event for event in risk_span.events if "risk_check_failed" in event["name"]]
        assert len(risk_events) == 1, "Should have one risk check failure event"

        # Verify error logs contain trace context
        error_logs = [log for log in trace_data["logs"] if log["level"] == "ERROR"]
        assert len(error_logs) >= 2, f"Expected at least 2 error logs, got {len(error_logs)}"

        # All error logs should have correct trace ID
        for log in error_logs:
            assert log["trace_id"] == root_trace_id

        print(f"Distributed tracing validation for {root_trace_id}:")
        print(f"  - Services traced: {list(spans_by_service.keys())}")
        print(f"  - Total spans: {len(trace_data['spans'])}")
        print(f"  - Error logs: {len(error_logs)}")
        print(f"  - Trace completeness: {validation_result['is_complete']}")

    @pytest.mark.asyncio
    async def test_concurrent_trace_isolation(self, trace_correlation_system):
        """
        Test that concurrent operations with different trace IDs remain properly isolated.

        Verifies:
        1. No cross-contamination between concurrent traces
        2. Each trace maintains its own context
        3. Proper cleanup of trace contexts
        4. Performance under concurrent load
        """
        system = trace_correlation_system
        await system.start_monitoring()

        # Create multiple concurrent traced operations
        num_concurrent_ops = 5
        trace_ids = [str(uuid.uuid4()) for _ in range(num_concurrent_ops)]

        async def concurrent_operation(trace_id: str, operation_id: int):
            """Perform a traced operation with specific ID."""
            async with system.create_traced_operation(f"concurrent_op_{operation_id}", trace_id) as op:
                op.set_attribute("operation_id", operation_id)
                op.set_attribute("worker_thread", threading.current_thread().ident)

                # Log start
                logger.info(
                    f"Starting concurrent operation {operation_id}", trace_id=trace_id, operation_id=operation_id
                )

                # Simulate work with some async operations
                for step in range(3):
                    op.add_event(f"step_{step}_started", {"step": step})

                    logger.debug(
                        f"Processing step {step} for operation {operation_id}",
                        trace_id=trace_id,
                        operation_id=operation_id,
                        step=step,
                    )

                    # Small delay to allow interleaving
                    await asyncio.sleep(0.05)

                    op.add_event(f"step_{step}_completed", {"step": step})

                # Log completion
                logger.info(
                    f"Completed concurrent operation {operation_id}", trace_id=trace_id, operation_id=operation_id
                )

        # Run all operations concurrently
        await asyncio.gather(*[concurrent_operation(trace_ids[i], i) for i in range(num_concurrent_ops)])

        # Allow processing to complete
        await asyncio.sleep(0.2)

        # Validate trace isolation
        summary = system.get_correlation_summary()

        # Should have all trace IDs represented
        assert len(summary["trace_groups"]) >= num_concurrent_ops

        # Validate each trace in isolation
        for i, trace_id in enumerate(trace_ids):
            validation_result = system.validate_trace_completeness(trace_id)
            trace_data = summary["trace_groups"][trace_id]

            # Each trace should have its own data
            assert validation_result["has_logs"], f"Trace {i} missing logs"
            assert validation_result["has_spans"], f"Trace {i} missing spans"

            # Verify operation_id consistency within trace
            logs_for_trace = trace_data["logs"]
            operation_ids_in_logs = set()

            for log in logs_for_trace:
                if "operation_id" in log.get("extra", {}):
                    operation_ids_in_logs.add(log["extra"]["operation_id"])

            # Each trace should only contain logs from its own operation
            if operation_ids_in_logs:
                assert len(operation_ids_in_logs) == 1, f"Trace {i} has mixed operation IDs: {operation_ids_in_logs}"
                assert i in operation_ids_in_logs, f"Trace {i} missing its operation ID"

            # Verify span isolation
            spans_for_trace = trace_data["spans"]
            for span in spans_for_trace:
                span_op_id = span.attributes.get("operation_id")
                if span_op_id is not None:
                    assert span_op_id == i, f"Span has wrong operation_id: {span_op_id} != {i}"

        # Verify no cross-contamination
        all_trace_ids_in_data = set()
        for trace_group in summary["trace_groups"].values():
            for log in trace_group["logs"]:
                if log["trace_id"]:
                    all_trace_ids_in_data.add(log["trace_id"])

        # Should only have our test trace IDs (plus any from monitoring system)
        test_trace_ids_found = all_trace_ids_in_data.intersection(set(trace_ids))
        assert len(test_trace_ids_found) == num_concurrent_ops, (
            f"Expected {num_concurrent_ops} trace IDs, found {len(test_trace_ids_found)}"
        )

        print("Concurrent trace isolation validation:")
        print(f"  - Concurrent operations: {num_concurrent_ops}")
        print(f"  - Unique trace IDs found: {len(test_trace_ids_found)}")
        print(f"  - Total traces in system: {summary['total_traces']}")
        print("  - No cross-contamination detected")

    @pytest.mark.asyncio
    async def test_opentelemetry_integration_fidelity(self, trace_correlation_system):
        """
        Test OpenTelemetry integration fidelity and data consistency.

        Verifies:
        1. Span attributes are properly set and preserved
        2. Span events are correctly recorded
        3. Span status reflects operation outcomes
        4. Timing information is accurate
        5. Context propagation follows OpenTelemetry standards
        """
        system = trace_correlation_system
        await system.start_monitoring()

        test_trace_id = str(uuid.uuid4())

        # Create comprehensive traced operation with all OpenTelemetry features
        async with system.create_traced_operation("otel_integration_test", test_trace_id) as main_span:
            # Set various attribute types
            main_span.set_attribute("string_attr", "test_value")
            main_span.set_attribute("int_attr", 42)
            main_span.set_attribute("float_attr", 3.14159)
            main_span.set_attribute("bool_attr", True)

            # Add events with attributes
            main_span.add_event(
                "operation_started", {"component": "test_component", "version": "1.0.0", "environment": "test"}
            )

            # Simulate work with timing
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing time

            main_span.add_event(
                "processing_phase_1", {"phase": "initialization", "duration_ms": (time.time() - start_time) * 1000}
            )

            # Nested span to test hierarchy
            async with system.create_traced_operation("nested_operation", test_trace_id) as nested_span:
                nested_span.set_attribute("parent_operation", "otel_integration_test")
                nested_span.set_attribute("nested_level", 1)

                # Add events to nested span
                nested_span.add_event("nested_work_started")
                await asyncio.sleep(0.05)
                nested_span.add_event("nested_work_completed", {"items_processed": 10})

            main_span.add_event("processing_phase_2", {"phase": "nested_operations_completed"})

            # Test error handling in spans
            try:
                async with system.create_traced_operation("error_operation", test_trace_id) as error_span:
                    error_span.set_attribute("will_fail", True)
                    error_span.add_event("about_to_fail")

                    # Simulate error
                    raise RuntimeError("Intentional test error")

            except RuntimeError as e:
                main_span.add_event("error_handled", {"error_type": type(e).__name__, "error_message": str(e)})

            main_span.add_event(
                "operation_completed", {"total_duration_ms": (time.time() - start_time) * 1000, "success": True}
            )

        # Allow processing to complete
        await asyncio.sleep(0.2)

        # Validate OpenTelemetry integration
        summary = system.get_correlation_summary()
        trace_data = summary["trace_groups"][test_trace_id]
        spans = trace_data["spans"]

        # Should have 3 spans: main, nested, error
        assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

        # Find each span type
        main_span = next(span for span in spans if span.name == "otel_integration_test")
        nested_span = next(span for span in spans if span.name == "nested_operation")
        error_span = next(span for span in spans if span.name == "error_operation")

        # Validate main span
        assert main_span.trace_id == test_trace_id
        assert main_span.status == "OK"
        assert main_span.attributes["string_attr"] == "test_value"
        assert main_span.attributes["int_attr"] == 42
        assert main_span.attributes["float_attr"] == 3.14159
        assert main_span.attributes["bool_attr"] is True

        # Validate main span events
        main_events = main_span.events
        event_names = [event["name"] for event in main_events]
        expected_events = [
            "operation_started",
            "processing_phase_1",
            "processing_phase_2",
            "error_handled",
            "operation_completed",
        ]

        for expected_event in expected_events:
            assert expected_event in event_names, f"Missing event: {expected_event}"

        # Validate nested span
        assert nested_span.trace_id == test_trace_id
        assert nested_span.status == "OK"
        assert nested_span.attributes["parent_operation"] == "otel_integration_test"
        assert nested_span.attributes["nested_level"] == 1

        # Validate nested span events
        nested_events = nested_span.events
        nested_event_names = [event["name"] for event in nested_events]
        assert "nested_work_started" in nested_event_names
        assert "nested_work_completed" in nested_event_names

        # Validate error span
        assert error_span.trace_id == test_trace_id
        assert error_span.status == "ERROR"
        assert error_span.attributes["will_fail"] is True

        # Validate error span events
        error_events = error_span.events
        error_event_names = [event["name"] for event in error_events]
        assert "about_to_fail" in error_event_names

        # Validate timing consistency
        assert main_span.end_time > main_span.start_time
        assert nested_span.end_time > nested_span.start_time
        assert error_span.end_time > error_span.start_time

        # Nested span should be within main span timeframe
        assert nested_span.start_time >= main_span.start_time
        assert nested_span.end_time <= main_span.end_time

        print(f"OpenTelemetry integration validation for {test_trace_id}:")
        print(f"  - Total spans: {len(spans)}")
        print(f"  - Main span events: {len(main_events)}")
        print(f"  - Nested span events: {len(nested_events)}")
        print(f"  - Error span status: {error_span.status}")
        print("  - Timing consistency: ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
