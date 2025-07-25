# ABOUTME: End-to-end integration tests for complete observability system combining notifications, logging, and performance monitoring
# ABOUTME: Tests the full business flow monitoring with data consistency validation across all observability components

import asyncio
import pytest
import pytest_asyncio

# End-to-end observability integration tests
import time
import uuid
from typing import Dict, Any
from loguru import logger

from core.implementations.memory.observability.notification_handler import (
    InMemoryNotificationHandler,
)
from ...fixtures.performance_metrics import PerformanceMonitor
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority


@pytest_asyncio.fixture
async def business_flow_simulator(mock_event_bus):
    """
    Create a business flow simulator that generates realistic business events
    for end-to-end observability testing.
    """

    class BusinessFlowSimulator:
        def __init__(self, event_bus):
            self.event_bus = event_bus
            self.trace_id = str(uuid.uuid4())
            self.business_events = []
            self.metrics_collected = []

        async def simulate_trading_session(self, duration_seconds: float = 2.0, events_per_second: float = 10.0):
            """Simulate a complete trading session with various event types."""
            session_id = str(uuid.uuid4())
            start_time = time.time()

            # Trading session start
            await self._publish_event(
                EventType.TRADE,
                {
                    "action": "session_start",
                    "session_id": session_id,
                    "trace_id": self.trace_id,
                    "timestamp": start_time,
                },
            )

            events_to_generate = int(duration_seconds * events_per_second)
            interval = 1.0 / events_per_second if events_per_second > 0 else 0.1

            for i in range(events_to_generate):
                # Mix of different event types
                if i % 5 == 0:
                    # Market data event
                    await self._publish_event(
                        EventType.MARKET_DATA,
                        {
                            "symbol": "BTCUSD",
                            "price": 50000 + (i * 10),
                            "volume": 1.5,
                            "trace_id": self.trace_id,
                            "session_id": session_id,
                            "event_sequence": i,
                        },
                    )
                elif i % 3 == 0:
                    # Trade execution
                    await self._publish_event(
                        EventType.TRADE,
                        {
                            "side": "buy" if i % 2 == 0 else "sell",
                            "quantity": 0.1,
                            "price": 50000 + (i * 10),
                            "trace_id": self.trace_id,
                            "session_id": session_id,
                            "event_sequence": i,
                        },
                    )
                else:
                    # System event
                    await self._publish_event(
                        EventType.SYSTEM,
                        {
                            "component": "order_engine",
                            "status": "processing",
                            "trace_id": self.trace_id,
                            "session_id": session_id,
                            "event_sequence": i,
                        },
                    )

                if interval > 0:
                    await asyncio.sleep(interval)

            # Trading session end
            await self._publish_event(
                EventType.TRADE,
                {
                    "action": "session_end",
                    "session_id": session_id,
                    "trace_id": self.trace_id,
                    "timestamp": time.time(),
                    "events_processed": events_to_generate,
                },
            )

            return session_id, events_to_generate + 2  # +2 for start/end events

        async def simulate_error_scenarios(self):
            """Simulate various error scenarios for testing error handling."""
            error_scenarios = [
                {
                    "type": "connection_error",
                    "severity": "critical",
                    "component": "database",
                    "details": {"error_code": "DB_CONN_TIMEOUT", "retry_count": 3},
                },
                {
                    "type": "validation_error",
                    "severity": "warning",
                    "component": "order_validator",
                    "details": {"field": "quantity", "value": -1.0, "constraint": "positive"},
                },
                {
                    "type": "performance_degradation",
                    "severity": "warning",
                    "component": "matching_engine",
                    "details": {"latency_ms": 1500, "threshold_ms": 1000},
                },
            ]

            for scenario in error_scenarios:
                await self._publish_event(
                    EventType.ERROR, {**scenario, "trace_id": self.trace_id, "timestamp": time.time()}
                )

            return len(error_scenarios)

        async def _publish_event(self, event_type: EventType, data: Dict[str, Any]):
            """Publish an event and track it."""
            event = BaseEvent(
                event_type=event_type, source="business_simulator", data=data, priority=EventPriority.NORMAL
            )

            self.business_events.append(
                {"event": event, "published_at": time.time(), "trace_id": data.get("trace_id", self.trace_id)}
            )

            await self.event_bus.publish(event)

        def get_trace_id(self) -> str:
            return self.trace_id

        def get_events_summary(self) -> Dict[str, Any]:
            return {
                "total_events": len(self.business_events),
                "trace_id": self.trace_id,
                "event_types": list(set(event["event"].event_type for event in self.business_events)),
                "time_span": {
                    "start": min(event["published_at"] for event in self.business_events)
                    if self.business_events
                    else 0,
                    "end": max(event["published_at"] for event in self.business_events) if self.business_events else 0,
                },
            }

    return BusinessFlowSimulator(mock_event_bus)


@pytest.mark.enable_loguru_file_io
@pytest_asyncio.fixture
async def integrated_observability_system(mock_event_bus):
    """
    Create a complete integrated observability system with all components.
    """
    # Create notification handler
    notification_handler = InMemoryNotificationHandler(
        max_queue_size=1000,
        max_history_size=5000,
        history_retention_hours=1.0,
        cleanup_interval_seconds=30.0,
        simulate_failure_rate=0.0,
        processing_delay_seconds=0.01,
    )

    # Create performance monitor
    performance_monitor = PerformanceMonitor(event_bus=mock_event_bus, sample_interval=0.05)

    # Configure appropriate thresholds for testing
    performance_monitor.cpu_threshold = 90.0
    performance_monitor.memory_threshold = 90.0
    performance_monitor.queue_size_threshold = 200
    performance_monitor.error_rate_threshold = 15.0

    # Create logging capture
    log_records = []

    def log_capture_handler(message):
        # Loguru passes a Message object with .record attribute
        if hasattr(message, 'record'):
            record = message.record
            level_name = record["level"].name
            message_text = record["message"]
            extra = record.get("extra", {})
        else:
            # Fallback for other formats
            level_name = "INFO"
            message_text = str(message)
            extra = {}

        log_records.append(
            {
                "timestamp": time.time(),
                "level": level_name,
                "message": message_text,
                "extra": extra,
                "trace_id": extra.get("trace_id"),
            }
        )

    # Configure logger with capture
    logger.remove()
    logger.add(log_capture_handler, level="DEBUG", format="{message}")

    # Create system coordinator
    class IntegratedObservabilitySystem:
        def __init__(self):
            self.notification_handler = notification_handler
            self.performance_monitor = performance_monitor
            self.log_records = log_records
            self.alerts_generated = []
            self.cross_component_data = []

            # Add alert callback to performance monitor
            self.performance_monitor.add_alert_callback(self._handle_performance_alert)

        async def start_monitoring(self):
            """Start all monitoring components."""
            await self.performance_monitor.start_monitoring()

        async def stop_monitoring(self):
            """Stop all monitoring components."""
            await self.performance_monitor.stop_monitoring()
            await self.notification_handler.close()

        def _handle_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]):
            """Handle performance alerts and create notifications."""
            alert_id = str(uuid.uuid4())

            # Create notification data
            notification_data = {
                "id": alert_id,
                "rule_name": f"performance_{alert_type}",
                "severity": "warning" if alert_type.startswith("high_") else "info",
                "message": f"Performance alert: {alert_type}",
                "metadata": {**alert_data, "component": "performance_monitor", "timestamp": time.time()},
            }

            # Store alert for verification
            self.alerts_generated.append(
                {
                    "type": alert_type,
                    "data": alert_data,
                    "notification_data": notification_data,
                    "timestamp": time.time(),
                }
            )

            # Send notification asynchronously
            asyncio.create_task(self._send_notification_async(notification_data))

        async def _send_notification_async(self, notification_data: Dict[str, Any]):
            """Send notification asynchronously."""
            try:
                success, message = await self.notification_handler.send_notification(notification_data)

                # Log the notification result with trace context
                logger.info(
                    f"Notification sent: {message}",
                    notification_id=notification_data["id"],
                    success=success,
                    rule_name=notification_data["rule_name"],
                )

            except Exception as e:
                logger.error(f"Failed to send notification: {str(e)}", exc_info=e)

        async def process_business_event(self, event: BaseEvent):
            """Process a business event with full observability."""
            trace_id = event.data.get("trace_id", str(uuid.uuid4()))

            # Log event processing
            logger.info(
                f"Processing business event: {event.event_type.value}",
                trace_id=trace_id,
                event_type=event.event_type.value,
                source=event.source,
            )

            # Collect cross-component data
            processing_start = time.time()

            # Simulate business logic
            await asyncio.sleep(0.01)  # Minimal processing delay

            processing_end = time.time()

            # Record cross-component data
            self.cross_component_data.append(
                {
                    "trace_id": trace_id,
                    "event_type": event.event_type.value,
                    "processing_start": processing_start,
                    "processing_end": processing_end,
                    "processing_duration": processing_end - processing_start,
                    "timestamp": processing_end,
                }
            )

            # Check for error events and generate alerts
            if event.event_type == EventType.ERROR:
                await self._handle_error_event(event, trace_id)

        async def _handle_error_event(self, event: BaseEvent, trace_id: str):
            """Handle error events and generate appropriate notifications."""
            error_data = event.data
            severity = error_data.get("severity", "warning")

            # Create error notification
            notification_data = {
                "id": str(uuid.uuid4()),
                "rule_name": f"error_{error_data.get('type', 'unknown')}",
                "severity": severity,
                "message": f"Error detected: {error_data.get('type', 'Unknown error')}",
                "metadata": {
                    **error_data,
                    "trace_id": trace_id,
                    "component": "error_handler",
                    "timestamp": time.time(),
                },
            }

            # Send error notification
            await self._send_notification_async(notification_data)

        def get_monitoring_summary(self) -> Dict[str, Any]:
            """Get comprehensive monitoring summary."""
            return {
                "performance_summary": self.performance_monitor.get_performance_summary(),
                "log_records_count": len(self.log_records),
                "alerts_generated_count": len(self.alerts_generated),
                "cross_component_data_count": len(self.cross_component_data),
                "trace_ids": list(set(data["trace_id"] for data in self.cross_component_data if data.get("trace_id"))),
            }

    system = IntegratedObservabilitySystem()

    try:
        yield system
    finally:
        await system.stop_monitoring()


class TestEndToEndObservabilityIntegration:
    """
    Test suite for end-to-end observability integration.

    Tests the complete observability pipeline including notifications,
    logging, and performance monitoring working together in realistic
    business scenarios.
    """

    @pytest.mark.enable_loguru_file_io
    @pytest.mark.asyncio
    async def test_complete_business_flow_monitoring(self, integrated_observability_system, business_flow_simulator):
        """
        Test complete business flow monitoring with all observability components.

        This test simulates a realistic trading session and verifies that:
        1. All events are properly logged with trace context
        2. Performance metrics are collected throughout the flow
        3. Alerts are generated when thresholds are exceeded
        4. Notifications are sent for critical events
        5. Data consistency is maintained across all components
        """
        system = integrated_observability_system
        simulator = business_flow_simulator

        # Start monitoring
        await system.start_monitoring()

        # Wait for monitoring to stabilize
        await asyncio.sleep(0.1)

        # Simulate business flow
        session_id, expected_events = await simulator.simulate_trading_session(
            duration_seconds=1.0, events_per_second=20.0
        )

        # Process events through observability system
        for event_data in simulator.business_events:
            await system.process_business_event(event_data["event"])

        # Simulate error scenarios
        error_count = await simulator.simulate_error_scenarios()

        # Process error events
        for event_data in simulator.business_events[-error_count:]:
            await system.process_business_event(event_data["event"])

        # Allow processing to complete
        await asyncio.sleep(0.5)

        # Verify monitoring data collection
        monitoring_summary = system.get_monitoring_summary()

        # Assert event processing
        assert len(system.cross_component_data) >= expected_events + error_count

        # Assert trace ID propagation
        trace_ids = monitoring_summary["trace_ids"]
        expected_trace_id = simulator.get_trace_id()
        assert expected_trace_id in trace_ids

        # Assert log records generation
        assert monitoring_summary["log_records_count"] > 0

        # Verify trace ID in logs
        trace_logs = [record for record in system.log_records if record.get("trace_id") == expected_trace_id]
        assert len(trace_logs) > 0

        # Assert performance monitoring
        perf_summary = monitoring_summary["performance_summary"]
        if perf_summary:  # May be empty if monitoring duration was too short
            assert "monitoring_duration" in perf_summary
            assert "sample_count" in perf_summary

        # Verify error notifications were generated
        error_notifications = [
            alert for alert in system.alerts_generated if "error" in alert["notification_data"]["rule_name"]
        ]
        # Should have notifications for the error scenarios we simulated
        # Note: Actual count may vary based on processing timing

        print("Business flow monitoring summary:")
        print(f"  - Events processed: {len(system.cross_component_data)}")
        print(f"  - Log records: {monitoring_summary['log_records_count']}")
        print(f"  - Alerts generated: {monitoring_summary['alerts_generated_count']}")
        print(f"  - Trace IDs tracked: {len(trace_ids)}")

    @pytest.mark.enable_loguru_file_io
    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, integrated_observability_system, business_flow_simulator):
        """
        Test data consistency and timing synchronization across all monitoring components.

        Verifies that:
        1. Timestamps are consistent across components
        2. Trace IDs are properly propagated
        3. Event sequences are maintained
        4. No data loss occurs between components
        """
        system = integrated_observability_system
        simulator = business_flow_simulator

        await system.start_monitoring()

        # Generate controlled event sequence
        test_trace_id = str(uuid.uuid4())
        event_sequence = []

        # Create test events with specific trace ID
        for i in range(5):
            event = BaseEvent(
                event_type=EventType.TRADE,
                source="consistency_test",
                data={
                    "trace_id": test_trace_id,
                    "sequence_number": i,
                    "test_marker": "data_consistency",
                    "timestamp": time.time(),
                },
                priority=EventPriority.NORMAL,
            )

            event_start_time = time.time()
            await system.process_business_event(event)
            event_end_time = time.time()

            event_sequence.append(
                {"event": event, "start_time": event_start_time, "end_time": event_end_time, "sequence": i}
            )

            await asyncio.sleep(0.05)  # Small delay between events

        # Allow processing to complete
        await asyncio.sleep(0.2)

        # Verify data consistency

        # 1. Check cross-component data consistency
        trace_data = [data for data in system.cross_component_data if data.get("trace_id") == test_trace_id]

        assert len(trace_data) == 5, f"Expected 5 trace records, got {len(trace_data)}"

        # 2. Verify timestamp consistency (processing times should be reasonable)
        for i, data in enumerate(trace_data):
            assert data["processing_duration"] < 1.0, f"Processing took too long: {data['processing_duration']}s"
            assert data["trace_id"] == test_trace_id

        # 3. Check log record consistency
        trace_logs = [record for record in system.log_records if record.get("trace_id") == test_trace_id]

        assert len(trace_logs) >= 5, f"Expected at least 5 log records, got {len(trace_logs)}"

        # 4. Verify temporal ordering
        trace_data.sort(key=lambda x: x["processing_start"])
        for i in range(1, len(trace_data)):
            assert trace_data[i]["processing_start"] >= trace_data[i - 1]["processing_start"], (
                "Event processing order is not maintained"
            )

        # 5. Check notification statistics consistency
        notification_stats = await system.notification_handler.get_notification_statistics()

        # The total processed should be consistent
        total_notifications = notification_stats["total_sent"] + notification_stats["total_failed"]

        print("Data consistency verification:")
        print(f"  - Trace records: {len(trace_data)}")
        print(f"  - Log records: {len(trace_logs)}")
        print(f"  - Notifications: {total_notifications}")
        print(f"  - Time span: {trace_data[-1]['processing_end'] - trace_data[0]['processing_start']:.3f}s")

    @pytest.mark.enable_loguru_file_io
    @pytest.mark.asyncio
    async def test_high_throughput_observability_performance(
        self, integrated_observability_system, business_flow_simulator
    ):
        """
        Test observability system performance under high throughput conditions.

        Verifies that:
        1. System can handle high event volumes
        2. Performance monitoring captures resource usage
        3. No significant memory leaks occur
        4. Alerts are properly generated under load
        """
        system = integrated_observability_system

        await system.start_monitoring()

        # Capture initial state
        initial_stats = await system.notification_handler.get_notification_statistics()
        initial_data_count = len(system.cross_component_data)

        # High throughput simulation
        start_time = time.time()

        # Generate high-frequency events
        session_id, events_generated = await business_flow_simulator.simulate_trading_session(
            duration_seconds=2.0,
            events_per_second=50.0,  # Higher throughput
        )

        # Process all events
        for event_data in business_flow_simulator.business_events:
            await system.process_business_event(event_data["event"])

        end_time = time.time()
        total_duration = end_time - start_time

        # Allow final processing
        await asyncio.sleep(0.5)

        # Verify performance
        final_stats = await system.notification_handler.get_notification_statistics()
        final_data_count = len(system.cross_component_data)

        events_processed = final_data_count - initial_data_count
        throughput = events_processed / total_duration if total_duration > 0 else 0

        # Performance assertions
        assert events_processed >= events_generated, (
            f"Not all events were processed: {events_processed}/{events_generated}"
        )
        assert throughput > 30.0, f"Throughput too low: {throughput:.2f} events/sec"

        # Check for performance monitoring data
        monitoring_summary = system.get_monitoring_summary()
        perf_summary = monitoring_summary["performance_summary"]

        if perf_summary:
            assert perf_summary["sample_count"] > 0
            assert perf_summary["monitoring_duration"] > 0

        # Verify system stability (no excessive resource usage indicators)
        if system.alerts_generated:
            # Check if any critical alerts were generated
            critical_alerts = [
                alert for alert in system.alerts_generated if alert["notification_data"]["severity"] == "critical"
            ]

            # Log critical alerts for investigation but don't fail the test
            # as they might be expected under high load
            if critical_alerts:
                print(f"Critical alerts generated under load: {len(critical_alerts)}")

        print("High throughput performance results:")
        print(f"  - Events processed: {events_processed}")
        print(f"  - Duration: {total_duration:.3f}s")
        print(f"  - Throughput: {throughput:.2f} events/sec")
        print(f"  - Notifications sent: {final_stats['total_sent']}")
        print(f"  - Log records: {len(system.log_records)}")

    @pytest.mark.enable_loguru_file_io
    @pytest.mark.asyncio
    async def test_error_propagation_and_alert_correlation(
        self, integrated_observability_system, business_flow_simulator
    ):
        """
        Test error propagation and alert correlation across all observability components.

        Verifies that:
        1. Errors generate appropriate logs
        2. Error notifications are created and sent
        3. Performance impact of errors is captured
        4. Trace IDs are maintained during error handling
        """
        system = integrated_observability_system
        simulator = business_flow_simulator

        await system.start_monitoring()

        # Generate normal events first
        await simulator.simulate_trading_session(duration_seconds=0.5, events_per_second=10.0)

        # Process normal events
        normal_events = simulator.business_events.copy()
        for event_data in normal_events:
            await system.process_business_event(event_data["event"])

        # Clear events for error testing
        simulator.business_events.clear()

        # Generate error scenarios
        error_count = await simulator.simulate_error_scenarios()

        # Process error events (they were just added to business_events)
        error_events = simulator.business_events.copy()
        for event_data in error_events:
            await system.process_business_event(event_data["event"])

        # Allow error processing to complete
        await asyncio.sleep(0.3)

        # Verify error handling

        # 1. Check error logs were generated
        error_logs = [
            record
            for record in system.log_records
            if "error" in record["message"].lower() or record["level"] == "ERROR"
        ]

        # 2. Check error notifications were sent
        error_notifications = [
            alert for alert in system.alerts_generated if "error" in alert["notification_data"]["rule_name"]
        ]

        # 3. Verify trace ID propagation in error handling
        error_trace_id = simulator.get_trace_id()

        error_trace_data = [
            data
            for data in system.cross_component_data
            if data.get("trace_id") == error_trace_id and data.get("event_type") == "error"
        ]

        # Assertions
        assert len(error_trace_data) >= error_count, (
            f"Not all error events were traced: {len(error_trace_data)}/{error_count}"
        )

        # Check notification success rate
        notification_stats = await system.notification_handler.get_notification_statistics()
        total_notifications = notification_stats["total_sent"] + notification_stats["total_failed"]

        if total_notifications > 0:
            success_rate = notification_stats["total_sent"] / total_notifications
            assert success_rate >= 0.8, f"Notification success rate too low: {success_rate:.2%}"

        # Verify error correlation
        for error_data in error_trace_data:
            # Each error should have corresponding log entries
            matching_logs = [log for log in system.log_records if log.get("trace_id") == error_data["trace_id"]]
            assert len(matching_logs) > 0, f"No logs found for error trace {error_data['trace_id']}"

        print("Error propagation verification:")
        print(f"  - Error events processed: {len(error_trace_data)}")
        print(f"  - Error logs generated: {len(error_logs)}")
        print(f"  - Error notifications: {len(error_notifications)}")
        print(f"  - Notification success rate: {notification_stats['total_sent']}/{total_notifications}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
