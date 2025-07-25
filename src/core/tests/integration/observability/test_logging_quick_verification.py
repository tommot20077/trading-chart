# ABOUTME: Quick verification tests for logging system core functionality
# ABOUTME: Simple tests to verify basic logging integration works without complex async operations

import tempfile
import json
import time
from pathlib import Path
from loguru import logger

import pytest

# File I/O tests restored - using direct loguru handler configuration to bypass configure_for_testing() limitations
# NOTE: These tests bypass the disable_loguru_for_tests fixture using pytest markers

from core.config.logging import LoggerConfig, setup_logging


# Skip the disable_loguru fixture for these specific tests
pytestmark = pytest.mark.enable_loguru_file_io


class TestLoggingQuickVerification:
    """Quick verification tests for logging system."""

    def test_basic_multi_output_logging(self):
        """Test basic multi-output logging functionality."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="quick_test_")
        temp_path = Path(temp_dir)

        try:
            # Store original handlers to restore later
            original_handlers = logger._core.handlers.copy()
            
            # Remove existing handlers first
            logger.remove()
            
            # Add file handlers directly to avoid testing environment interference
            log_file = temp_path / "test.log"
            structured_file = temp_path / "test.jsonl" 
            error_file = temp_path / "errors.log"
            
            # Ensure parent directories exist
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add handlers directly with explicit flush configuration
            handler_ids = []
            handler_ids.append(logger.add(log_file, level="DEBUG", enqueue=False, catch=False))
            handler_ids.append(logger.add(structured_file, level="DEBUG", serialize=True, enqueue=False, catch=False))
            handler_ids.append(logger.add(error_file, level="ERROR", enqueue=False, catch=False))

            # Create test logger
            test_logger = logger.bind(name="quick_test")

            # Log messages at different levels
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")

            # Ensure all writes are flushed
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Multiple attempts to ensure file completion
            for _ in range(3):
                logger.complete()
                time.sleep(0.05)

            # Verify regular file
            assert log_file.exists(), f"Regular log file should exist at {log_file}"
            file_content = log_file.read_text(encoding='utf-8')
            assert "Debug message" in file_content, f"Debug message not found in: {file_content}"
            assert "Info message" in file_content, f"Info message not found in: {file_content}"
            assert "Warning message" in file_content, f"Warning message not found in: {file_content}"
            assert "Error message" in file_content, f"Error message not found in: {file_content}"

            # Verify structured file  
            assert structured_file.exists(), f"Structured log file should exist at {structured_file}"
            with open(structured_file, "r", encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                assert len(lines) >= 4, f"Should have at least 4 log entries, got {len(lines)}"

                # Verify JSON format
                for i, line in enumerate(lines):
                    try:
                        entry = json.loads(line)
                        assert "record" in entry, f"Missing 'record' in entry {i}: {entry}"
                        assert "message" in entry["record"], f"Missing 'message' in record {i}: {entry['record']}"
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON in line {i}: {line} - Error: {e}")

            # Verify error file
            assert error_file.exists(), f"Error log file should exist at {error_file}"
            error_content = error_file.read_text(encoding='utf-8')
            assert "Error message" in error_content, f"Error message not found in: {error_content}"
            assert "Debug message" not in error_content, f"Debug message should not be in error file: {error_content}"

        finally:
            # Cleanup loguru handlers
            logger.remove()
            # Restore original handlers
            for handler_id, handler in original_handlers.items():
                try:
                    logger.add(handler.sink, **handler._kwargs)
                except Exception:
                    pass  # Ignore restoration errors
            
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    def test_basic_notification_handler_integration(self):
        """Test basic notification handler functionality."""
        from core.implementations.memory.observability.notification_handler import InMemoryNotificationHandler
        import asyncio

        async def run_test():
            # Create notification handler
            handler = InMemoryNotificationHandler(
                max_queue_size=100,
                max_history_size=500,
                simulate_failure_rate=0.0,
                processing_delay_seconds=0.0,
            )

            try:
                # Send test notification
                alert_data = {
                    "id": "test-001",
                    "rule_name": "test_rule",
                    "severity": "error",
                    "message": "Test error notification",
                    "metadata": {"test": True},
                }

                await handler.send_notification(alert_data)

                # Allow processing time
                await asyncio.sleep(0.2)

                # Verify notification was processed
                stats = await handler.get_notification_statistics()
                assert stats["total_sent"] >= 1, f"Should have sent at least one notification, got: {stats}"

                # Verify notification in history
                history = await handler.get_notification_history()
                assert len(history) >= 1, f"Should have notification in history, got {len(history)} entries"

                test_notification = None
                for notification in history:
                    if notification["alert_data"]["id"] == "test-001":
                        test_notification = notification
                        break

                assert test_notification is not None, f"Should find test notification in history: {[n['alert_data']['id'] for n in history]}"
                assert test_notification["alert_data"]["severity"] == "error", f"Expected severity 'error', got: {test_notification['alert_data']['severity']}"
                assert test_notification["status"] == "sent", f"Expected status 'sent', got: {test_notification['status']}"

            finally:
                await handler.close()

        # Run async test with proper event loop handling
        try:
            # Try to get current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use asyncio.create_task
                import pytest
                pytest.skip("Cannot run nested asyncio.run() in pytest-asyncio environment")
            else:
                asyncio.run(run_test())
        except RuntimeError:
            # No event loop, safe to use asyncio.run
            asyncio.run(run_test())

    def test_error_logging_with_structured_data(self):
        """Test error logging with structured data preservation."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="error_test_")
        temp_path = Path(temp_dir)

        try:
            # Store original handlers to restore later
            original_handlers = logger._core.handlers.copy()
            
            logger.remove()
            
            # Add structured logging handler directly  
            structured_file = temp_path / "structured.jsonl"
            structured_file.parent.mkdir(parents=True, exist_ok=True)
            
            handler_id = logger.add(structured_file, level="DEBUG", serialize=True, enqueue=False, catch=False)

            test_logger = logger.bind(name="error_test")

            # Log error with structured data
            error_data = {
                "error_type": "DatabaseError",
                "error_code": "DB_CONNECTION_FAILED",
                "user_id": "user_123",
                "retry_count": 3,
                "context": {"endpoint": "/api/users", "method": "GET"},
            }

            test_logger.error("Database connection failed", extra=error_data)

            # Ensure all writes are flushed
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Multiple attempts to ensure file completion
            for _ in range(3):
                logger.complete()
                time.sleep(0.05)

            # Verify structured log preservation
            assert structured_file.exists(), f"Structured file should exist at {structured_file}"

            with open(structured_file, "r", encoding='utf-8') as f:
                entries = [json.loads(line.strip()) for line in f if line.strip()]

            assert len(entries) >= 1, f"Should have log entry, got {len(entries)} entries"

            # Find our error entry
            error_entry = None
            for i, entry in enumerate(entries):
                try:
                    if "Database connection failed" in entry.get("record", {}).get("message", ""):
                        error_entry = entry
                        break
                except (KeyError, TypeError) as e:
                    pytest.fail(f"Invalid entry structure at index {i}: {entry} - Error: {e}")

            assert error_entry is not None, f"Should find error entry in {len(entries)} entries: {[e.get('record', {}).get('message', 'NO_MESSAGE') for e in entries]}"

            # Verify structured data preservation
            record = error_entry["record"]
            extra_data = record.get("extra", {}).get("extra", {})

            assert extra_data["error_type"] == "DatabaseError", f"Expected error_type 'DatabaseError', got: {extra_data.get('error_type')}"
            assert extra_data["error_code"] == "DB_CONNECTION_FAILED", f"Expected error_code 'DB_CONNECTION_FAILED', got: {extra_data.get('error_code')}"
            assert extra_data["user_id"] == "user_123", f"Expected user_id 'user_123', got: {extra_data.get('user_id')}"
            assert extra_data["retry_count"] == 3, f"Expected retry_count 3, got: {extra_data.get('retry_count')}"
            assert extra_data["context"]["endpoint"] == "/api/users", f"Expected endpoint '/api/users', got: {extra_data.get('context', {}).get('endpoint')}"
            assert extra_data["context"]["method"] == "GET", f"Expected method 'GET', got: {extra_data.get('context', {}).get('method')}"

        finally:
            logger.remove()
            # Restore original handlers
            for handler_id, handler in original_handlers.items():
                try:
                    logger.add(handler.sink, **handler._kwargs)
                except Exception:
                    pass  # Ignore restoration errors
            
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


@pytest.mark.integration
def test_logging_system_integration_basic():
    """Basic integration test to verify logging system works."""
    temp_dir = tempfile.mkdtemp(prefix="integration_basic_")
    temp_path = Path(temp_dir)

    try:
        # Store original handlers to restore later
        original_handlers = logger._core.handlers.copy()
        
        logger.remove()
        
        # Add handlers directly
        log_file = temp_path / "basic.log"
        structured_file = temp_path / "basic.jsonl"
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(log_file, level="INFO", enqueue=False, catch=False)
        logger.add(structured_file, level="INFO", serialize=True, enqueue=False, catch=False)

        test_logger = logger.bind(name="integration_basic")
        test_logger.info("Integration test message", extra={"test_id": "basic_001"})

        # Ensure all writes are flushed
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Multiple attempts to ensure file completion
        for _ in range(3):
            logger.complete()
            time.sleep(0.05)

        # Verify files exist and contain data
        assert log_file.exists(), f"Log file should exist at {log_file}"
        assert structured_file.exists(), f"Structured file should exist at {structured_file}"

        content = log_file.read_text(encoding='utf-8')
        assert "Integration test message" in content, f"Message not found in log content: {content}"

        # Verify structured file has valid JSON
        with open(structured_file, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) >= 1, f"Should have at least 1 log entry, got {len(lines)}"
            
            # Verify JSON format
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                    assert "record" in entry, f"Missing 'record' in entry {i}: {entry}"
                    if "Integration test message" in entry["record"].get("message", ""):
                        # Found our test message
                        extra_data = entry["record"].get("extra", {}).get("extra", {})
                        assert extra_data.get("test_id") == "basic_001", f"Expected test_id 'basic_001', got: {extra_data.get('test_id')}"
                        break
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in line {i}: {line} - Error: {e}")

    finally:
        logger.remove()
        # Restore original handlers
        for handler_id, handler in original_handlers.items():
            try:
                logger.add(handler.sink, **handler._kwargs)
            except Exception:
                pass  # Ignore restoration errors
        
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
