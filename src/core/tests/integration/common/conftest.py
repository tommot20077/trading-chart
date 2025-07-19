# ABOUTME: Test fixtures and configuration for common components integration tests
# ABOUTME: Provides shared testing utilities and mock components

import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
from typing import Dict, Any, Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter
from core.implementations.noop.common.rate_limiter import NoOpRateLimiter
from core.exceptions.base import (
    CoreException, ValidationException, BusinessLogicException,
    DataNotFoundException, ExternalServiceException, ConfigurationException,
    AuthenticationException, AuthorizationError, RateLimitExceededException
)


@pytest.fixture
def test_rate_limiter() -> InMemoryRateLimiter:
    """Fixture providing a configured InMemoryRateLimiter for testing."""
    return InMemoryRateLimiter(
        capacity=10,
        refill_rate=2.0,
        cleanup_interval=1.0,
        identifier_key="test_user"
    )


@pytest.fixture
def noop_rate_limiter() -> NoOpRateLimiter:
    """Fixture providing a NoOp rate limiter for testing."""
    return NoOpRateLimiter()


@pytest.fixture
def multi_tenant_rate_limiter() -> InMemoryRateLimiter:
    """Fixture providing a rate limiter configured for multi-tenant testing."""
    return InMemoryRateLimiter(
        capacity=5,
        refill_rate=1.0,
        cleanup_interval=0.5,
        identifier_key="default_tenant"
    )


@pytest.fixture
async def async_rate_limiter() -> AsyncGenerator[InMemoryRateLimiter, None]:
    """Async fixture providing a rate limiter with automatic cleanup."""
    limiter = InMemoryRateLimiter(
        capacity=20,
        refill_rate=5.0,
        cleanup_interval=2.0
    )
    try:
        yield limiter
    finally:
        await limiter.close()


@pytest.fixture
def mock_settings() -> Dict[str, Any]:
    """Fixture providing mock configuration settings."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        },
        "logging": {
            "level": "DEBUG",
            "format": "json",
            "output": ["console", "file"]
        },
        "rate_limiting": {
            "default_capacity": 100,
            "default_refill_rate": 10.0
        }
    }


@pytest.fixture
def test_config_file() -> Generator[str, None, None]:
    """Fixture providing a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"test_key": "test_value", "nested": {"key": "value"}}')
        config_path = f.name
    
    try:
        yield config_path
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


@pytest.fixture
def exception_simulator():
    """Fixture providing utilities to simulate various exceptions."""
    class ExceptionSimulator:
        @staticmethod
        def create_validation_error() -> ValidationException:
            return ValidationException(
                "Invalid input data",
                code="VALIDATION_001",
                details={"field": "price", "value": -10.5}
            )
        
        @staticmethod
        def create_business_logic_error() -> BusinessLogicException:
            return BusinessLogicException(
                "Insufficient balance",
                code="BUSINESS_001",
                details={"required": 1000, "available": 500}
            )
        
        @staticmethod
        def create_data_not_found_error() -> DataNotFoundException:
            return DataNotFoundException(
                "Trading pair not found",
                code="DATA_001",
                details={"symbol": "UNKNOWN/USD"}
            )
        
        @staticmethod
        def create_external_service_error() -> ExternalServiceException:
            return ExternalServiceException(
                "Exchange API error",
                code="EXTERNAL_001",
                details={"service": "binance", "status": 503}
            )
        
        @staticmethod
        def create_rate_limit_error() -> RateLimitExceededException:
            return RateLimitExceededException(
                "Rate limit exceeded",
                code="RATE_001",
                details={"limit": 100, "current": 150}
            )
    
    return ExceptionSimulator()


@pytest.fixture
def mock_middleware_context():
    """Fixture providing a mock middleware context."""
    class MockMiddlewareContext:
        def __init__(self):
            self.data = {}
            self.metadata = {}
            self.execution_time = 0.0
            self.success = True
            self.error = None
        
        def set_data(self, key: str, value: Any):
            self.data[key] = value
        
        def get_data(self, key: str, default: Any = None) -> Any:
            return self.data.get(key, default)
        
        def set_metadata(self, key: str, value: Any):
            self.metadata[key] = value
        
        def get_metadata(self, key: str, default: Any = None) -> Any:
            return self.metadata.get(key, default)
    
    return MockMiddlewareContext()


@pytest.fixture
def concurrent_executor():
    """Fixture providing utilities for concurrent testing."""
    class ConcurrentExecutor:
        @staticmethod
        async def run_concurrent_tasks(tasks, max_concurrent=10):
            """Run tasks concurrently with controlled concurrency."""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task
            
            return await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
        
        @staticmethod
        async def stress_test_rate_limiter(rate_limiter, num_requests=100, 
                                         concurrent_users=5, identifier_prefix="user"):
            """Stress test a rate limiter with concurrent requests."""
            async def make_requests(user_id: str, num_requests: int):
                results = []
                for _ in range(num_requests):
                    result = await rate_limiter.acquire_for_identifier(user_id)
                    results.append(result)
                    await asyncio.sleep(0.01)  # Small delay between requests
                return results
            
            tasks = [
                make_requests(f"{identifier_prefix}_{i}", num_requests)
                for i in range(concurrent_users)
            ]
            
            return await asyncio.gather(*tasks)
    
    return ConcurrentExecutor()


@pytest.fixture
def resource_monitor():
    """Fixture providing resource monitoring utilities."""
    class ResourceMonitor:
        def __init__(self):
            self.initial_memory = None
            self.initial_threads = None
        
        def start_monitoring(self):
            """Start monitoring system resources."""
            import psutil
            import threading
            
            process = psutil.Process()
            self.initial_memory = process.memory_info().rss
            self.initial_threads = threading.active_count()
        
        def get_memory_delta(self) -> int:
            """Get memory usage delta since monitoring started."""
            import psutil
            
            if self.initial_memory is None:
                return 0
            
            process = psutil.Process()
            current_memory = process.memory_info().rss
            return current_memory - self.initial_memory
        
        def get_thread_delta(self) -> int:
            """Get thread count delta since monitoring started."""
            import threading
            
            if self.initial_threads is None:
                return 0
            
            current_threads = threading.active_count()
            return current_threads - self.initial_threads
        
        def check_for_leaks(self, memory_threshold_mb=10, thread_threshold=5) -> Dict[str, bool]:
            """Check for potential resource leaks."""
            memory_delta_mb = self.get_memory_delta() / (1024 * 1024)
            thread_delta = self.get_thread_delta()
            
            return {
                "memory_leak": memory_delta_mb > memory_threshold_mb,
                "thread_leak": thread_delta > thread_threshold,
                "memory_delta_mb": memory_delta_mb,
                "thread_delta": thread_delta
            }
    
    return ResourceMonitor()


# Auto-cleanup fixture removed due to timeout issues
# Individual tests should handle their own cleanup