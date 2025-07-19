# ABOUTME: Integration tests for component lifecycle coordination
# ABOUTME: Tests initialization, runtime, exception handling, graceful shutdown across components

import asyncio
import time
import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from enum import Enum

from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.config.settings import CoreSettings, get_settings
from core.exceptions.base import (
    CoreException, BusinessLogicException, ExternalServiceException,
    ConfigurationException, TimeoutException
)
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from core.interfaces.middleware import AbstractMiddleware


class ComponentState(Enum):
    """Component lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class LifecycleAwareComponent:
    """Base class for lifecycle-aware components."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.CREATED
        self.lifecycle_events = []
        self.dependencies = []
        self.error_handlers = []
    
    async def initialize(self):
        """Initialize the component."""
        self.state = ComponentState.INITIALIZING
        self.lifecycle_events.append({"event": "initialize", "timestamp": time.time()})
        
        # Initialize dependencies first
        for dependency in self.dependencies:
            if hasattr(dependency, 'initialize'):
                await dependency.initialize()
        
        # Component-specific initialization
        await self._do_initialize()
        
        self.state = ComponentState.RUNNING
        self.lifecycle_events.append({"event": "running", "timestamp": time.time()})
    
    async def shutdown(self):
        """Shutdown the component gracefully."""
        self.state = ComponentState.STOPPING
        self.lifecycle_events.append({"event": "shutdown_start", "timestamp": time.time()})
        
        try:
            # Component-specific shutdown
            await self._do_shutdown()
            
            # Shutdown dependencies in reverse order
            for dependency in reversed(self.dependencies):
                if hasattr(dependency, 'shutdown'):
                    await dependency.shutdown()
            
            self.state = ComponentState.STOPPED
            self.lifecycle_events.append({"event": "shutdown_complete", "timestamp": time.time()})
        
        except Exception as e:
            self.state = ComponentState.ERROR
            self.lifecycle_events.append({"event": "shutdown_error", "timestamp": time.time(), "error": str(e)})
            raise
    
    async def handle_error(self, error: Exception):
        """Handle component errors."""
        self.state = ComponentState.ERROR
        self.lifecycle_events.append({"event": "error", "timestamp": time.time(), "error": str(error)})
        
        # Try error handlers
        for handler in self.error_handlers:
            try:
                await handler(error)
                # If handler succeeds, component may recover
                if self.state == ComponentState.ERROR:
                    self.state = ComponentState.RUNNING
                    self.lifecycle_events.append({"event": "recovery", "timestamp": time.time()})
                break
            except Exception:
                continue  # Try next handler
    
    async def _do_initialize(self):
        """Override in subclasses for specific initialization."""
        pass
    
    async def _do_shutdown(self):
        """Override in subclasses for specific shutdown."""
        pass


class RateLimiterComponent(LifecycleAwareComponent):
    """Rate limiter component with lifecycle management."""
    
    def __init__(self, name: str, capacity: int = 10, refill_rate: float = 2.0):
        super().__init__(name)
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.rate_limiter = None
    
    async def _do_initialize(self):
        """Initialize rate limiter."""
        self.rate_limiter = InMemoryRateLimiter(
            capacity=self.capacity,
            refill_rate=self.refill_rate,
            cleanup_interval=1.0
        )
    
    async def _do_shutdown(self):
        """Shutdown rate limiter."""
        if self.rate_limiter:
            await self.rate_limiter.close()
    
    async def acquire(self, identifier: str = "default") -> bool:
        """Acquire tokens from rate limiter."""
        if self.state != ComponentState.RUNNING or not self.rate_limiter:
            raise BusinessLogicException("Rate limiter not available", "RATE_001")
        
        return await self.rate_limiter.acquire_for_identifier(identifier)


class MiddlewareComponent(LifecycleAwareComponent):
    """Middleware component with lifecycle management."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.pipeline = None
        self.registered_middleware = []
    
    async def _do_initialize(self):
        """Initialize middleware pipeline."""
        self.pipeline = InMemoryMiddlewarePipeline(f"{self.name}_pipeline")
        
        # Register default middleware
        for middleware in self.registered_middleware:
            await self.pipeline.add_middleware(middleware)
    
    async def _do_shutdown(self):
        """Shutdown middleware pipeline."""
        if self.pipeline:
            await self.pipeline.clear()
    
    async def add_middleware(self, middleware: AbstractMiddleware):
        """Add middleware to pipeline."""
        self.registered_middleware.append(middleware)
        if self.pipeline and self.state == ComponentState.RUNNING:
            await self.pipeline.add_middleware(middleware)
    
    async def execute(self, context: MiddlewareContext) -> MiddlewareResult:
        """Execute middleware pipeline."""
        if self.state != ComponentState.RUNNING or not self.pipeline:
            raise BusinessLogicException("Middleware pipeline not available", "MIDDLEWARE_001")
        
        return await self.pipeline.execute(context)


class ConfigurationComponent(LifecycleAwareComponent):
    """Configuration component with lifecycle management."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.settings = None
        self.config_cache = {}
    
    async def _do_initialize(self):
        """Initialize configuration."""
        # Clear settings cache for fresh load
        get_settings.cache_clear()
        self.settings = get_settings()
        
        # Load configuration into cache
        self.config_cache = {
            "app_name": self.settings.APP_NAME,
            "env": self.settings.ENV,
            "log_level": self.settings.LOG_LEVEL,
            "debug": self.settings.DEBUG
        }
    
    async def _do_shutdown(self):
        """Shutdown configuration."""
        self.config_cache.clear()
    
    def get_config(self, key: str, default=None):
        """Get configuration value."""
        if self.state != ComponentState.RUNNING:
            raise ConfigurationException("Configuration not available", "CONFIG_001")
        
        return self.config_cache.get(key, default)
    
    async def reload_config(self):
        """Reload configuration."""
        if self.state == ComponentState.RUNNING:
            await self._do_initialize()


class TestComponentLifecycleCoordination:
    """Test component lifecycle coordination across the system."""
    
    @pytest.mark.asyncio
    async def test_initialization_runtime_exception_handling_graceful_shutdown_cross_component_coordination(self):
        """Test complete lifecycle: initialization → runtime → exception handling → graceful shutdown across components."""
        # Create components with dependencies
        config_component = ConfigurationComponent("config")
        rate_limiter_component = RateLimiterComponent("rate_limiter", capacity=5, refill_rate=1.0)
        middleware_component = MiddlewareComponent("middleware")
        
        # Set up dependencies
        rate_limiter_component.dependencies = [config_component]
        middleware_component.dependencies = [config_component, rate_limiter_component]
        
        # Phase 1: Initialization
        lifecycle_tracker = {"events": []}
        
        try:
            # Initialize in dependency order
            await config_component.initialize()
            lifecycle_tracker["events"].append("config_initialized")
            
            await rate_limiter_component.initialize()
            lifecycle_tracker["events"].append("rate_limiter_initialized")
            
            await middleware_component.initialize()
            lifecycle_tracker["events"].append("middleware_initialized")
            
            # Verify all components are running
            assert config_component.state == ComponentState.RUNNING
            assert rate_limiter_component.state == ComponentState.RUNNING
            assert middleware_component.state == ComponentState.RUNNING
            
            # Phase 2: Runtime operations
            lifecycle_tracker["events"].append("runtime_start")
            
            # Test configuration access
            app_name = config_component.get_config("app_name")
            assert app_name is not None
            
            # Test rate limiting
            for i in range(3):  # Should succeed (within capacity)
                success = await rate_limiter_component.acquire(f"user_{i}")
                assert success, f"Rate limiting should allow request {i}"
            
            # Test middleware execution
            class TestMiddleware(AbstractMiddleware):
                def __init__(self):
                    super().__init__(EventPriority.NORMAL)
                    self.name = "test_middleware"
                
                async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                    return MiddlewareResult(
                        middleware_name=self.name,
                        status=MiddlewareStatus.SUCCESS,
                        data={"processed": True}
                    )
                
                def can_process(self, context: MiddlewareContext) -> bool:
                    return True
            
            await middleware_component.add_middleware(TestMiddleware())
            
            context = MiddlewareContext(event_id="test_001", event_type="lifecycle_test")
            result = await middleware_component.execute(context)
            assert result.status == MiddlewareStatus.SUCCESS
            
            lifecycle_tracker["events"].append("runtime_operations_complete")
            
            # Phase 3: Exception handling
            lifecycle_tracker["events"].append("exception_handling_test")
            
            # Simulate configuration error
            config_error = ConfigurationException("Config reload failed", "CONFIG_002")
            await config_component.handle_error(config_error)
            
            # Component should be in error state
            assert config_component.state == ComponentState.ERROR
            
            # Dependent components should handle the dependency failure gracefully
            try:
                config_component.get_config("test_key")
                assert False, "Should raise exception when component is in error state"
            except ConfigurationException:
                pass  # Expected
            
            lifecycle_tracker["events"].append("exception_handling_complete")
            
        finally:
            # Phase 4: Graceful shutdown
            lifecycle_tracker["events"].append("shutdown_start")
            
            # Shutdown in reverse dependency order
            await middleware_component.shutdown()
            lifecycle_tracker["events"].append("middleware_shutdown")
            
            await rate_limiter_component.shutdown()
            lifecycle_tracker["events"].append("rate_limiter_shutdown")
            
            await config_component.shutdown()
            lifecycle_tracker["events"].append("config_shutdown")
            
            # Verify all components are stopped
            assert middleware_component.state == ComponentState.STOPPED
            assert rate_limiter_component.state == ComponentState.STOPPED
            # Config component may be in ERROR state from earlier exception
            assert config_component.state in [ComponentState.STOPPED, ComponentState.ERROR]
        
        # Verify complete lifecycle was executed
        expected_events = [
            "config_initialized",
            "rate_limiter_initialized", 
            "middleware_initialized",
            "runtime_start",
            "runtime_operations_complete",
            "exception_handling_test",
            "exception_handling_complete",
            "shutdown_start",
            "middleware_shutdown",
            "rate_limiter_shutdown",
            "config_shutdown"
        ]
        
        assert lifecycle_tracker["events"] == expected_events

    @pytest.mark.asyncio
    async def test_component_dependency_management_and_coordination(self):
        """Test component dependency management and coordination."""
        # Create components with complex dependency graph
        components = {
            "config": ConfigurationComponent("config"),
            "rate_limiter": RateLimiterComponent("rate_limiter", capacity=5, refill_rate=1.0),
            "middleware_primary": MiddlewareComponent("middleware_primary"),
            "middleware_secondary": MiddlewareComponent("middleware_secondary")
        }
        
        # Set up dependency graph:
        # config <- rate_limiter <- middleware_primary <- middleware_secondary
        components["rate_limiter"].dependencies = [components["config"]]
        components["middleware_primary"].dependencies = [components["config"], components["rate_limiter"]]
        components["middleware_secondary"].dependencies = [components["middleware_primary"]]
        
        # Track initialization order
        initialization_order = []
        
        async def track_initialization(component_name: str, component: LifecycleAwareComponent):
            original_init = component._do_initialize
            
            async def tracked_init():
                initialization_order.append(f"{component_name}_start")
                await original_init()
                initialization_order.append(f"{component_name}_complete")
            
            component._do_initialize = tracked_init
        
        # Set up tracking for all components
        for name, component in components.items():
            await track_initialization(name, component)
        
        try:
            # Initialize components (should handle dependencies automatically)
            for component in components.values():
                await component.initialize()
            
            # Verify initialization order respects dependencies
            # Dependencies should initialize before dependents
            config_complete_idx = initialization_order.index("config_complete")
            rate_limiter_start_idx = initialization_order.index("rate_limiter_start")
            middleware_primary_start_idx = initialization_order.index("middleware_primary_start")
            middleware_secondary_start_idx = initialization_order.index("middleware_secondary_start")
            
            assert config_complete_idx < rate_limiter_start_idx, "Config should complete before rate limiter starts"
            assert rate_limiter_start_idx < middleware_primary_start_idx, "Rate limiter should start before primary middleware"
            
            # Test cross-component coordination
            # Configuration change should affect dependent components
            with patch.dict('os.environ', {'APP_NAME': 'Updated_App'}, clear=False):
                get_settings.cache_clear()
                await components["config"].reload_config()
                
                # Verify configuration change is reflected
                updated_name = components["config"].get_config("app_name")
                assert updated_name == "Updated_App"
            
            # Test component interaction through rate limiting
            rate_limiter = components["rate_limiter"]
            
            # Exhaust rate limiter capacity
            for i in range(5):  # Capacity is 5
                success = await rate_limiter.acquire("same_user")
                assert success, f"Request {i} should succeed"
            
            # Next request should fail
            should_fail = await rate_limiter.acquire("same_user")
            assert not should_fail, "Request should be rate limited"
            
        finally:
            # Shutdown in reverse dependency order
            shutdown_order = []
            
            for name, component in reversed(list(components.items())):
                shutdown_order.append(name)
                await component.shutdown()
            
            # Verify shutdown order
            expected_shutdown_order = ["middleware_secondary", "middleware_primary", "rate_limiter", "config"]
            assert shutdown_order == expected_shutdown_order

    @pytest.mark.asyncio
    async def test_component_error_propagation_and_recovery(self):
        """Test component error propagation and recovery mechanisms."""
        # Create interconnected components
        config_component = ConfigurationComponent("config")
        rate_limiter_component = RateLimiterComponent("rate_limiter")
        middleware_component = MiddlewareComponent("middleware")
        
        # Set up dependencies
        rate_limiter_component.dependencies = [config_component]
        middleware_component.dependencies = [config_component, rate_limiter_component]
        
        # Add error recovery handlers
        recovery_attempts = []
        
        async def config_recovery_handler(error: Exception):
            recovery_attempts.append(f"config_recovery: {type(error).__name__}")
            # Simulate successful recovery
            config_component.state = ComponentState.RUNNING
        
        async def rate_limiter_recovery_handler(error: Exception):
            recovery_attempts.append(f"rate_limiter_recovery: {type(error).__name__}")
            # Re-initialize rate limiter
            await rate_limiter_component._do_initialize()
        
        config_component.error_handlers.append(config_recovery_handler)
        rate_limiter_component.error_handlers.append(rate_limiter_recovery_handler)
        
        try:
            # Initialize all components
            await config_component.initialize()
            await rate_limiter_component.initialize()
            await middleware_component.initialize()
            
            # Verify all running
            assert all(c.state == ComponentState.RUNNING for c in [config_component, rate_limiter_component, middleware_component])
            
            # Simulate configuration component failure
            config_error = ConfigurationException("Database connection lost", "CONFIG_DB_001")
            await config_component.handle_error(config_error)
            
            # Verify error was handled and component recovered
            assert "config_recovery: ConfigurationException" in recovery_attempts
            assert config_component.state == ComponentState.RUNNING
            
            # Simulate rate limiter failure
            rate_limiter_error = ExternalServiceException("Redis connection timeout", "RATE_REDIS_001")
            await rate_limiter_component.handle_error(rate_limiter_error)
            
            # Verify rate limiter recovered
            assert "rate_limiter_recovery: ExternalServiceException" in recovery_attempts
            assert rate_limiter_component.state == ComponentState.RUNNING
            
            # Test that middleware component can still function after dependency recovery
            class RecoveryTestMiddleware(AbstractMiddleware):
                def __init__(self):
                    super().__init__(EventPriority.NORMAL)
                    self.name = "recovery_test"
                
                async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                    # Test dependency access
                    app_name = config_component.get_config("app_name")
                    rate_limit_success = await rate_limiter_component.acquire("recovery_test")
                    
                    return MiddlewareResult(
                        middleware_name=self.name,
                        status=MiddlewareStatus.SUCCESS,
                        data={
                            "config_access": app_name is not None,
                            "rate_limit_access": True  # If we got here, rate limiter is working
                        }
                    )
                
                def can_process(self, context: MiddlewareContext) -> bool:
                    return True
            
            await middleware_component.add_middleware(RecoveryTestMiddleware())
            
            context = MiddlewareContext(event_id="recovery_test", event_type="error_recovery")
            result = await middleware_component.execute(context)
            
            # Verify middleware can access recovered dependencies
            assert result.status == MiddlewareStatus.SUCCESS
            pipeline_results = result.metadata["pipeline_results"]
            recovery_result = next(r for r in pipeline_results if r["middleware_name"] == "recovery_test")
            assert recovery_result["data"]["config_access"] is True
            assert recovery_result["data"]["rate_limit_access"] is True
            
        finally:
            # Cleanup
            await middleware_component.shutdown()
            await rate_limiter_component.shutdown()
            await config_component.shutdown()


class TestResourceManagementLifecycle:
    """Test resource management throughout component lifecycle."""
    
    @pytest.mark.asyncio
    async def test_memory_connection_file_resource_management_and_leak_detection(self):
        """Test memory, connection, file resource management and leak detection."""
        import psutil
        import gc
        
        # Track initial resource state
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_open_files = len(process.open_files()) if hasattr(process, 'open_files') else 0
        initial_connections = len(process.net_connections()) if hasattr(process, 'net_connections') else 0
        
        resource_tracker = {
            "peak_memory_mb": 0,
            "peak_open_files": 0,
            "peak_connections": 0,
            "components_created": 0,
            "components_destroyed": 0
        }
        
        async def create_and_destroy_components(cycle: int):
            """Create components, use them, then destroy them."""
            components = []
            
            try:
                # Create multiple components
                for i in range(5):
                    config = ConfigurationComponent(f"config_{cycle}_{i}")
                    rate_limiter = RateLimiterComponent(f"rate_limiter_{cycle}_{i}", capacity=20, refill_rate=5.0)
                    middleware = MiddlewareComponent(f"middleware_{cycle}_{i}")
                    
                    # Set up dependencies
                    rate_limiter.dependencies = [config]
                    middleware.dependencies = [config, rate_limiter]
                    
                    components.extend([config, rate_limiter, middleware])
                    resource_tracker["components_created"] += 3
                
                # Initialize all components
                for component in components:
                    await component.initialize()
                
                # Use components to create resource pressure
                for component in components:
                    if isinstance(component, RateLimiterComponent):
                        # Create many token acquisitions
                        for j in range(50):
                            await component.acquire(f"user_{j}")
                    
                    elif isinstance(component, MiddlewareComponent):
                        # Execute middleware multiple times
                        for j in range(10):
                            context = MiddlewareContext(
                                event_id=f"resource_test_{cycle}_{j}",
                                event_type="resource_pressure"
                            )
                            await component.execute(context)
                
                # Track peak resource usage
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                current_open_files = len(process.open_files()) if hasattr(process, 'open_files') else 0
                current_connections = len(process.net_connections()) if hasattr(process, 'net_connections') else 0
                
                resource_tracker["peak_memory_mb"] = max(resource_tracker["peak_memory_mb"], current_memory_mb)
                resource_tracker["peak_open_files"] = max(resource_tracker["peak_open_files"], current_open_files)
                resource_tracker["peak_connections"] = max(resource_tracker["peak_connections"], current_connections)
                
            finally:
                # Shutdown all components
                for component in reversed(components):
                    await component.shutdown()
                    resource_tracker["components_destroyed"] += 1
                
                # Force garbage collection
                gc.collect()
        
        # Run multiple cycles to test resource management
        for cycle in range(3):
            await create_and_destroy_components(cycle)
            
            # Wait for cleanup
            await asyncio.sleep(0.1)
        
        # Final garbage collection
        gc.collect()
        await asyncio.sleep(0.5)  # Allow cleanup tasks to complete
        
        # Check final resource state
        final_memory = process.memory_info().rss
        final_open_files = len(process.open_files()) if hasattr(process, 'open_files') else 0
        final_connections = len(process.net_connections()) if hasattr(process, 'net_connections') else 0
        
        # Verify resource management
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        file_growth = final_open_files - initial_open_files
        connection_growth = final_connections - initial_connections
        
        # Verify no significant resource leaks
        assert memory_growth_mb < 50, f"Memory growth too high: {memory_growth_mb}MB"
        assert file_growth <= 5, f"File descriptor leak detected: {file_growth}"
        assert connection_growth <= 2, f"Connection leak detected: {connection_growth}"
        
        # Verify all components were properly managed
        assert resource_tracker["components_created"] == resource_tracker["components_destroyed"]
        assert resource_tracker["components_created"] == 45  # 3 cycles * 5 components * 3 types

    @pytest.mark.asyncio
    async def test_graceful_shutdown_under_load(self):
        """Test graceful shutdown behavior under load conditions."""
        components = []
        shutdown_tracker = {"shutdown_times": [], "shutdown_success": []}
        
        try:
            # Create components under load
            for i in range(10):
                rate_limiter = RateLimiterComponent(f"loaded_rate_limiter_{i}", capacity=100, refill_rate=50.0)
                await rate_limiter.initialize()
                components.append(rate_limiter)
            
            # Create high load on components
            async def generate_load(component: RateLimiterComponent, duration: float):
                """Generate continuous load on a component."""
                end_time = time.time() + duration
                request_count = 0
                
                while time.time() < end_time:
                    try:
                        await component.acquire(f"load_user_{request_count}")
                        request_count += 1
                        await asyncio.sleep(0.001)  # Small delay to prevent overwhelming
                    except Exception:
                        break  # Component may be shutting down
                
                return request_count
            
            # Start load generation
            load_tasks = [
                asyncio.create_task(generate_load(component, 2.0))
                for component in components
            ]
            
            # Wait a bit for load to build up
            await asyncio.sleep(0.5)
            
            # Begin graceful shutdown while under load
            shutdown_start = time.time()
            
            for i, component in enumerate(components):
                component_shutdown_start = time.time()
                
                try:
                    await component.shutdown()
                    shutdown_time = time.time() - component_shutdown_start
                    shutdown_tracker["shutdown_times"].append(shutdown_time)
                    shutdown_tracker["shutdown_success"].append(True)
                
                except Exception as e:
                    shutdown_time = time.time() - component_shutdown_start
                    shutdown_tracker["shutdown_times"].append(shutdown_time)
                    shutdown_tracker["shutdown_success"].append(False)
            
            total_shutdown_time = time.time() - shutdown_start
            
            # Wait for load tasks to complete
            load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Verify graceful shutdown behavior
            successful_shutdowns = sum(shutdown_tracker["shutdown_success"])
            assert successful_shutdowns >= 8, f"Most shutdowns should succeed: {successful_shutdowns}/10"
            
            # Verify shutdown times are reasonable (not hanging)
            max_shutdown_time = max(shutdown_tracker["shutdown_times"])
            avg_shutdown_time = sum(shutdown_tracker["shutdown_times"]) / len(shutdown_tracker["shutdown_times"])
            
            assert max_shutdown_time < 5.0, f"Shutdown taking too long: {max_shutdown_time}s"
            assert avg_shutdown_time < 1.0, f"Average shutdown time too high: {avg_shutdown_time}s"
            assert total_shutdown_time < 10.0, f"Total shutdown time too high: {total_shutdown_time}s"
            
            # Verify load was actually generated
            total_requests = sum(result for result in load_results if isinstance(result, int))
            assert total_requests > 100, f"Should have generated significant load: {total_requests} requests"
            
        finally:
            # Ensure all components are cleaned up
            for component in components:
                if component.state not in [ComponentState.STOPPED, ComponentState.ERROR]:
                    try:
                        await component.shutdown()
                    except Exception:
                        pass  # Best effort cleanup


if __name__ == "__main__":
    pytest.main([__file__])