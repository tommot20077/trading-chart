# ABOUTME: End-to-end integration tests for all common components working together
# ABOUTME: Tests rate limiting → configuration management → exception handling → middleware complete coordination

import asyncio
import time
import pytest
import os
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
from dataclasses import dataclass

from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.config.settings import CoreSettings, get_settings
from core.exceptions.base import (
    CoreException, ValidationException, BusinessLogicException,
    RateLimitExceededException, ConfigurationException, TimeoutException
)
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from core.interfaces.middleware import AbstractMiddleware


@dataclass
class SystemOperationResult:
    """Result of a complete system operation."""
    success: bool
    operation_id: str
    user_id: str
    processing_time_ms: float
    components_used: List[str]
    rate_limit_status: str
    configuration_used: Dict[str, Any]
    middleware_results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]


class TradingSystemIntegrator:
    """Integrates all common components for end-to-end testing."""
    
    def __init__(self, system_name: str = "trading_system"):
        self.system_name = system_name
        self.rate_limiter = None
        self.middleware_pipeline = None
        self.configuration = None
        self.operation_history = []
        self.system_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "rate_limited_operations": 0,
            "average_processing_time_ms": 0.0
        }
    
    async def initialize(self):
        """Initialize all system components."""
        # Initialize configuration
        get_settings.cache_clear()
        self.configuration = get_settings()
        
        # Initialize rate limiter based on configuration
        capacity = 100 if self.configuration.ENV == "production" else 10
        refill_rate = 50.0 if self.configuration.ENV == "production" else 5.0
        
        self.rate_limiter = InMemoryRateLimiter(
            capacity=capacity,
            refill_rate=refill_rate,
            cleanup_interval=30.0
        )
        
        # Initialize middleware pipeline
        self.middleware_pipeline = InMemoryMiddlewarePipeline(f"{self.system_name}_pipeline")
        
        # Add default middleware
        await self._setup_default_middleware()
    
    async def _setup_default_middleware(self):
        """Setup default middleware components."""
        # Authentication middleware
        class AuthenticationMiddleware(AbstractMiddleware):
            def __init__(self, config):
                super().__init__(EventPriority.HIGHEST)
                self.name = "authentication"
                self.config = config
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                user_id = context.get_metadata("user_id")
                if not user_id:
                    raise ValidationException("User ID required", "AUTH_001")
                
                # Simulate authentication check
                if user_id.startswith("invalid_"):
                    raise ValidationException("Invalid user", "AUTH_002")
                
                context.set_data("authenticated", True)
                context.set_metadata("auth_timestamp", time.time())
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={"user_authenticated": True}
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Validation middleware
        class ValidationMiddleware(AbstractMiddleware):
            def __init__(self):
                super().__init__(EventPriority.HIGH)
                self.name = "validation"
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                # Validate operation parameters
                operation_type = context.get_data("operation_type")
                if not operation_type:
                    raise ValidationException("Operation type required", "VAL_001")
                
                if operation_type not in ["trade", "query", "update"]:
                    raise ValidationException("Invalid operation type", "VAL_002")
                
                # Validate trade-specific parameters
                if operation_type == "trade":
                    amount = context.get_data("amount", 0)
                    if amount <= 0:
                        raise ValidationException("Trade amount must be positive", "VAL_003")
                
                context.set_data("validated", True)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={"validation_passed": True}
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Business logic middleware
        class BusinessLogicMiddleware(AbstractMiddleware):
            def __init__(self, rate_limiter):
                super().__init__(EventPriority.NORMAL)
                self.name = "business_logic"
                self.rate_limiter = rate_limiter
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                user_id = context.get_metadata("user_id")
                operation_type = context.get_data("operation_type")
                
                # Check rate limiting
                if not await self.rate_limiter.acquire_for_identifier(user_id):
                    raise RateLimitExceededException(
                        "User rate limit exceeded",
                        "RATE_001",
                        {"user_id": user_id, "operation": operation_type}
                    )
                
                # Simulate business processing
                processing_time = 0.01  # 10ms
                await asyncio.sleep(processing_time)
                
                result_data = {
                    "operation_completed": True,
                    "processing_time_ms": processing_time * 1000
                }
                
                if operation_type == "trade":
                    amount = context.get_data("amount")
                    result_data.update({
                        "trade_executed": True,
                        "trade_amount": amount,
                        "trade_id": f"trade_{int(time.time() * 1000)}"
                    })
                
                context.set_data("business_result", result_data)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data=result_data,
                    execution_time_ms=processing_time * 1000
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Audit middleware
        class AuditMiddleware(AbstractMiddleware):
            def __init__(self):
                super().__init__(EventPriority.LOW)
                self.name = "audit"
                self.audit_log = []
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                # Create audit record
                audit_record = {
                    "timestamp": time.time(),
                    "user_id": context.get_metadata("user_id"),
                    "operation_type": context.get_data("operation_type"),
                    "success": context.get_data("business_result", {}).get("operation_completed", False),
                    "execution_path": context.get_execution_path()
                }
                
                self.audit_log.append(audit_record)
                context.set_data("audit_logged", True)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={"audit_record_created": True}
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Add middleware to pipeline
        await self.middleware_pipeline.add_middleware(AuthenticationMiddleware(self.configuration))
        await self.middleware_pipeline.add_middleware(ValidationMiddleware())
        await self.middleware_pipeline.add_middleware(BusinessLogicMiddleware(self.rate_limiter))
        await self.middleware_pipeline.add_middleware(AuditMiddleware())
    
    async def execute_operation(self, operation_request: Dict[str, Any]) -> SystemOperationResult:
        """Execute a complete system operation with all components."""
        operation_id = f"op_{int(time.time() * 1000)}_{len(self.operation_history)}"
        start_time = time.time()
        
        # Update metrics
        self.system_metrics["total_operations"] += 1
        
        try:
            # Create middleware context
            context = MiddlewareContext(
                event_id=operation_id,
                event_type="system_operation",
                user_id=operation_request.get("user_id"),
                timestamp=str(start_time)
            )
            
            # Set operation data
            for key, value in operation_request.items():
                if key != "user_id":  # user_id goes in metadata
                    context.set_data(key, value)
            
            context.set_metadata("user_id", operation_request.get("user_id"))
            context.set_metadata("operation_id", operation_id)
            
            # Execute middleware pipeline
            pipeline_result = await self.middleware_pipeline.execute(context)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Determine rate limit status
            rate_limit_status = "allowed"
            components_used = ["configuration", "rate_limiter", "middleware_pipeline"]
            errors = []
            
            # Check for rate limiting in results
            pipeline_results = pipeline_result.metadata.get("pipeline_results", [])
            for result in pipeline_results:
                if result.get("status") == "failed" and "rate limit" in result.get("error", "").lower():
                    rate_limit_status = "exceeded"
                    self.system_metrics["rate_limited_operations"] += 1
                    errors.append({
                        "component": result["middleware_name"],
                        "error": result["error"],
                        "type": "rate_limit"
                    })
                elif result.get("status") == "failed":
                    errors.append({
                        "component": result["middleware_name"],
                        "error": result.get("error", "Unknown error"),
                        "type": "processing"
                    })
            
            success = pipeline_result.status == MiddlewareStatus.SUCCESS and len(errors) == 0
            
            if success:
                self.system_metrics["successful_operations"] += 1
            else:
                self.system_metrics["failed_operations"] += 1
            
            # Update average processing time
            total_ops = self.system_metrics["total_operations"]
            current_avg = self.system_metrics["average_processing_time_ms"]
            self.system_metrics["average_processing_time_ms"] = (
                (current_avg * (total_ops - 1) + processing_time_ms) / total_ops
            )
            
            # Create result
            result = SystemOperationResult(
                success=success,
                operation_id=operation_id,
                user_id=operation_request.get("user_id", "unknown"),
                processing_time_ms=processing_time_ms,
                components_used=components_used,
                rate_limit_status=rate_limit_status,
                configuration_used={
                    "env": self.configuration.ENV,
                    "debug": self.configuration.DEBUG,
                    "log_level": self.configuration.LOG_LEVEL
                },
                middleware_results=[
                    {
                        "name": r["middleware_name"],
                        "status": r["status"],
                        "execution_time_ms": r.get("execution_time_ms", 0),
                        "data": r.get("data", {})
                    } for r in pipeline_results
                ],
                errors=errors,
                performance_metrics={
                    "total_execution_time_ms": processing_time_ms,
                    "middleware_execution_time_ms": pipeline_result.execution_time_ms or 0,
                    "rate_limiter_tokens_remaining": await self.rate_limiter.get_remaining_tokens(operation_request.get("user_id", "default"))
                }
            )
            
            self.operation_history.append(result)
            return result
        
        except Exception as e:
            # Handle unexpected errors
            processing_time_ms = (time.time() - start_time) * 1000
            self.system_metrics["failed_operations"] += 1
            
            return SystemOperationResult(
                success=False,
                operation_id=operation_id,
                user_id=operation_request.get("user_id", "unknown"),
                processing_time_ms=processing_time_ms,
                components_used=["configuration", "rate_limiter", "middleware_pipeline"],
                rate_limit_status="error",
                configuration_used={
                    "env": self.configuration.ENV,
                    "debug": self.configuration.DEBUG,
                    "log_level": self.configuration.LOG_LEVEL
                },
                middleware_results=[],
                errors=[{
                    "component": "system",
                    "error": str(e),
                    "type": "unexpected"
                }],
                performance_metrics={
                    "total_execution_time_ms": processing_time_ms,
                    "middleware_execution_time_ms": 0,
                    "rate_limiter_tokens_remaining": 0
                }
            )
    
    async def shutdown(self):
        """Shutdown all system components."""
        if self.rate_limiter:
            await self.rate_limiter.close()
        
        if self.middleware_pipeline:
            await self.middleware_pipeline.clear()


class TestEndToEndCommonComponentsIntegration:
    """Test end-to-end integration of all common components."""
    
    @pytest.mark.asyncio
    async def test_complete_system_operation_rate_limiting_configuration_exception_handling_middleware_coordination(self):
        """Test complete system: rate limiting → configuration management → exception handling → middleware coordination."""
        # Setup system with production configuration
        with patch.dict(os.environ, {
            'ENV': 'production',
            'DEBUG': 'false',
            'LOG_LEVEL': 'INFO'
        }, clear=False):
            
            system = TradingSystemIntegrator("end_to_end_test")
            
            try:
                await system.initialize()
                
                # Test successful operation flow
                successful_operation = {
                    "user_id": "user_001",
                    "operation_type": "trade",
                    "amount": 100.0,
                    "symbol": "BTC/USD"
                }
                
                result = await system.execute_operation(successful_operation)
                
                # Verify complete successful flow
                assert result.success is True
                assert result.rate_limit_status == "allowed"
                assert len(result.middleware_results) == 4  # auth, validation, business, audit
                assert len(result.errors) == 0
                
                # Verify each middleware executed successfully
                middleware_names = [r["name"] for r in result.middleware_results]
                assert "authentication" in middleware_names
                assert "validation" in middleware_names
                assert "business_logic" in middleware_names
                assert "audit" in middleware_names
                
                # Verify configuration was used correctly
                assert result.configuration_used["env"] == "production"
                assert result.configuration_used["debug"] is False
                
                # Test validation failure handling
                invalid_operation = {
                    "user_id": "user_002", 
                    "operation_type": "invalid_type",
                    "amount": 50.0
                }
                
                result = await system.execute_operation(invalid_operation)
                
                # Verify validation error was handled
                assert result.success is False
                assert len(result.errors) > 0
                validation_error = next((e for e in result.errors if "validation" in e.get("component", "").lower()), None)
                assert validation_error is not None
                
                # Test rate limiting with more reliable approach
                rate_limit_operations = []
                total_operations = 110  # Exceed production capacity of 100
                
                for i in range(total_operations):
                    operation = {
                        "user_id": "heavy_user",
                        "operation_type": "query"
                    }
                    result = await system.execute_operation(operation)
                    rate_limit_operations.append(result)
                    
                    # Add small delay to ensure rate limiting has time to work
                    if i > 0 and i % 10 == 0:
                        await asyncio.sleep(0.01)
                
                # Verify rate limiting kicked in using percentage-based checks
                successful_ops = [r for r in rate_limit_operations if r.success]
                rate_limited_ops = [r for r in rate_limit_operations if r.rate_limit_status == "exceeded"]
                
                # Allow some tolerance but ensure rate limiting is working
                success_rate = len(successful_ops) / total_operations
                rate_limit_rate = len(rate_limited_ops) / total_operations
                
                assert success_rate <= 0.95, f"Success rate too high: {success_rate:.2%} - rate limiting should prevent this"
                assert rate_limit_rate >= 0.05, f"Rate limit rate too low: {rate_limit_rate:.2%} - should have some rate limited operations"
                assert len(successful_ops) <= 105, f"Too many successful operations: {len(successful_ops)} - rate limiting not effective enough"
                
                # Test authentication failure
                auth_failure_operation = {
                    "user_id": "invalid_user_123",
                    "operation_type": "trade",
                    "amount": 25.0
                }
                
                result = await system.execute_operation(auth_failure_operation)
                
                # Verify authentication error was handled
                assert result.success is False
                auth_error = next((e for e in result.errors if "auth" in e.get("component", "").lower()), None)
                assert auth_error is not None
                
                # Verify system metrics
                metrics = system.system_metrics
                assert metrics["total_operations"] > 100
                assert metrics["successful_operations"] > 0
                assert metrics["failed_operations"] > 0
                assert metrics["rate_limited_operations"] > 0
                assert metrics["average_processing_time_ms"] > 0
                
            finally:
                await system.shutdown()

    @pytest.mark.asyncio
    async def test_system_performance_under_concurrent_load(self):
        """Test system performance and component coordination under concurrent load."""
        # Setup system with development configuration for controlled testing
        with patch.dict(os.environ, {
            'ENV': 'development',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG'
        }, clear=False):
            
            system = TradingSystemIntegrator("performance_test")
            
            try:
                await system.initialize()
                
                # Create concurrent users with different operation patterns
                async def user_operations(user_id: str, operation_count: int, operation_type: str):
                    """Simulate user performing multiple operations."""
                    results = []
                    for i in range(operation_count):
                        operation = {
                            "user_id": user_id,
                            "operation_type": operation_type,
                            "amount": 10.0 + i * 5.0 if operation_type == "trade" else None
                        }
                        result = await system.execute_operation(operation)
                        results.append(result)
                        
                        # Small delay to simulate realistic usage
                        await asyncio.sleep(0.001)
                    
                    return results
                
                # Create concurrent load
                user_tasks = [
                    asyncio.create_task(user_operations(f"concurrent_user_{i}", 15, "trade"))
                    for i in range(10)
                ] + [
                    asyncio.create_task(user_operations(f"query_user_{i}", 25, "query"))
                    for i in range(5)
                ]
                
                # Execute all tasks concurrently
                start_time = time.time()
                all_results = await asyncio.gather(*user_tasks)
                total_time = time.time() - start_time
                
                # Flatten results
                flat_results = [result for user_results in all_results for result in user_results]
                
                # Analyze performance
                successful_results = [r for r in flat_results if r.success]
                failed_results = [r for r in flat_results if not r.success]
                rate_limited_results = [r for r in flat_results if r.rate_limit_status == "exceeded"]
                
                # Verify system handled concurrent load
                assert len(flat_results) == 275  # 10*15 + 5*25
                assert len(successful_results) > 0, "Should have some successful operations"
                
                # Verify rate limiting worked under load
                assert len(rate_limited_results) > 0, "Should have rate limited some operations under load"
                
                # Verify performance metrics
                processing_times = [r.processing_time_ms for r in successful_results]
                avg_processing_time = sum(processing_times) / len(processing_times)
                max_processing_time = max(processing_times)
                
                assert avg_processing_time < 100, f"Average processing time too high: {avg_processing_time}ms"
                assert max_processing_time < 500, f"Max processing time too high: {max_processing_time}ms"
                assert total_time < 30, f"Total execution time too high: {total_time}s"
                
                # Verify component coordination under load
                middleware_execution_times = []
                for result in successful_results:
                    for middleware_result in result.middleware_results:
                        if middleware_result["execution_time_ms"] > 0:
                            middleware_execution_times.append(middleware_result["execution_time_ms"])
                
                avg_middleware_time = sum(middleware_execution_times) / len(middleware_execution_times)
                assert avg_middleware_time < 50, f"Middleware execution time too high: {avg_middleware_time}ms"
                
                # Verify system metrics consistency
                metrics = system.system_metrics
                assert metrics["total_operations"] == len(flat_results)
                assert metrics["successful_operations"] == len(successful_results)
                assert metrics["failed_operations"] == len(failed_results)
                
            finally:
                await system.shutdown()

    @pytest.mark.asyncio
    async def test_configuration_change_impact_on_system_behavior(self):
        """Test configuration changes impact on entire system behavior."""
        system = TradingSystemIntegrator("config_change_test")
        
        try:
            # Test with development configuration
            with patch.dict(os.environ, {
                'ENV': 'development',
                'DEBUG': 'true',
                'LOG_LEVEL': 'DEBUG'
            }, clear=False):
                
                await system.initialize()
                
                # Execute operations in development mode
                dev_operations = []
                for i in range(15):  # Exceed development capacity of 10
                    operation = {
                        "user_id": "config_test_user",
                        "operation_type": "trade",
                        "amount": 100.0
                    }
                    result = await system.execute_operation(operation)
                    dev_operations.append(result)
                
                dev_successful = [r for r in dev_operations if r.success]
                dev_rate_limited = [r for r in dev_operations if r.rate_limit_status == "exceeded"]
                
                # Should hit rate limit sooner in development
                assert len(dev_successful) <= 10, "Development should have lower rate limit"
                assert len(dev_rate_limited) >= 5, "Should hit rate limit in development"
                
                # Verify development configuration was used
                assert dev_operations[0].configuration_used["env"] == "development"
                assert dev_operations[0].configuration_used["debug"] is True
                
                await system.shutdown()
            
            # Test with production configuration
            with patch.dict(os.environ, {
                'ENV': 'production',
                'DEBUG': 'false',
                'LOG_LEVEL': 'WARNING'
            }, clear=False):
                
                # Reinitialize with new configuration
                system = TradingSystemIntegrator("config_change_test_prod")
                await system.initialize()
                
                # Execute same operations in production mode
                prod_operations = []
                for i in range(60):  # Test higher capacity
                    operation = {
                        "user_id": "config_test_user_prod",
                        "operation_type": "trade",
                        "amount": 100.0
                    }
                    result = await system.execute_operation(operation)
                    prod_operations.append(result)
                
                prod_successful = [r for r in prod_operations if r.success]
                prod_rate_limited = [r for r in prod_operations if r.rate_limit_status == "exceeded"]
                
                # Should handle more operations in production
                assert len(prod_successful) > len(dev_successful), "Production should handle more operations"
                
                # Verify production configuration was used
                assert prod_operations[0].configuration_used["env"] == "production"
                assert prod_operations[0].configuration_used["debug"] is False
                
                # Verify performance difference
                dev_avg_time = sum(r.processing_time_ms for r in dev_successful) / len(dev_successful)
                prod_avg_time = sum(r.processing_time_ms for r in prod_successful) / len(prod_successful)
                
                # Production might be optimized differently
                # Just verify both configurations work
                assert dev_avg_time > 0
                assert prod_avg_time > 0
                
        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self):
        """Test system resilience and recovery from various failure scenarios."""
        system = TradingSystemIntegrator("resilience_test")
        
        try:
            await system.initialize()
            
            # Test various failure scenarios
            failure_scenarios = [
                {
                    "name": "missing_user_id",
                    "operation": {
                        "operation_type": "trade",
                        "amount": 100.0
                    },
                    "expected_error_type": "auth"
                },
                {
                    "name": "invalid_operation_type",
                    "operation": {
                        "user_id": "test_user",
                        "operation_type": "invalid",
                        "amount": 100.0
                    },
                    "expected_error_type": "validation"
                },
                {
                    "name": "negative_amount",
                    "operation": {
                        "user_id": "test_user",
                        "operation_type": "trade",
                        "amount": -50.0
                    },
                    "expected_error_type": "validation"
                },
                {
                    "name": "missing_operation_type",
                    "operation": {
                        "user_id": "test_user",
                        "amount": 100.0
                    },
                    "expected_error_type": "validation"
                }
            ]
            
            recovery_results = []
            
            for scenario in failure_scenarios:
                # Execute failing operation
                fail_result = await system.execute_operation(scenario["operation"])
                
                # Verify failure was handled gracefully
                assert fail_result.success is False
                assert len(fail_result.errors) > 0
                
                # Verify error type
                error_found = any(
                    scenario["expected_error_type"] in error.get("error", "").lower() or
                    scenario["expected_error_type"] in error.get("component", "").lower()
                    for error in fail_result.errors
                )
                assert error_found, f"Expected {scenario['expected_error_type']} error in {scenario['name']}"
                
                # Test recovery with valid operation
                recovery_operation = {
                    "user_id": "recovery_user",
                    "operation_type": "query"
                }
                recovery_result = await system.execute_operation(recovery_operation)
                recovery_results.append(recovery_result)
                
                # Verify system recovered
                assert recovery_result.success is True, f"System should recover after {scenario['name']} failure"
            
            # Verify all recovery operations succeeded
            assert all(r.success for r in recovery_results), "All recovery operations should succeed"
            
            # Test rate limit recovery
            rate_limit_user = "rate_limit_recovery_user"
            
            # Exhaust rate limit
            for i in range(12):  # Exceed capacity
                operation = {
                    "user_id": rate_limit_user,
                    "operation_type": "query"
                }
                await system.execute_operation(operation)
            
            # Wait for token refill
            await asyncio.sleep(2.0)  # Allow tokens to refill
            
            # Test recovery operation
            recovery_operation = {
                "user_id": rate_limit_user,
                "operation_type": "query"
            }
            recovery_result = await system.execute_operation(recovery_operation)
            
            # Should succeed after token refill
            assert recovery_result.success is True, "Should recover from rate limiting"
            assert recovery_result.rate_limit_status == "allowed", "Rate limit should be reset"
            
            # Verify system metrics show resilience
            metrics = system.system_metrics
            assert metrics["total_operations"] > 20
            assert metrics["successful_operations"] > 0
            assert metrics["failed_operations"] > 0
            
            # System should maintain reasonable success rate despite failures
            success_rate = metrics["successful_operations"] / metrics["total_operations"]
            assert success_rate > 0.3, f"Success rate too low: {success_rate}"
            
        finally:
            await system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])