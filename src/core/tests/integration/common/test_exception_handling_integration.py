# ABOUTME: Integration tests for unified exception handling across modules
# ABOUTME: Tests 12 exception types, cross-module propagation, and unified processing

import asyncio
import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from core.exceptions.base import (
    CoreException, ValidationException, BusinessLogicException,
    DataNotFoundException, ExternalServiceException, ConfigurationException,
    AuthenticationException, AuthorizationError, RateLimitExceededException,
    DataIntegrityException, TimeoutException, EventSerializationError,
    EventDeserializationError, StorageError, NotSupportedError
)
from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter


class TestUnifiedExceptionHandlingCrossModule:
    """Test unified exception handling across different modules."""
    
    def test_twelve_exception_types_cross_module_propagation(self, exception_simulator):
        """Test 12 exception types and their cross-module propagation and unified handling."""
        # Define all 12 exception types with their contexts
        exception_scenarios = [
            {
                'type': ValidationException,
                'factory': exception_simulator.create_validation_error,
                'module': 'data_validation',
                'expected_code': 'VALIDATION_001'
            },
            {
                'type': BusinessLogicException,
                'factory': exception_simulator.create_business_logic_error,
                'module': 'business_engine',
                'expected_code': 'BUSINESS_001'
            },
            {
                'type': DataNotFoundException,
                'factory': exception_simulator.create_data_not_found_error,
                'module': 'data_access',
                'expected_code': 'DATA_001'
            },
            {
                'type': ExternalServiceException,
                'factory': exception_simulator.create_external_service_error,
                'module': 'external_api',
                'expected_code': 'EXTERNAL_001'
            },
            {
                'type': RateLimitExceededException,
                'factory': exception_simulator.create_rate_limit_error,
                'module': 'rate_limiting',
                'expected_code': 'RATE_001'
            },
            {
                'type': ConfigurationException,
                'factory': lambda: ConfigurationException("Config error", "CONFIG_001", {"setting": "invalid"}),
                'module': 'configuration',
                'expected_code': 'CONFIG_001'
            },
            {
                'type': AuthenticationException,
                'factory': lambda: AuthenticationException("Auth failed", "AUTH_001", {"user": "test"}),
                'module': 'authentication',
                'expected_code': 'AUTH_001'
            },
            {
                'type': AuthorizationError,
                'factory': lambda: AuthorizationError("Access denied", "AUTHZ_001", {"resource": "/api/data"}),
                'module': 'authorization',
                'expected_code': 'AUTHZ_001'
            },
            {
                'type': DataIntegrityException,
                'factory': lambda: DataIntegrityException("Constraint violation", "INTEGRITY_001", {"constraint": "unique"}),
                'module': 'data_storage',
                'expected_code': 'INTEGRITY_001'
            },
            {
                'type': TimeoutException,
                'factory': lambda: TimeoutException("Operation timeout", "TIMEOUT_001", {"duration": 30}),
                'module': 'async_operations',
                'expected_code': 'TIMEOUT_001'
            },
            {
                'type': EventSerializationError,
                'factory': lambda: EventSerializationError("Serialization failed", "SERIALIZE_001", {"event": "trade"}),
                'module': 'event_system',
                'expected_code': 'SERIALIZE_001'
            },
            {
                'type': StorageError,
                'factory': lambda: StorageError("Storage failure", "STORAGE_001", {"operation": "write"}),
                'module': 'storage_system',
                'expected_code': 'STORAGE_001'
            }
        ]
        
        # Simulate cross-module exception propagation
        exception_results = []
        
        for scenario in exception_scenarios:
            try:
                # Simulate module operation that raises exception
                exception = scenario['factory']()
                
                # Simulate cross-module propagation
                def module_operation():
                    raise exception
                
                def calling_module():
                    try:
                        module_operation()
                    except CoreException as e:
                        # Unified exception handling
                        return {
                            'exception_type': type(e).__name__,
                            'message': e.message,
                            'code': e.code,
                            'details': e.details,
                            'source_module': scenario['module'],
                            'handled': True
                        }
                
                result = calling_module()
                exception_results.append(result)
                
            except Exception as e:
                # Fallback for any unexpected exceptions
                exception_results.append({
                    'exception_type': type(e).__name__,
                    'message': str(e),
                    'source_module': scenario['module'],
                    'handled': False
                })
        
        # Verify all exceptions were handled uniformly
        assert len(exception_results) == 12, "All 12 exception types should be tested"
        
        for result in exception_results:
            assert result['handled'] is True, f"Exception {result['exception_type']} should be handled"
            assert 'message' in result
            assert 'code' in result
        
        # Verify each exception type was encountered
        exception_types = [result['exception_type'] for result in exception_results]
        expected_types = [scenario['type'].__name__ for scenario in exception_scenarios]
        
        for expected_type in expected_types:
            assert expected_type in exception_types, f"Exception type {expected_type} should be present"

    def test_exception_propagation_through_async_components(self):
        """Test exception propagation through async components like rate limiters."""
        async def test_async_exception_propagation():
            rate_limiter = InMemoryRateLimiter(capacity=1, refill_rate=0.1)
            
            try:
                # Exhaust rate limiter
                await rate_limiter.acquire()
                
                # Simulate business operation that checks rate limit
                async def business_operation():
                    if not await rate_limiter.acquire():
                        raise RateLimitExceededException(
                            "Rate limit exceeded in business operation",
                            "BUSINESS_RATE_001",
                            {"operation": "trade_execution", "limit": 1}
                        )
                    return "success"
                
                # This should raise rate limit exception
                result = await business_operation()
                assert False, "Should have raised RateLimitExceededException"
                
            except RateLimitExceededException as e:
                # Verify exception propagated correctly
                assert e.code == "BUSINESS_RATE_001"
                assert e.details["operation"] == "trade_execution"
                return True
            
            finally:
                await rate_limiter.close()
        
        # Run async test
        result = asyncio.run(test_async_exception_propagation())
        assert result is True, "Async exception propagation should work"

    def test_exception_context_preservation_across_modules(self):
        """Test exception context and details preservation across module boundaries."""
        def data_access_layer():
            """Simulates data access layer raising an exception."""
            raise DataNotFoundException(
                "User profile not found",
                "DATA_404",
                {
                    "user_id": "12345",
                    "table": "user_profiles",
                    "query": "SELECT * FROM user_profiles WHERE id = 12345",
                    "timestamp": "2025-01-20T10:30:00Z"
                }
            )
        
        def business_logic_layer():
            """Simulates business logic layer handling data access."""
            try:
                data_access_layer()
            except DataNotFoundException as e:
                # Business layer adds its own context
                e.details.update({
                    "business_operation": "get_user_profile",
                    "caller": "profile_service",
                    "impact": "user_dashboard_unavailable"
                })
                raise  # Re-raise with enhanced context
        
        def api_layer():
            """Simulates API layer handling business logic."""
            try:
                business_logic_layer()
            except DataNotFoundException as e:
                # API layer adds request context
                e.details.update({
                    "request_id": "req_12345",
                    "endpoint": "/api/v1/users/12345/profile",
                    "method": "GET",
                    "user_agent": "test_client/1.0"
                })
                return {
                    "error": True,
                    "exception_type": type(e).__name__,
                    "message": e.message,
                    "code": e.code,
                    "context": e.details
                }
        
        # Execute cross-module operation
        result = api_layer()
        
        # Verify context preservation and enhancement
        assert result["error"] is True
        assert result["code"] == "DATA_404"
        assert result["context"]["user_id"] == "12345"
        assert result["context"]["business_operation"] == "get_user_profile"
        assert result["context"]["request_id"] == "req_12345"
        assert result["context"]["endpoint"] == "/api/v1/users/12345/profile"


class TestExceptionRecoveryMechanismIntegration:
    """Test exception recovery mechanisms and business continuity."""
    
    def test_exception_recovery_strategies_business_continuity(self):
        """Test exception recovery strategies and business continuity guarantee."""
        class MockTradingSystem:
            def __init__(self):
                self.fallback_exchange = "backup_exchange"
                self.retry_count = 0
                self.max_retries = 3
                self.circuit_breaker_open = False
            
            def execute_trade(self, symbol: str, amount: float):
                """Simulate trade execution with various failure scenarios."""
                self.retry_count += 1
                
                # Simulate different types of failures
                if self.retry_count == 1:
                    raise ExternalServiceException(
                        "Primary exchange unavailable",
                        "EXCHANGE_001",
                        {"exchange": "primary", "symbol": symbol}
                    )
                elif self.retry_count == 2:
                    raise TimeoutException(
                        "Trade execution timeout",
                        "TIMEOUT_001",
                        {"symbol": symbol, "timeout": 5.0}
                    )
                elif self.retry_count == 3:
                    # Recovery successful
                    return {
                        "status": "success",
                        "symbol": symbol,
                        "amount": amount,
                        "exchange": self.fallback_exchange,
                        "retry_count": self.retry_count
                    }
        
        def robust_trading_operation(symbol: str, amount: float):
            """Implements robust trading with exception recovery."""
            trading_system = MockTradingSystem()
            last_exception = None
            
            for attempt in range(trading_system.max_retries):
                try:
                    result = trading_system.execute_trade(symbol, amount)
                    return {
                        "success": True,
                        "result": result,
                        "attempts": attempt + 1,
                        "recovered": attempt > 0
                    }
                
                except ExternalServiceException as e:
                    last_exception = e
                    # Implement fallback strategy
                    print(f"Attempt {attempt + 1}: External service error, trying fallback")
                    continue
                
                except TimeoutException as e:
                    last_exception = e
                    # Implement retry with backoff
                    print(f"Attempt {attempt + 1}: Timeout error, retrying")
                    continue
                
                except Exception as e:
                    # Unexpected error, break circuit
                    return {
                        "success": False,
                        "error": "circuit_breaker_open",
                        "last_exception": str(e),
                        "attempts": attempt + 1
                    }
            
            # All retries exhausted
            return {
                "success": False,
                "error": "max_retries_exceeded",
                "last_exception": last_exception.message if last_exception else "Unknown",
                "attempts": trading_system.max_retries
            }
        
        # Test recovery mechanism
        result = robust_trading_operation("BTC/USD", 1.5)
        
        # Verify business continuity was maintained
        assert result["success"] is True, "Business operation should eventually succeed"
        assert result["recovered"] is True, "System should have recovered from failures"
        assert result["attempts"] == 3, "Should have taken 3 attempts"
        assert result["result"]["exchange"] == "backup_exchange", "Should use fallback exchange"

    def test_exception_recovery_with_async_components(self):
        """Test exception recovery mechanisms with async components."""
        async def test_async_recovery():
            # Simulate async service with failure and recovery
            class AsyncServiceWithRecovery:
                def __init__(self):
                    self.attempt_count = 0
                    self.rate_limiter = InMemoryRateLimiter(capacity=2, refill_rate=1.0)
                
                async def process_request(self, request_id: str):
                    self.attempt_count += 1
                    
                    # Check rate limit first
                    if not await self.rate_limiter.acquire():
                        raise RateLimitExceededException(
                            "Service rate limit exceeded",
                            "RATE_002",
                            {"request_id": request_id, "attempt": self.attempt_count}
                        )
                    
                    # Simulate temporary failure
                    if self.attempt_count == 1:
                        raise ExternalServiceException(
                            "Temporary service unavailable",
                            "SERVICE_001",
                            {"request_id": request_id}
                        )
                    
                    # Success on second attempt
                    return {
                        "status": "processed",
                        "request_id": request_id,
                        "attempt": self.attempt_count
                    }
                
                async def close(self):
                    await self.rate_limiter.close()
            
            async def resilient_request_handler(request_id: str):
                """Handles requests with automatic retry and recovery."""
                service = AsyncServiceWithRecovery()
                max_attempts = 3
                
                try:
                    for attempt in range(max_attempts):
                        try:
                            result = await service.process_request(request_id)
                            return {
                                "success": True,
                                "result": result,
                                "attempts": attempt + 1
                            }
                        
                        except RateLimitExceededException as e:
                            if attempt < max_attempts - 1:
                                # Wait for rate limit to reset
                                await asyncio.sleep(1.1)
                                continue
                            else:
                                raise
                        
                        except ExternalServiceException as e:
                            if attempt < max_attempts - 1:
                                # Exponential backoff
                                await asyncio.sleep(0.5 * (2 ** attempt))
                                continue
                            else:
                                raise
                    
                    return {"success": False, "error": "max_attempts_exceeded"}
                
                finally:
                    await service.close()
            
            # Test async recovery
            result = await resilient_request_handler("req_001")
            
            # Verify async recovery worked
            assert result["success"] is True
            assert result["attempts"] == 2  # Failed once, succeeded on second attempt
            assert result["result"]["status"] == "processed"
            
            return True
        
        # Run async test
        success = asyncio.run(test_async_recovery())
        assert success is True

    def test_cascading_exception_recovery(self):
        """Test recovery from cascading exceptions across multiple components."""
        class CascadingFailureScenario:
            def __init__(self):
                self.database_healthy = False
                self.cache_healthy = False
                self.external_api_healthy = False
                self.recovery_attempts = 0
            
            def attempt_database_operation(self):
                if not self.database_healthy:
                    raise DataIntegrityException(
                        "Database connection failed",
                        "DB_001",
                        {"component": "database", "health": "unhealthy"}
                    )
                return {"source": "database", "data": "primary_data"}
            
            def attempt_cache_operation(self):
                if not self.cache_healthy:
                    raise ExternalServiceException(
                        "Cache service unavailable",
                        "CACHE_001",
                        {"component": "cache", "health": "unhealthy"}
                    )
                return {"source": "cache", "data": "cached_data"}
            
            def attempt_external_api_operation(self):
                if not self.external_api_healthy:
                    raise ExternalServiceException(
                        "External API unreachable",
                        "API_001",
                        {"component": "external_api", "health": "unhealthy"}
                    )
                return {"source": "external_api", "data": "external_data"}
            
            def trigger_recovery(self):
                """Simulate system recovery after failures."""
                self.recovery_attempts += 1
                if self.recovery_attempts >= 2:
                    self.external_api_healthy = True  # External API recovers last
        
        def resilient_data_access(scenario: CascadingFailureScenario):
            """Implements cascading fallback strategy."""
            fallback_chain = [
                ("primary", scenario.attempt_database_operation),
                ("cache", scenario.attempt_cache_operation),
                ("external", scenario.attempt_external_api_operation)
            ]
            
            errors = []
            
            for source_name, operation in fallback_chain:
                try:
                    result = operation()
                    return {
                        "success": True,
                        "source": source_name,
                        "data": result,
                        "fallback_used": source_name != "primary",
                        "errors": errors
                    }
                
                except CoreException as e:
                    errors.append({
                        "source": source_name,
                        "error": e.message,
                        "code": e.code
                    })
                    
                    # Trigger recovery attempt
                    scenario.trigger_recovery()
                    continue
            
            return {
                "success": False,
                "errors": errors,
                "total_failures": len(errors)
            }
        
        # Test cascading failure and recovery
        scenario = CascadingFailureScenario()
        result = resilient_data_access(scenario)
        
        # Verify cascading recovery worked
        assert result["success"] is True, "Should eventually succeed through fallback chain"
        assert result["source"] == "external", "Should succeed via external API"
        assert result["fallback_used"] is True, "Should have used fallback"
        assert len(result["errors"]) == 2, "Should have encountered 2 failures before success"


class TestErrorContextTrackingIntegration:
    """Test error context and tracking integration across the system."""
    
    def test_error_code_context_data_complete_correlation(self):
        """Test error codes, context data, and tracking information complete correlation."""
        class DistributedOperationTracker:
            def __init__(self):
                self.operation_id = "op_12345"
                self.trace_id = "trace_67890"
                self.user_context = {"user_id": "user_001", "session_id": "sess_abc123"}
                self.operation_stack = []
            
            def enter_operation(self, operation_name: str, component: str):
                self.operation_stack.append({
                    "operation": operation_name,
                    "component": component,
                    "timestamp": "2025-01-20T10:30:00Z"
                })
            
            def create_contextual_exception(self, exception_type, message: str, error_code: str, component_details: Dict[str, Any]):
                """Create exception with full context correlation."""
                full_context = {
                    "operation_id": self.operation_id,
                    "trace_id": self.trace_id,
                    "user_context": self.user_context,
                    "operation_stack": self.operation_stack.copy(),
                    "component_details": component_details,
                    "correlation_data": {
                        "timestamp": "2025-01-20T10:30:00Z",
                        "environment": "production",
                        "service_version": "1.2.3"
                    }
                }
                
                return exception_type(message, error_code, full_context)
        
        # Simulate distributed operation with context tracking
        tracker = DistributedOperationTracker()
        
        # Step 1: User authentication
        tracker.enter_operation("authenticate_user", "auth_service")
        
        try:
            # Step 2: Data validation
            tracker.enter_operation("validate_request", "validation_service")
            
            # Step 3: Business logic (where error occurs)
            tracker.enter_operation("execute_trade", "trading_service")
            
            # Simulate validation failure
            raise tracker.create_contextual_exception(
                ValidationException,
                "Invalid trading parameters",
                "TRADE_VALIDATION_001",
                {
                    "validation_rules": ["amount_positive", "symbol_exists", "balance_sufficient"],
                    "failed_rule": "balance_sufficient",
                    "provided_amount": 1000.0,
                    "available_balance": 500.0,
                    "symbol": "BTC/USD"
                }
            )
        
        except ValidationException as e:
            # Verify complete context correlation
            assert e.code == "TRADE_VALIDATION_001"
            assert e.details["operation_id"] == "op_12345"
            assert e.details["trace_id"] == "trace_67890"
            assert e.details["user_context"]["user_id"] == "user_001"
            
            # Verify operation stack tracking
            assert len(e.details["operation_stack"]) == 3
            operations = [op["operation"] for op in e.details["operation_stack"]]
            assert "authenticate_user" in operations
            assert "validate_request" in operations
            assert "execute_trade" in operations
            
            # Verify component-specific details
            component_details = e.details["component_details"]
            assert component_details["failed_rule"] == "balance_sufficient"
            assert component_details["provided_amount"] == 1000.0
            assert component_details["available_balance"] == 500.0
            
            # Verify correlation data
            correlation = e.details["correlation_data"]
            assert correlation["environment"] == "production"
            assert correlation["service_version"] == "1.2.3"

    def test_cross_service_error_correlation_tracking(self):
        """Test cross-service error correlation and tracking."""
        def microservice_a():
            """Simulates microservice A operation."""
            try:
                # Simulate calling microservice B
                microservice_b()
            except ExternalServiceException as e:
                # Add service A context to the exception
                e.details.update({
                    "service_a_context": {
                        "service": "trading_engine",
                        "operation": "execute_order",
                        "order_id": "order_123",
                        "timestamp": "2025-01-20T10:30:00Z"
                    }
                })
                raise
        
        def microservice_b():
            """Simulates microservice B operation."""
            try:
                # Simulate calling external service
                external_service_call()
            except TimeoutException as e:
                # Transform timeout to external service exception with context
                raise ExternalServiceException(
                    "Downstream service timeout",
                    "DOWNSTREAM_001",
                    {
                        "service_b_context": {
                            "service": "market_data_service",
                            "operation": "get_price_feed",
                            "symbol": "BTC/USD",
                            "timeout_duration": 5.0
                        },
                        "original_error": {
                            "type": "TimeoutException",
                            "message": e.message,
                            "code": e.code,
                            "details": e.details
                        }
                    }
                )
        
        def external_service_call():
            """Simulates external service call that times out."""
            raise TimeoutException(
                "External API timeout",
                "EXTERNAL_TIMEOUT_001",
                {
                    "external_service": {
                        "name": "binance_api",
                        "endpoint": "/api/v3/ticker/price",
                        "timeout": 5.0,
                        "attempt": 1
                    }
                }
            )
        
        # Test cross-service error correlation
        try:
            microservice_a()
            assert False, "Should have raised an exception"
        
        except ExternalServiceException as e:
            # Verify cross-service correlation
            assert e.code == "DOWNSTREAM_001"
            
            # Verify service A context
            service_a_context = e.details["service_a_context"]
            assert service_a_context["service"] == "trading_engine"
            assert service_a_context["order_id"] == "order_123"
            
            # Verify service B context
            service_b_context = e.details["service_b_context"]
            assert service_b_context["service"] == "market_data_service"
            assert service_b_context["symbol"] == "BTC/USD"
            
            # Verify original error context
            original_error = e.details["original_error"]
            assert original_error["type"] == "TimeoutException"
            assert original_error["code"] == "EXTERNAL_TIMEOUT_001"
            
            # Verify external service details are preserved
            external_details = original_error["details"]["external_service"]
            assert external_details["name"] == "binance_api"
            assert external_details["endpoint"] == "/api/v3/ticker/price"


if __name__ == "__main__":
    pytest.main([__file__])