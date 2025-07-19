# ABOUTME: Integration tests for error context and tracking across system components
# ABOUTME: Tests error codes, context data correlation, and trace information completeness

import asyncio
import time
import uuid
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from core.exceptions.base import (
    CoreException, ValidationException, BusinessLogicException,
    DataNotFoundException, ExternalServiceException, TimeoutException
)


class ErrorContextTracker:
    """Utility class for tracking error context across operations."""
    
    def __init__(self, operation_id: Optional[str] = None):
        self.operation_id = operation_id or str(uuid.uuid4())
        self.trace_id = str(uuid.uuid4())
        self.context_stack = []
        self.metadata = {}
    
    def push_context(self, component: str, operation: str, **kwargs):
        """Add context to the tracking stack."""
        context = {
            "component": component,
            "operation": operation,
            "timestamp": time.time(),
            "metadata": kwargs
        }
        self.context_stack.append(context)
    
    def pop_context(self):
        """Remove the latest context from the stack."""
        if self.context_stack:
            return self.context_stack.pop()
        return None
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context information."""
        return {
            "operation_id": self.operation_id,
            "trace_id": self.trace_id,
            "context_stack": self.context_stack.copy(),
            "metadata": self.metadata.copy(),
            "timestamp": time.time()
        }
    
    def create_contextual_exception(self, exception_class, message: str, 
                                  error_code: str, component_details: Dict[str, Any]):
        """Create an exception with full context information."""
        full_context = self.get_full_context()
        full_context.update(component_details)
        
        return exception_class(message, error_code, full_context)


class TestErrorCodeContextCorrelation:
    """Test error codes and context data correlation across components."""
    
    def test_error_code_hierarchy_and_context_correlation(self):
        """Test error code hierarchy and context data correlation."""
        tracker = ErrorContextTracker()
        
        # Simulate hierarchical operation with error codes
        def level_1_operation():
            tracker.push_context("api_gateway", "handle_request", 
                                endpoint="/api/v1/trading/orders", method="POST")
            try:
                level_2_operation()
            except Exception as e:
                # Add API gateway context
                if isinstance(e, CoreException):
                    e.details.update({
                        "api_context": {
                            "request_id": "req_12345",
                            "client_ip": "192.168.1.100",
                            "user_agent": "trading-client/1.0"
                        }
                    })
                raise
            finally:
                tracker.pop_context()
        
        def level_2_operation():
            tracker.push_context("order_service", "create_order",
                                order_type="market", symbol="BTC/USD")
            try:
                level_3_operation()
            except Exception as e:
                # Add order service context
                if isinstance(e, CoreException):
                    e.details.update({
                        "order_context": {
                            "order_id": "ord_67890",
                            "symbol": "BTC/USD",
                            "order_type": "market",
                            "amount": 1.5
                        }
                    })
                raise
            finally:
                tracker.pop_context()
        
        def level_3_operation():
            tracker.push_context("validation_service", "validate_balance",
                                user_id="user_001", required_balance=1500.0)
            try:
                # Simulate validation failure
                raise tracker.create_contextual_exception(
                    ValidationException,
                    "Insufficient balance for order",
                    "VAL_BALANCE_001",
                    {
                        "validation_context": {
                            "rule": "sufficient_balance",
                            "required_balance": 1500.0,
                            "available_balance": 1200.0,
                            "currency": "USD",
                            "deficit": 300.0
                        }
                    }
                )
            finally:
                tracker.pop_context()
        
        # Execute hierarchical operation
        try:
            level_1_operation()
            assert False, "Should have raised ValidationException"
        
        except ValidationException as e:
            # Verify error code hierarchy
            assert e.code == "VAL_BALANCE_001"
            
            # Verify context correlation at all levels
            assert "api_context" in e.details
            assert "order_context" in e.details
            assert "validation_context" in e.details
            
            # Verify API gateway context
            api_context = e.details["api_context"]
            assert api_context["request_id"] == "req_12345"
            assert api_context["client_ip"] == "192.168.1.100"
            
            # Verify order service context
            order_context = e.details["order_context"]
            assert order_context["order_id"] == "ord_67890"
            assert order_context["symbol"] == "BTC/USD"
            assert order_context["amount"] == 1.5
            
            # Verify validation context
            validation_context = e.details["validation_context"]
            assert validation_context["rule"] == "sufficient_balance"
            assert validation_context["deficit"] == 300.0
            
            # Verify operation tracking
            assert e.details["operation_id"] == tracker.operation_id
            assert e.details["trace_id"] == tracker.trace_id

    def test_cross_component_error_propagation_with_context_preservation(self):
        """Test cross-component error propagation with context preservation."""
        tracker = ErrorContextTracker()
        
        # Define components with different error handling strategies
        class DatabaseComponent:
            def query_user_data(self, user_id: str):
                tracker.push_context("database", "query_user_data", 
                                    table="users", user_id=user_id)
                try:
                    raise tracker.create_contextual_exception(
                        DataNotFoundException,
                        "User not found in database",
                        "DB_USER_404",
                        {
                            "database_context": {
                                "table": "users",
                                "query": f"SELECT * FROM users WHERE id = '{user_id}'",
                                "affected_rows": 0,
                                "execution_time_ms": 15
                            }
                        }
                    )
                finally:
                    tracker.pop_context()
        
        class CacheComponent:
            def __init__(self):
                self.db = DatabaseComponent()
            
            def get_user_data(self, user_id: str):
                tracker.push_context("cache", "get_user_data",
                                    cache_key=f"user:{user_id}")
                try:
                    # Cache miss, try database
                    return self.db.query_user_data(user_id)
                except DataNotFoundException as e:
                    # Add cache context to existing exception
                    e.details.update({
                        "cache_context": {
                            "cache_key": f"user:{user_id}",
                            "cache_hit": False,
                            "ttl": None,
                            "fallback_attempted": True
                        }
                    })
                    raise
                finally:
                    tracker.pop_context()
        
        class UserService:
            def __init__(self):
                self.cache = CacheComponent()
            
            def get_user_profile(self, user_id: str):
                tracker.push_context("user_service", "get_user_profile",
                                    profile_operation="get_profile")
                try:
                    return self.cache.get_user_data(user_id)
                except DataNotFoundException as e:
                    # Transform to business logic exception with service context
                    raise tracker.create_contextual_exception(
                        BusinessLogicException,
                        "Cannot retrieve user profile",
                        "USER_PROFILE_UNAVAILABLE",
                        {
                            "service_context": {
                                "service": "user_service",
                                "operation": "get_user_profile",
                                "user_id": user_id,
                                "fallback_available": False
                            },
                            "original_error": {
                                "type": type(e).__name__,
                                "code": e.code,
                                "message": e.message,
                                "details": e.details
                            }
                        }
                    )
                finally:
                    tracker.pop_context()
        
        # Execute cross-component operation
        user_service = UserService()
        
        try:
            user_service.get_user_profile("nonexistent_user")
            assert False, "Should have raised BusinessLogicException"
        
        except BusinessLogicException as e:
            # Verify final error context
            assert e.code == "USER_PROFILE_UNAVAILABLE"
            
            # Verify service context
            service_context = e.details["service_context"]
            assert service_context["service"] == "user_service"
            assert service_context["user_id"] == "nonexistent_user"
            
            # Verify original error is preserved
            original_error = e.details["original_error"]
            assert original_error["type"] == "DataNotFoundException"
            assert original_error["code"] == "DB_USER_404"
            
            # Verify nested context preservation
            original_details = original_error["details"]
            assert "database_context" in original_details
            assert "cache_context" in original_details
            
            # Verify database context
            db_context = original_details["database_context"]
            assert db_context["table"] == "users"
            assert db_context["affected_rows"] == 0
            
            # Verify cache context
            cache_context = original_details["cache_context"]
            assert cache_context["cache_hit"] is False
            assert cache_context["fallback_attempted"] is True

    def test_async_operation_error_context_tracking(self):
        """Test error context tracking in async operations."""
        async def test_async_context_tracking():
            tracker = ErrorContextTracker()
            
            async def async_database_operation():
                tracker.push_context("async_db", "execute_query",
                                    connection_pool="primary")
                try:
                    # Simulate async operation failure
                    await asyncio.sleep(0.1)  # Simulate async work
                    raise tracker.create_contextual_exception(
                        TimeoutException,
                        "Database query timeout",
                        "DB_TIMEOUT_001",
                        {
                            "async_context": {
                                "operation": "SELECT * FROM orders WHERE status = 'pending'",
                                "timeout_ms": 5000,
                                "actual_duration_ms": 5100,
                                "connection_id": "conn_123",
                                "pool_stats": {
                                    "active_connections": 8,
                                    "idle_connections": 2,
                                    "max_connections": 10
                                }
                            }
                        }
                    )
                finally:
                    tracker.pop_context()
            
            async def async_business_operation():
                tracker.push_context("async_business", "process_pending_orders",
                                    batch_size=100)
                try:
                    await async_database_operation()
                except TimeoutException as e:
                    # Add business context to async error
                    e.details.update({
                        "business_context": {
                            "operation": "process_pending_orders",
                            "batch_size": 100,
                            "processing_time": 0.15,
                            "orders_processed": 0,
                            "retry_strategy": "exponential_backoff"
                        }
                    })
                    raise
                finally:
                    tracker.pop_context()
            
            # Execute async operation
            try:
                await async_business_operation()
                assert False, "Should have raised TimeoutException"
            
            except TimeoutException as e:
                # Verify async context tracking
                assert e.code == "DB_TIMEOUT_001"
                
                # Verify async database context
                async_context = e.details["async_context"]
                assert async_context["timeout_ms"] == 5000
                assert async_context["actual_duration_ms"] == 5100
                assert async_context["connection_id"] == "conn_123"
                
                # Verify pool statistics
                pool_stats = async_context["pool_stats"]
                assert pool_stats["active_connections"] == 8
                assert pool_stats["max_connections"] == 10
                
                # Verify business context
                business_context = e.details["business_context"]
                assert business_context["operation"] == "process_pending_orders"
                assert business_context["batch_size"] == 100
                assert business_context["orders_processed"] == 0
                
                # Verify operation tracking
                assert e.details["operation_id"] == tracker.operation_id
                assert e.details["trace_id"] == tracker.trace_id
                
                return True
            
            return False
        
        # Run async test
        result = asyncio.run(test_async_context_tracking())
        assert result is True


class TestDistributedErrorTracking:
    """Test error tracking across distributed components."""
    
    def test_microservice_error_correlation_tracking(self):
        """Test error correlation tracking across microservices."""
        # Simulate distributed operation across multiple services
        class DistributedOperationContext:
            def __init__(self):
                self.correlation_id = str(uuid.uuid4())
                self.span_id = str(uuid.uuid4())
                self.service_call_chain = []
            
            def add_service_call(self, service_name: str, operation: str, **kwargs):
                self.service_call_chain.append({
                    "service": service_name,
                    "operation": operation,
                    "span_id": str(uuid.uuid4()),
                    "timestamp": time.time(),
                    "metadata": kwargs
                })
            
            def create_distributed_exception(self, exception_class, message: str,
                                           error_code: str, service_details: Dict[str, Any]):
                """Create exception with distributed tracing context."""
                distributed_context = {
                    "correlation_id": self.correlation_id,
                    "root_span_id": self.span_id,
                    "service_call_chain": self.service_call_chain.copy(),
                    "distributed_trace": {
                        "total_services": len(self.service_call_chain),
                        "call_duration_ms": (time.time() - self.service_call_chain[0]["timestamp"]) * 1000 if self.service_call_chain else 0
                    }
                }
                distributed_context.update(service_details)
                
                return exception_class(message, error_code, distributed_context)
        
        # Simulate service call chain
        def order_processing_service():
            """Entry point service."""
            context = DistributedOperationContext()
            context.add_service_call("order_service", "process_order", order_id="ord_123")
            
            try:
                payment_processing_service(context)
            except ExternalServiceException as e:
                # Add order service context
                e.details.update({
                    "order_service_context": {
                        "order_id": "ord_123",
                        "customer_id": "cust_456",
                        "amount": 299.99,
                        "currency": "USD",
                        "payment_method": "credit_card"
                    }
                })
                raise
        
        def payment_processing_service(context: DistributedOperationContext):
            """Payment processing service."""
            context.add_service_call("payment_service", "process_payment", 
                                   payment_id="pay_789")
            
            try:
                external_payment_gateway_service(context)
            except TimeoutException as e:
                # Transform timeout to external service exception
                raise context.create_distributed_exception(
                    ExternalServiceException,
                    "Payment gateway timeout",
                    "PAYMENT_GATEWAY_TIMEOUT",
                    {
                        "payment_service_context": {
                            "payment_id": "pay_789",
                            "gateway": "stripe",
                            "amount": 299.99,
                            "attempt": 1,
                            "retry_policy": "exponential_backoff"
                        },
                        "original_timeout": {
                            "type": "TimeoutException",
                            "message": e.message,
                            "code": e.code,
                            "details": e.details
                        }
                    }
                )
        
        def external_payment_gateway_service(context: DistributedOperationContext):
            """External payment gateway service."""
            context.add_service_call("payment_gateway", "charge_card",
                                   gateway="stripe", timeout=30)
            
            # Simulate timeout
            raise context.create_distributed_exception(
                TimeoutException,
                "Payment gateway request timeout",
                "GATEWAY_TIMEOUT_001",
                {
                    "gateway_context": {
                        "provider": "stripe",
                        "endpoint": "/v1/charges",
                        "timeout_seconds": 30,
                        "actual_duration_seconds": 31,
                        "http_status": None,
                        "retry_attempt": 1
                    }
                }
            )
        
        # Execute distributed operation
        try:
            order_processing_service()
            assert False, "Should have raised ExternalServiceException"
        
        except ExternalServiceException as e:
            # Verify distributed error correlation
            assert e.code == "PAYMENT_GATEWAY_TIMEOUT"
            
            # Verify correlation tracking
            assert "correlation_id" in e.details
            assert "root_span_id" in e.details
            assert "service_call_chain" in e.details
            
            # Verify service call chain
            call_chain = e.details["service_call_chain"]
            assert len(call_chain) == 3
            
            services = [call["service"] for call in call_chain]
            assert "order_service" in services
            assert "payment_service" in services
            assert "payment_gateway" in services
            
            # Verify order service context
            order_context = e.details["order_service_context"]
            assert order_context["order_id"] == "ord_123"
            assert order_context["amount"] == 299.99
            
            # Verify payment service context
            payment_context = e.details["payment_service_context"]
            assert payment_context["payment_id"] == "pay_789"
            assert payment_context["gateway"] == "stripe"
            
            # Verify original timeout context
            original_timeout = e.details["original_timeout"]
            assert original_timeout["type"] == "TimeoutException"
            assert original_timeout["code"] == "GATEWAY_TIMEOUT_001"
            
            # Verify gateway context in original error
            original_details = original_timeout["details"]
            gateway_context = original_details["gateway_context"]
            assert gateway_context["provider"] == "stripe"
            assert gateway_context["timeout_seconds"] == 30
            assert gateway_context["actual_duration_seconds"] == 31

    def test_complex_error_aggregation_and_correlation(self):
        """Test complex error aggregation and correlation across multiple failure points."""
        class ErrorAggregator:
            def __init__(self):
                self.errors = []
                self.correlation_id = str(uuid.uuid4())
            
            def add_error(self, error: CoreException, component: str, operation: str):
                """Add error to aggregation with component context."""
                error_info = {
                    "component": component,
                    "operation": operation,
                    "error_type": type(error).__name__,
                    "error_code": error.code,
                    "error_message": error.message,
                    "error_details": error.details,
                    "timestamp": time.time()
                }
                self.errors.append(error_info)
            
            def create_aggregated_exception(self):
                """Create aggregated exception with all error context."""
                return BusinessLogicException(
                    "Multiple component failures detected",
                    "AGGREGATE_FAILURE_001",
                    {
                        "correlation_id": self.correlation_id,
                        "total_errors": len(self.errors),
                        "error_summary": {
                            "components_affected": list(set(e["component"] for e in self.errors)),
                            "error_types": list(set(e["error_type"] for e in self.errors)),
                            "first_error_time": min(e["timestamp"] for e in self.errors) if self.errors else None,
                            "last_error_time": max(e["timestamp"] for e in self.errors) if self.errors else None
                        },
                        "detailed_errors": self.errors
                    }
                )
        
        # Simulate multiple component failures
        aggregator = ErrorAggregator()
        
        # Component 1 failure
        try:
            raise ValidationException(
                "Invalid input parameters",
                "VAL_001",
                {"field": "amount", "value": -100, "constraint": "positive"}
            )
        except ValidationException as e:
            aggregator.add_error(e, "validation_service", "validate_order_params")
        
        # Component 2 failure
        try:
            raise ExternalServiceException(
                "Market data service unavailable",
                "MARKET_001",
                {"service": "market_data", "endpoint": "/prices", "status": 503}
            )
        except ExternalServiceException as e:
            aggregator.add_error(e, "market_data_service", "get_current_price")
        
        # Component 3 failure
        try:
            raise DataNotFoundException(
                "User account not found",
                "ACCOUNT_001",
                {"user_id": "user_123", "account_type": "trading"}
            )
        except DataNotFoundException as e:
            aggregator.add_error(e, "account_service", "get_user_account")
        
        # Create aggregated exception
        aggregated_error = aggregator.create_aggregated_exception()
        
        # Verify error aggregation
        assert aggregated_error.code == "AGGREGATE_FAILURE_001"
        assert aggregated_error.details["total_errors"] == 3
        
        # Verify error summary
        error_summary = aggregated_error.details["error_summary"]
        assert len(error_summary["components_affected"]) == 3
        assert "validation_service" in error_summary["components_affected"]
        assert "market_data_service" in error_summary["components_affected"]
        assert "account_service" in error_summary["components_affected"]
        
        assert len(error_summary["error_types"]) == 3
        assert "ValidationException" in error_summary["error_types"]
        assert "ExternalServiceException" in error_summary["error_types"]
        assert "DataNotFoundException" in error_summary["error_types"]
        
        # Verify detailed errors
        detailed_errors = aggregated_error.details["detailed_errors"]
        assert len(detailed_errors) == 3
        
        # Verify each error maintains its context
        validation_error = next(e for e in detailed_errors if e["error_code"] == "VAL_001")
        assert validation_error["component"] == "validation_service"
        assert validation_error["error_details"]["field"] == "amount"
        
        market_error = next(e for e in detailed_errors if e["error_code"] == "MARKET_001")
        assert market_error["component"] == "market_data_service"
        assert market_error["error_details"]["status"] == 503
        
        account_error = next(e for e in detailed_errors if e["error_code"] == "ACCOUNT_001")
        assert account_error["component"] == "account_service"
        assert account_error["error_details"]["user_id"] == "user_123"


if __name__ == "__main__":
    pytest.main([__file__])