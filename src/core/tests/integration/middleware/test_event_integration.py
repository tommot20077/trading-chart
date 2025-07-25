# ABOUTME: Integration tests for middleware and event system interactions
# ABOUTME: Tests middleware processing flow, priority interactions, and handler collaboration

import pytest
import pytest_asyncio
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock
from datetime import datetime, UTC

from core.interfaces.event.event_bus import AbstractEventBus
from core.interfaces.middleware import AbstractMiddleware, AbstractMiddlewarePipeline
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority


def create_order_event(order_id: str, user_id: str, amount: float, **kwargs) -> BaseEvent:
    """Create a test order event."""
    kwargs.setdefault('source', 'test_system')
    kwargs.setdefault('data', {
        'order_id': order_id,
        'user_id': user_id,
        'amount': amount
    })
    # Add user_id to metadata for middleware access
    if 'metadata' not in kwargs:
        kwargs['metadata'] = {}
    kwargs['metadata']['user_id'] = user_id
    kwargs['metadata']['event_type'] = EventType.ORDER.value
    return BaseEvent(event_type=EventType.ORDER, **kwargs)


def create_trade_event(trade_id: str, symbol: str, price: float, **kwargs) -> BaseEvent:
    """Create a test trade event."""
    kwargs.setdefault('source', 'test_system')
    kwargs.setdefault('data', {
        'trade_id': trade_id,
        'symbol': symbol,
        'price': price
    })
    # Add metadata for middleware access
    if 'metadata' not in kwargs:
        kwargs['metadata'] = {}
    kwargs['metadata']['event_type'] = EventType.TRADE.value
    return BaseEvent(event_type=EventType.TRADE, **kwargs)


class SecurityMiddleware(AbstractMiddleware):
    """Security middleware for testing authentication and authorization."""
    
    def __init__(self, priority: EventPriority = EventPriority.CRITICAL):
        super().__init__(priority)
        self.processed_events = []
        self.blocked_events = []
        self.authorized_users = {"user_1", "user_2", "admin_user"}
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process security checks."""
        self.processed_events.append(context.event_id)
        
        user_id = context.metadata.get("user_id", "anonymous")
        event_type = context.metadata.get("event_type", "unknown")
        
        if user_id not in self.authorized_users:
            self.blocked_events.append(context.event_id)
            return MiddlewareResult(
                middleware_name="SecurityMiddleware",
                status=MiddlewareStatus.FAILED,
                error="Unauthorized user",
                should_continue=False,
                execution_time_ms=2.0,
                metadata={"blocked_user": user_id, "event_type": event_type}
            )
        
        # Add security context
        context.set_metadata("security_cleared", True)
        context.set_metadata("cleared_by", "SecurityMiddleware")
        
        return MiddlewareResult(
            middleware_name="SecurityMiddleware",
            status=MiddlewareStatus.SUCCESS,
            data={"user_authorized": True, "user_id": user_id},
            should_continue=True,
            execution_time_ms=1.5,
            metadata={"processed_at": datetime.now(UTC).isoformat()}
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can process all events for security checks."""
        return True


class LoggingMiddleware(AbstractMiddleware):
    """Logging middleware for audit trail."""
    
    def __init__(self, priority: EventPriority = EventPriority.HIGH):
        super().__init__(priority)
        self.logged_events = []
        self.audit_trail = []
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process logging."""
        event_id = context.event_id
        self.logged_events.append(event_id)
        
        audit_entry = {
            "event_id": event_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": context.metadata.get("user_id", "unknown"),
            "event_type": context.metadata.get("event_type", "unknown"),
            "security_cleared": context.metadata.get("security_cleared", False),
            "execution_path": context.execution_path.copy()
        }
        self.audit_trail.append(audit_entry)
        
        # Add logging metadata
        context.set_metadata("logged_at", audit_entry["timestamp"])
        context.set_metadata("audit_id", len(self.audit_trail))
        
        return MiddlewareResult(
            middleware_name="LoggingMiddleware",
            status=MiddlewareStatus.SUCCESS,
            data={"logged": True, "audit_id": len(self.audit_trail)},
            should_continue=True,
            execution_time_ms=0.5,
            metadata={"audit_entry": audit_entry}
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can process all events for logging."""
        return True


class ValidationMiddleware(AbstractMiddleware):
    """Validation middleware for business rules."""
    
    def __init__(self, priority: EventPriority = EventPriority.NORMAL):
        super().__init__(priority)
        self.validated_events = []
        self.validation_errors = []
        
        # Business rules
        self.max_order_amount = 10000.0
        self.min_trade_price = 0.01
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process validation."""
        event_id = context.event_id
        event_type = context.metadata.get("event_type")
        
        validation_errors = []
        
        # Validate order events
        if event_type == EventType.ORDER.value:
            amount = context.data.get("amount", 0)
            if amount > self.max_order_amount:
                validation_errors.append(f"Order amount {amount} exceeds maximum {self.max_order_amount}")
            if amount <= 0:
                validation_errors.append("Order amount must be positive")
        
        # Validate trade events
        elif event_type == EventType.TRADE.value:
            price = context.data.get("price", 0)
            if price < self.min_trade_price:
                validation_errors.append(f"Trade price {price} below minimum {self.min_trade_price}")
        
        if validation_errors:
            self.validation_errors.extend(validation_errors)
            return MiddlewareResult(
                middleware_name="ValidationMiddleware",
                status=MiddlewareStatus.FAILED,
                error="; ".join(validation_errors),
                should_continue=False,
                execution_time_ms=1.0,
                metadata={"validation_errors": validation_errors}
            )
        
        self.validated_events.append(event_id)
        
        # Add validation metadata
        context.set_metadata("validation_passed", True)
        context.set_metadata("validated_by", "ValidationMiddleware")
        
        return MiddlewareResult(
            middleware_name="ValidationMiddleware",
            status=MiddlewareStatus.SUCCESS,
            data={"validation_passed": True},
            should_continue=True,
            execution_time_ms=0.8,
            metadata={"validation_rules_applied": ["amount_check", "price_check"]}
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can process order and trade events."""
        event_type = context.metadata.get("event_type")
        return event_type in [EventType.ORDER.value, EventType.TRADE.value]


class TestEventMiddlewareIntegration:
    """Integration tests for event-middleware system interactions."""
    
    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create event bus with middleware pipeline."""
        bus = InMemoryEventBus()
        pipeline = InMemoryMiddlewarePipeline("TestEventPipeline")
        
        # Set up middleware pipeline in event bus
        await bus.set_middleware_pipeline(pipeline)
        
        return bus
    
    @pytest.fixture
    def security_middleware(self):
        """Create security middleware."""
        return SecurityMiddleware()
    
    @pytest.fixture
    def logging_middleware(self):
        """Create logging middleware."""
        return LoggingMiddleware()
    
    @pytest.fixture
    def validation_middleware(self):
        """Create validation middleware."""
        return ValidationMiddleware()
    
    @pytest.mark.asyncio 
    async def test_event_publishing_middleware_flow(self, event_bus, security_middleware, logging_middleware):
        """Test middleware processing flow during event publishing."""
        # Add middleware to pipeline
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(security_middleware)
        await pipeline.add_middleware(logging_middleware)
        
        # Create test event
        event = create_order_event(
            order_id="order_123",
            user_id="user_1", 
            amount=1000.0,
            priority=EventPriority.NORMAL
        )
        
        # Mock event handler
        handler = AsyncMock()
        event_bus.subscribe(EventType.ORDER, handler)
        
        # Publish event
        await event_bus.publish(event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify middleware execution order (CRITICAL -> HIGH priority)
        assert len(security_middleware.processed_events) == 1
        assert len(logging_middleware.logged_events) == 1
        
        # Verify security middleware processed first (higher priority)
        audit_entry = logging_middleware.audit_trail[0]
        assert audit_entry["security_cleared"] is True
        # LoggingMiddleware sees what executed before it (SecurityMiddleware)
        assert audit_entry["execution_path"] == ["SecurityMiddleware"]
        
        # Verify handler was called
        handler.assert_called_once()
        
        # Verify event data was enriched by middleware
        call_args = handler.call_args[0][0]
        assert call_args.data['order_id'] == "order_123"
        assert call_args.data['user_id'] == "user_1"
    
    @pytest.mark.asyncio
    async def test_middleware_priority_vs_event_priority_interaction(self, event_bus):
        """Test interaction between middleware priority and event priority."""
        # Create middleware with different priorities
        critical_middleware = SecurityMiddleware(EventPriority.CRITICAL)
        high_middleware = LoggingMiddleware(EventPriority.HIGH) 
        normal_middleware = ValidationMiddleware(EventPriority.NORMAL)
        
        # Add middleware in reverse priority order
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(normal_middleware)
        await pipeline.add_middleware(high_middleware)
        await pipeline.add_middleware(critical_middleware)
        
        # Create events with different priorities
        high_priority_event = create_order_event(
            order_id="high_order",
            user_id="user_1",
            amount=500.0,
            priority=EventPriority.HIGH
        )
        
        low_priority_event = create_trade_event(
            trade_id="low_trade", 
            symbol="BTC/USD",
            price=50000.0,
            priority=EventPriority.LOW,
            metadata={'user_id': 'user_1'}  # Add user metadata for security middleware
        )
        
        # Mock handlers
        order_handler = AsyncMock()
        trade_handler = AsyncMock()
        event_bus.subscribe(EventType.ORDER, order_handler)
        event_bus.subscribe(EventType.TRADE, trade_handler)
        
        # Publish events
        await event_bus.publish(high_priority_event)
        await event_bus.publish(low_priority_event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify middleware executed in priority order regardless of event priority
        # Check execution path in audit trail
        order_audit = high_middleware.audit_trail[0]
        trade_audit = high_middleware.audit_trail[1]
        
        # Both events should have SecurityMiddleware executed before LoggingMiddleware  
        # LoggingMiddleware sees only what executed before it (SecurityMiddleware)
        assert order_audit["execution_path"] == ["SecurityMiddleware"]
        assert trade_audit["execution_path"] == ["SecurityMiddleware"]
        
        # Verify all middleware processed both events
        assert len(critical_middleware.processed_events) == 2
        assert len(high_middleware.logged_events) == 2
        assert len(normal_middleware.validated_events) == 2  # Both events are processable
    
    @pytest.mark.asyncio
    async def test_event_handler_middleware_collaboration(self, event_bus, security_middleware, validation_middleware):
        """Test collaboration between event handlers and middleware."""
        # Add middleware
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(security_middleware)
        await pipeline.add_middleware(validation_middleware)
        
        # Create event handler that depends on middleware context
        processed_events = []
        
        async def context_aware_handler(event):
            """Handler that uses middleware-enriched context."""
            processed_events.append({
                "event_id": event.event_id,
                "order_id": event.data.get('order_id'),
                "user_id": event.data.get('user_id'),
                "amount": event.data.get('amount'),
                # The event passes through middleware, so we can verify it was processed
                "event_metadata": event.metadata
            })
        
        event_bus.subscribe(EventType.ORDER, context_aware_handler)
        
        # Test valid event
        valid_event = create_order_event(
            order_id="valid_order",
            user_id="user_1",
            amount=500.0
        )
        
        await event_bus.publish(valid_event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify handler received middleware-processed event
        assert len(processed_events) == 1
        processed = processed_events[0]
        assert processed["order_id"] == "valid_order"
        assert processed["user_id"] == "user_1"
        assert processed["amount"] == 500.0
        
        # Verify middleware processing
        assert len(security_middleware.processed_events) == 1
        assert len(validation_middleware.validated_events) == 1
        assert len(security_middleware.blocked_events) == 0
        assert len(validation_middleware.validation_errors) == 0
    
    @pytest.mark.asyncio
    async def test_middleware_blocks_invalid_events(self, event_bus, security_middleware, validation_middleware):
        """Test middleware blocking invalid events from reaching handlers."""
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(security_middleware)
        await pipeline.add_middleware(validation_middleware)
        
        handler_calls = []
        
        async def test_handler(event):
            handler_calls.append(event.id)
        
        event_bus.subscribe(EventType.ORDER, test_handler)
        
        # Test unauthorized user event
        unauthorized_event = create_order_event(
            order_id="unauthorized_order",
            user_id="unauthorized_user",
            amount=1000.0
        )
        
        await event_bus.publish(unauthorized_event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Handler should not be called due to security middleware blocking
        assert len(handler_calls) == 0
        assert len(security_middleware.blocked_events) == 1
        
        # Test validation error event
        invalid_event = create_order_event(
            order_id="invalid_order", 
            user_id="user_1",
            amount=20000.0  # Exceeds max amount
        )
        
        await event_bus.publish(invalid_event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Handler should not be called due to validation middleware blocking
        assert len(handler_calls) == 0
        assert len(validation_middleware.validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_middleware_error_handling_in_event_flow(self, event_bus):
        """Test error handling when middleware fails during event processing."""
        
        class FailingMiddleware(AbstractMiddleware):
            def __init__(self):
                super().__init__(EventPriority.HIGH)
                self.failure_count = 0
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                self.failure_count += 1
                raise ValueError("Middleware processing failed")
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        failing_middleware = FailingMiddleware()
        logging_middleware = LoggingMiddleware(EventPriority.LOW)
        
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(failing_middleware) 
        await pipeline.add_middleware(logging_middleware)
        
        handler_calls = []
        
        async def test_handler(event):
            handler_calls.append(event.id)
        
        event_bus.subscribe(EventType.ORDER, test_handler)
        
        # Publish event
        event = create_order_event(
            order_id="test_order",
            user_id="user_1", 
            amount=1000.0
        )
        
        await event_bus.publish(event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify failing middleware was called
        assert failing_middleware.failure_count == 1
        
        # Verify subsequent middleware was not called due to failure
        assert len(logging_middleware.logged_events) == 0
        
        # Verify handler was not called due to middleware failure
        assert len(handler_calls) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_event_types_middleware_processing(self, event_bus, security_middleware, validation_middleware):
        """Test middleware processing with multiple event types."""
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(security_middleware)
        await pipeline.add_middleware(validation_middleware)
        
        order_handler_calls = []
        trade_handler_calls = []
        
        async def order_handler(event):
            order_handler_calls.append(event.event_id)
        
        async def trade_handler(event):
            trade_handler_calls.append(event.event_id)
        
        event_bus.subscribe(EventType.ORDER, order_handler)
        event_bus.subscribe(EventType.TRADE, trade_handler)
        
        # Publish different event types
        order_event = create_order_event(
            order_id="order_1",
            user_id="user_1",
            amount=1000.0
        )
        
        trade_event = create_trade_event(
            trade_id="trade_1",
            symbol="BTC/USD", 
            price=50000.0,
            metadata={'user_id': 'user_1'}  # Add user metadata for security middleware
        )
        
        await event_bus.publish(order_event)
        await event_bus.publish(trade_event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify both events were processed by security middleware
        assert len(security_middleware.processed_events) == 2
        
        # Verify validation middleware processed both (both are processable event types)
        assert len(validation_middleware.validated_events) == 2
        
        # Verify handlers were called
        assert len(order_handler_calls) == 1
        assert len(trade_handler_calls) == 1
    
    @pytest.mark.asyncio
    async def test_middleware_context_data_propagation(self, event_bus):
        """Test that middleware context data is properly propagated through the pipeline."""
        
        class ContextEnrichingMiddleware(AbstractMiddleware):
            def __init__(self, name: str, priority: EventPriority):
                super().__init__(priority)
                self.name = name
                self.processed_contexts = []
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                # Store context snapshot
                self.processed_contexts.append({
                    "event_id": context.event_id,
                    "data": dict(context.data),
                    "metadata": dict(context.metadata),
                    "execution_path": context.execution_path.copy()
                })
                
                # Enrich context
                context.set_data(f"{self.name}_processed", True)
                context.set_metadata(f"{self.name}_timestamp", datetime.now(UTC).isoformat())
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={f"{self.name}_data": "processed"},
                    should_continue=True,
                    execution_time_ms=1.0
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Create middleware chain
        middleware_a = ContextEnrichingMiddleware("MiddlewareA", EventPriority.HIGH)
        middleware_b = ContextEnrichingMiddleware("MiddlewareB", EventPriority.NORMAL)
        middleware_c = ContextEnrichingMiddleware("MiddlewareC", EventPriority.LOW)
        
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(middleware_a)
        await pipeline.add_middleware(middleware_b) 
        await pipeline.add_middleware(middleware_c)
        
        # Publish event
        event = create_order_event(
            order_id="context_test",
            user_id="user_1",
            amount=1000.0
        )
        
        await event_bus.publish(event)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify context data propagation
        # MiddlewareA should see empty execution path
        context_a = middleware_a.processed_contexts[0]
        assert context_a["execution_path"] == []
        
        # MiddlewareB should see MiddlewareA in execution path
        context_b = middleware_b.processed_contexts[0]
        assert "MiddlewareA" in context_b["execution_path"]
        assert context_b["data"]["MiddlewareA_processed"] is True
        assert "MiddlewareA_timestamp" in context_b["metadata"]
        
        # MiddlewareC should see both previous middleware in execution path and their data
        context_c = middleware_c.processed_contexts[0]
        assert context_c["execution_path"] == ["MiddlewareA", "MiddlewareB"]
        assert context_c["data"]["MiddlewareA_processed"] is True
        assert context_c["data"]["MiddlewareB_processed"] is True
        assert "MiddlewareA_timestamp" in context_c["metadata"]
        assert "MiddlewareB_timestamp" in context_c["metadata"]
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing_with_middleware(self, event_bus, security_middleware, logging_middleware):
        """Test concurrent event processing through middleware pipeline."""
        pipeline = await event_bus.get_middleware_pipeline()
        await pipeline.add_middleware(security_middleware)
        await pipeline.add_middleware(logging_middleware)
        
        handler_calls = []
        
        async def concurrent_handler(event):
            handler_calls.append(event.event_id)
        
        event_bus.subscribe(EventType.ORDER, concurrent_handler)
        
        # Create multiple events
        events = []
        for i in range(10):
            event = create_order_event(
                order_id=f"concurrent_order_{i}",
                user_id="user_1",
                amount=1000.0 + i * 100
            )
            events.append(event)
        
        # Publish events concurrently
        tasks = []
        for event in events:
            task = asyncio.create_task(event_bus.publish(event))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Wait for event processing to complete
        await event_bus.flush_queue(timeout=5.0)
        
        # Verify all events were processed
        assert len(security_middleware.processed_events) == 10
        assert len(logging_middleware.logged_events) == 10
        assert len(handler_calls) == 10
        
        # Verify no events were blocked
        assert len(security_middleware.blocked_events) == 0
        
        # Verify audit trail integrity
        assert len(logging_middleware.audit_trail) == 10
        for entry in logging_middleware.audit_trail:
            assert entry["security_cleared"] is True
            assert entry["user_id"] == "user_1"