# ABOUTME: Cross-module collaboration integration tests
# ABOUTME: Tests integration between different system modules, data flow validation, and end-to-end processes

import pytest
import pytest_asyncio
import asyncio
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock
from datetime import datetime, UTC
from uuid import uuid4, UUID

# Core system imports
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.implementations.memory.auth.token_manager import InMemoryTokenManager
from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter
from core.implementations.noop.data.provider import NoOpDataProvider

# Models and interfaces
from core.models.data.event import BaseEvent
from core.models.data.market_data import MarketData
from core.models.data.order import Order
from core.models.data.trading_pair import TradingPair
from core.models.event.event_type import EventType
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from core.interfaces.middleware import AbstractMiddleware
from core.interfaces.auth.token_manager import AbstractTokenManager
from core.interfaces.common.rate_limiter import AbstractRateLimiter

# Data models enums
from core.models.data.order_enums import OrderType, OrderSide, OrderStatus


class TradingSystemMiddleware(AbstractMiddleware):
    """Middleware that simulates trading system operations."""
    
    def __init__(self, token_manager: AbstractTokenManager, rate_limiter: AbstractRateLimiter):
        super().__init__(EventPriority.HIGH)
        self.token_manager = token_manager
        self.rate_limiter = rate_limiter
        self.processed_orders = []
        self.processed_market_data = []
        self.authentication_failures = []
        self.rate_limit_violations = []
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process trading system operations with auth and rate limiting."""
        user_id = context.metadata.get("user_id", "anonymous")
        event_type = context.metadata.get("event_type")
        
        # Authentication check
        token = context.metadata.get("auth_token")
        if token:
            try:
                auth_token = self.token_manager.validate_token(token)
                if auth_token.is_expired():
                    self.authentication_failures.append({
                        "user_id": user_id,
                        "token": token,
                        "reason": "expired_token",
                        "timestamp": datetime.now(UTC)
                    })
                    return MiddlewareResult(
                        middleware_name="TradingSystemMiddleware",
                        status=MiddlewareStatus.FAILED,
                        error="Authentication failed - token expired",
                        should_continue=False,
                        execution_time_ms=1.0
                    )
                context.metadata["authenticated_user"] = auth_token.user_id
            except Exception as e:
                self.authentication_failures.append({
                    "user_id": user_id,
                    "token": token,
                    "reason": "invalid_token",
                    "error": str(e),
                    "timestamp": datetime.now(UTC)
                })
                return MiddlewareResult(
                    middleware_name="TradingSystemMiddleware",
                    status=MiddlewareStatus.FAILED,
                    error="Authentication failed",
                    should_continue=False,
                    execution_time_ms=1.0
                )
        
        # Rate limiting check
        try:
            rate_limit_key = f"trading_operations:{user_id}"
            can_proceed = await self.rate_limiter.acquire_for_identifier(rate_limit_key)
            if not can_proceed:
                self.rate_limit_violations.append({
                    "user_id": user_id,
                    "key": rate_limit_key,
                    "timestamp": datetime.now(UTC)
                })
                return MiddlewareResult(
                    middleware_name="TradingSystemMiddleware",
                    status=MiddlewareStatus.FAILED,
                    error="Rate limit exceeded",
                    should_continue=False,
                    execution_time_ms=0.5
                )
        except Exception as e:
            return MiddlewareResult(
                middleware_name="TradingSystemMiddleware",
                status=MiddlewareStatus.FAILED,
                error=f"Rate limiting error: {str(e)}",
                should_continue=False,
                execution_time_ms=0.5
            )
        
        # Process different event types
        if event_type == EventType.ORDER.value:
            await self._process_order(context)
        elif event_type == EventType.MARKET_DATA.value:
            await self._process_market_data(context)
        
        context.metadata["trading_system_processed"] = True
        context.metadata["processing_timestamp"] = datetime.now(UTC).isoformat()
        
        return MiddlewareResult(
            middleware_name="TradingSystemMiddleware",
            status=MiddlewareStatus.SUCCESS,
            data={"trading_system_validation": "passed"},
            should_continue=True,
            execution_time_ms=2.0,
            metadata={"authenticated": token is not None}
        )
    
    async def _process_order(self, context: MiddlewareContext) -> None:
        """Process order-specific logic."""
        # Get order data from context data (middleware context unpacks event data)
        order_data = context.data.get("order")
        if order_data:
            self.processed_orders.append({
                "order_id": order_data.get("order_id"),
                "user_id": context.metadata.get("user_id"),
                "order_type": order_data.get("order_type"),
                "timestamp": datetime.now(UTC)
            })
    
    async def _process_market_data(self, context: MiddlewareContext) -> None:
        """Process market data specific logic."""
        # Get market data from context data (middleware context unpacks event data)
        market_data = context.data.get("market_data")
        if market_data:
            self.processed_market_data.append({
                "symbol": market_data.get("symbol"),
                "price": market_data.get("price"),
                "timestamp": datetime.now(UTC)
            })
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can process order and market data events."""
        event_type = context.metadata.get("event_type")
        return event_type in [EventType.ORDER.value, EventType.MARKET_DATA.value]


class DataValidationMiddleware(AbstractMiddleware):
    """Middleware that validates data model integrity across modules."""
    
    def __init__(self):
        super().__init__(EventPriority.CRITICAL)
        self.validation_results = []
        self.model_validations = []
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Validate data models and cross-module consistency."""
        event_type = context.metadata.get("event_type")
        validation_errors = []
        
        if event_type == EventType.ORDER.value:
            validation_errors.extend(await self._validate_order_data(context))
        elif event_type == EventType.MARKET_DATA.value:
            validation_errors.extend(await self._validate_market_data(context))
        elif event_type == EventType.TRADE.value:
            validation_errors.extend(await self._validate_trade_data(context))
        
        validation_result = {
            "context_id": context.id,
            "event_type": event_type,
            "validation_errors": validation_errors,
            "is_valid": len(validation_errors) == 0,
            "timestamp": datetime.now(UTC)
        }
        self.validation_results.append(validation_result)
        
        if validation_errors:
            return MiddlewareResult(
                middleware_name="DataValidationMiddleware",
                status=MiddlewareStatus.FAILED,
                error=f"Validation failed: {'; '.join(validation_errors)}",
                should_continue=False,
                execution_time_ms=1.5,
                metadata={"validation_errors": validation_errors}
            )
        
        context.metadata["data_validation_passed"] = True
        return MiddlewareResult(
            middleware_name="DataValidationMiddleware",
            status=MiddlewareStatus.SUCCESS,
            data={"validation_status": "passed"},
            should_continue=True,
            execution_time_ms=1.5
        )
    
    async def _validate_order_data(self, context: MiddlewareContext) -> List[str]:
        """Validate order data model."""
        errors = []
        # Get order data from context data (middleware context unpacks event data)
        order_data = context.data.get("order")
        
        if not order_data:
            errors.append("Missing order data")
            return errors
        
        try:
            # Validate Order model
            order = order_data if isinstance(order_data, Order) else Order(**order_data)
            self.model_validations.append({
                "model_type": "Order",
                "order_id": order.order_id,
                "is_valid": True,
                "timestamp": datetime.now(UTC)
            })
            
            # Validate related TradingPair (trading_pair is now a string in Order model)
            if hasattr(order, 'trading_pair') and order.trading_pair:
                trading_pair_str = order.trading_pair
                if isinstance(trading_pair_str, str):
                    # For string trading pairs, we can't validate TradingPair model
                    # but we can record that the trading pair format is valid
                    self.model_validations.append({
                        "model_type": "TradingPair",
                        "symbol": trading_pair_str,
                        "is_valid": True,
                        "timestamp": datetime.now(UTC)
                    })
                elif isinstance(trading_pair_str, dict):
                    # If it's still a dict (legacy), create TradingPair object
                    trading_pair = TradingPair(**trading_pair_str)
                    self.model_validations.append({
                        "model_type": "TradingPair",
                        "symbol": trading_pair.symbol,
                        "is_valid": True,
                        "timestamp": datetime.now(UTC)
                    })
        
        except Exception as e:
            errors.append(f"Order validation error: {str(e)}")
            self.model_validations.append({
                "model_type": "Order",
                "is_valid": False,
                "error": str(e),
                "timestamp": datetime.now(UTC)
            })
        
        return errors
    
    async def _validate_market_data(self, context: MiddlewareContext) -> List[str]:
        """Validate market data model."""
        errors = []
        # Get market data from context data (middleware context unpacks event data)
        market_data = context.data.get("market_data")
        
        if not market_data:
            errors.append("Missing market data")
            return errors
        
        try:
            # For simple dict-based market data, just validate basic structure
            if isinstance(market_data, dict):
                symbol = market_data.get("symbol")
                if not symbol:
                    errors.append("Missing symbol in market data")
                else:
                    self.model_validations.append({
                        "model_type": "MarketData",
                        "symbol": symbol,
                        "is_valid": True,
                        "timestamp": datetime.now(UTC)
                    })
            else:
                # Try to validate as MarketData model
                data = market_data if isinstance(market_data, MarketData) else MarketData(**market_data)
                self.model_validations.append({
                    "model_type": "MarketData",
                    "symbol": data.symbol,
                    "is_valid": True,
                    "timestamp": datetime.now(UTC)
                })
        except Exception as e:
            errors.append(f"Market data validation error: {str(e)}")
            self.model_validations.append({
                "model_type": "MarketData",
                "is_valid": False,
                "error": str(e),
                "timestamp": datetime.now(UTC)
            })
        
        return errors
    
    async def _validate_trade_data(self, context: MiddlewareContext) -> List[str]:
        """Validate trade data and related models."""
        errors = []
        # Get trade data from context data (middleware context unpacks event data)
        trade_data = context.data.get("trade")
        
        if not trade_data:
            errors.append("Missing trade data")
            return errors
        
        # Validate trade data structure
        required_fields = ["trade_id", "symbol", "price", "quantity", "timestamp"]
        for field in required_fields:
            if field not in trade_data:
                errors.append(f"Missing required trade field: {field}")
        
        return errors
    
    def can_process(self, context: MiddlewareContext) -> bool:
        """Can validate all event types."""
        return True


class TestCrossModuleCollaboration:
    """Integration tests for cross-module collaboration."""
    
    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create event bus with middleware pipeline."""
        bus = InMemoryEventBus()
        pipeline = InMemoryMiddlewarePipeline("CrossModuleTestPipeline")
        await bus.set_middleware_pipeline(pipeline)
        return bus
    
    @pytest_asyncio.fixture
    async def token_manager(self):
        """Create token manager for testing."""
        return InMemoryTokenManager()
    
    @pytest_asyncio.fixture
    async def rate_limiter(self):
        """Create rate limiter for testing."""
        return InMemoryRateLimiter(capacity=5, refill_rate=1.0)
    
    @pytest_asyncio.fixture
    async def data_provider(self):
        """Create data provider for testing."""
        return NoOpDataProvider()
    
    @pytest_asyncio.fixture
    async def integrated_system(self, event_bus, token_manager, rate_limiter):
        """Create fully integrated system with all components."""
        # Set up middleware pipeline
        pipeline = await event_bus.get_middleware_pipeline()
        
        data_validation_middleware = DataValidationMiddleware()
        trading_system_middleware = TradingSystemMiddleware(token_manager, rate_limiter)
        
        await pipeline.add_middleware(data_validation_middleware)
        await pipeline.add_middleware(trading_system_middleware)
        
        return {
            "event_bus": event_bus,
            "token_manager": token_manager,
            "rate_limiter": rate_limiter,
            "data_validation": data_validation_middleware,
            "trading_system": trading_system_middleware
        }
    
    @pytest.mark.asyncio
    async def test_order_processing_integration(self, integrated_system):
        """Test end-to-end order processing with all system components."""
        system = integrated_system
        event_bus = system["event_bus"]
        token_manager = system["token_manager"]
        
        # Create user authentication token
        auth_token = token_manager.generate_token({
            "user_id": "12345678-1234-5678-9012-123456789012",
            "username": "test_trader_001"
        })
        
        # Create order event with proper models
        order = Order(
            order_id=uuid4(),
            user_id=UUID("12345678-1234-5678-9012-123456789012"),
            trading_pair="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=1.5,
            price=50000.0,
            status=OrderStatus.PENDING
        )
        
        # Create event
        order_event = BaseEvent(
            event_type=EventType.ORDER,
            source="trading_client",
            data={"order": order.model_dump()},
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789012",
                "event_type": EventType.ORDER.value,
                "auth_token": auth_token
            }
        )
        
        # Set up event handler
        processed_events = []
        
        async def order_handler(event):
            processed_events.append({
                "event_id": event.event_id,
                "order_id": event.data["order"]["order_id"],
                "user_id": str(event.data["order"]["user_id"]),
                "processed_at": datetime.now(UTC)
            })
        
        event_bus.subscribe(EventType.ORDER, order_handler)
        
        # Process event through integrated system
        await event_bus.publish(order_event)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.1)
        
        # Verify all components processed the event
        data_validation = system["data_validation"]
        trading_system = system["trading_system"]
        
        # Verify data validation
        assert len(data_validation.validation_results) == 1
        validation_result = data_validation.validation_results[0]
        assert validation_result["is_valid"] is True
        assert validation_result["event_type"] == EventType.ORDER.value
        
        # Verify model validations
        assert len(data_validation.model_validations) >= 2  # Order and TradingPair
        order_validation = next(
            (v for v in data_validation.model_validations if v["model_type"] == "Order"), 
            None
        )
        assert order_validation is not None
        assert order_validation["is_valid"] is True
        
        # Verify trading system processing
        assert len(trading_system.processed_orders) == 1
        processed_order = trading_system.processed_orders[0]
        assert str(processed_order["order_id"]) == str(order.order_id)
        assert processed_order["user_id"] == "12345678-1234-5678-9012-123456789012"
        
        # Verify authentication succeeded
        assert len(trading_system.authentication_failures) == 0
        
        # Verify rate limiting worked
        assert len(trading_system.rate_limit_violations) == 0
        
        # Verify event handler was called
        assert len(processed_events) == 1
        assert str(processed_events[0]["order_id"]) == str(order.order_id)
    
    @pytest.mark.asyncio
    async def test_market_data_processing_integration(self, integrated_system):
        """Test market data processing across all system modules."""
        system = integrated_system
        event_bus = system["event_bus"]
        
        # Create market data event (using dict instead of MarketData model)
        market_data_dict = {
            "symbol": "ETHUSDT",
            "price": 3000.50,
            "volume": 1500.0,
            "high_24h": 3100.0,
            "low_24h": 2950.0,
            "change_24h": 2.5
        }
        
        market_event = BaseEvent(
            event_type=EventType.MARKET_DATA,
            source="market_feed",
            data={"market_data": market_data_dict},
            metadata={
                "user_id": "market_service",
                "event_type": EventType.MARKET_DATA.value
            }
        )
        
        # Set up event handler
        processed_events = []
        
        async def market_data_handler(event):
            processed_events.append({
                "event_id": event.event_id,
                "symbol": event.data["market_data"]["symbol"],
                "price": event.data["market_data"]["price"],
                "processed_at": datetime.now(UTC)
            })
        
        event_bus.subscribe(EventType.MARKET_DATA, market_data_handler)
        
        # Process event
        await event_bus.publish(market_event)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.1)
        
        # Verify processing
        data_validation = system["data_validation"]
        trading_system = system["trading_system"]
        
        # Verify data validation
        assert len(data_validation.validation_results) == 1
        validation_result = data_validation.validation_results[0]
        assert validation_result["is_valid"] is True
        assert validation_result["event_type"] == EventType.MARKET_DATA.value
        
        # Verify MarketData model validation
        market_data_validation = next(
            (v for v in data_validation.model_validations if v["model_type"] == "MarketData"),
            None
        )
        assert market_data_validation is not None
        assert market_data_validation["is_valid"] is True
        assert market_data_validation["symbol"] == "ETHUSDT"
        
        # Verify trading system processed market data
        assert len(trading_system.processed_market_data) == 1
        processed_data = trading_system.processed_market_data[0]
        assert processed_data["symbol"] == "ETHUSDT"
        assert processed_data["price"] == 3000.50
        
        # Verify event handler was called
        assert len(processed_events) == 1
        assert processed_events[0]["symbol"] == "ETHUSDT"
        assert processed_events[0]["price"] == 3000.50
    
    @pytest.mark.asyncio
    async def test_authentication_and_authorization_flow(self, integrated_system):
        """Test authentication and authorization across system modules."""
        system = integrated_system
        event_bus = system["event_bus"]
        token_manager = system["token_manager"]
        
        # Test 1: Valid authentication
        valid_token = token_manager.generate_token({
            "user_id": "authorized_user",
            "username": "authorized_user"
        })
        
        order_event_valid = BaseEvent(
            event_type=EventType.ORDER,
            source="trading_client",
            data={
                "order": {
                    "order_id": str(uuid4()),
                    "user_id": "12345678-1234-5678-9012-123456789012",
                    "trading_pair": "BTC/USDT",
                    "order_type": "limit",
                    "side": "buy",
                    "quantity": 0.1,
                    "price": 50000.0,
                    "status": "pending"
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789012",
                "event_type": EventType.ORDER.value,
                "auth_token": valid_token
            }
        )
        
        valid_events = []
        
        async def valid_handler(event):
            valid_events.append(event.event_id)
        
        event_bus.subscribe(EventType.ORDER, valid_handler)
        
        # Process valid event
        await event_bus.publish(order_event_valid)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.1)
        
        # Verify valid authentication succeeded
        trading_system = system["trading_system"]
        assert len(trading_system.authentication_failures) == 0
        assert len(valid_events) == 1
        
        # Test 2: Invalid authentication
        order_event_invalid = BaseEvent(
            event_type=EventType.ORDER,
            source="trading_client",
            data={
                "order": {
                    "order_id": str(uuid4()),
                    "user_id": "12345678-1234-5678-9012-123456789013",
                    "trading_pair": "BTC/USDT",
                    "order_type": "limit",
                    "side": "buy",
                    "quantity": 0.1,
                    "price": 50000.0,
                    "status": "pending"
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789013",
                "event_type": EventType.ORDER.value,
                "auth_token": "invalid_token_12345"
            }
        )
        
        # Process invalid event
        await event_bus.publish(order_event_invalid)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.1)
        
        # Verify invalid authentication was blocked
        assert len(trading_system.authentication_failures) == 1
        auth_failure = trading_system.authentication_failures[0]
        assert auth_failure["token"] == "invalid_token_12345"
        assert auth_failure["reason"] == "invalid_token"
        
        # Verify only valid event reached handler
        assert len(valid_events) == 1  # Still only 1 from valid auth
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, integrated_system):
        """Test rate limiting across system modules."""
        system = integrated_system
        event_bus = system["event_bus"]
        token_manager = system["token_manager"]
        rate_limiter = system["rate_limiter"]
        
        # Create user token
        auth_token = token_manager.generate_token({
            "user_id": "12345678-1234-5678-9012-123456789014",
            "username": "rate_test_user"
        })
        
        # Create multiple events to trigger rate limiting
        events = []
        for i in range(7):  # More than the rate limit of 5
            event = BaseEvent(
                event_type=EventType.ORDER,
                source="trading_client",
                data={
                    "order": {
                        "order_id": str(uuid4()),
                        "user_id": "12345678-1234-5678-9012-123456789014",
                        "trading_pair": "BTC/USDT",
                        "order_type": "limit",
                        "side": "buy",
                        "quantity": 0.1,
                        "price": 50000.0,
                        "status": "pending"
                    }
                },
                metadata={
                    "user_id": "12345678-1234-5678-9012-123456789014",
                    "event_type": EventType.ORDER.value,
                    "auth_token": auth_token
                }
            )
            events.append(event)
        
        processed_events = []
        
        async def rate_test_handler(event):
            processed_events.append(event.data["order"]["order_id"])
        
        event_bus.subscribe(EventType.ORDER, rate_test_handler)
        
        # Process all events
        for event in events:
            await event_bus.publish(event)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.2)
        
        # Verify rate limiting occurred
        trading_system = system["trading_system"]
        assert len(trading_system.rate_limit_violations) >= 2  # At least 2 violations
        
        # Verify some events were blocked
        assert len(processed_events) <= 5  # At most 5 should succeed
        
        # Verify rate limit violations were recorded
        for violation in trading_system.rate_limit_violations:
            assert violation["user_id"] == "12345678-1234-5678-9012-123456789014"
            assert "trading_operations:" in violation["key"]
    
    @pytest.mark.asyncio
    async def test_data_model_validation_across_modules(self, integrated_system):
        """Test data model validation across different system modules."""
        system = integrated_system
        event_bus = system["event_bus"]
        
        # Test 1: Valid data models
        valid_order_event = BaseEvent(
            event_type=EventType.ORDER,
            source="trading_client",
            data={
                "order": {
                    "order_id": str(uuid4()),
                    "user_id": "12345678-1234-5678-9012-123456789015",
                    "trading_pair": "ADA/USDT",
                    "order_type": "market",
                    "side": "sell",
                    "quantity": 100.0,
                    "status": "pending"
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789015",
                "event_type": EventType.ORDER.value
            }
        )
        
        # Test 2: Invalid data models
        invalid_order_event = BaseEvent(
            event_type=EventType.ORDER,
            source="trading_client",
            data={
                "order": {
                    "order_id": str(uuid4()),
                    "user_id": "12345678-1234-5678-9012-123456789016",
                    "trading_pair": "",  # Invalid: empty trading pair
                    "order_type": "invalid_type",  # Invalid order type
                    "side": "buy",
                    "quantity": 0.1,
                    "status": "pending"
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789016",
                "event_type": EventType.ORDER.value
            }
        )
        
        processed_events = []
        
        async def validation_test_handler(event):
            processed_events.append(event.event_id)
        
        event_bus.subscribe(EventType.ORDER, validation_test_handler)
        
        # Process valid event
        await event_bus.publish(valid_order_event)
        
        # Process invalid event
        await event_bus.publish(invalid_order_event)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.1)
        
        # Verify validation results
        data_validation = system["data_validation"]
        assert len(data_validation.validation_results) == 2
        
        # Find validation results
        valid_result = None
        invalid_result = None
        
        for result in data_validation.validation_results:
            if result["is_valid"]:
                valid_result = result
            else:
                invalid_result = result
        
        # Verify valid event passed validation
        assert valid_result is not None
        assert valid_result["is_valid"] is True
        assert len(valid_result["validation_errors"]) == 0
        
        # Verify invalid event failed validation
        assert invalid_result is not None
        assert invalid_result["is_valid"] is False
        assert len(invalid_result["validation_errors"]) > 0
        
        # Verify model validations were recorded
        assert len(data_validation.model_validations) >= 2
        
        # Verify only valid event reached handler
        assert len(processed_events) == 1  # Only valid event processed
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading_workflow(self, integrated_system, data_provider):
        """Test complete end-to-end trading workflow integration."""
        system = integrated_system
        event_bus = system["event_bus"]
        token_manager = system["token_manager"]
        
        # Step 1: User authentication
        auth_token = token_manager.generate_token({
            "user_id": "12345678-1234-5678-9012-123456789017",
            "username": "end_to_end_trader"
        })
        
        # Step 2: Market data event
        market_data_event = BaseEvent(
            event_type=EventType.MARKET_DATA,
            source="market_feed",
            data={
                "market_data": {
                    "symbol": "SOLUSDT",
                    "price": 100.25,
                    "volume": 50000.0,
                    "high_24h": 105.0,
                    "low_24h": 95.5,
                    "change_24h": -2.1
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-999999999999",
                "event_type": EventType.MARKET_DATA.value
            }
        )
        
        # Step 3: Order placement
        order_event = BaseEvent(
            event_type=EventType.ORDER,
            source="trading_client",
            data={
                "order": {
                    "order_id": str(uuid4()),
                    "user_id": "12345678-1234-5678-9012-123456789017",
                    "trading_pair": "SOL/USDT",
                    "order_type": "limit",
                    "side": "buy",
                    "quantity": 10.0,
                    "price": 100.00,
                    "status": "pending"
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789017",
                "event_type": EventType.ORDER.value,
                "auth_token": auth_token
            }
        )
        
        # Step 4: Trade execution (simulated)
        trade_event = BaseEvent(
            event_type=EventType.TRADE,
            source="matching_engine",
            data={
                "trade": {
                    "trade_id": "trade_001",
                    "order_id": str(uuid4()),
                    "symbol": "SOLUSDT",
                    "price": 100.00,
                    "quantity": 10.0,
                    "side": "buy",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            },
            metadata={
                "user_id": "12345678-1234-5678-9012-123456789017",
                "event_type": EventType.TRADE.value
            }
        )
        
        # Set up event handlers to track workflow
        workflow_events = {
            "market_data": [],
            "orders": [],
            "trades": []
        }
        
        async def market_handler(event):
            workflow_events["market_data"].append(event.data["market_data"])
        
        async def order_handler(event):
            workflow_events["orders"].append(event.data["order"])
        
        async def trade_handler(event):
            workflow_events["trades"].append(event.data["trade"])
        
        event_bus.subscribe(EventType.MARKET_DATA, market_handler)
        event_bus.subscribe(EventType.ORDER, order_handler)
        event_bus.subscribe(EventType.TRADE, trade_handler)
        
        # Execute complete workflow
        await event_bus.publish(market_data_event)
        await event_bus.publish(order_event)
        await event_bus.publish(trade_event)
        
        # Wait for async processing to complete
        await asyncio.sleep(0.2)
        
        # Verify end-to-end workflow
        data_validation = system["data_validation"]
        trading_system = system["trading_system"]
        
        # Verify all events were processed
        assert len(workflow_events["market_data"]) == 1
        assert len(workflow_events["orders"]) == 1
        assert len(workflow_events["trades"]) == 1
        
        # Verify validation results for all event types
        assert len(data_validation.validation_results) == 3
        
        # Verify trading system processed relevant events
        assert len(trading_system.processed_market_data) == 1
        assert len(trading_system.processed_orders) == 1
        
        # Verify authentication succeeded for order
        assert len(trading_system.authentication_failures) == 0
        
        # Verify data flow consistency
        market_data = workflow_events["market_data"][0]
        order_data = workflow_events["orders"][0]
        trade_data = workflow_events["trades"][0]
        
        # All events should reference the same symbol
        assert market_data["symbol"] == "SOLUSDT"
        assert order_data["trading_pair"] == "SOL/USDT"
        assert trade_data["symbol"] == "SOLUSDT"
        
        # Order and trade data should be valid (order_ids are different by design)
        assert "order_id" in order_data
        assert "order_id" in trade_data
        assert order_data["quantity"] == trade_data["quantity"]
        assert order_data["price"] == trade_data["price"]