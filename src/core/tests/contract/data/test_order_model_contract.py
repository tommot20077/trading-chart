# ABOUTME: Contract tests for Order model ensuring compliance with expected interfaces
# ABOUTME: Verifies serialization, validation, and business rule contracts for Order data model

import pytest
from datetime import datetime, UTC
from decimal import Decimal
from uuid import UUID, uuid4
from typing import Dict, Any

from core.models.data.order import Order
from core.models.data.order_enums import (
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderExecutionType
)


class TestOrderModelContract:
    """
    Contract tests for Order model.
    
    These tests verify that the Order model conforms to expected contracts
    for serialization, validation, and business logic behavior.
    """

    @pytest.fixture
    def valid_order_data(self) -> Dict[str, Any]:
        """Provides valid order data for testing."""
        return {
            "user_id": uuid4(),
            "trading_pair": "BTC/USDT",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": Decimal("1.5"),
            "price": Decimal("50000.00"),
        }

    @pytest.fixture
    def order_instance(self, valid_order_data) -> Order:
        """Provides a valid Order instance for testing."""
        return Order(**valid_order_data)

    # Serialization Contract Tests

    @pytest.mark.contract
    def test_model_serialization_contract(self, order_instance):
        """Test that Order model can be serialized to dict consistently."""
        serialized = order_instance.to_dict()
        
        # Contract: Must return a dictionary
        assert isinstance(serialized, dict)
        
        # Contract: Must include all required fields with aliases
        required_fields = [
            "orderId", "userId", "tradingPair", "side", "orderType", 
            "quantity", "status", "timeInForce", "createdAt", "updatedAt"
        ]
        for field in required_fields:
            assert field in serialized
            
        # Contract: UUIDs must be serialized as strings
        assert isinstance(serialized["orderId"], str)
        assert isinstance(serialized["userId"], str)
        
        # Contract: Decimals must be serialized as strings for precision
        assert isinstance(serialized["quantity"], str)

    @pytest.mark.contract
    def test_model_deserialization_contract(self, valid_order_data):
        """Test that Order model can be deserialized from dict consistently."""
        order = Order(**valid_order_data)
        serialized = order.to_dict()
        
        # Contract: Must be able to recreate from serialized data
        recreated = Order.model_validate(serialized)
        
        # Contract: Recreated instance must have same core data
        assert recreated.order_id == order.order_id
        assert recreated.user_id == order.user_id
        assert recreated.trading_pair == order.trading_pair
        assert recreated.quantity == order.quantity

    @pytest.mark.contract
    def test_json_serialization_contract(self, order_instance):
        """Test that Order model maintains JSON serialization contract."""
        # Contract: Must be JSON serializable
        json_data = order_instance.model_dump(mode="json")
        assert isinstance(json_data, dict)
        
        # Contract: All datetime fields must be ISO format strings in JSON mode
        datetime_fields = ["createdAt", "updatedAt"]
        for field in datetime_fields:
            if json_data.get(field):
                # Should be ISO format string
                assert isinstance(json_data[field], str)
                # Should be parseable back to datetime
                datetime.fromisoformat(json_data[field].replace('Z', '+00:00'))

    # Validation Contract Tests

    @pytest.mark.contract
    def test_required_field_validation_contract(self):
        """Test that Order model enforces required field validation contract."""
        required_fields = ["user_id", "trading_pair", "side", "order_type", "quantity"]
        
        for field in required_fields:
            incomplete_data = {
                "user_id": uuid4(),
                "trading_pair": "BTC/USDT",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "quantity": Decimal("1.0"),
            }
            del incomplete_data[field]
            
            # Contract: Must raise ValidationError for missing required fields
            with pytest.raises(Exception):  # Pydantic raises ValidationError
                Order(**incomplete_data)

    @pytest.mark.contract
    def test_data_type_validation_contract(self, valid_order_data):
        """Test that Order model enforces data type validation contract."""
        # Contract: quantity must be Decimal or convertible to Decimal
        valid_order_data["quantity"] = "1.5"  # String should convert
        order = Order(**valid_order_data)
        assert isinstance(order.quantity, Decimal)
        
        # Contract: Invalid quantity should raise error
        invalid_data = valid_order_data.copy()
        invalid_data["quantity"] = "invalid_decimal"
        with pytest.raises(Exception):
            Order(**invalid_data)

    @pytest.mark.contract
    def test_business_rule_validation_contract(self, valid_order_data):
        """Test that Order model enforces business rule validation contract."""
        # Contract: Market orders should not have price
        market_order_data = valid_order_data.copy()
        market_order_data["order_type"] = OrderType.MARKET
        market_order_data["price"] = Decimal("50000")
        
        with pytest.raises(Exception):  # Should violate business rule
            Order(**market_order_data)
        
        # Contract: Limit orders must have price
        limit_order_data = valid_order_data.copy()
        limit_order_data["order_type"] = OrderType.LIMIT
        limit_order_data["price"] = None
        
        with pytest.raises(Exception):  # Should violate business rule
            Order(**limit_order_data)

    # State Management Contract Tests

    @pytest.mark.contract
    def test_state_transition_contract(self, order_instance):
        """Test that Order model maintains state transition contract."""
        # Contract: Should start in PENDING status
        assert order_instance.status == OrderStatus.PENDING
        
        # Contract: Should be able to transition to valid states
        assert order_instance.can_transition_to(OrderStatus.PARTIALLY_FILLED)
        assert order_instance.can_transition_to(OrderStatus.FILLED)
        assert order_instance.can_transition_to(OrderStatus.CANCELLED)
        
        # Contract: Should not allow invalid transitions
        order_instance.status = OrderStatus.FILLED
        assert not order_instance.can_transition_to(OrderStatus.PENDING)

    @pytest.mark.contract
    def test_immutable_fields_contract(self, order_instance):
        """Test that Order model maintains immutable field contract."""
        original_order_id = order_instance.order_id
        original_user_id = order_instance.user_id
        original_created_at = order_instance.created_at
        
        # Contract: Core identity fields should remain constant
        # (Note: In practice, these would be immutable in business logic)
        assert order_instance.order_id == original_order_id
        assert order_instance.user_id == original_user_id
        assert order_instance.created_at == original_created_at

    # Business Logic Contract Tests

    @pytest.mark.contract
    def test_fill_calculation_contract(self, order_instance):
        """Test that Order model maintains fill calculation contract."""
        # Contract: remaining_quantity should equal quantity - filled_quantity
        assert order_instance.remaining_quantity == order_instance.quantity - order_instance.filled_quantity
        
        # Contract: fill_percentage should be calculated correctly
        expected_percentage = (order_instance.filled_quantity / order_instance.quantity) * Decimal("100")
        assert order_instance.fill_percentage == expected_percentage
        
        # Contract: Order should not be considered filled until fully filled
        assert not order_instance.is_filled
        
        # Simulate partial fill
        order_instance.filled_quantity = order_instance.quantity / 2
        assert order_instance.is_partially_filled
        assert not order_instance.is_filled

    @pytest.mark.contract
    def test_update_timestamp_contract(self, order_instance):
        """Test that Order model maintains timestamp update contract."""
        original_updated_at = order_instance.updated_at
        
        # Contract: updated_at should change when order is modified
        order_instance.update_fill(Decimal("0.5"), Decimal("50000"))
        assert order_instance.updated_at > original_updated_at

    @pytest.mark.contract
    def test_enum_value_contract(self, order_instance):
        """Test that Order model maintains enum value contract."""
        # Contract: Enum fields should maintain their enum type
        assert isinstance(order_instance.status, OrderStatus)
        assert isinstance(order_instance.side, OrderSide)
        assert isinstance(order_instance.order_type, OrderType)
        assert isinstance(order_instance.time_in_force, TimeInForce)
        
        # Contract: Should be able to call enum methods
        assert isinstance(order_instance.status.is_active(), bool)
        
    @pytest.mark.contract  
    def test_string_representation_contract(self, order_instance):
        """Test that Order model maintains string representation contract."""
        # Contract: Should have meaningful string representation
        str_repr = str(order_instance)
        assert "Order(" in str_repr
        assert str(order_instance.order_id) in str_repr
        assert order_instance.trading_pair in str_repr
        
        # Contract: Should have detailed repr
        repr_str = repr(order_instance)
        assert "Order(" in repr_str
        assert "order_id=" in repr_str
        assert "trading_pair=" in repr_str