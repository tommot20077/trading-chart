# ABOUTME: 訂單模型的單元測試
# ABOUTME: 測試訂單模型的創建、驗證、狀態轉換和業務邏輯

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
from pydantic import ValidationError

from core.models.data.order import Order
from core.models.data.order_enums import (
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderExecutionType
)


class TestOrderCreation:
    """測試訂單創建"""
    
    def test_minimal_order_creation(self):
        """測試最小化訂單創建"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        assert order.user_id == user_id
        assert order.trading_pair == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == Decimal("1.0")
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal("0")
        assert order.time_in_force == TimeInForce.GTC
        assert order.price is None  # 市價訂單無價格
        assert isinstance(order.order_id, UUID)
    
    def test_limit_order_creation(self):
        """測試限價訂單創建"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.5"),
            price=Decimal("2000.50")
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("2000.50")
        assert order.side == OrderSide.SELL
    
    def test_stop_loss_order_creation(self):
        """測試停損訂單創建"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.5"),
            trigger_price=Decimal("45000.0")
        )
        
        assert order.order_type == OrderType.STOP_LOSS
        assert order.trigger_price == Decimal("45000.0")
        assert order.price is None  # 停損訂單觸發後轉為市價


class TestOrderValidation:
    """測試訂單驗證"""
    
    def test_trading_pair_validation(self):
        """測試交易對驗證"""
        user_id = uuid4()
        
        # 有效交易對
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        assert order.trading_pair == "BTC/USDT"
        
        # 小寫會被轉為大寫
        order = Order(
            user_id=user_id,
            trading_pair="eth/btc",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        assert order.trading_pair == "ETH/BTC"
        
        # 無效格式
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTCUSDT",  # 缺少分隔符
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            )
        
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/",  # 缺少報價貨幣
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            )
        
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/BTC",  # 相同貨幣
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0")
            )
    
    def test_quantity_validation(self):
        """測試數量驗證"""
        user_id = uuid4()
        
        # 負數量
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("-1.0")
            )
        
        # 零數量
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0")
            )
    
    def test_price_requirements(self):
        """測試價格要求驗證"""
        user_id = uuid4()
        
        # 限價訂單必須有價格
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0")
                # 缺少價格
            )
        
        # 市價訂單不應有價格
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=Decimal("50000.0")  # 市價訂單設置價格
            )
    
    def test_trigger_price_requirements(self):
        """測試觸發價格要求驗證"""
        user_id = uuid4()
        
        # 停損訂單必須有觸發價格
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP_LOSS,
                quantity=Decimal("1.0")
                # 缺少觸發價格
            )
    
    def test_filled_quantity_validation(self):
        """測試已成交數量驗證"""
        user_id = uuid4()
        
        # 已成交數量不能超過訂單數量
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                filled_quantity=Decimal("2.0")  # 超過訂單數量
            )
    
    def test_time_in_force_requirements(self):
        """測試有效期要求驗證"""
        user_id = uuid4()
        
        # GTD 訂單必須有過期時間
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                time_in_force=TimeInForce.GTD
                # 缺少過期時間
            )


class TestOrderProperties:
    """測試訂單屬性"""
    
    def test_remaining_quantity(self):
        """測試剩餘數量計算"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            filled_quantity=Decimal("3.5")
        )
        
        assert order.remaining_quantity == Decimal("6.5")
    
    def test_fill_percentage(self):
        """測試成交百分比計算"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            filled_quantity=Decimal("2.5")
        )
        
        assert order.fill_percentage == Decimal("25.0")
    
    def test_status_properties(self):
        """測試狀態屬性"""
        user_id = uuid4()
        
        # 活躍訂單
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.PENDING
        )
        assert order.is_active is True
        assert order.is_filled is False
        assert order.is_partially_filled is False
        
        # 部分成交 - 需要有成交數量才能設置部分成交狀態
        order.filled_quantity = Decimal("0.5")
        object.__setattr__(order, 'status', OrderStatus.PARTIALLY_FILLED)
        assert order.is_active is True
        assert order.is_partially_filled is True
        
        # 完全成交
        object.__setattr__(order, 'status', OrderStatus.FILLED)
        assert order.is_active is False
        assert order.is_filled is True
    
    def test_total_fill_value(self):
        """測試總成交價值計算"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            filled_quantity=Decimal("5.0"),
            average_fill_price=Decimal("50000.0")
        )
        
        assert order.total_fill_value == Decimal("250000.0")
        
        # 未成交時返回 None - 創建一個新的訂單
        empty_order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            filled_quantity=Decimal("0")
        )
        assert empty_order.total_fill_value is None


class TestOrderStatusTransitions:
    """測試訂單狀態轉換"""
    
    def test_can_transition_to(self):
        """測試狀態轉換檢查"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.PENDING
        )
        
        # 從 PENDING 可以轉換到其他狀態
        assert order.can_transition_to(OrderStatus.PARTIALLY_FILLED) is True
        assert order.can_transition_to(OrderStatus.FILLED) is True
        assert order.can_transition_to(OrderStatus.CANCELLED) is True
        
        # 設置為已完成後不能轉換
        order.status = OrderStatus.FILLED
        assert order.can_transition_to(OrderStatus.CANCELLED) is False
        assert order.can_transition_to(OrderStatus.PENDING) is False
    
    def test_cancel_order(self):
        """測試取消訂單"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0")
        )
        
        # 取消訂單
        order.cancel("用戶取消")
        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None
        assert order.metadata["cancel_reason"] == "用戶取消"
        
        # 已取消的訂單不能再次取消
        with pytest.raises(ValueError):
            order.cancel()
    
    def test_reject_order(self):
        """測試拒絕訂單"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0")
        )
        
        # 拒絕訂單
        order.reject("資金不足")
        assert order.status == OrderStatus.REJECTED
        assert order.metadata["reject_reason"] == "資金不足"
    
    def test_expire_order(self):
        """測試訂單過期"""
        user_id = uuid4()
        expiry_time = datetime.now(timezone.utc) + timedelta(hours=1)
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            time_in_force=TimeInForce.GTD,
            expired_at=expiry_time
        )
        
        # 使訂單過期
        order.expire()
        assert order.status == OrderStatus.EXPIRED


class TestOrderFillUpdates:
    """測試訂單成交更新"""
    
    def test_update_fill_partial(self):
        """測試部分成交更新"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("50000.0")
        )
        
        # 第一次成交
        order.update_fill(
            fill_quantity=Decimal("3.0"),
            fill_price=Decimal("49500.0"),
            execution_type=OrderExecutionType.MAKER
        )
        
        assert order.filled_quantity == Decimal("3.0")
        assert order.average_fill_price == Decimal("49500.0")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.execution_type == OrderExecutionType.MAKER
        
        # 第二次成交
        order.update_fill(
            fill_quantity=Decimal("2.0"),
            fill_price=Decimal("50500.0")
        )
        
        assert order.filled_quantity == Decimal("5.0")
        # 平均價格 = (3.0 * 49500 + 2.0 * 50500) / 5.0 = 49900
        assert order.average_fill_price == Decimal("49900.0")
        assert order.status == OrderStatus.PARTIALLY_FILLED
    
    def test_update_fill_complete(self):
        """測試完全成交更新"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("50000.0")
        )
        
        # 完全成交
        order.update_fill(
            fill_quantity=Decimal("5.0"),
            fill_price=Decimal("50000.0")
        )
        
        assert order.filled_quantity == Decimal("5.0")
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None
        assert order.remaining_quantity == Decimal("0")
    
    def test_update_fill_validation(self):
        """測試成交更新驗證"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("50000.0")
        )
        
        # 成交數量不能為負
        with pytest.raises(ValueError):
            order.update_fill(
                fill_quantity=Decimal("-1.0"),
                fill_price=Decimal("50000.0")
            )
        
        # 成交價格不能為負
        with pytest.raises(ValueError):
            order.update_fill(
                fill_quantity=Decimal("1.0"),
                fill_price=Decimal("-50000.0")
            )
        
        # 成交數量不能超過剩餘數量
        with pytest.raises(ValueError):
            order.update_fill(
                fill_quantity=Decimal("10.0"),  # 超過訂單數量
                fill_price=Decimal("50000.0")
            )


class TestOrderSerialization:
    """測試訂單序列化"""
    
    def test_to_dict(self):
        """測試轉換為字典"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0")
        )
        
        order_dict = order.to_dict()
        assert isinstance(order_dict, dict)
        assert order_dict["userId"] == str(user_id)  # 使用別名
        assert order_dict["tradingPair"] == "BTC/USDT"
        assert order_dict["side"] == "buy"
        assert order_dict["orderType"] == "limit"
    
    def test_model_dump_exclude_none(self):
        """測試模型導出（排除空值）"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        # 包含 None 值
        data_with_none = order.model_dump(by_alias=True, exclude_none=False)
        assert "price" in data_with_none
        assert data_with_none["price"] is None
        
        # 排除 None 值
        data_without_none = order.model_dump(by_alias=True, exclude_none=True)
        assert "price" not in data_without_none
    
    def test_json_serialization(self):
        """測試 JSON 序列化"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0")
        )
        
        # 序列化為 JSON
        json_str = order.model_dump_json(by_alias=True)
        assert isinstance(json_str, str)
        
        # 反序列化
        order_data = order.model_validate_json(json_str)
        assert order_data.user_id == user_id
        assert order_data.trading_pair == "BTC/USDT"


class TestOrderStringRepresentation:
    """測試訂單字符串表示"""
    
    def test_str_representation(self):
        """測試字符串表示"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0")
        )
        
        str_repr = str(order)
        assert "BTC/USDT" in str_repr
        assert "buy" in str_repr
        assert "limit" in str_repr
        assert str(order.order_id) in str_repr
    
    def test_repr_representation(self):
        """測試詳細字符串表示"""
        user_id = uuid4()
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0")
        )
        
        repr_str = repr(order)
        assert "Order(" in repr_str
        assert "BTC/USDT" in repr_str
        assert "OrderSide.BUY" in repr_str
        assert "OrderType.LIMIT" in repr_str


class TestOrderEdgeCases:
    """測試訂單邊界情況"""
    
    def test_automatic_status_adjustment(self):
        """測試自動狀態調整"""
        user_id = uuid4()
        
        # 當已成交數量等於訂單數量時，自動調整為已完成
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("50000.0"),
            filled_quantity=Decimal("5.0"),  # 完全成交
            status=OrderStatus.PENDING  # 但狀態仍為待處理
        )
        
        # 模型驗證會自動調整狀態
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None
    
    def test_zero_quantity_edge_case(self):
        """測試零數量邊界情況"""
        user_id = uuid4()
        
        # 零數量訂單應該被拒絕
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0")
            )
    
    def test_time_validation_edge_cases(self):
        """測試時間驗證邊界情況"""
        user_id = uuid4()
        created_time = datetime.now(timezone.utc)
        
        # 成交時間早於創建時間
        with pytest.raises(ValidationError):
            Order(
                user_id=user_id,
                trading_pair="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                created_at=created_time,
                filled_at=created_time - timedelta(seconds=1)  # 早於創建時間
            )
    
    def test_decimal_precision(self):
        """測試小數精度處理"""
        user_id = uuid4()
        
        # 高精度小數
        order = Order(
            user_id=user_id,
            trading_pair="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.12345678"),  # 8位小數
            price=Decimal("50000.12345678")
        )
        
        assert order.quantity == Decimal("1.12345678")
        assert order.price == Decimal("50000.12345678")