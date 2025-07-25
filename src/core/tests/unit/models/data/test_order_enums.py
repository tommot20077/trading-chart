# ABOUTME: 訂單枚舉類型的單元測試
# ABOUTME: 測試所有訂單相關枚舉的功能和業務邏輯

import pytest
from core.models.data.order_enums import (
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderExecutionType
)


class TestOrderStatus:
    """測試訂單狀態枚舉"""
    
    def test_order_status_values(self):
        """測試訂單狀態的值"""
        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.PARTIALLY_FILLED == "partially_filled"
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.CANCELLED == "cancelled"
        assert OrderStatus.REJECTED == "rejected"
        assert OrderStatus.EXPIRED == "expired"
    
    def test_is_active(self):
        """測試活躍狀態檢查"""
        assert OrderStatus.PENDING.is_active() is True
        assert OrderStatus.PARTIALLY_FILLED.is_active() is True
        assert OrderStatus.FILLED.is_active() is False
        assert OrderStatus.CANCELLED.is_active() is False
        assert OrderStatus.REJECTED.is_active() is False
        assert OrderStatus.EXPIRED.is_active() is False
    
    def test_is_final(self):
        """測試最終狀態檢查"""
        assert OrderStatus.PENDING.is_final() is False
        assert OrderStatus.PARTIALLY_FILLED.is_final() is False
        assert OrderStatus.FILLED.is_final() is True
        assert OrderStatus.CANCELLED.is_final() is True
        assert OrderStatus.REJECTED.is_final() is True
        assert OrderStatus.EXPIRED.is_final() is True
    
    def test_valid_transitions(self):
        """測試有效狀態轉換"""
        transitions = OrderStatus.get_valid_transitions()
        
        # PENDING 可以轉換到其他狀態
        assert OrderStatus.PARTIALLY_FILLED in transitions[OrderStatus.PENDING]
        assert OrderStatus.FILLED in transitions[OrderStatus.PENDING]
        assert OrderStatus.CANCELLED in transitions[OrderStatus.PENDING]
        assert OrderStatus.REJECTED in transitions[OrderStatus.PENDING]
        assert OrderStatus.EXPIRED in transitions[OrderStatus.PENDING]
        
        # PARTIALLY_FILLED 可以轉換到完成、取消或過期
        assert OrderStatus.FILLED in transitions[OrderStatus.PARTIALLY_FILLED]
        assert OrderStatus.CANCELLED in transitions[OrderStatus.PARTIALLY_FILLED]
        assert OrderStatus.EXPIRED in transitions[OrderStatus.PARTIALLY_FILLED]
        assert OrderStatus.PENDING not in transitions[OrderStatus.PARTIALLY_FILLED]
        
        # 最終狀態不能轉換
        assert len(transitions[OrderStatus.FILLED]) == 0
        assert len(transitions[OrderStatus.CANCELLED]) == 0
        assert len(transitions[OrderStatus.REJECTED]) == 0
        assert len(transitions[OrderStatus.EXPIRED]) == 0


class TestOrderType:
    """測試訂單類型枚舉"""
    
    def test_order_type_values(self):
        """測試訂單類型的值"""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP_LOSS == "stop_loss"
        assert OrderType.STOP_LIMIT == "stop_limit"
        assert OrderType.TAKE_PROFIT == "take_profit"
        assert OrderType.TRAILING_STOP == "trailing_stop"
    
    def test_requires_price(self):
        """測試是否需要價格"""
        assert OrderType.MARKET.requires_price() is False
        assert OrderType.LIMIT.requires_price() is True
        assert OrderType.STOP_LOSS.requires_price() is False
        assert OrderType.STOP_LIMIT.requires_price() is True
        assert OrderType.TAKE_PROFIT.requires_price() is True
        assert OrderType.TRAILING_STOP.requires_price() is False
    
    def test_requires_trigger_price(self):
        """測試是否需要觸發價格"""
        assert OrderType.MARKET.requires_trigger_price() is False
        assert OrderType.LIMIT.requires_trigger_price() is False
        assert OrderType.STOP_LOSS.requires_trigger_price() is True
        assert OrderType.STOP_LIMIT.requires_trigger_price() is True
        assert OrderType.TAKE_PROFIT.requires_trigger_price() is False
        assert OrderType.TRAILING_STOP.requires_trigger_price() is True
    
    def test_is_conditional(self):
        """測試是否為條件訂單"""
        assert OrderType.MARKET.is_conditional() is False
        assert OrderType.LIMIT.is_conditional() is False
        assert OrderType.STOP_LOSS.is_conditional() is True
        assert OrderType.STOP_LIMIT.is_conditional() is True
        assert OrderType.TAKE_PROFIT.is_conditional() is True
        assert OrderType.TRAILING_STOP.is_conditional() is True
    
    def test_is_immediate(self):
        """測試是否為即時執行"""
        assert OrderType.MARKET.is_immediate() is True
        assert OrderType.LIMIT.is_immediate() is False
        assert OrderType.STOP_LOSS.is_immediate() is False
        assert OrderType.STOP_LIMIT.is_immediate() is False
        assert OrderType.TAKE_PROFIT.is_immediate() is False
        assert OrderType.TRAILING_STOP.is_immediate() is False


class TestOrderSide:
    """測試訂單方向枚舉"""
    
    def test_order_side_values(self):
        """測試訂單方向的值"""
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"
    
    def test_opposite(self):
        """測試相反方向"""
        assert OrderSide.BUY.opposite() == OrderSide.SELL
        assert OrderSide.SELL.opposite() == OrderSide.BUY
    
    def test_is_buy(self):
        """測試是否為買入"""
        assert OrderSide.BUY.is_buy() is True
        assert OrderSide.SELL.is_buy() is False
    
    def test_is_sell(self):
        """測試是否為賣出"""
        assert OrderSide.BUY.is_sell() is False
        assert OrderSide.SELL.is_sell() is True


class TestTimeInForce:
    """測試訂單有效期類型枚舉"""
    
    def test_time_in_force_values(self):
        """測試訂單有效期類型的值"""
        assert TimeInForce.GTC == "gtc"
        assert TimeInForce.IOC == "ioc"
        assert TimeInForce.FOK == "fok"
        assert TimeInForce.GTD == "gtd"
        assert TimeInForce.DAY == "day"
    
    def test_allows_partial_fill(self):
        """測試是否允許部分成交"""
        assert TimeInForce.GTC.allows_partial_fill() is True
        assert TimeInForce.IOC.allows_partial_fill() is True
        assert TimeInForce.FOK.allows_partial_fill() is False
        assert TimeInForce.GTD.allows_partial_fill() is True
        assert TimeInForce.DAY.allows_partial_fill() is True
    
    def test_requires_immediate_execution(self):
        """測試是否要求立即執行"""
        assert TimeInForce.GTC.requires_immediate_execution() is False
        assert TimeInForce.IOC.requires_immediate_execution() is True
        assert TimeInForce.FOK.requires_immediate_execution() is True
        assert TimeInForce.GTD.requires_immediate_execution() is False
        assert TimeInForce.DAY.requires_immediate_execution() is False
    
    def test_requires_expiry_time(self):
        """測試是否需要過期時間"""
        assert TimeInForce.GTC.requires_expiry_time() is False
        assert TimeInForce.IOC.requires_expiry_time() is False
        assert TimeInForce.FOK.requires_expiry_time() is False
        assert TimeInForce.GTD.requires_expiry_time() is True
        assert TimeInForce.DAY.requires_expiry_time() is False


class TestOrderExecutionType:
    """測試訂單執行類型枚舉"""
    
    def test_order_execution_type_values(self):
        """測試訂單執行類型的值"""
        assert OrderExecutionType.MAKER == "maker"
        assert OrderExecutionType.TAKER == "taker"
        assert OrderExecutionType.MIXED == "mixed"
    
    def test_provides_liquidity(self):
        """測試是否提供流動性"""
        assert OrderExecutionType.MAKER.provides_liquidity() is True
        assert OrderExecutionType.TAKER.provides_liquidity() is False
        assert OrderExecutionType.MIXED.provides_liquidity() is True
    
    def test_consumes_liquidity(self):
        """測試是否消耗流動性"""
        assert OrderExecutionType.MAKER.consumes_liquidity() is False
        assert OrderExecutionType.TAKER.consumes_liquidity() is True
        assert OrderExecutionType.MIXED.consumes_liquidity() is True