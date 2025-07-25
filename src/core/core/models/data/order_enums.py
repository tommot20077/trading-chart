# ABOUTME: 訂單相關的枚舉定義，包含訂單狀態、類型、方向等業務枚舉
# ABOUTME: 提供訂單系統所需的所有枚舉類型，支援完整的訂單生命週期管理

from enum import Enum
from typing import Set


class OrderStatus(str, Enum):
    """
    訂單狀態枚舉。

    定義訂單在整個生命週期中的所有可能狀態，
    從創建到最終完成或取消的完整狀態轉換。

    Attributes:
        PENDING: 訂單已創建但尚未執行
        PARTIALLY_FILLED: 訂單部分成交
        FILLED: 訂單完全成交
        CANCELLED: 訂單已取消
        REJECTED: 訂單被拒絕
        EXPIRED: 訂單已過期
    """

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    def is_active(self) -> bool:
        """
        檢查訂單狀態是否為活躍狀態。

        Returns:
            如果訂單仍在活躍狀態（可以繼續執行或修改）則返回 True
        """
        return self in {self.PENDING, self.PARTIALLY_FILLED}

    def is_final(self) -> bool:
        """
        檢查訂單狀態是否為最終狀態。

        Returns:
            如果訂單已經處於最終狀態（無法再變更）則返回 True
        """
        return self in {self.FILLED, self.CANCELLED, self.REJECTED, self.EXPIRED}

    @classmethod
    def get_valid_transitions(cls) -> dict[str, Set[str]]:
        """
        獲取有效的狀態轉換映射。

        Returns:
            字典，鍵為當前狀態，值為可轉換到的狀態集合
        """
        return {
            cls.PENDING: {cls.PARTIALLY_FILLED, cls.FILLED, cls.CANCELLED, cls.REJECTED, cls.EXPIRED},
            cls.PARTIALLY_FILLED: {cls.FILLED, cls.CANCELLED, cls.EXPIRED},
            cls.FILLED: set(),  # 已完成的訂單無法轉換
            cls.CANCELLED: set(),  # 已取消的訂單無法轉換
            cls.REJECTED: set(),  # 被拒絕的訂單無法轉換
            cls.EXPIRED: set(),  # 已過期的訂單無法轉換
        }


class OrderType(str, Enum):
    """
    訂單類型枚舉。

    定義不同的訂單執行方式和觸發條件。

    Attributes:
        MARKET: 市價訂單，立即以市場價格執行
        LIMIT: 限價訂單，指定價格執行
        STOP_LOSS: 停損訂單，觸發價格後轉為市價訂單
        STOP_LIMIT: 停損限價訂單，觸發後轉為限價訂單
        TAKE_PROFIT: 止盈訂單，達到目標價格後執行
        TRAILING_STOP: 追蹤停損訂單，根據價格變動調整停損點
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

    def requires_price(self) -> bool:
        """
        檢查此訂單類型是否需要指定價格。

        Returns:
            如果訂單類型需要指定執行價格則返回 True
        """
        return self in {self.LIMIT, self.STOP_LIMIT, self.TAKE_PROFIT}

    def requires_trigger_price(self) -> bool:
        """
        檢查此訂單類型是否需要觸發價格。

        Returns:
            如果訂單類型需要觸發價格則返回 True
        """
        return self in {self.STOP_LOSS, self.STOP_LIMIT, self.TRAILING_STOP}

    def is_conditional(self) -> bool:
        """
        檢查此訂單類型是否為條件訂單。

        Returns:
            如果訂單需要滿足特定條件才執行則返回 True
        """
        return self in {self.STOP_LOSS, self.STOP_LIMIT, self.TAKE_PROFIT, self.TRAILING_STOP}

    def is_immediate(self) -> bool:
        """
        檢查此訂單類型是否為即時執行。

        Returns:
            如果訂單會立即執行則返回 True
        """
        return self == self.MARKET


class OrderSide(str, Enum):
    """
    訂單方向枚舉。

    定義訂單的買賣方向。

    Attributes:
        BUY: 買入訂單
        SELL: 賣出訂單
    """

    BUY = "buy"
    SELL = "sell"

    def opposite(self) -> "OrderSide":
        """
        獲取相反方向。

        Returns:
            相反的訂單方向
        """
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY

    def is_buy(self) -> bool:
        """
        檢查是否為買入訂單。

        Returns:
            如果是買入訂單則返回 True
        """
        return self == self.BUY

    def is_sell(self) -> bool:
        """
        檢查是否為賣出訂單。

        Returns:
            如果是賣出訂單則返回 True
        """
        return self == self.SELL


class TimeInForce(str, Enum):
    """
    訂單有效期類型枚舉。

    定義訂單的有效期限制。

    Attributes:
        GTC: 撤銷前有效（Good Till Cancelled）
        IOC: 立即成交或取消（Immediate Or Cancel）
        FOK: 全額成交或取消（Fill Or Kill）
        GTD: 當日有效（Good Till Date）
        DAY: 當日有效（Day Order）
    """

    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTD = "gtd"  # Good Till Date
    DAY = "day"  # Day Order

    def allows_partial_fill(self) -> bool:
        """
        檢查是否允許部分成交。

        Returns:
            如果允許部分成交則返回 True
        """
        return self in {self.GTC, self.IOC, self.GTD, self.DAY}

    def requires_immediate_execution(self) -> bool:
        """
        檢查是否要求立即執行。

        Returns:
            如果要求立即執行則返回 True
        """
        return self in {self.IOC, self.FOK}

    def requires_expiry_time(self) -> bool:
        """
        檢查是否需要指定過期時間。

        Returns:
            如果需要指定過期時間則返回 True
        """
        return self == self.GTD


class OrderExecutionType(str, Enum):
    """
    訂單執行類型枚舉。

    定義訂單的執行方式和優先級。

    Attributes:
        MAKER: 掛單者，提供流動性
        TAKER: 吃單者，消耗流動性
        MIXED: 混合執行，部分掛單部分吃單
    """

    MAKER = "maker"
    TAKER = "taker"
    MIXED = "mixed"

    def provides_liquidity(self) -> bool:
        """
        檢查是否提供流動性。

        Returns:
            如果提供流動性則返回 True
        """
        return self in {self.MAKER, self.MIXED}

    def consumes_liquidity(self) -> bool:
        """
        檢查是否消耗流動性。

        Returns:
            如果消耗流動性則返回 True
        """
        return self in {self.TAKER, self.MIXED}
