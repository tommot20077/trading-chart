# ABOUTME: 訂單數據模型，基於 Pydantic 2.0+ 實現的完整訂單實體
# ABOUTME: 包含訂單的所有屬性、驗證邏輯和業務方法，支援完整的訂單生命週期管理

from datetime import datetime, UTC
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .order_enums import OrderStatus, OrderType, OrderSide, TimeInForce, OrderExecutionType


class Order(BaseModel):
    """
    訂單數據模型。

    完整的訂單實體，包含訂單執行所需的所有信息和業務邏輯。
    基於 Pydantic 2.0+ 實現，提供完整的數據驗證和序列化功能。

    Attributes:
        order_id: 訂單唯一標識符
        user_id: 用戶唯一標識符
        trading_pair: 交易對（如 BTC/USDT）
        side: 訂單方向（買入/賣出）
        order_type: 訂單類型（市價/限價等）
        quantity: 訂單數量
        price: 訂單價格（市價訂單可為 None）
        trigger_price: 觸發價格（條件訂單使用）
        filled_quantity: 已成交數量
        remaining_quantity: 剩餘數量
        average_fill_price: 平均成交價格
        status: 訂單狀態
        time_in_force: 訂單有效期類型
        execution_type: 執行類型（maker/taker）
        created_at: 創建時間
        updated_at: 最後更新時間
        expired_at: 過期時間（如適用）
        filled_at: 完全成交時間（如適用）
        cancelled_at: 取消時間（如適用）
        metadata: 額外元數據
    """

    model_config = ConfigDict(
        frozen=False,  # 允許狀態更新
        validate_assignment=True,  # 賦值時驗證
        use_enum_values=False,  # 保持枚舉對象
        # ser_by_alias=True,  # 序列化時使用別名 (removed for mypy compatibility)
        populate_by_name=True,  # 允許使用字段名或別名
        str_strip_whitespace=True,  # 自動去除字符串空白
        validate_default=True,  # 驗證默認值
        extra="forbid",  # 禁止額外字段
    )

    # 基本標識信息
    order_id: UUID = Field(default_factory=uuid4, description="訂單唯一標識符", alias="orderId")

    user_id: UUID = Field(description="用戶唯一標識符", alias="userId")

    trading_pair: str = Field(min_length=3, max_length=20, description="交易對符號，如 BTC/USDT", alias="tradingPair")

    # 訂單執行參數
    side: OrderSide = Field(description="訂單方向（買入/賣出）")

    order_type: OrderType = Field(description="訂單類型", alias="orderType")

    quantity: Decimal = Field(gt=0, decimal_places=8, description="訂單數量，必須大於 0")

    price: Optional[Decimal] = Field(default=None, gt=0, decimal_places=8, description="訂單價格，市價訂單可為空")

    trigger_price: Optional[Decimal] = Field(
        default=None, gt=0, decimal_places=8, description="觸發價格，條件訂單使用", alias="triggerPrice"
    )

    # 成交信息
    filled_quantity: Decimal = Field(
        default=Decimal("0"), ge=0, decimal_places=8, description="已成交數量", alias="filledQuantity"
    )

    average_fill_price: Optional[Decimal] = Field(
        default=None, gt=0, decimal_places=8, description="平均成交價格", alias="averageFillPrice"
    )

    # 狀態信息
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="訂單狀態")

    time_in_force: TimeInForce = Field(default=TimeInForce.GTC, description="訂單有效期類型", alias="timeInForce")

    execution_type: Optional[OrderExecutionType] = Field(
        default=None, description="執行類型（maker/taker）", alias="executionType"
    )

    # 時間戳信息
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="創建時間（UTC）", alias="createdAt"
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="最後更新時間（UTC）", alias="updatedAt"
    )

    expired_at: Optional[datetime] = Field(default=None, description="過期時間（UTC），如適用", alias="expiredAt")

    filled_at: Optional[datetime] = Field(default=None, description="完全成交時間（UTC），如適用", alias="filledAt")

    cancelled_at: Optional[datetime] = Field(default=None, description="取消時間（UTC），如適用", alias="cancelledAt")

    # 元數據
    metadata: Dict[str, Any] = Field(default_factory=dict, description="額外的訂單元數據")

    @property
    def remaining_quantity(self) -> Decimal:
        """
        計算剩餘數量。

        Returns:
            訂單剩餘未成交的數量
        """
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> Decimal:
        """
        計算成交百分比。

        Returns:
            成交百分比（0-100）
        """
        if self.quantity == 0:
            return Decimal("0")
        return (self.filled_quantity / self.quantity) * Decimal("100")

    @property
    def is_active(self) -> bool:
        """
        檢查訂單是否處於活躍狀態。

        Returns:
            如果訂單仍可執行或修改則返回 True
        """
        return self.status.is_active()

    @property
    def is_filled(self) -> bool:
        """
        檢查訂單是否完全成交。

        Returns:
            如果訂單完全成交則返回 True
        """
        return self.status == OrderStatus.FILLED

    @property
    def is_partially_filled(self) -> bool:
        """
        檢查訂單是否部分成交。

        Returns:
            如果訂單部分成交則返回 True
        """
        return self.status == OrderStatus.PARTIALLY_FILLED

    @property
    def total_fill_value(self) -> Optional[Decimal]:
        """
        計算總成交價值。

        Returns:
            總成交價值，如果沒有成交則返回 None
        """
        if self.filled_quantity == 0 or self.average_fill_price is None:
            return None
        return self.filled_quantity * self.average_fill_price

    @field_validator("trading_pair")
    @classmethod
    def validate_trading_pair(cls, v: str) -> str:
        """
        驗證交易對格式。

        Args:
            v: 交易對字符串

        Returns:
            規範化的交易對字符串

        Raises:
            ValueError: 如果交易對格式無效
        """
        v = v.upper().strip()
        if "/" not in v:
            raise ValueError('交易對必須包含 "/" 分隔符')

        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("交易對格式必須為 BASE/QUOTE")

        base, quote = parts
        if not base or not quote:
            raise ValueError("基礎貨幣和報價貨幣都不能為空")

        if base == quote:
            raise ValueError("基礎貨幣和報價貨幣不能相同")

        return v

    @field_validator("filled_quantity")
    @classmethod
    def validate_filled_quantity(cls, v: Decimal, info) -> Decimal:
        """
        驗證已成交數量。

        Args:
            v: 已成交數量
            info: 驗證上下文信息

        Returns:
            驗證後的已成交數量

        Raises:
            ValueError: 如果已成交數量無效
        """
        if v < 0:
            raise ValueError("已成交數量不能為負數")

        # 如果有訂單數量信息，檢查不能超過訂單數量
        if hasattr(info, "data") and "quantity" in info.data:
            quantity = info.data["quantity"]
            if isinstance(quantity, (int, float, Decimal)) and v > Decimal(str(quantity)):
                raise ValueError("已成交數量不能超過訂單數量")

        return v

    @model_validator(mode="after")
    def validate_order_consistency(self) -> "Order":
        """
        驗證訂單整體一致性。

        Returns:
            驗證後的訂單實例

        Raises:
            ValueError: 如果訂單數據不一致
        """
        # 檢查價格要求
        if self.order_type.requires_price() and self.price is None:
            raise ValueError(f"{self.order_type.value} 訂單必須指定價格")

        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("市價訂單不應指定價格")

        # 檢查觸發價格要求
        if self.order_type.requires_trigger_price() and self.trigger_price is None:
            raise ValueError(f"{self.order_type.value} 訂單必須指定觸發價格")

        # 檢查已成交數量
        if self.filled_quantity > self.quantity:
            raise ValueError("已成交數量不能超過訂單數量")

        # 檢查狀態一致性
        if self.filled_quantity == 0 and self.status == OrderStatus.PARTIALLY_FILLED:
            raise ValueError("已成交數量為 0 時狀態不能為部分成交")

        if self.filled_quantity == self.quantity and self.filled_quantity > 0:
            if self.status not in {OrderStatus.FILLED}:
                # 自動調整狀態
                object.__setattr__(self, "status", OrderStatus.FILLED)
                if self.filled_at is None:
                    object.__setattr__(self, "filled_at", datetime.now(UTC))

        if 0 < self.filled_quantity < self.quantity:
            if self.status == OrderStatus.PENDING:
                # 自動調整狀態
                object.__setattr__(self, "status", OrderStatus.PARTIALLY_FILLED)

        # 檢查時間邏輯
        if self.filled_at and self.filled_at < self.created_at:
            raise ValueError("成交時間不能早於創建時間")

        if self.cancelled_at and self.cancelled_at < self.created_at:
            raise ValueError("取消時間不能早於創建時間")

        if self.expired_at and self.expired_at < self.created_at:
            raise ValueError("過期時間不能早於創建時間")

        # 檢查有效期要求
        if self.time_in_force.requires_expiry_time() and self.expired_at is None:
            raise ValueError(f"{self.time_in_force.value} 訂單必須指定過期時間")

        # 更新最後修改時間（避免遞歸）
        object.__setattr__(self, "updated_at", datetime.now(UTC))

        return self

    def can_transition_to(self, new_status: OrderStatus) -> bool:
        """
        檢查是否可以轉換到指定狀態。

        Args:
            new_status: 目標狀態

        Returns:
            如果可以轉換則返回 True
        """
        valid_transitions = OrderStatus.get_valid_transitions()
        return new_status in valid_transitions.get(self.status, set())

    def update_fill(
        self, fill_quantity: Decimal, fill_price: Decimal, execution_type: Optional[OrderExecutionType] = None
    ) -> None:
        """
        更新訂單成交信息。

        Args:
            fill_quantity: 本次成交數量
            fill_price: 本次成交價格
            execution_type: 執行類型

        Raises:
            ValueError: 如果成交信息無效
        """
        if fill_quantity <= 0:
            raise ValueError("成交數量必須大於 0")

        if fill_price <= 0:
            raise ValueError("成交價格必須大於 0")

        new_filled_quantity = self.filled_quantity + fill_quantity
        if new_filled_quantity > self.quantity:
            raise ValueError("成交數量超過訂單數量")

        # 計算新的平均成交價格
        if self.filled_quantity == 0:
            new_average_price = fill_price
        else:
            total_value = (self.filled_quantity * (self.average_fill_price or Decimal("0"))) + (
                fill_quantity * fill_price
            )
            new_average_price = total_value / new_filled_quantity

        # 更新成交信息
        self.filled_quantity = new_filled_quantity
        self.average_fill_price = new_average_price

        if execution_type:
            self.execution_type = execution_type

        # 更新狀態
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now(UTC)
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.updated_at = datetime.now(UTC)

    def cancel(self, reason: Optional[str] = None) -> None:
        """
        取消訂單。

        Args:
            reason: 取消原因

        Raises:
            ValueError: 如果訂單無法取消
        """
        if not self.can_transition_to(OrderStatus.CANCELLED):
            raise ValueError(f"訂單狀態 {self.status.value} 無法取消")

        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

        if reason:
            self.metadata["cancel_reason"] = reason

    def expire(self) -> None:
        """
        使訂單過期。

        Raises:
            ValueError: 如果訂單無法過期
        """
        if not self.can_transition_to(OrderStatus.EXPIRED):
            raise ValueError(f"訂單狀態 {self.status.value} 無法過期")

        self.status = OrderStatus.EXPIRED
        if self.expired_at is None:
            self.expired_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def reject(self, reason: str) -> None:
        """
        拒絕訂單。

        Args:
            reason: 拒絕原因

        Raises:
            ValueError: 如果訂單無法拒絕
        """
        if not self.can_transition_to(OrderStatus.REJECTED):
            raise ValueError(f"訂單狀態 {self.status.value} 無法拒絕")

        self.status = OrderStatus.REJECTED
        self.updated_at = datetime.now(UTC)
        self.metadata["reject_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """
        轉換為字典格式。

        Returns:
            訂單的字典表示
        """
        return self.model_dump(by_alias=True, exclude_none=False, mode="json")

    def __str__(self) -> str:
        """
        字符串表示。

        Returns:
            訂單的字符串描述
        """
        return (
            f"Order({self.order_id}, {self.trading_pair}, "
            f"{self.side.value}, {self.order_type.value}, "
            f"{self.quantity}, {self.status.value})"
        )

    def __repr__(self) -> str:
        """
        詳細字符串表示。

        Returns:
            訂單的詳細字符串描述
        """
        return (
            f"Order(order_id={self.order_id}, trading_pair='{self.trading_pair}', "
            f"side={self.side}, order_type={self.order_type}, "
            f"quantity={self.quantity}, price={self.price}, "
            f"filled_quantity={self.filled_quantity}, status={self.status})"
        )
