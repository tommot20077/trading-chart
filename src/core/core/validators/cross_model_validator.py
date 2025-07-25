# ABOUTME: 跨模型業務規則驗證器，實現Order、TradingPair、MarketData等模型間的一致性檢查
# ABOUTME: 提供訂單與交易對一致性、交易數據合規性、市場數據完整性等跨模型驗證功能

from typing import List, Optional, Set
from decimal import Decimal
from datetime import timedelta

from core.models.data.order import Order
from core.models.data.trading_pair import TradingPair
from core.models.data.trade import Trade
from core.models.data.kline import Kline
from core.models.data.market_data import MarketData
from core.config.market_limits import MarketLimitsConfig
from .validation_result import ValidationResult, ValidationSeverity


class CrossModelValidator:
    """跨模型業務規則驗證器.

    提供模型間一致性檢查和業務規則驗證功能，確保數據的完整性和合規性。
    """

    def __init__(self, market_limits: Optional[MarketLimitsConfig] = None):
        """初始化驗證器.

        Args:
            market_limits: 市場限制配置，用於驗證交易規則
        """
        self.market_limits = market_limits or MarketLimitsConfig()

    def validate_order_trading_pair_consistency(self, order: Order, trading_pair: TradingPair) -> ValidationResult:
        """驗證訂單與交易對配置一致性.

        Args:
            order: 訂單實例
            trading_pair: 交易對配置

        Returns:
            驗證結果，包含所有發現的問題和建議
        """
        result = ValidationResult(is_valid=True, validated_models=["Order", "TradingPair"])

        # 檢查交易對符號一致性
        self._check_symbol_consistency(order, trading_pair, result)

        # 檢查交易對狀態
        self._check_trading_pair_status(trading_pair, result)

        # 檢查數量限制
        self._check_quantity_limits(order, trading_pair, result)

        # 檢查價格精度
        self._check_price_precision(order, trading_pair, result)

        # 檢查最小名義價值
        self._check_min_notional_value(order, trading_pair, result)

        return result

    def validate_trade_market_rules(self, trade: Trade, trading_pair: TradingPair) -> ValidationResult:
        """驗證交易數據符合市場規則.

        Args:
            trade: 交易記錄
            trading_pair: 交易對配置

        Returns:
            驗證結果
        """
        result = ValidationResult(is_valid=True, validated_models=["Trade", "TradingPair"])

        # 檢查符號一致性
        if trade.symbol != trading_pair.symbol:
            result.add_issue(
                code="TRADE_SYMBOL_MISMATCH",
                severity=ValidationSeverity.ERROR,
                message=f"交易符號 {trade.symbol} 與交易對 {trading_pair.symbol} 不匹配",
                field_path="symbol",
                expected_value=trading_pair.symbol,
                actual_value=trade.symbol,
                suggestion="確保交易記錄使用正確的交易對符號",
            )

        # 檢查價格精度
        if not self._check_decimal_precision(trade.price, trading_pair.price_precision):
            result.add_issue(
                code="TRADE_PRICE_PRECISION",
                severity=ValidationSeverity.ERROR,
                message=f"交易價格精度超出限制，期望 {trading_pair.price_precision} 位小數",
                field_path="price",
                expected_value=f"{trading_pair.price_precision} decimal places",
                actual_value=str(trade.price),
                suggestion=f"將價格調整為 {trading_pair.price_precision} 位小數精度",
            )

        # 檢查數量精度
        if not self._check_decimal_precision(trade.quantity, trading_pair.quantity_precision):
            result.add_issue(
                code="TRADE_QUANTITY_PRECISION",
                severity=ValidationSeverity.ERROR,
                message=f"交易數量精度超出限制，期望 {trading_pair.quantity_precision} 位小數",
                field_path="quantity",
                expected_value=f"{trading_pair.quantity_precision} decimal places",
                actual_value=str(trade.quantity),
                suggestion=f"將數量調整為 {trading_pair.quantity_precision} 位小數精度",
            )

        # 檢查最小交易數量
        if trade.quantity < trading_pair.min_trade_quantity:
            result.add_issue(
                code="TRADE_MIN_QUANTITY",
                severity=ValidationSeverity.ERROR,
                message=f"交易數量 {trade.quantity} 小於最小限制 {trading_pair.min_trade_quantity}",
                field_path="quantity",
                expected_value=f">= {trading_pair.min_trade_quantity}",
                actual_value=str(trade.quantity),
                suggestion=f"增加交易數量至最少 {trading_pair.min_trade_quantity}",
            )

        # 檢查最大交易數量
        if trade.quantity > trading_pair.max_trade_quantity:
            result.add_issue(
                code="TRADE_MAX_QUANTITY",
                severity=ValidationSeverity.ERROR,
                message=f"交易數量 {trade.quantity} 超過最大限制 {trading_pair.max_trade_quantity}",
                field_path="quantity",
                expected_value=f"<= {trading_pair.max_trade_quantity}",
                actual_value=str(trade.quantity),
                suggestion=f"減少交易數量至最多 {trading_pair.max_trade_quantity}",
            )

        return result

    def validate_kline_trade_consistency(self, klines: List[Kline], trades: List[Trade]) -> ValidationResult:
        """驗證K線與交易數據的一致性.

        Args:
            klines: K線數據列表
            trades: 交易記錄列表

        Returns:
            驗證結果
        """
        result = ValidationResult(is_valid=True, validated_models=["Kline", "Trade"])

        if not klines or not trades:
            return result

        # 檢查符號一致性
        self._check_data_symbol_consistency(klines + trades, result)

        # 檢查時間範圍一致性
        self._check_time_range_consistency(klines, trades, result)

        # 檢查價格範圍一致性
        self._check_price_range_consistency(klines, trades, result)

        return result

    def validate_market_data_integrity(self, market_data: MarketData) -> ValidationResult:
        """驗證市場數據的完整性和一致性.

        Args:
            market_data: 市場數據容器

        Returns:
            驗證結果
        """
        result = ValidationResult(is_valid=True, validated_models=["MarketData", "Kline", "Trade"])

        # 檢查基本數據完整性
        if not market_data.validate_data_integrity():
            result.add_issue(
                code="MARKET_DATA_INTEGRITY",
                severity=ValidationSeverity.ERROR,
                message="市場數據基礎完整性檢查失敗",
                suggestion="檢查數據符號一致性和時間戳邏輯",
            )

        # 檢查數據時間連續性
        self._check_data_time_continuity(market_data, result)

        # 檢查數據量合理性
        self._check_data_volume_reasonableness(market_data, result)

        return result

    # 私有輔助方法

    def _check_symbol_consistency(self, order: Order, trading_pair: TradingPair, result: ValidationResult) -> None:
        """檢查訂單與交易對的符號一致性."""
        # 將order的trading_pair格式轉換為與trading_pair.symbol一致的格式進行比較
        order_symbol = order.trading_pair.replace("/", "")  # BTC/USDT -> BTCUSDT
        if order_symbol != trading_pair.symbol:
            result.add_issue(
                code="ORDER_SYMBOL_MISMATCH",
                severity=ValidationSeverity.ERROR,
                message=f"訂單交易對 {order.trading_pair} 與配置 {trading_pair.symbol} 不匹配",
                field_path="trading_pair",
                expected_value=trading_pair.symbol,
                actual_value=order.trading_pair,
                suggestion="確保訂單使用正確的交易對標識符",
            )

    def _check_trading_pair_status(self, trading_pair: TradingPair, result: ValidationResult) -> None:
        """檢查交易對是否處於可交易狀態."""
        # 處理use_enum_values=True的情況，status可能是字符串
        if isinstance(trading_pair.status, str):
            from core.models.data.trading_pair import TradingPairStatus

            status_enum = TradingPairStatus(trading_pair.status)
            is_tradeable = status_enum.is_tradeable()
        else:
            is_tradeable = trading_pair.status.is_tradeable()

        if not is_tradeable:
            result.add_issue(
                code="TRADING_PAIR_NOT_ACTIVE",
                severity=ValidationSeverity.ERROR,
                message=f"交易對 {trading_pair.symbol} 狀態為 {trading_pair.status}，不可交易",
                field_path="status",
                expected_value="active",
                actual_value=trading_pair.status,
                suggestion="等待交易對恢復活躍狀態或選擇其他交易對",
            )

    def _check_quantity_limits(self, order: Order, trading_pair: TradingPair, result: ValidationResult) -> None:
        """檢查訂單數量是否在允許範圍內."""
        if order.quantity < trading_pair.min_trade_quantity:
            result.add_issue(
                code="ORDER_MIN_QUANTITY",
                severity=ValidationSeverity.ERROR,
                message=f"訂單數量 {order.quantity} 小於最小限制 {trading_pair.min_trade_quantity}",
                field_path="quantity",
                expected_value=f">= {trading_pair.min_trade_quantity}",
                actual_value=str(order.quantity),
                suggestion=f"增加訂單數量至最少 {trading_pair.min_trade_quantity}",
            )

        if order.quantity > trading_pair.max_trade_quantity:
            result.add_issue(
                code="ORDER_MAX_QUANTITY",
                severity=ValidationSeverity.ERROR,
                message=f"訂單數量 {order.quantity} 超過最大限制 {trading_pair.max_trade_quantity}",
                field_path="quantity",
                expected_value=f"<= {trading_pair.max_trade_quantity}",
                actual_value=str(order.quantity),
                suggestion=f"減少訂單數量至最多 {trading_pair.max_trade_quantity}",
            )

    def _check_price_precision(self, order: Order, trading_pair: TradingPair, result: ValidationResult) -> None:
        """檢查訂單價格精度是否符合要求."""
        if order.price and not self._check_decimal_precision(order.price, trading_pair.price_precision):
            result.add_issue(
                code="ORDER_PRICE_PRECISION",
                severity=ValidationSeverity.ERROR,
                message=f"訂單價格精度超出限制，期望 {trading_pair.price_precision} 位小數",
                field_path="price",
                expected_value=f"{trading_pair.price_precision} decimal places",
                actual_value=str(order.price),
                suggestion=f"將價格調整為 {trading_pair.price_precision} 位小數精度",
            )

    def _check_min_notional_value(self, order: Order, trading_pair: TradingPair, result: ValidationResult) -> None:
        """檢查訂單最小名義價值."""
        if order.price and trading_pair.min_notional is not None:
            notional_value = order.quantity * order.price
            if notional_value < trading_pair.min_notional:
                result.add_issue(
                    code="ORDER_MIN_NOTIONAL",
                    severity=ValidationSeverity.ERROR,
                    message=f"訂單名義價值 {notional_value} 小於最小限制 {trading_pair.min_notional}",
                    field_path="quantity,price",
                    expected_value=f">= {trading_pair.min_notional}",
                    actual_value=str(notional_value),
                    suggestion=f"增加訂單數量或價格使名義價值達到 {trading_pair.min_notional}",
                )

    def _check_decimal_precision(self, value: Decimal, max_places: int) -> bool:
        """檢查小數精度是否符合要求."""
        exponent = value.as_tuple().exponent
        decimal_places = abs(exponent) if isinstance(exponent, int) else 0
        return decimal_places <= max_places

    def _check_data_symbol_consistency(self, data_items: List, result: ValidationResult) -> None:
        """檢查數據項的符號一致性."""
        symbols: Set[str] = set()
        for item in data_items:
            if hasattr(item, "symbol"):
                symbols.add(item.symbol)

        if len(symbols) > 1:
            result.add_issue(
                code="DATA_SYMBOL_INCONSISTENCY",
                severity=ValidationSeverity.ERROR,
                message=f"數據包含多個不同符號: {', '.join(symbols)}",
                field_path="symbol",
                actual_value=list(symbols),
                suggestion="確保所有數據項使用相同的交易對符號",
            )

    def _check_time_range_consistency(self, klines: List[Kline], trades: List[Trade], result: ValidationResult) -> None:
        """檢查K線和交易數據的時間範圍一致性."""
        if not klines or not trades:
            return

        # 獲取時間範圍
        kline_start = min(kline.open_time for kline in klines)
        kline_end = max(kline.close_time for kline in klines)

        trade_start = min(trade.timestamp for trade in trades)
        trade_end = max(trade.timestamp for trade in trades)

        # 檢查交易時間是否在K線時間範圍內
        if trade_start < kline_start or trade_end > kline_end:
            result.add_issue(
                code="TIME_RANGE_MISMATCH",
                severity=ValidationSeverity.WARNING,
                message="交易數據時間範圍與K線數據不匹配",
                field_path="timestamp",
                expected_value=f"{kline_start} - {kline_end}",
                actual_value=f"{trade_start} - {trade_end}",
                suggestion="檢查數據來源時間同步性",
            )

    def _check_price_range_consistency(
        self, klines: List[Kline], trades: List[Trade], result: ValidationResult
    ) -> None:
        """檢查交易價格是否在K線價格範圍內."""
        for trade in trades:
            # 找到對應時間範圍的K線
            matching_klines = [kline for kline in klines if kline.open_time <= trade.timestamp <= kline.close_time]

            for kline in matching_klines:
                if not (kline.low_price <= trade.price <= kline.high_price):
                    result.add_issue(
                        code="TRADE_PRICE_OUT_OF_RANGE",
                        severity=ValidationSeverity.WARNING,
                        message=f"交易價格 {trade.price} 超出K線價格範圍 [{kline.low_price}, {kline.high_price}]",
                        field_path="price",
                        expected_value=f"[{kline.low_price}, {kline.high_price}]",
                        actual_value=str(trade.price),
                        suggestion="檢查數據來源準確性或時間同步",
                    )

    def _check_data_time_continuity(self, market_data: MarketData, result: ValidationResult) -> None:
        """檢查市場數據的時間連續性."""
        # 檢查K線時間連續性
        if len(market_data.klines) > 1:
            sorted_klines = sorted(market_data.klines, key=lambda k: k.open_time)
            for i in range(1, len(sorted_klines)):
                current = sorted_klines[i]
                previous = sorted_klines[i - 1]

                # 檢查時間間隔是否合理
                time_gap = current.open_time - previous.close_time
                if time_gap > timedelta(minutes=5):  # 允許5分鐘的間隙
                    result.add_issue(
                        code="KLINE_TIME_GAP",
                        severity=ValidationSeverity.WARNING,
                        message=f"K線數據存在 {time_gap} 的時間間隙",
                        field_path="open_time",
                        actual_value=str(time_gap),
                        suggestion="檢查是否存在數據缺失",
                    )

    def _check_data_volume_reasonableness(self, market_data: MarketData, result: ValidationResult) -> None:
        """檢查市場數據量的合理性."""
        # 檢查異常大的交易量
        for trade in market_data.trades:
            if trade.quantity > Decimal("1000000"):  # 假設100萬為異常大交易量
                result.add_issue(
                    code="ABNORMAL_TRADE_VOLUME",
                    severity=ValidationSeverity.WARNING,
                    message=f"發現異常大的交易量: {trade.quantity}",
                    field_path="quantity",
                    actual_value=str(trade.quantity),
                    suggestion="檢查交易數據是否正確",
                )

        # 檢查異常的價格波動
        if len(market_data.klines) > 1:
            sorted_klines = sorted(market_data.klines, key=lambda k: k.open_time)
            for i in range(1, len(sorted_klines)):
                current = sorted_klines[i]
                previous = sorted_klines[i - 1]

                price_change = abs(current.open_price - previous.close_price) / previous.close_price
                if price_change > Decimal("0.5"):  # 50%的價格變化
                    result.add_issue(
                        code="ABNORMAL_PRICE_CHANGE",
                        severity=ValidationSeverity.WARNING,
                        message=f"發現異常價格變化: {price_change:.2%}",
                        field_path="open_price,close_price",
                        actual_value=f"{price_change:.2%}",
                        suggestion="檢查價格數據是否存在錯誤",
                    )
