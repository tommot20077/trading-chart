# ABOUTME: 跨模型驗證器的單元測試，測試Order、TradingPair、Trade等模型間的一致性檢查
# ABOUTME: 驗證業務規則驗證功能，包括價格精度、數量限制、時間一致性等驗證邏輯

import pytest
from decimal import Decimal
from datetime import datetime, UTC, timedelta
from uuid import uuid4

from core.validators.cross_model_validator import CrossModelValidator
from core.validators.validation_result import ValidationSeverity
from core.models.data.order import Order, OrderType, OrderSide, OrderStatus
from core.models.data.trading_pair import TradingPair, TradingPairStatus, AssetClass
from core.models.data.trade import Trade, TradeSide
from core.models.data.kline import Kline
from core.models.data.market_data import MarketData
from core.config.market_limits import MarketLimitsConfig, MarketLimits


class TestCrossModelValidator:
    """跨模型驗證器測試類."""

    @pytest.fixture
    def validator(self):
        """初始化驗證器."""
        config = MarketLimitsConfig()
        return CrossModelValidator(config)

    @pytest.fixture
    def valid_trading_pair(self):
        """創建有效的交易對."""
        return TradingPair(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            asset_class=AssetClass.DIGITAL,
            status=TradingPairStatus.ACTIVE,
            price_precision=2,
            quantity_precision=8,
            min_trade_quantity=Decimal("0.00100000"),
            max_trade_quantity=Decimal("10000.00000000"),
            min_notional=Decimal("10.00"),
            taker_fee=Decimal("0.001"),
            maker_fee=Decimal("0.001"),
        )

    @pytest.fixture
    def valid_order(self):
        """創建有效的訂單."""
        return Order(
            user_id=uuid4(),
            trading_pair="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1.00000000"),
            price=Decimal("50000.00"),
            status=OrderStatus.PENDING,
        )

    @pytest.fixture
    def valid_trade(self):
        """創建有效的交易記錄."""
        return Trade(
            trade_id="trade_789",
            symbol="BTCUSDT",
            price=Decimal("50000.00"),
            quantity=Decimal("1.00000000"),
            side=TradeSide.BUY,
            timestamp=datetime.now(UTC),
            is_buyer_maker=True,
        )

    def test_order_trading_pair_consistency_valid(self, validator, valid_order, valid_trading_pair):
        """測試訂單與交易對一致性驗證 - 成功案例."""
        result = validator.validate_order_trading_pair_consistency(valid_order, valid_trading_pair)
        
        assert result.is_valid
        assert len(result.issues) == 0
        assert "Order" in result.validated_models
        assert "TradingPair" in result.validated_models

    def test_order_symbol_mismatch(self, validator, valid_order, valid_trading_pair):
        """測試訂單符號不匹配."""
        # 創建一個新的Order實例而不是修改現有的
        mismatched_order = Order(
            user_id=valid_order.user_id,
            trading_pair="ETH/USDT",  # 使用不同的交易對
            order_type=valid_order.order_type,
            side=valid_order.side,
            quantity=valid_order.quantity,
            price=valid_order.price,
            status=valid_order.status,
        )
        
        result = validator.validate_order_trading_pair_consistency(mismatched_order, valid_trading_pair)
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].code == "ORDER_SYMBOL_MISMATCH"
        assert result.issues[0].severity == ValidationSeverity.ERROR

    def test_trading_pair_not_active(self, validator, valid_order, valid_trading_pair):
        """測試交易對非活躍狀態."""
        valid_trading_pair.status = TradingPairStatus.SUSPENDED
        
        result = validator.validate_order_trading_pair_consistency(valid_order, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "TRADING_PAIR_NOT_ACTIVE" for issue in result.issues)

    def test_order_quantity_too_small(self, validator, valid_order, valid_trading_pair):
        """測試訂單數量小於最小限制."""
        small_quantity_order = Order(
            user_id=valid_order.user_id,
            trading_pair=valid_order.trading_pair,
            order_type=valid_order.order_type,
            side=valid_order.side,
            quantity=Decimal("0.00050000"),  # 小於最小交易量
            price=valid_order.price,
            status=valid_order.status,
        )
        
        result = validator.validate_order_trading_pair_consistency(small_quantity_order, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "ORDER_MIN_QUANTITY" for issue in result.issues)

    def test_order_quantity_too_large(self, validator, valid_order, valid_trading_pair):
        """測試訂單數量超過最大限制."""
        large_quantity_order = Order(
            user_id=valid_order.user_id,
            trading_pair=valid_order.trading_pair,
            order_type=valid_order.order_type,
            side=valid_order.side,
            quantity=Decimal("20000.00000000"),  # 超過最大交易量
            price=valid_order.price,
            status=valid_order.status,
        )
        
        result = validator.validate_order_trading_pair_consistency(large_quantity_order, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "ORDER_MAX_QUANTITY" for issue in result.issues)

    def test_order_price_precision_error(self, validator, valid_order, valid_trading_pair):
        """測試訂單價格精度錯誤."""
        precision_error_order = Order(
            user_id=valid_order.user_id,
            trading_pair=valid_order.trading_pair,
            order_type=valid_order.order_type,
            side=valid_order.side,
            quantity=valid_order.quantity,
            price=Decimal("50000.123"),  # 超過2位小數精度
            status=valid_order.status,
        )
        
        result = validator.validate_order_trading_pair_consistency(precision_error_order, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "ORDER_PRICE_PRECISION" for issue in result.issues)

    def test_order_min_notional_error(self, validator, valid_order, valid_trading_pair):
        """測試訂單最小名義價值錯誤."""
        min_notional_error_order = Order(
            user_id=valid_order.user_id,
            trading_pair=valid_order.trading_pair,
            order_type=valid_order.order_type,
            side=valid_order.side,
            quantity=Decimal("0.00100000"),
            price=Decimal("5.00"),  # 名義價值 = 0.001 * 5 = 0.005 < 10
            status=valid_order.status,
        )
        
        result = validator.validate_order_trading_pair_consistency(min_notional_error_order, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "ORDER_MIN_NOTIONAL" for issue in result.issues)

    def test_trade_market_rules_valid(self, validator, valid_trade, valid_trading_pair):
        """測試交易市場規則驗證 - 成功案例."""
        result = validator.validate_trade_market_rules(valid_trade, valid_trading_pair)
        
        assert result.is_valid
        assert len(result.issues) == 0

    def test_trade_symbol_mismatch(self, validator, valid_trade, valid_trading_pair):
        """測試交易符號不匹配."""
        mismatched_trade = Trade(
            trade_id=valid_trade.trade_id,
            symbol="ETHUSDT",
            price=valid_trade.price,
            quantity=valid_trade.quantity,
            side=valid_trade.side,
            timestamp=valid_trade.timestamp,
            is_buyer_maker=valid_trade.is_buyer_maker,
        )
        
        result = validator.validate_trade_market_rules(mismatched_trade, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "TRADE_SYMBOL_MISMATCH" for issue in result.issues)

    def test_trade_price_precision_error(self, validator, valid_trade, valid_trading_pair):
        """測試交易價格精度錯誤."""
        precision_error_trade = Trade(
            trade_id=valid_trade.trade_id,
            symbol=valid_trade.symbol,
            price=Decimal("50000.123"),  # 超過2位小數精度
            quantity=valid_trade.quantity,
            side=valid_trade.side,
            timestamp=valid_trade.timestamp,
            is_buyer_maker=valid_trade.is_buyer_maker,
        )
        
        result = validator.validate_trade_market_rules(precision_error_trade, valid_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "TRADE_PRICE_PRECISION" for issue in result.issues)

    def test_trade_quantity_precision_error(self, validator, valid_trade, valid_trading_pair):
        """測試交易數量精度錯誤."""
        # 由於Trade模型內部也會進行精度驗證，我們需要直接測試驗證器的精度檢查邏輯
        # 創建一個精度為8位的有效trade，但在驗證器中使用更嚴格的trading_pair限制
        valid_trade_copy = Trade(
            trade_id=valid_trade.trade_id,
            symbol=valid_trade.symbol,
            price=valid_trade.price,
            quantity=Decimal("1.12345678"),  # 8位精度，在模型內有效
            side=valid_trade.side,
            timestamp=valid_trade.timestamp,
            is_buyer_maker=valid_trade.is_buyer_maker,
        )
        
        # 創建一個要求更高精度限制的trading_pair
        strict_trading_pair = TradingPair(
            symbol=valid_trading_pair.symbol,
            base_currency=valid_trading_pair.base_currency,
            quote_currency=valid_trading_pair.quote_currency,
            asset_class=valid_trading_pair.asset_class,
            status=valid_trading_pair.status,
            price_precision=valid_trading_pair.price_precision,
            quantity_precision=6,  # 更嚴格的6位精度限制
            min_trade_quantity=valid_trading_pair.min_trade_quantity,
            max_trade_quantity=valid_trading_pair.max_trade_quantity,
            min_notional=valid_trading_pair.min_notional,
            maker_fee_rate=valid_trading_pair.maker_fee_rate,
            taker_fee_rate=valid_trading_pair.taker_fee_rate,
        )
        
        result = validator.validate_trade_market_rules(valid_trade_copy, strict_trading_pair)
        
        assert not result.is_valid
        assert any(issue.code == "TRADE_QUANTITY_PRECISION" for issue in result.issues)

    def test_trade_quantity_limits(self, validator, valid_trade, valid_trading_pair):
        """測試交易數量限制."""
        # 測試數量過小
        small_quantity_trade = Trade(
            trade_id=valid_trade.trade_id,
            symbol=valid_trade.symbol,
            price=valid_trade.price,
            quantity=Decimal("0.00050000"),
            side=valid_trade.side,
            timestamp=valid_trade.timestamp,
            is_buyer_maker=valid_trade.is_buyer_maker,
        )
        result = validator.validate_trade_market_rules(small_quantity_trade, valid_trading_pair)
        assert not result.is_valid
        assert any(issue.code == "TRADE_MIN_QUANTITY" for issue in result.issues)

        # 測試數量過大
        large_quantity_trade = Trade(
            trade_id=valid_trade.trade_id,
            symbol=valid_trade.symbol,
            price=valid_trade.price,
            quantity=Decimal("20000.00000000"),
            side=valid_trade.side,
            timestamp=valid_trade.timestamp,
            is_buyer_maker=valid_trade.is_buyer_maker,
        )
        result = validator.validate_trade_market_rules(large_quantity_trade, valid_trading_pair)
        assert not result.is_valid
        assert any(issue.code == "TRADE_MAX_QUANTITY" for issue in result.issues)

    def test_kline_trade_consistency_empty_data(self, validator):
        """測試空數據的K線交易一致性."""
        result = validator.validate_kline_trade_consistency([], [])
        assert result.is_valid
        assert len(result.issues) == 0

    def test_kline_trade_symbol_consistency(self, validator):
        """測試K線和交易數據的符號一致性."""
        base_time = datetime.now(UTC)
        
        klines = [
            Kline(
                symbol="BTCUSDT",
                interval="1m",
                open_time=base_time,
                close_time=base_time + timedelta(minutes=1),
                open_price=Decimal("50000.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49900.00"),
                close_price=Decimal("50050.00"),
                volume=Decimal("10.0"),
                quote_volume=Decimal("500000.0"),
                trades_count=100,
                taker_buy_volume=Decimal("5.0"),
                taker_buy_quote_volume=Decimal("250000.0"),
            )
        ]
        
        trades = [
            Trade(
                trade_id="trade_1",
                symbol="ETHUSDT",  # 不同符號
                price=Decimal("50000.00"),
                quantity=Decimal("1.0"),
                side=TradeSide.BUY,
                timestamp=base_time + timedelta(seconds=30),
                is_buyer_maker=True,
            )
        ]
        
        result = validator.validate_kline_trade_consistency(klines, trades)
        
        assert not result.is_valid
        assert any(issue.code == "DATA_SYMBOL_INCONSISTENCY" for issue in result.issues)

    def test_market_data_integrity_valid(self, validator):
        """測試市場數據完整性驗證 - 成功案例."""
        base_time = datetime.now(UTC)
        
        market_data = MarketData(
            symbol="BTCUSDT",
            klines=[
                Kline(
                    symbol="BTCUSDT",
                    interval="1m",
                    open_time=base_time,
                    close_time=base_time + timedelta(minutes=1),
                    open_price=Decimal("50000.00"),
                    high_price=Decimal("50100.00"),
                    low_price=Decimal("49900.00"),
                    close_price=Decimal("50050.00"),
                    volume=Decimal("10.0"),
                    quote_volume=Decimal("500000.0"),
                    trades_count=100,
                    taker_buy_volume=Decimal("5.0"),
                    taker_buy_quote_volume=Decimal("250000.0"),
                )
            ],
            trades=[
                Trade(
                    trade_id="trade_1",
                    symbol="BTCUSDT",
                    price=Decimal("50000.00"),
                    quantity=Decimal("1.0"),
                    side=TradeSide.BUY,
                    timestamp=base_time + timedelta(seconds=30),
                    is_buyer_maker=True,
                )
            ],
        )
        
        result = validator.validate_market_data_integrity(market_data)
        
        assert result.is_valid

    def test_decimal_precision_check_helper(self, validator):
        """測試小數精度檢查輔助方法."""
        # 精度符合要求
        assert validator._check_decimal_precision(Decimal("123.45"), 2)
        assert validator._check_decimal_precision(Decimal("123.4"), 2)
        assert validator._check_decimal_precision(Decimal("123"), 2)
        
        # 精度超出要求
        assert not validator._check_decimal_precision(Decimal("123.456"), 2)

    def test_validation_result_properties(self, validator, valid_order, valid_trading_pair):
        """測試驗證結果的屬性方法."""
        # 創建有多個問題的場景
        problematic_order = Order(
            user_id=valid_order.user_id,
            trading_pair="ETH/USDT",  # 符號不匹配
            order_type=valid_order.order_type,
            side=valid_order.side,
            quantity=Decimal("0.00050000"),  # 數量過小
            price=valid_order.price,
            status=valid_order.status,
        )
        valid_trading_pair.status = TradingPairStatus.SUSPENDED  # 非活躍狀態
        
        result = validator.validate_order_trading_pair_consistency(problematic_order, valid_trading_pair)
        
        assert not result.is_valid
        assert result.has_errors
        assert result.error_count > 0
        assert len(result.get_issues_by_severity(ValidationSeverity.ERROR)) > 0
        
        summary = result.get_summary()
        assert summary["is_valid"] is False
        assert summary["total_issues"] > 0
        assert summary["error_count"] > 0

    def test_abnormal_volume_detection(self, validator):
        """測試異常交易量檢測."""
        base_time = datetime.now(UTC)
        
        market_data = MarketData(
            symbol="BTCUSDT",
            klines=[],
            trades=[
                Trade(
                    trade_id="trade_large",
                    symbol="BTCUSDT",
                    price=Decimal("50000.00"),
                    quantity=Decimal("2000000.0"),  # 異常大交易量
                    side=TradeSide.BUY,
                    timestamp=base_time,
                    is_buyer_maker=True,
                )
            ],
        )
        
        result = validator.validate_market_data_integrity(market_data)
        
        assert any(issue.code == "ABNORMAL_TRADE_VOLUME" for issue in result.issues)

    def test_abnormal_price_change_detection(self, validator):
        """測試異常價格變化檢測."""
        base_time = datetime.now(UTC)
        
        market_data = MarketData(
            symbol="BTCUSDT",
            klines=[
                Kline(
                    symbol="BTCUSDT",
                    interval="1m",
                    open_time=base_time,
                    close_time=base_time + timedelta(minutes=1),
                    open_price=Decimal("50000.00"),
                    high_price=Decimal("50000.00"),
                    low_price=Decimal("50000.00"),
                    close_price=Decimal("50000.00"),
                    volume=Decimal("10.0"),
                    quote_volume=Decimal("500000.0"),
                    trades_count=100,
                    taker_buy_volume=Decimal("5.0"),
                    taker_buy_quote_volume=Decimal("250000.0"),
                ),
                Kline(
                    symbol="BTCUSDT",
                    interval="1m",
                    open_time=base_time + timedelta(minutes=1),
                    close_time=base_time + timedelta(minutes=2),
                    open_price=Decimal("100000.00"),  # 100%價格變化
                    high_price=Decimal("100000.00"),
                    low_price=Decimal("100000.00"),
                    close_price=Decimal("100000.00"),
                    volume=Decimal("10.0"),
                    quote_volume=Decimal("1000000.0"),
                    trades_count=100,
                    taker_buy_volume=Decimal("5.0"),
                    taker_buy_quote_volume=Decimal("500000.0"),
                ),
            ],
            trades=[],
        )
        
        result = validator.validate_market_data_integrity(market_data)
        
        assert any(issue.code == "ABNORMAL_PRICE_CHANGE" for issue in result.issues)