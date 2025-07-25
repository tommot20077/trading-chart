# ABOUTME: TradingPair 模型的單元測試
# ABOUTME: 測試交易對模型的創建、驗證、狀態管理和業務邏輯

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import ValidationError

from core.models.data.trading_pair import TradingPair, TradingPairStatus
from core.models.data.enum import AssetClass


class TestTradingPairCreation:
    """測試交易對創建"""
    
    def test_minimal_trading_pair_creation(self):
        """測試最小化交易對創建"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT"
        )
        
        assert trading_pair.symbol == "BTC/USDT"
        assert trading_pair.base_currency == "BTC"
        assert trading_pair.quote_currency == "USDT"
        assert trading_pair.status == TradingPairStatus.ACTIVE
        assert trading_pair.price_precision == 8
        assert trading_pair.quantity_precision == 8
        assert trading_pair.min_trade_quantity == Decimal("0.00000001")
        assert trading_pair.max_trade_quantity == Decimal("1000000")
        assert trading_pair.maker_fee_rate == Decimal("0.001")
        assert trading_pair.taker_fee_rate == Decimal("0.001")
        assert trading_pair.asset_class == AssetClass.DIGITAL
        assert isinstance(trading_pair.created_at, datetime)
        assert isinstance(trading_pair.updated_at, datetime)
    
    def test_full_trading_pair_creation(self):
        """測試完整交易對創建"""
        trading_pair = TradingPair(
            symbol="ETH/BTC",
            base_currency="ETH",
            quote_currency="BTC",
            status=TradingPairStatus.DISABLED,
            price_precision=6,
            quantity_precision=4,
            min_trade_quantity=Decimal("0.01"),
            max_trade_quantity=Decimal("1000"),
            maker_fee_rate=Decimal("0.0005"),
            taker_fee_rate=Decimal("0.001"),
            asset_class=AssetClass.DIGITAL
        )
        
        assert trading_pair.symbol == "ETH/BTC"
        assert trading_pair.base_currency == "ETH"
        assert trading_pair.quote_currency == "BTC"
        assert trading_pair.status == TradingPairStatus.DISABLED
        assert trading_pair.price_precision == 6
        assert trading_pair.quantity_precision == 4
        assert trading_pair.min_trade_quantity == Decimal("0.01")
        assert trading_pair.max_trade_quantity == Decimal("1000")
        assert trading_pair.maker_fee_rate == Decimal("0.0005")
        assert trading_pair.taker_fee_rate == Decimal("0.001")


class TestTradingPairValidation:
    """測試交易對驗證"""
    
    def test_symbol_validation(self):
        """測試交易對符號驗證"""
        # 測試有效符號
        valid_symbols = ["BTC/USDT", "eth/btc", "DOT/USD"]
        for symbol in valid_symbols:
            trading_pair = TradingPair(
                symbol=symbol,
                base_currency="BTC",
                quote_currency="USDT"
            )
            assert trading_pair.symbol == symbol.upper()
        
        # 測試沒有分隔符的符號 (現在支援)
        trading_pair = TradingPair(
            symbol="BTCUSDT",  # 沒有分隔符也支援
            base_currency="BTC",
            quote_currency="USDT"
        )
        assert trading_pair.symbol == "BTCUSDT"
        
        with pytest.raises(ValidationError, match="交易對格式必須為"):
            TradingPair(
                symbol="BTC/USDT/ETH",  # 多個分隔符
                base_currency="BTC",
                quote_currency="USDT"
            )
        
        with pytest.raises(ValidationError, match="基礎貨幣和報價貨幣都不能為空"):
            TradingPair(
                symbol="/USDT",  # 空基礎貨幣
                base_currency="",
                quote_currency="USDT"
            )
    
    def test_currency_consistency_validation(self):
        """測試貨幣一致性驗證"""
        # 測試貨幣與符號不一致
        with pytest.raises(ValidationError, match="符號中的基礎貨幣必須與"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="ETH",  # 與符號不匹配
                quote_currency="USDT"
            )
        
        with pytest.raises(ValidationError, match="符號中的報價貨幣必須與"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="BTC"  # 與符號不匹配
            )
    
    def test_precision_validation(self):
        """測試精度驗證"""
        # 測試負精度
        with pytest.raises(ValidationError, match="精度必須大於等於0"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="USDT",
                price_precision=-1
            )
        
        # 測試過大精度
        with pytest.raises(ValidationError, match="精度不能超過18"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="USDT",
                price_precision=20
            )
    
    def test_trade_quantity_validation(self):
        """測試交易量限制驗證"""
        # 測試負數最小交易量
        with pytest.raises(ValidationError, match="最小交易量必須大於0"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="USDT",
                min_trade_quantity=Decimal("-1")
            )
        
        # 測試最大交易量小於最小交易量
        with pytest.raises(ValidationError, match="最大交易量必須大於等於最小交易量"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="USDT",
                min_trade_quantity=Decimal("10"),
                max_trade_quantity=Decimal("5")
            )
    
    def test_fee_rate_validation(self):
        """測試手續費率驗證"""
        # 測試負手續費率  
        with pytest.raises(ValidationError, match="手續費率必須大於等於0"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="USDT",
                maker_fee_rate=Decimal("-0.01")
            )
        
        # 測試過高手續費率
        with pytest.raises(ValidationError, match="手續費率不能超過10%"):
            TradingPair(
                symbol="BTC/USDT",
                base_currency="BTC",
                quote_currency="USDT",
                taker_fee_rate=Decimal("0.2")
            )


class TestTradingPairProperties:
    """測試交易對屬性"""
    
    def test_is_active_property(self):
        """測試 is_active 屬性"""
        active_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            status=TradingPairStatus.ACTIVE
        )
        assert active_pair.is_active is True
        
        disabled_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            status=TradingPairStatus.DISABLED
        )
        assert disabled_pair.is_active is False
    
    def test_display_name_property(self):
        """測試 display_name 屬性"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT"
        )
        assert trading_pair.display_name == "BTC/USDT"
    
    def test_inverse_symbol_property(self):
        """測試 inverse_symbol 屬性"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT"
        )
        assert trading_pair.inverse_symbol == "USDT/BTC"


class TestTradingPairMethods:
    """測試交易對方法"""
    
    def test_update_status(self):
        """測試狀態更新"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            status=TradingPairStatus.ACTIVE
        )
        
        original_updated_at = trading_pair.updated_at
        
        # 測試狀態變更
        trading_pair.update_status(TradingPairStatus.DISABLED)
        assert trading_pair.status == TradingPairStatus.DISABLED
        assert trading_pair.updated_at > original_updated_at
        
        # 測試相同狀態不改變時間戳
        same_status_updated_at = trading_pair.updated_at
        trading_pair.update_status(TradingPairStatus.DISABLED)
        assert trading_pair.updated_at == same_status_updated_at
    
    def test_calculate_maker_fee(self):
        """測試 maker 手續費計算"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            maker_fee_rate=Decimal("0.001")
        )
        
        quantity = Decimal("1.5")
        price = Decimal("50000")
        fee = trading_pair.calculate_maker_fee(quantity, price)
        expected_fee = quantity * price * Decimal("0.001")
        assert fee == expected_fee
    
    def test_calculate_taker_fee(self):
        """測試 taker 手續費計算"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            taker_fee_rate=Decimal("0.0015")
        )
        
        quantity = Decimal("2.0")
        price = Decimal("45000")
        fee = trading_pair.calculate_taker_fee(quantity, price)
        expected_fee = quantity * price * Decimal("0.0015")
        assert fee == expected_fee
    
    def test_is_quantity_valid(self):
        """測試交易量驗證"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            min_trade_quantity=Decimal("0.01"),
            max_trade_quantity=Decimal("100")
        )
        
        # 有效數量
        assert trading_pair.is_quantity_valid(Decimal("1.0")) is True
        assert trading_pair.is_quantity_valid(Decimal("0.01")) is True
        assert trading_pair.is_quantity_valid(Decimal("100")) is True
        
        # 無效數量
        assert trading_pair.is_quantity_valid(Decimal("0.005")) is False
        assert trading_pair.is_quantity_valid(Decimal("150")) is False
        assert trading_pair.is_quantity_valid(Decimal("0")) is False
    
    def test_format_price(self):
        """測試價格格式化"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            price_precision=2
        )
        
        # 測試正常價格
        formatted = trading_pair.format_price(Decimal("50000.123456"))
        assert formatted == Decimal("50000.12")
        
        # 測試精度截斷
        formatted = trading_pair.format_price(Decimal("1.999"))
        assert formatted == Decimal("1.99")
    
    def test_format_quantity(self):
        """測試數量格式化"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            quantity_precision=4
        )
        
        # 測試正常數量
        formatted = trading_pair.format_quantity(Decimal("1.123456789"))
        assert formatted == Decimal("1.1234")
        
        # 測試精度截斷
        formatted = trading_pair.format_quantity(Decimal("0.99999"))
        assert formatted == Decimal("0.9999")


class TestTradingPairSerialization:
    """測試交易對序列化"""
    
    def test_to_dict(self):
        """測試轉換為字典"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT",
            price_precision=2,
            maker_fee_rate=Decimal("0.001")
        )
        
        data = trading_pair.to_dict()
        
        assert data["symbol"] == "BTC/USDT"
        assert data["base_currency"] == "BTC"
        assert data["quote_currency"] == "USDT"
        assert data["price_precision"] == 2
        assert data["maker_fee_rate"] == "0.001"
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_string_representation(self):
        """測試字符串表示"""
        trading_pair = TradingPair(
            symbol="BTC/USDT",
            base_currency="BTC",
            quote_currency="USDT"
        )
        
        str_repr = str(trading_pair)
        assert "BTC/USDT" in str_repr
        assert "ACTIVE" in str_repr
        
        repr_str = repr(trading_pair)
        assert "TradingPair" in repr_str
        assert "BTC/USDT" in repr_str


class TestTradingPairStatus:
    """測試交易對狀態枚舉"""
    
    def test_status_values(self):
        """測試狀態值"""
        assert TradingPairStatus.ACTIVE.value == "active"
        assert TradingPairStatus.DISABLED.value == "disabled"
        assert TradingPairStatus.SUSPENDED.value == "suspended"
        assert TradingPairStatus.DELISTED.value == "delisted"
    
    def test_is_tradeable_method(self):
        """測試 is_tradeable 方法"""
        assert TradingPairStatus.ACTIVE.is_tradeable() is True
        assert TradingPairStatus.DISABLED.is_tradeable() is False
        assert TradingPairStatus.SUSPENDED.is_tradeable() is False
        assert TradingPairStatus.DELISTED.is_tradeable() is False