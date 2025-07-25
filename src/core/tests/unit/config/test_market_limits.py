# ABOUTME: Unit tests for market limits configuration
# ABOUTME: Tests validation limits, precision checks, and asset class configurations

import pytest
from decimal import Decimal

from core.config.market_limits import (
    MarketLimits,
    MarketLimitsConfig,
    DEFAULT_LIMITS,
    get_market_limits_config,
)


class TestMarketLimits:
    """Test suite for MarketLimits dataclass."""
    
    def test_valid_market_limits(self):
        """Test creating valid market limits."""
        limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("10000.00"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("1000000.00"),
            price_precision=2,
            quantity_precision=3,
        )
        
        assert limits.min_price == Decimal("0.01")
        assert limits.max_price == Decimal("10000.00")
        assert limits.min_quantity == Decimal("0.001")
        assert limits.max_quantity == Decimal("1000000.00")
        assert limits.price_precision == 2
        assert limits.quantity_precision == 3
    
    def test_min_price_validation(self):
        """Test that min_price must be greater than 0."""
        with pytest.raises(ValueError, match="min_price must be greater than 0"):
            MarketLimits(
                min_price=Decimal("0"),
                max_price=Decimal("100"),
                min_quantity=Decimal("1"),
                max_quantity=Decimal("1000"),
            )
    
    def test_max_price_validation(self):
        """Test that max_price must be greater than min_price."""
        with pytest.raises(ValueError, match="max_price must be greater than min_price"):
            MarketLimits(
                min_price=Decimal("100"),
                max_price=Decimal("100"),
                min_quantity=Decimal("1"),
                max_quantity=Decimal("1000"),
            )
    
    def test_min_quantity_validation(self):
        """Test that min_quantity must be greater than 0."""
        with pytest.raises(ValueError, match="min_quantity must be greater than 0"):
            MarketLimits(
                min_price=Decimal("1"),
                max_price=Decimal("100"),
                min_quantity=Decimal("0"),
                max_quantity=Decimal("1000"),
            )
    
    def test_max_quantity_validation(self):
        """Test that max_quantity must be greater than min_quantity."""
        with pytest.raises(ValueError, match="max_quantity must be greater than min_quantity"):
            MarketLimits(
                min_price=Decimal("1"),
                max_price=Decimal("100"),
                min_quantity=Decimal("100"),
                max_quantity=Decimal("100"),
            )
    
    def test_precision_validation(self):
        """Test that precision values must be non-negative."""
        with pytest.raises(ValueError, match="price_precision must be non-negative"):
            MarketLimits(
                min_price=Decimal("1"),
                max_price=Decimal("100"),
                min_quantity=Decimal("1"),
                max_quantity=Decimal("1000"),
                price_precision=-1,
            )
        
        with pytest.raises(ValueError, match="quantity_precision must be non-negative"):
            MarketLimits(
                min_price=Decimal("1"),
                max_price=Decimal("100"),
                min_quantity=Decimal("1"),
                max_quantity=Decimal("1000"),
                quantity_precision=-1,
            )


class TestDefaultLimits:
    """Test suite for default market limits."""
    
    def test_crypto_limits(self):
        """Test default crypto market limits."""
        limits = DEFAULT_LIMITS["CRYPTO"]
        assert limits.min_price == Decimal("0.00000001")
        assert limits.max_price == Decimal("10000000.00")
        assert limits.min_quantity == Decimal("0.00000001")
        assert limits.max_quantity == Decimal("1000000000.00")
        assert limits.price_precision == 8
        assert limits.quantity_precision == 8
    
    def test_forex_limits(self):
        """Test default forex market limits."""
        limits = DEFAULT_LIMITS["FOREX"]
        assert limits.min_price == Decimal("0.00001")
        assert limits.max_price == Decimal("10000.00")
        assert limits.min_quantity == Decimal("0.01")
        assert limits.max_quantity == Decimal("100000000.00")
        assert limits.price_precision == 5
        assert limits.quantity_precision == 2
    
    def test_stock_limits(self):
        """Test default stock market limits."""
        limits = DEFAULT_LIMITS["STOCK"]
        assert limits.min_price == Decimal("0.01")
        assert limits.max_price == Decimal("1000000.00")
        assert limits.min_quantity == Decimal("1")
        assert limits.max_quantity == Decimal("1000000000.00")
        assert limits.price_precision == 2
        assert limits.quantity_precision == 0
    
    def test_commodity_limits(self):
        """Test default commodity market limits."""
        limits = DEFAULT_LIMITS["COMMODITY"]
        assert limits.min_price == Decimal("0.01")
        assert limits.max_price == Decimal("100000.00")
        assert limits.min_quantity == Decimal("0.001")
        assert limits.max_quantity == Decimal("10000000.00")
        assert limits.price_precision == 2
        assert limits.quantity_precision == 3


class TestMarketLimitsConfig:
    """Test suite for MarketLimitsConfig."""
    
    def test_default_initialization(self):
        """Test default initialization of config."""
        config = MarketLimitsConfig()
        assert config._default_asset_class == "CRYPTO"
        assert len(config._symbol_limits) == 0
    
    def test_set_and_get_symbol_limits(self):
        """Test setting and getting symbol-specific limits."""
        config = MarketLimitsConfig()
        custom_limits = MarketLimits(
            min_price=Decimal("0.1"),
            max_price=Decimal("1000.00"),
            min_quantity=Decimal("10"),
            max_quantity=Decimal("10000.00"),
            price_precision=1,
            quantity_precision=0,
        )
        
        config.set_symbol_limits("AAPL", custom_limits)
        retrieved_limits = config.get_limits("AAPL")
        
        assert retrieved_limits == custom_limits
        assert retrieved_limits.min_price == Decimal("0.1")
        assert retrieved_limits.price_precision == 1
    
    def test_get_limits_fallback_to_asset_class(self):
        """Test fallback to asset class defaults when symbol not found."""
        config = MarketLimitsConfig()
        
        # Default is CRYPTO
        limits = config.get_limits("BTCUSDT")
        assert limits == DEFAULT_LIMITS["CRYPTO"]
        
        # With explicit asset class
        limits = config.get_limits("EURUSD", asset_class="FOREX")
        assert limits == DEFAULT_LIMITS["FOREX"]
    
    def test_set_default_asset_class(self):
        """Test setting default asset class."""
        config = MarketLimitsConfig()
        
        config.set_default_asset_class("STOCK")
        assert config._default_asset_class == "STOCK"
        
        # Should now use STOCK as default
        limits = config.get_limits("UNKNOWN")
        assert limits == DEFAULT_LIMITS["STOCK"]
    
    def test_set_invalid_asset_class(self):
        """Test that invalid asset class raises error."""
        config = MarketLimitsConfig()
        
        with pytest.raises(ValueError, match="Unknown asset class: INVALID"):
            config.set_default_asset_class("INVALID")
    
    def test_symbol_limits_override_asset_class(self):
        """Test that symbol-specific limits override asset class."""
        config = MarketLimitsConfig()
        custom_limits = MarketLimits(
            min_price=Decimal("1"),
            max_price=Decimal("100"),
            min_quantity=Decimal("1"),
            max_quantity=Decimal("1000"),
        )
        
        config.set_symbol_limits("BTCUSDT", custom_limits)
        
        # Even with asset class specified, symbol limits take precedence
        limits = config.get_limits("BTCUSDT", asset_class="FOREX")
        assert limits == custom_limits


class TestGlobalConfig:
    """Test suite for global configuration instance."""
    
    def test_get_market_limits_config(self):
        """Test getting the global config instance."""
        config1 = get_market_limits_config()
        config2 = get_market_limits_config()
        
        # Should be the same instance
        assert config1 is config2
        
        # Should be a MarketLimitsConfig instance
        assert isinstance(config1, MarketLimitsConfig)