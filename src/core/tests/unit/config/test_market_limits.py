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

    @pytest.mark.unit
    @pytest.mark.config
    def test_market_limits_frozen_immutable(self):
        """Test that MarketLimits is immutable (frozen)."""
        limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("100.00"),
            min_quantity=Decimal("1"),
            max_quantity=Decimal("1000"),
        )
        
        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            limits.min_price = Decimal("0.02")
        
        with pytest.raises(AttributeError):
            limits.price_precision = 4

    @pytest.mark.unit
    @pytest.mark.config
    def test_negative_price_values(self):
        """Test that negative price values are rejected."""
        with pytest.raises(ValueError, match="min_price must be greater than 0"):
            MarketLimits(
                min_price=Decimal("-0.01"),
                max_price=Decimal("100"),
                min_quantity=Decimal("1"),
                max_quantity=Decimal("1000"),
            )

    @pytest.mark.unit
    @pytest.mark.config
    def test_negative_quantity_values(self):
        """Test that negative quantity values are rejected."""
        with pytest.raises(ValueError, match="min_quantity must be greater than 0"):
            MarketLimits(
                min_price=Decimal("0.01"),
                max_price=Decimal("100"),
                min_quantity=Decimal("-1"),
                max_quantity=Decimal("1000"),
            )

    @pytest.mark.unit
    @pytest.mark.config
    def test_default_precision_values(self):
        """Test that default precision values are applied correctly."""
        limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("100.00"),
            min_quantity=Decimal("1"),
            max_quantity=Decimal("1000"),
        )
        
        # Should use default precision of 8
        assert limits.price_precision == 8
        assert limits.quantity_precision == 8

    @pytest.mark.unit
    @pytest.mark.config
    def test_zero_precision_allowed(self):
        """Test that zero precision values are allowed."""
        limits = MarketLimits(
            min_price=Decimal("1"),
            max_price=Decimal("100"),
            min_quantity=Decimal("1"),
            max_quantity=Decimal("1000"),
            price_precision=0,
            quantity_precision=0,
        )
        
        assert limits.price_precision == 0
        assert limits.quantity_precision == 0

    @pytest.mark.unit
    @pytest.mark.config
    def test_very_small_decimal_precision(self):
        """Test handling of very small decimal values (satoshi-level)."""
        limits = MarketLimits(
            min_price=Decimal("0.00000001"),  # 1 satoshi
            max_price=Decimal("0.00000002"),
            min_quantity=Decimal("0.00000001"),
            max_quantity=Decimal("0.00000002"),
        )
        
        assert limits.min_price == Decimal("0.00000001")
        assert limits.max_price == Decimal("0.00000002")

    @pytest.mark.unit
    @pytest.mark.config
    def test_very_large_decimal_values(self):
        """Test handling of very large decimal values."""
        limits = MarketLimits(
            min_price=Decimal("1000000"),
            max_price=Decimal("999999999999.99"),
            min_quantity=Decimal("1000000"),
            max_quantity=Decimal("999999999999.99"),
        )
        
        assert limits.max_price == Decimal("999999999999.99")
        assert limits.max_quantity == Decimal("999999999999.99")

    @pytest.mark.unit
    @pytest.mark.config
    def test_all_default_limits_are_valid(self):
        """Test that all predefined default limits are valid."""
        for asset_class, limits in DEFAULT_LIMITS.items():
            # Each limit should satisfy validation constraints
            assert limits.min_price > 0
            assert limits.max_price > limits.min_price
            assert limits.min_quantity > 0
            assert limits.max_quantity > limits.min_quantity
            assert limits.price_precision >= 0
            assert limits.quantity_precision >= 0

    @pytest.mark.unit
    @pytest.mark.config
    def test_expected_default_asset_classes(self):
        """Test that all expected default asset classes are present."""
        expected_classes = {"CRYPTO", "FOREX", "STOCK", "COMMODITY"}
        assert set(DEFAULT_LIMITS.keys()) == expected_classes

    @pytest.mark.unit
    @pytest.mark.config
    def test_config_multiple_symbols(self):
        """Test configuration with multiple symbol-specific limits."""
        config = MarketLimitsConfig()
        
        btc_limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("100000"),
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("1000"),
        )
        
        eth_limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("10000"),
            min_quantity=Decimal("0.0001"),
            max_quantity=Decimal("10000"),
        )
        
        config.set_symbol_limits("BTCUSDT", btc_limits)
        config.set_symbol_limits("ETHUSDT", eth_limits)
        
        assert config.get_limits("BTCUSDT") == btc_limits
        assert config.get_limits("ETHUSDT") == eth_limits
        assert config.get_limits("ADAUSDT") == DEFAULT_LIMITS["CRYPTO"]  # Fallback

    @pytest.mark.unit
    @pytest.mark.config
    def test_config_overwrite_symbol_limits(self):
        """Test overwriting existing symbol limits."""
        config = MarketLimitsConfig()
        
        original_limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("1000"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("10000"),
        )
        
        new_limits = MarketLimits(
            min_price=Decimal("0.1"),
            max_price=Decimal("10000"),
            min_quantity=Decimal("0.01"),
            max_quantity=Decimal("100000"),
        )
        
        config.set_symbol_limits("BTCUSDT", original_limits)
        config.set_symbol_limits("BTCUSDT", new_limits)  # Overwrite
        
        assert config.get_limits("BTCUSDT") == new_limits

    @pytest.mark.unit
    @pytest.mark.config
    def test_case_sensitive_symbol_names(self):
        """Test that symbol names are case-sensitive."""
        config = MarketLimitsConfig()
        
        limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("1000"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("10000"),
        )
        
        config.set_symbol_limits("BTCUSDT", limits)
        
        # Exact match should return custom limits
        assert config.get_limits("BTCUSDT") == limits
        
        # Case mismatch should return default limits
        assert config.get_limits("btcusdt") == DEFAULT_LIMITS["CRYPTO"]
        assert config.get_limits("BtcUsdt") == DEFAULT_LIMITS["CRYPTO"]

    @pytest.mark.unit
    @pytest.mark.config
    def test_unknown_asset_class_fallback(self):
        """Test fallback behavior for unknown asset classes."""
        config = MarketLimitsConfig()
        
        # Unknown asset class should fallback to CRYPTO (default)
        limits = config.get_limits("UNKNOWN", asset_class="UNKNOWN_ASSET")
        assert limits == DEFAULT_LIMITS["CRYPTO"]

    @pytest.mark.unit
    @pytest.mark.config
    def test_integration_realistic_scenario(self):
        """Test a realistic trading scenario with mixed configurations."""
        config = MarketLimitsConfig()
        
        # Set up Bitcoin with custom limits
        btc_limits = MarketLimits(
            min_price=Decimal("1.00"),
            max_price=Decimal("200000.00"),
            min_quantity=Decimal("0.00001"),
            max_quantity=Decimal("100.00"),
            price_precision=2,
            quantity_precision=5,
        )
        config.set_symbol_limits("BTCUSDT", btc_limits)
        
        # Set default to CRYPTO for other symbols
        config.set_default_asset_class("CRYPTO")
        
        # Test various symbols
        assert config.get_limits("BTCUSDT") == btc_limits  # Custom
        assert config.get_limits("ETHUSDT") == DEFAULT_LIMITS["CRYPTO"]  # Default
        assert config.get_limits("EURUSD", "FOREX") == DEFAULT_LIMITS["FOREX"]  # Explicit
        assert config.get_limits("AAPL", "STOCK") == DEFAULT_LIMITS["STOCK"]  # Explicit

    @pytest.mark.unit
    @pytest.mark.config
    def test_global_config_state_persistence(self):
        """Test that global config state persists across calls."""
        # Get global config and modify it
        config = get_market_limits_config()
        
        test_limits = MarketLimits(
            min_price=Decimal("0.01"),
            max_price=Decimal("1000"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("1000"),
        )
        
        config.set_symbol_limits("TEST_SYMBOL", test_limits)
        
        # Get config again and verify persistence
        config_again = get_market_limits_config()
        assert config_again is config  # Same instance
        assert config_again.get_limits("TEST_SYMBOL") == test_limits