# ABOUTME: Market-specific validation limits configuration for trading data
# ABOUTME: Provides configurable limits for price and volume validation based on market and asset class

"""Market-specific validation limits configuration.

This module provides configurable limits for price and volume validation
that can be customized based on asset class, market conditions, and exchange
requirements. These limits are used by Trade and Kline models for data validation.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional


@dataclass(frozen=True)
class MarketLimits:
    """Market-specific limits for data validation.

    Attributes:
        min_price: Minimum allowed price (must be > 0).
        max_price: Maximum allowed price.
        min_quantity: Minimum allowed quantity/volume (must be > 0).
        max_quantity: Maximum allowed quantity/volume.
        price_precision: Number of decimal places allowed for prices.
        quantity_precision: Number of decimal places allowed for quantities.
    """

    min_price: Decimal
    max_price: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    price_precision: int = 8
    quantity_precision: int = 8

    def __post_init__(self) -> None:
        """Validate the market limits configuration."""
        if self.min_price <= 0:
            raise ValueError("min_price must be greater than 0")
        if self.max_price <= self.min_price:
            raise ValueError("max_price must be greater than min_price")
        if self.min_quantity <= 0:
            raise ValueError("min_quantity must be greater than 0")
        if self.max_quantity <= self.min_quantity:
            raise ValueError("max_quantity must be greater than min_quantity")
        if self.price_precision < 0:
            raise ValueError("price_precision must be non-negative")
        if self.quantity_precision < 0:
            raise ValueError("quantity_precision must be non-negative")


# Default market limits for different asset classes
DEFAULT_LIMITS: Dict[str, MarketLimits] = {
    "CRYPTO": MarketLimits(
        min_price=Decimal("0.00000001"),  # 1 satoshi
        max_price=Decimal("10000000.00"),  # 10M
        min_quantity=Decimal("0.00000001"),
        max_quantity=Decimal("1000000000.00"),  # 1B
        price_precision=8,
        quantity_precision=8,
    ),
    "FOREX": MarketLimits(
        min_price=Decimal("0.00001"),
        max_price=Decimal("10000.00"),
        min_quantity=Decimal("0.01"),
        max_quantity=Decimal("100000000.00"),  # 100M
        price_precision=5,
        quantity_precision=2,
    ),
    "STOCK": MarketLimits(
        min_price=Decimal("0.01"),
        max_price=Decimal("1000000.00"),  # 1M
        min_quantity=Decimal("1"),
        max_quantity=Decimal("1000000000.00"),  # 1B shares
        price_precision=2,
        quantity_precision=0,
    ),
    "COMMODITY": MarketLimits(
        min_price=Decimal("0.01"),
        max_price=Decimal("100000.00"),
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("10000000.00"),  # 10M
        price_precision=2,
        quantity_precision=3,
    ),
}


class MarketLimitsConfig:
    """Configuration manager for market-specific validation limits.

    This class manages market limits configuration and provides methods
    to retrieve limits based on symbol or asset class.
    """

    def __init__(self) -> None:
        """Initialize the market limits configuration."""
        self._symbol_limits: Dict[str, MarketLimits] = {}
        self._default_asset_class = "CRYPTO"

    def set_symbol_limits(self, symbol: str, limits: MarketLimits) -> None:
        """Set custom limits for a specific symbol.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").
            limits: The market limits to apply for this symbol.
        """
        self._symbol_limits[symbol] = limits

    def get_limits(self, symbol: str, asset_class: Optional[str] = None) -> MarketLimits:
        """Get market limits for a symbol.

        Args:
            symbol: The trading symbol.
            asset_class: Optional asset class override. If not provided,
                uses default asset class.

        Returns:
            MarketLimits for the symbol.
        """
        # Check for symbol-specific limits first
        if symbol in self._symbol_limits:
            return self._symbol_limits[symbol]

        # Fall back to asset class defaults
        asset_class = asset_class or self._default_asset_class
        return DEFAULT_LIMITS.get(asset_class, DEFAULT_LIMITS["CRYPTO"])

    def set_default_asset_class(self, asset_class: str) -> None:
        """Set the default asset class for limit lookup.

        Args:
            asset_class: The asset class to use as default (e.g., "CRYPTO", "FOREX").

        Raises:
            ValueError: If the asset class is not recognized.
        """
        if asset_class not in DEFAULT_LIMITS:
            raise ValueError(f"Unknown asset class: {asset_class}. Valid options: {list(DEFAULT_LIMITS.keys())}")
        self._default_asset_class = asset_class


# Global configuration instance
_market_limits_config = MarketLimitsConfig()


def get_market_limits_config() -> MarketLimitsConfig:
    """Get the global market limits configuration instance.

    Returns:
        The global MarketLimitsConfig instance.
    """
    return _market_limits_config
