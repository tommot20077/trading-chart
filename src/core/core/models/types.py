# ABOUTME: Common type definitions for improved type safety across the Core system
# ABOUTME: Provides TypedDict classes and type aliases for commonly used data structures

from typing import TypedDict, Any, Union, Optional
from datetime import datetime


class UserData(TypedDict, total=False):
    """Type definition for user data used in token generation.

    All fields are optional to support various authentication scenarios.
    """

    user_id: str
    username: str
    email: str
    roles: list[str]
    permissions: list[str]
    metadata: dict[str, Any]


class AlertData(TypedDict):
    """Type definition for alert data used in notifications."""

    id: str
    rule_name: str
    severity: str
    status: str
    message: str
    labels: dict[str, str]
    annotations: dict[str, str]
    fired_at: datetime
    trace_id: str


class ExchangeInfo(TypedDict, total=False):
    """Type definition for exchange information."""

    name: str
    timezone: str
    server_time: datetime
    rate_limits: list[dict[str, Any]]
    exchange_filters: list[dict[str, Any]]
    symbols: list[str]


class SymbolInfo(TypedDict, total=False):
    """Type definition for trading symbol information."""

    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    base_precision: int
    quote_precision: int
    order_types: list[str]
    filters: list[dict[str, Any]]
    is_spot_trading_allowed: bool
    is_margin_trading_allowed: bool


class RawTradeData(TypedDict):
    """Type definition for raw trade data from exchanges."""

    id: Union[str, int]
    price: Union[str, float]
    quantity: Union[str, float]
    time: Union[int, str, datetime]
    is_buyer_maker: bool
    symbol: str


class RawKlineData(TypedDict):
    """Type definition for raw kline/candlestick data from exchanges."""

    open_time: Union[int, str, datetime]
    close_time: Union[int, str, datetime]
    open_price: Union[str, float]
    high_price: Union[str, float]
    low_price: Union[str, float]
    close_price: Union[str, float]
    volume: Union[str, float]
    number_of_trades: Union[int, str]
    taker_buy_base_volume: Union[str, float]
    taker_buy_quote_volume: Union[str, float]
    symbol: str
    interval: str


class StatisticsResult(TypedDict, total=False):
    """Type definition for storage statistics results."""

    count: int
    min_timestamp: Optional[datetime]
    max_timestamp: Optional[datetime]
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]
    storage_size_bytes: int
    avg_size_bytes: float
    price_statistics: dict[str, float]
    volume_statistics: dict[str, float]
    data_statistics: dict[str, Any]
    # Additional fields for test compatibility
    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]
    time_span_seconds: Union[int, float]
    # Flat fields expected by tests
    volume_total: float
    quote_volume_total: float
    price_high: Optional[float]
    price_low: Optional[float]
    avg_price: Optional[float]
    avg_volume: Optional[float]


class BackfillStatus(TypedDict):
    """Type definition for backfill status information."""

    symbol: str
    start_time: datetime
    end_time: datetime
    status: str
    progress: float
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class ProviderConfig(TypedDict, total=False):
    """Type definition for data provider configuration."""

    api_key: str
    api_secret: str
    base_url: str
    timeout: int
    rate_limit: int
    retry_count: int
    retry_delay: float
    proxy_url: Optional[str]
    use_testnet: bool
    custom_headers: dict[str, str]


# Type aliases for commonly used types
TimestampType = Union[int, float, str, datetime]
RawDataType = Union[RawTradeData, RawKlineData, dict[str, Any]]
MetadataValue = Union[str, int, float, bool, list[Any], dict[str, Any], None]
