# ABOUTME: MarketData container model for integrating multiple market data types
# ABOUTME: Provides unified interface for Kline, Trade data with time-based operations and statistics

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from collections import defaultdict
from statistics import mean, median

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.models.data.enum import AssetClass, KlineInterval
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.trading_pair import TradingPair


class MarketDataSummary(BaseModel):
    """
    Statistical summary of market data.

    Provides statistical insights including price ranges, volume metrics,
    and trading activity summaries for a given time period.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    # Price statistics
    min_price: Optional[Decimal] = Field(None, description="Minimum price in the dataset")
    max_price: Optional[Decimal] = Field(None, description="Maximum price in the dataset")
    avg_price: Optional[Decimal] = Field(None, description="Average price in the dataset")
    median_price: Optional[Decimal] = Field(None, description="Median price in the dataset")

    # Volume statistics
    total_volume: Decimal = Field(default=Decimal("0"), description="Total volume across all data points")
    avg_volume: Optional[Decimal] = Field(None, description="Average volume per data point")

    # Trading activity
    total_trades: int = Field(default=0, description="Total number of trades")
    unique_symbols: int = Field(default=0, description="Number of unique trading symbols")

    # Time range
    start_time: Optional[datetime] = Field(None, description="Earliest timestamp in the dataset")
    end_time: Optional[datetime] = Field(None, description="Latest timestamp in the dataset")

    # Data composition
    kline_count: int = Field(default=0, description="Number of Kline data points")
    trade_count: int = Field(default=0, description="Number of Trade data points")


class MarketData(BaseModel):
    """
    Unified container for multiple types of market data.

    This model serves as a comprehensive container for integrating various
    market data types including Kline (candlestick) data and Trade data.
    It provides unified time-based operations, data validation, statistics,
    and querying capabilities across different data types.

    Key Features:
    - Unified timestamp handling and timezone management
    - Cross-data type validation and consistency checks
    - Time range queries and filtering
    - Statistical analysis and summary generation
    - Serialization and deserialization support
    - Performance-optimized data access patterns

    Attributes:
        symbol (str): The primary trading symbol for this market data
        klines (List[Kline]): Collection of Kline (candlestick) data points
        trades (List[Trade]): Collection of Trade data points
        trading_pair (TradingPair | None): Optional trading pair metadata
        asset_class (AssetClass): Asset classification for this market data
        created_at (datetime): When this MarketData instance was created
        updated_at (datetime): When this MarketData instance was last updated
        metadata (Dict[str, Any]): Additional structured metadata
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
        validate_assignment=True,
    )

    # Core identification
    symbol: str = Field(..., description="Primary trading symbol (e.g., BTCUSDT)")

    # Data collections
    klines: List[Kline] = Field(default_factory=list, description="Collection of Kline data points")
    trades: List[Trade] = Field(default_factory=list, description="Collection of Trade data points")

    # Associated metadata
    trading_pair: Optional[TradingPair] = Field(None, description="Trading pair configuration and metadata")
    asset_class: AssetClass = Field(default=AssetClass.DIGITAL, description="Asset class classification")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When this MarketData instance was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When this MarketData instance was last updated"
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional structured metadata")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """
        Validates and normalizes the trading symbol.

        Args:
            v: The symbol string to validate.

        Returns:
            The normalized symbol string.

        Raises:
            ValueError: If the symbol is invalid.
        """
        if v is None:
            raise ValueError("Symbol cannot be None")
        if not isinstance(v, str):
            raise ValueError("Symbol must be a string")

        v = v.strip().upper()
        if not v:
            raise ValueError("Symbol cannot be empty")

        return v

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        """
        Ensures that datetime fields are timezone-aware UTC.

        Args:
            v: The datetime object to validate.

        Returns:
            The validated datetime object in UTC.
        """
        if not isinstance(v, datetime):
            raise ValueError("Value must be a datetime object")

        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)

        return v.astimezone(UTC)

    @model_validator(mode="after")
    def validate_data_consistency(self) -> "MarketData":
        """
        Validates data consistency across all contained data points.

        Ensures that all Kline and Trade data points have consistent
        symbols and timestamps, and validates cross-data relationships.

        Returns:
            The validated MarketData instance.

        Raises:
            ValueError: If data consistency checks fail.
        """
        # Validate symbol consistency in klines
        for kline in self.klines:
            if kline.symbol != self.symbol:
                raise ValueError(f"Kline symbol {kline.symbol} does not match MarketData symbol {self.symbol}")

        # Validate symbol consistency in trades
        for trade in self.trades:
            if trade.symbol != self.symbol:
                raise ValueError(f"Trade symbol {trade.symbol} does not match MarketData symbol {self.symbol}")

        # Validate trading pair consistency if present
        if self.trading_pair:
            # Extract symbol from trading pair format (e.g., "BTC/USDT" -> "BTCUSDT")
            pair_symbol = self.trading_pair.symbol.replace("/", "")
            if pair_symbol != self.symbol:
                # Allow some flexibility - check if symbols are related
                base_quote = f"{self.trading_pair.base_currency}{self.trading_pair.quote_currency}"
                if base_quote != self.symbol:
                    raise ValueError(
                        f"Trading pair symbol {self.trading_pair.symbol} is not compatible with MarketData symbol {self.symbol}"
                    )

        return self

    def add_kline(self, kline: Kline) -> None:
        """
        Add a Kline data point to the collection.

        Validates symbol consistency and maintains chronological order.

        Args:
            kline: The Kline data point to add.

        Raises:
            ValueError: If the kline symbol doesn't match this MarketData's symbol.
        """
        if kline.symbol != self.symbol:
            raise ValueError(f"Kline symbol {kline.symbol} does not match MarketData symbol {self.symbol}")

        self.klines.append(kline)
        self._sort_klines()
        self.updated_at = datetime.now(UTC)

    def add_trade(self, trade: Trade) -> None:
        """
        Add a Trade data point to the collection.

        Validates symbol consistency and maintains chronological order.

        Args:
            trade: The Trade data point to add.

        Raises:
            ValueError: If the trade symbol doesn't match this MarketData's symbol.
        """
        if trade.symbol != self.symbol:
            raise ValueError(f"Trade symbol {trade.symbol} does not match MarketData symbol {self.symbol}")

        self.trades.append(trade)
        self._sort_trades()
        self.updated_at = datetime.now(UTC)

    def add_klines(self, klines: List[Kline]) -> None:
        """
        Add multiple Kline data points to the collection.

        Args:
            klines: List of Kline data points to add.
        """
        for kline in klines:
            if kline.symbol != self.symbol:
                raise ValueError(f"Kline symbol {kline.symbol} does not match MarketData symbol {self.symbol}")

        self.klines.extend(klines)
        self._sort_klines()
        self.updated_at = datetime.now(UTC)

    def add_trades(self, trades: List[Trade]) -> None:
        """
        Add multiple Trade data points to the collection.

        Args:
            trades: List of Trade data points to add.
        """
        for trade in trades:
            if trade.symbol != self.symbol:
                raise ValueError(f"Trade symbol {trade.symbol} does not match MarketData symbol {self.symbol}")

        self.trades.extend(trades)
        self._sort_trades()
        self.updated_at = datetime.now(UTC)

    def _sort_klines(self) -> None:
        """Sort klines by open_time in chronological order."""
        self.klines.sort(key=lambda k: k.open_time)

    def _sort_trades(self) -> None:
        """Sort trades by timestamp in chronological order."""
        self.trades.sort(key=lambda t: t.timestamp)

    def get_klines_in_range(self, start_time: datetime, end_time: datetime) -> List[Kline]:
        """
        Retrieve Kline data points within a specified time range.

        Args:
            start_time: Start of the time range (inclusive).
            end_time: End of the time range (inclusive).

        Returns:
            List of Kline data points within the specified range.
        """
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)

        return [kline for kline in self.klines if start_time <= kline.open_time <= end_time]

    def get_trades_in_range(self, start_time: datetime, end_time: datetime) -> List[Trade]:
        """
        Retrieve Trade data points within a specified time range.

        Args:
            start_time: Start of the time range (inclusive).
            end_time: End of the time range (inclusive).

        Returns:
            List of Trade data points within the specified range.
        """
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)

        return [trade for trade in self.trades if start_time <= trade.timestamp <= end_time]

    def get_latest_kline(self) -> Optional[Kline]:
        """
        Get the most recent Kline data point.

        Returns:
            The latest Kline data point, or None if no klines exist.
        """
        if not self.klines:
            return None
        return max(self.klines, key=lambda k: k.open_time)

    def get_latest_trade(self) -> Optional[Trade]:
        """
        Get the most recent Trade data point.

        Returns:
            The latest Trade data point, or None if no trades exist.
        """
        if not self.trades:
            return None
        return max(self.trades, key=lambda t: t.timestamp)

    def get_earliest_timestamp(self) -> Optional[datetime]:
        """
        Get the earliest timestamp across all data points.

        Returns:
            The earliest timestamp, or None if no data exists.
        """
        timestamps = []

        if self.klines:
            timestamps.append(min(k.open_time for k in self.klines))
        if self.trades:
            timestamps.append(min(t.timestamp for t in self.trades))

        return min(timestamps) if timestamps else None

    def get_latest_timestamp(self) -> Optional[datetime]:
        """
        Get the latest timestamp across all data points.

        Returns:
            The latest timestamp, or None if no data exists.
        """
        timestamps = []

        if self.klines:
            timestamps.append(max(k.close_time for k in self.klines))
        if self.trades:
            timestamps.append(max(t.timestamp for t in self.trades))

        return max(timestamps) if timestamps else None

    def get_klines_by_interval(self, interval: KlineInterval) -> List[Kline]:
        """
        Get Kline data points filtered by interval.

        Args:
            interval: The Kline interval to filter by.

        Returns:
            List of Kline data points with the specified interval.
        """
        return [kline for kline in self.klines if kline.interval == interval]

    def calculate_summary(self) -> MarketDataSummary:
        """
        Calculate comprehensive statistical summary of all market data.

        Returns:
            MarketDataSummary containing statistical insights.
        """
        summary = MarketDataSummary(
            min_price=None,
            max_price=None,
            avg_price=None,
            median_price=None,
            avg_volume=None,
            start_time=None,
            end_time=None,
        )

        # Collect all prices
        prices = []
        total_volume = Decimal("0")
        total_trades = 0

        # Process klines
        for kline in self.klines:
            prices.extend([kline.open_price, kline.high_price, kline.low_price, kline.close_price])
            total_volume += kline.volume
            total_trades += kline.trades_count

        # Process trades
        for trade in self.trades:
            prices.append(trade.price)
            total_volume += trade.quantity
            total_trades += 1

        # Calculate price statistics
        if prices:
            summary.min_price = min(prices)
            summary.max_price = max(prices)
            summary.avg_price = Decimal(str(mean(float(p) for p in prices)))
            summary.median_price = Decimal(str(median(float(p) for p in prices)))

        # Volume statistics
        summary.total_volume = total_volume
        data_point_count = len(self.klines) + len(self.trades)
        if data_point_count > 0:
            summary.avg_volume = total_volume / data_point_count

        # Trading activity
        summary.total_trades = total_trades
        summary.unique_symbols = 1  # Since MarketData contains data for a single symbol

        # Time range
        summary.start_time = self.get_earliest_timestamp()
        summary.end_time = self.get_latest_timestamp()

        # Data composition
        summary.kline_count = len(self.klines)
        summary.trade_count = len(self.trades)

        return summary

    def validate_data_integrity(self) -> bool:
        """
        Perform comprehensive data integrity validation.

        Checks for data consistency, chronological order, and reasonable values.

        Returns:
            True if all data passes integrity checks.

        Raises:
            ValueError: If data integrity issues are found.
        """
        # Validate kline sequence continuity for each interval
        klines_by_interval = defaultdict(list)
        for kline in self.klines:
            klines_by_interval[kline.interval].append(kline)

        for interval, klines in klines_by_interval.items():
            if len(klines) > 1:
                try:
                    Kline.validate_sequence_continuity(klines)
                except ValueError as e:
                    raise ValueError(f"Kline sequence validation failed for interval {interval}: {e}")

        # Validate trades are in chronological order
        if len(self.trades) > 1:
            for i in range(1, len(self.trades)):
                if self.trades[i].timestamp < self.trades[i - 1].timestamp:
                    raise ValueError(f"Trades are not in chronological order at index {i}")

        # Cross-validate trade prices against kline ranges where applicable
        for trade in self.trades:
            matching_klines = [k for k in self.klines if k.open_time <= trade.timestamp <= k.close_time]
            for kline in matching_klines:
                if not (kline.low_price <= trade.price <= kline.high_price):
                    raise ValueError(
                        f"Trade price {trade.price} at {trade.timestamp} is outside "
                        f"kline range [{kline.low_price}, {kline.high_price}] for period "
                        f"[{kline.open_time}, {kline.close_time}]"
                    )

        return True

    @property
    def is_empty(self) -> bool:
        """
        Check if the MarketData contains any data points.

        Returns:
            True if no klines or trades exist, False otherwise.
        """
        return len(self.klines) == 0 and len(self.trades) == 0

    @property
    def data_count(self) -> int:
        """
        Get the total number of data points (klines + trades).

        Returns:
            Total count of all data points.
        """
        return len(self.klines) + len(self.trades)

    @property
    def time_span(self) -> Optional[timedelta]:
        """
        Calculate the time span covered by this market data.

        Returns:
            The time difference between earliest and latest data points,
            or None if insufficient data exists.
        """
        start = self.get_earliest_timestamp()
        end = self.get_latest_timestamp()

        if start and end:
            return end - start
        return None

    def merge(self, other: "MarketData") -> "MarketData":
        """
        Merge another MarketData instance into this one.

        Args:
            other: Another MarketData instance to merge.

        Returns:
            A new MarketData instance containing merged data.

        Raises:
            ValueError: If the symbols don't match.
        """
        if other.symbol != self.symbol:
            raise ValueError(f"Cannot merge MarketData with different symbols: {self.symbol} vs {other.symbol}")

        merged = MarketData(
            symbol=self.symbol,
            klines=self.klines + other.klines,
            trades=self.trades + other.trades,
            trading_pair=self.trading_pair or other.trading_pair,
            asset_class=self.asset_class,
            metadata={**self.metadata, **other.metadata},
        )

        # Remove duplicates and sort
        merged._remove_duplicate_klines()
        merged._remove_duplicate_trades()
        merged._sort_klines()
        merged._sort_trades()

        return merged

    def _remove_duplicate_klines(self) -> None:
        """Remove duplicate klines based on symbol, interval, and open_time."""
        seen = set()
        unique_klines = []

        for kline in self.klines:
            key = (kline.symbol, kline.interval, kline.open_time)
            if key not in seen:
                seen.add(key)
                unique_klines.append(kline)

        self.klines = unique_klines

    def _remove_duplicate_trades(self) -> None:
        """Remove duplicate trades based on symbol, trade_id, and timestamp."""
        seen = set()
        unique_trades = []

        for trade in self.trades:
            key = (trade.symbol, trade.trade_id, trade.timestamp)
            if key not in seen:
                seen.add(key)
                unique_trades.append(trade)

        self.trades = unique_trades

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MarketData to dictionary format suitable for JSON serialization.

        Returns:
            Dictionary representation of the MarketData.
        """
        return {
            "symbol": self.symbol,
            "klines": [kline.to_dict() for kline in self.klines],
            "trades": [trade.to_dict() for trade in self.trades],
            "trading_pair": self.trading_pair.to_dict() if self.trading_pair else None,
            "asset_class": str(self.asset_class),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "summary": self.calculate_summary().model_dump(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        """
        Create MarketData instance from dictionary.

        Args:
            data: Dictionary containing MarketData fields.

        Returns:
            MarketData instance created from the dictionary.
        """
        # Parse klines
        klines = []
        for kline_data in data.get("klines", []):
            klines.append(Kline.model_validate(kline_data))

        # Parse trades
        trades = []
        for trade_data in data.get("trades", []):
            trades.append(Trade.model_validate(trade_data))

        # Parse trading pair
        trading_pair = None
        if data.get("trading_pair"):
            trading_pair = TradingPair.model_validate(data["trading_pair"])

        return cls(
            symbol=data["symbol"],
            klines=klines,
            trades=trades,
            trading_pair=trading_pair,
            asset_class=AssetClass(data.get("asset_class", AssetClass.DIGITAL)),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(UTC),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """
        String representation of MarketData.

        Returns:
            Human-readable string showing key information.
        """
        return (
            f"MarketData({self.symbol} [{self.asset_class}] "
            f"klines={len(self.klines)} trades={len(self.trades)} "
            f"span={self.time_span})"
        )

    def __repr__(self) -> str:
        """
        Developer representation of MarketData.

        Returns:
            Detailed string representation for debugging.
        """
        return (
            f"MarketData(symbol='{self.symbol}', asset_class={self.asset_class}, "
            f"klines_count={len(self.klines)}, trades_count={len(self.trades)}, "
            f"created_at='{self.created_at.isoformat()}')"
        )
