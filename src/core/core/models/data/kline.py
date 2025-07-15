"""Kline (candlestick) model definition."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.models.data.enum import KlineInterval, AssetClass


class Kline(BaseModel):
    """
    Represents a single candlestick (Kline) data point.

    This Pydantic model defines the structure for Kline data, including
    open, high, low, close prices, volume, and other relevant information
    for a specific time interval. It includes validation logic to ensure
    data integrity.

    Attributes:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").
        interval (KlineInterval): The time interval of the kline (e.g., "1m", "1h").
        open_time (datetime): The UTC timestamp of the kline's opening.
        close_time (datetime): The UTC timestamp of the kline's closing.
        open_price (Decimal): The price at the beginning of the interval.
        high_price (Decimal): The highest price during the interval.
        low_price (Decimal): The lowest price during the interval.
        close_price (Decimal): The price at the end of the interval.
        volume (Decimal): The volume of the base asset traded during the interval.
        quote_volume (Decimal): The volume of the quote asset traded during the interval.
        trades_count (int): The number of trades that occurred during the interval.
        asset_class (AssetClass): The asset class classification (defaults to AssetClass.DIGITAL).
        exchange (str | None): Optional. The name of the exchange where the kline originated.
        taker_buy_volume (Decimal | None): Optional. The volume of taker buy trades.
        taker_buy_quote_volume (Decimal | None): Optional. The quote volume of taker buy trades.
        is_closed (bool): Indicates if the kline is closed (i.e., the interval has ended).
        received_at (datetime | None): Optional. The UTC timestamp when the kline data was received.
        metadata (dict[str, Any]): Additional, unstructured metadata for the kline.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    # Core fields
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTCUSDT)")
    interval: KlineInterval = Field(..., description="Kline interval")
    open_time: datetime = Field(..., description="Kline open time (UTC)")
    close_time: datetime = Field(..., description="Kline close time (UTC)")

    # Price data
    open_price: Decimal = Field(..., description="Open price")
    high_price: Decimal = Field(..., description="High price")
    low_price: Decimal = Field(..., description="Low price")
    close_price: Decimal = Field(..., description="Close price")

    # Volume data
    volume: Decimal = Field(..., description="Volume in base asset")
    quote_volume: Decimal = Field(..., description="Volume in quote asset")

    # Trade statistics
    trades_count: int = Field(..., description="Number of trades", ge=0)

    # Asset classification
    asset_class: AssetClass = Field(default=AssetClass.DIGITAL, description="Asset class classification")

    # Optional fields
    exchange: str | None = Field(None, description="Exchange name")
    taker_buy_volume: Decimal | None = Field(None, description="Taker buy volume")
    taker_buy_quote_volume: Decimal | None = Field(None, description="Taker buy quote volume")

    # Status
    is_closed: bool = Field(True, description="Whether the kline is closed")

    # Metadata
    received_at: datetime | None = Field(None, description="When the kline was received")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """
        Validates and normalizes the trading symbol.

        The symbol is stripped of whitespace and converted to uppercase.

        Args:
            v: The symbol string to validate.

        Returns:
            The normalized symbol string.

        Raises:
            ValueError: If the symbol is `None`, not a string, or becomes empty after stripping.
        """
        # Handle None and non-string values
        if v is None:
            raise ValueError("Symbol cannot be None")
        if not isinstance(v, str):
            raise ValueError("Symbol must be a string")

        # Strip whitespace and convert to uppercase
        v = v.strip().upper()

        # Check for empty symbol after stripping
        if not v:
            raise ValueError("Symbol cannot be empty")

        return v

    @field_validator("open_price", "high_price", "low_price", "close_price")
    @classmethod
    def validate_prices(cls, v: Decimal) -> Decimal:
        """
        Validates that price values are positive.

        Args:
            v: The price value to validate.

        Returns:
            The validated price value.

        Raises:
            ValueError: If the price is not greater than 0.
        """
        if v <= 0:
            raise ValueError("Price must be greater than 0")

        # TODO: Dynamic validation - implement market-specific price limits
        # This should be configurable based on asset class and market conditions
        max_reasonable_price = Decimal("10000000.00")  # 10M as reasonable upper limit
        if v > max_reasonable_price:
            raise ValueError(f"Price {v} exceeds reasonable maximum of {max_reasonable_price}")

        return v

    @field_validator("volume", "quote_volume")
    @classmethod
    def validate_volumes(cls, v: Decimal) -> Decimal:
        """
        Validates that volume values are non-negative.

        Args:
            v: The volume value to validate.

        Returns:
            The validated volume value.

        Raises:
            ValueError: If the volume is less than 0.
        """
        if v < 0:
            raise ValueError("Volume must be greater than or equal to 0")

        # TODO: Dynamic validation - implement market-specific volume limits
        # This should be configurable based on asset class and market conditions
        max_reasonable_volume = Decimal("1000000000.00")  # 1B as reasonable upper limit
        if v > max_reasonable_volume:
            raise ValueError(f"Volume {v} exceeds reasonable maximum of {max_reasonable_volume}")

        return v

    @field_validator("taker_buy_volume", "taker_buy_quote_volume")
    @classmethod
    def validate_optional_volumes(cls, v: Decimal | None) -> Decimal | None:
        """
        Validates that optional volume values are non-negative if present.

        Args:
            v: The optional volume value to validate.

        Returns:
            The validated optional volume value, or `None`.

        Raises:
            ValueError: If the volume is not `None` and is less than 0.
        """
        if v is not None and v < 0:
            raise ValueError("Volume must be greater than or equal to 0")
        return v

    @field_validator("open_time", "close_time", "received_at")
    @classmethod
    def validate_timezone(cls, v: datetime | None) -> datetime | None:
        """
        Ensures that datetime fields are timezone-aware UTC.

        Args:
            v: The datetime object to validate.

        Returns:
            The validated datetime object, converted to UTC if necessary, or `None`.

        Raises:
            ValueError: If the value is not a datetime object.
        """
        if v is None:
            return None
        if not isinstance(v, datetime):
            raise ValueError("Value must be a datetime object")

        if v.tzinfo is None:
            # Assume UTC if no timezone
            return v.replace(tzinfo=UTC)

        # Convert to UTC if different timezone
        return v.astimezone(UTC)

    @model_validator(mode="after")
    def validate_price_relationships(self) -> "Kline":
        """
        Validates basic price relationships: high >= low.

        Note: We only validate the most critical relationship (high >= low)
        to avoid overly strict validation that might reject valid market data.

        Returns:
            The validated Kline instance.

        Raises:
            ValueError: If high price is less than low price.
        """
        if self.high_price < self.low_price:
            raise ValueError("High price cannot be less than low price")
        return self

    @field_validator("close_time")
    @classmethod
    def validate_close_time(cls, v: datetime, info: Any) -> datetime:
        """
        Validates that the close time is strictly after the open time.

        Args:
            v: The close time datetime object.
            info: Pydantic validation info, containing other field data.

        Returns:
            The validated close time.

        Raises:
            ValueError: If the close time is not after the open time.
        """
        data = info.data
        if "open_time" in data and v <= data["open_time"]:
            raise ValueError("Close time must be after open time")
        return v

    @model_validator(mode="after")
    def validate_basic_consistency(self) -> "Kline":
        """
        Validates basic consistency requirements for the Kline.

        Only validates essential business logic to avoid rejecting valid market data
        due to minor timing variations or exchange-specific behaviors.

        Returns:
            The validated Kline instance.

        Raises:
            ValueError: If there are critical consistency issues.
        """
        # Only validate that duration is reasonable (not negative or extremely large)
        actual_duration = self.duration
        if actual_duration.total_seconds() <= 0:
            raise ValueError("Kline duration must be positive")

        # Allow up to 24 hours for any interval (covers all reasonable cases)
        max_duration = timedelta(hours=24)
        if actual_duration > max_duration:
            raise ValueError(f"Kline duration {actual_duration} exceeds maximum allowed duration")

        return self

    @classmethod
    def validate_sequence_continuity(cls, klines: list["Kline"]) -> bool:
        """
        Validates that a sequence of Klines is continuous and ordered.

        This method checks if there are any time gaps between consecutive klines
        in a given list and ensures they belong to the same symbol and interval.

        Args:
            klines: A list of `Kline` objects to validate.

        Returns:
            `True` if the sequence is continuous and valid.

        Raises:
            ValueError: If a gap is detected, or if klines have inconsistent symbols or intervals.
        """
        if len(klines) < 2:
            return True

        # Sort klines by open_time
        sorted_klines = sorted(klines, key=lambda k: k.open_time)

        for i in range(1, len(sorted_klines)):
            current = sorted_klines[i]
            previous = sorted_klines[i - 1]

            # Check if klines are for the same symbol and interval
            if current.symbol != previous.symbol:
                raise ValueError(f"Klines must have the same symbol. Found {current.symbol} and {previous.symbol}")

            if current.interval != previous.interval:
                raise ValueError(
                    f"Klines must have the same interval. Found {current.interval} and {previous.interval}"
                )

            # Check for continuity: current.open_time should equal previous.close_time
            # Allow small tolerance for timing variations
            tolerance = timedelta(seconds=1)
            time_gap = abs(current.open_time - previous.close_time)

            if time_gap > tolerance:
                raise ValueError(
                    f"Gap detected between klines: {previous.close_time} to {current.open_time}, gap: {time_gap}"
                )

        return True

    @property
    def duration(self) -> timedelta:
        """
        Calculates the duration of the Kline (close_time - open_time).

        Returns:
            A `timedelta` object representing the duration.
        """
        return self.close_time - self.open_time

    @property
    def primary_timestamp(self) -> datetime:
        """
        Return the primary timestamp for this data point.

        For Kline data, this is the open_time field.
        This property is required by the TimeSeriesData protocol.

        Returns:
            The kline open time.
        """
        return self.open_time

    @property
    def price_change(self) -> Decimal:
        """
        Calculates the absolute price change (close_price - open_price).

        Returns:
            A `Decimal` representing the price change.
        """
        return self.close_price - self.open_price

    @property
    def price_change_percent(self) -> Decimal:
        """
        Calculates the percentage price change based on open and close prices.

        Returns:
            A `Decimal` representing the percentage change.
        """
        if self.open_price == 0:
            return Decimal("0")
        return (self.price_change / self.open_price) * 100

    @property
    def is_bullish(self) -> bool:
        """
        Checks if the Kline is bullish (close price is greater than open price).

        Returns:
            `True` if bullish, `False` otherwise.
        """
        return self.close_price > self.open_price

    @property
    def is_bearish(self) -> bool:
        """
        Checks if the Kline is bearish (close price is less than open price).

        Returns:
            `True` if bearish, `False` otherwise.
        """
        return self.close_price < self.open_price

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Kline object to a dictionary, with Decimal and datetime fields
        converted to string representations suitable for JSON serialization.

        Returns:
            A dictionary representation of the Kline.
        """
        data = self.model_dump()
        # Convert Decimal to string for JSON serialization
        for field in ["open_price", "high_price", "low_price", "close_price", "volume", "quote_volume"]:
            data[field] = str(data[field])
        if self.taker_buy_volume is not None:
            data["taker_buy_volume"] = str(self.taker_buy_volume)
        if self.taker_buy_quote_volume is not None:
            data["taker_buy_quote_volume"] = str(self.taker_buy_quote_volume)
        # Convert datetime to ISO format
        data["open_time"] = self.open_time.isoformat()
        data["close_time"] = self.close_time.isoformat()
        if self.received_at:
            data["received_at"] = self.received_at.isoformat()
        # Ensure asset_class is included as string
        data["asset_class"] = str(self.asset_class)
        # Add calculated fields
        data["price_change"] = str(self.price_change)
        data["price_change_percent"] = str(self.price_change_percent)
        return data

    def __str__(self) -> str:
        """
        Returns a string representation of the Kline object.

        Returns:
            A formatted string showing key Kline details.
        """
        return (
            f"Kline({self.symbol} {self.interval} [{self.asset_class}] "
            f"O:{self.open_price} H:{self.high_price} L:{self.low_price} C:{self.close_price} "
            f"V:{self.volume} at {self.open_time.isoformat()})"
        )

    @model_validator(mode="after")
    def validate_time_consistency(self) -> "Kline":
        """
        Validates that close_time is after open_time.

        Returns:
            The validated Kline instance.

        Raises:
            ValueError: If close_time is not after open_time.
        """
        if self.close_time <= self.open_time:
            raise ValueError("Close time must be after open time")
        return self

    @model_validator(mode="after")
    def validate_ohlc_consistency(self) -> "Kline":
        """
        Validates OHLC price consistency (high >= max(open, close), low <= min(open, close)).

        Returns:
            The validated Kline instance.

        Raises:
            ValueError: If OHLC prices are inconsistent.
        """
        max_oc = max(self.open_price, self.close_price)
        min_oc = min(self.open_price, self.close_price)

        if self.high_price < max_oc:
            raise ValueError("high_price must be >= max(open_price, close_price)")

        if self.low_price > min_oc:
            raise ValueError("low_price must be <= min(open_price, close_price)")

        return self
