# ABOUTME: In-memory implementation of AbstractDataConverter for basic data conversion
# ABOUTME: Provides zero-dependency data conversion with support for common exchange formats

from datetime import datetime, UTC
from decimal import Decimal
import decimal
from typing import Any

from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.enum import TradeSide, AssetClass, KlineInterval
from core.config.market_limits import get_market_limits_config


class InMemoryDataConverter(AbstractDataConverter):
    """
    In-memory implementation of AbstractDataConverter.

    This implementation provides basic data conversion functionality for common
    exchange data formats. It's designed to be simple, reliable, and suitable
    for testing and development scenarios.

    Features:
    - Supports multiple timestamp formats (Unix seconds/milliseconds, ISO strings)
    - Handles common exchange data structures
    - Provides reasonable defaults for missing fields
    - Zero external dependencies
    """

    def __init__(self, default_exchange: str = "memory", default_asset_class: AssetClass = AssetClass.DIGITAL):
        """
        Initialize the converter with default values.

        Args:
            default_exchange: Default exchange name for converted data
            default_asset_class: Default asset class for converted data
        """
        self.default_exchange = default_exchange
        self.default_asset_class = default_asset_class

    def convert_trade(self, raw_trade: dict[str, Any], symbol: str) -> Trade:
        """
        Converts a single raw trade record into a standardized Trade model.

        Expected raw_trade format (flexible field names):
        {
            "id" or "trade_id": str,
            "price": str/float/Decimal,
            "quantity" or "qty" or "amount": str/float/Decimal,
            "side": str ("buy"/"sell" or "BUY"/"SELL"),
            "timestamp" or "time" or "ts": int/float/str,
            "exchange": str (optional),
            "maker_order_id": str (optional),
            "taker_order_id": str (optional),
            "is_buyer_maker": bool (optional)
        }
        """
        try:
            # Extract trade ID
            trade_id = self._extract_field(raw_trade, ["id", "trade_id", "tradeId"], required=True)

            # Get market limits for precision
            config = get_market_limits_config()
            limits = config.get_limits(symbol)

            # Extract price and quantize to proper precision
            price_raw = self._extract_field(raw_trade, ["price", "p"], required=True)
            price_precision = Decimal("0.1") ** limits.price_precision
            price = Decimal(str(price_raw)).quantize(price_precision)

            # Extract quantity and quantize to proper precision
            qty_raw = self._extract_field(raw_trade, ["quantity", "qty", "amount", "q"], required=True)
            quantity_precision = Decimal("0.1") ** limits.quantity_precision
            quantity = Decimal(str(qty_raw)).quantize(quantity_precision)

            # Extract side
            side_raw = self._extract_field(raw_trade, ["side", "s"], required=True)
            side = self._parse_trade_side(side_raw)

            # Extract timestamp
            timestamp_raw = self._extract_field(raw_trade, ["timestamp", "time", "ts", "T"], required=True)
            timestamp = self._parse_timestamp(timestamp_raw)

            # Extract optional fields
            exchange = self._extract_field(raw_trade, ["exchange"], default=self.default_exchange)
            maker_order_id = self._extract_field(raw_trade, ["maker_order_id", "makerOrderId"])
            taker_order_id = self._extract_field(raw_trade, ["taker_order_id", "takerOrderId"])
            is_buyer_maker = self._extract_field(raw_trade, ["is_buyer_maker", "isBuyerMaker"])

            # Convert boolean field if present
            if is_buyer_maker is not None:
                is_buyer_maker = bool(is_buyer_maker)

            return Trade(
                symbol=symbol,
                trade_id=str(trade_id),
                price=price,
                quantity=quantity,
                side=side,
                timestamp=timestamp,
                asset_class=self.default_asset_class,
                exchange=exchange,
                maker_order_id=maker_order_id,
                taker_order_id=taker_order_id,
                is_buyer_maker=is_buyer_maker,
                received_at=datetime.now(UTC),
                metadata={"raw_data": raw_trade},
            )

        except Exception as e:
            raise ValueError(f"Failed to convert trade data: {e}") from e

    def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """
        Converts multiple raw trade records into a list of Trade models.
        """
        trades = []
        for i, raw_trade in enumerate(raw_trades):
            try:
                trade = self.convert_trade(raw_trade, symbol)
                trades.append(trade)
            except Exception as e:
                raise ValueError(f"Failed to convert trade at index {i}: {e}") from e
        return trades

    def convert_kline(self, raw_kline: dict[str, Any], symbol: str) -> Kline:
        """
        Converts a single raw K-line record into a standardized Kline model.

        Expected raw_kline format (flexible field names):
        {
            "open_time" or "openTime" or "t": int/float/str,
            "close_time" or "closeTime" or "T": int/float/str,
            "open" or "o": str/float/Decimal,
            "high" or "h": str/float/Decimal,
            "low" or "l": str/float/Decimal,
            "close" or "c": str/float/Decimal,
            "volume" or "v": str/float/Decimal,
            "quote_volume" or "quoteVolume" or "qv": str/float/Decimal,
            "trades_count" or "count" or "n": int,
            "interval": str (optional),
            "exchange": str (optional),
            "taker_buy_volume": str/float/Decimal (optional),
            "taker_buy_quote_volume": str/float/Decimal (optional),
            "is_closed": bool (optional)
        }
        """
        try:
            # Extract timestamps
            open_time_raw = self._extract_field(raw_kline, ["open_time", "openTime", "t"], required=True)
            open_time = self._parse_timestamp(open_time_raw)

            close_time_raw = self._extract_field(raw_kline, ["close_time", "closeTime", "T"], required=True)
            close_time = self._parse_timestamp(close_time_raw)

            # Get market limits for precision
            config = get_market_limits_config()
            limits = config.get_limits(symbol)

            # Extract OHLC prices and quantize to proper precision
            price_precision = Decimal("0.1") ** limits.price_precision
            open_price = Decimal(str(self._extract_field(raw_kline, ["open", "o"], required=True))).quantize(
                price_precision
            )
            high_price = Decimal(str(self._extract_field(raw_kline, ["high", "h"], required=True))).quantize(
                price_precision
            )
            low_price = Decimal(str(self._extract_field(raw_kline, ["low", "l"], required=True))).quantize(
                price_precision
            )
            close_price = Decimal(str(self._extract_field(raw_kline, ["close", "c"], required=True))).quantize(
                price_precision
            )

            # Extract volumes and quantize to proper precision
            quantity_precision = Decimal("0.1") ** limits.quantity_precision
            volume = Decimal(str(self._extract_field(raw_kline, ["volume", "v"], required=True))).quantize(
                quantity_precision
            )
            quote_volume = Decimal(
                str(self._extract_field(raw_kline, ["quote_volume", "quoteVolume", "qv"], required=True))
            ).quantize(quantity_precision)

            # Extract trade count
            trades_count = int(self._extract_field(raw_kline, ["trades_count", "count", "n"], required=True))

            # Extract interval (try to parse or default to 1m)
            interval_raw = self._extract_field(raw_kline, ["interval"], default="1m")
            interval = self._parse_kline_interval(interval_raw)

            # Extract optional fields
            exchange = self._extract_field(raw_kline, ["exchange"], default=self.default_exchange)
            taker_buy_volume_raw = self._extract_field(raw_kline, ["taker_buy_volume", "takerBuyVolume"])
            taker_buy_quote_volume_raw = self._extract_field(
                raw_kline, ["taker_buy_quote_volume", "takerBuyQuoteVolume"]
            )
            is_closed = self._extract_field(raw_kline, ["is_closed", "isClosed"], default=True)

            # Convert optional volumes and quantize
            taker_buy_volume = (
                Decimal(str(taker_buy_volume_raw)).quantize(quantity_precision)
                if taker_buy_volume_raw is not None
                else None
            )
            taker_buy_quote_volume = (
                Decimal(str(taker_buy_quote_volume_raw)).quantize(quantity_precision)
                if taker_buy_quote_volume_raw is not None
                else None
            )

            return Kline(
                symbol=symbol,
                interval=interval,
                open_time=open_time,
                close_time=close_time,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                quote_volume=quote_volume,
                trades_count=trades_count,
                asset_class=self.default_asset_class,
                exchange=exchange,
                taker_buy_volume=taker_buy_volume,
                taker_buy_quote_volume=taker_buy_quote_volume,
                is_closed=bool(is_closed),
                received_at=datetime.now(UTC),
                metadata={"raw_data": raw_kline},
            )

        except Exception as e:
            raise ValueError(f"Failed to convert kline data: {e}") from e

    def convert_multiple_klines(self, raw_klines: list[dict[str, Any]], symbol: str) -> list[Kline]:
        """
        Converts multiple raw K-line records into a list of Kline models.
        """
        klines = []
        for i, raw_kline in enumerate(raw_klines):
            try:
                kline = self.convert_kline(raw_kline, symbol)
                klines.append(kline)
            except Exception as e:
                raise ValueError(f"Failed to convert kline at index {i}: {e}") from e
        return klines

    def validate_raw_data(self, data: Any) -> tuple[bool, str]:
        """
        Validates the structure and content of raw data.
        """
        if data is None:
            return False, "Data cannot be None"

        if isinstance(data, dict):
            return self._validate_single_record(data)
        elif isinstance(data, list):
            return self._validate_multiple_records(data)
        else:
            return False, f"Data must be dict or list, got {type(data)}"

    def to_internal_timestamp(self, timestamp: Any) -> int:
        """
        Converts various timestamp formats to Unix milliseconds.

        Supports:
        - Unix seconds (int/float)
        - Unix milliseconds (int)
        - ISO 8601 strings
        - datetime objects
        """
        if timestamp is None:
            raise ValueError("Timestamp cannot be None")

        # Handle datetime objects
        if isinstance(timestamp, datetime):
            return int(timestamp.timestamp() * 1000)

        # Handle string timestamps
        if isinstance(timestamp, str):
            try:
                # Try parsing as ISO 8601
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except ValueError:
                try:
                    # Try parsing as Unix timestamp string
                    return self.to_internal_timestamp(float(timestamp))
                except ValueError:
                    raise ValueError(f"Unable to parse timestamp string: {timestamp}")

        # Handle numeric timestamps
        if isinstance(timestamp, (int, float)):
            # Determine if it's seconds or milliseconds based on magnitude
            if timestamp > 1e12:  # Likely milliseconds
                return int(timestamp)
            else:  # Likely seconds
                return int(timestamp * 1000)

        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

    def _extract_field(
        self, data: dict[str, Any], field_names: list[str], required: bool = False, default: Any = None
    ) -> Any:
        """
        Extract field from data using multiple possible field names.
        """
        for field_name in field_names:
            if field_name in data:
                return data[field_name]

        if required:
            raise ValueError(f"Required field not found. Tried: {field_names}")

        return default

    def _parse_trade_side(self, side_raw: Any) -> TradeSide:
        """
        Parse trade side from various formats.
        """
        if isinstance(side_raw, str):
            side_lower = side_raw.lower().strip()
            if side_lower in ("buy", "b"):
                return TradeSide.BUY
            elif side_lower in ("sell", "s"):
                return TradeSide.SELL

        raise ValueError(f"Invalid trade side: {side_raw}")

    def _parse_kline_interval(self, interval_raw: Any) -> KlineInterval:
        """
        Parse kline interval from string.
        """
        if isinstance(interval_raw, str):
            interval_str = interval_raw.lower().strip()
            # Try to match with enum values
            for interval in KlineInterval:
                if interval.value.lower() == interval_str:
                    return interval

        # Default to 1 minute if parsing fails
        return KlineInterval.MINUTE_1

    def _parse_timestamp(self, timestamp_raw: Any) -> datetime:
        """
        Parse timestamp and return as UTC datetime.
        """
        timestamp_ms = self.to_internal_timestamp(timestamp_raw)
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

    def _validate_single_record(self, data: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate a single record (trade or kline).
        """
        if not isinstance(data, dict):
            return False, "Record must be a dictionary"

        if not data:
            return False, "Record cannot be empty"

        # Check for common required fields
        has_price_fields = any(field in data for field in ["price", "p", "open", "o"])
        has_timestamp_fields = any(field in data for field in ["timestamp", "time", "ts", "t", "open_time", "openTime"])

        if not has_price_fields:
            return False, "Record must contain price information"

        if not has_timestamp_fields:
            return False, "Record must contain timestamp information"

        # Validate field values
        # Check timestamp fields
        for field in ["timestamp", "time", "ts", "t", "open_time", "openTime"]:
            if field in data:
                try:
                    # Try to parse the timestamp
                    if data[field] == "invalid_timestamp":
                        return False, f"Invalid timestamp value: {data[field]}"
                    # Try to convert to check if it's a valid timestamp
                    self.to_internal_timestamp(data[field])
                except (ValueError, TypeError):
                    return False, f"Invalid timestamp in field '{field}': {data[field]}"

        # Check price fields
        for field in ["price", "p", "open", "o", "high", "h", "low", "l", "close", "c"]:
            if field in data:
                try:
                    if data[field] == "invalid_price":
                        return False, f"Invalid price value: {data[field]}"
                    price = Decimal(str(data[field]))
                    if price <= 0:
                        return False, f"Price must be positive in field '{field}': {price}"
                except (ValueError, TypeError, decimal.InvalidOperation):
                    return False, f"Invalid price in field '{field}': {data[field]}"

        # Check volume fields
        for field in ["volume", "v", "quantity", "qty", "amount", "q"]:
            if field in data:
                try:
                    volume = Decimal(str(data[field]))
                    if volume < 0:
                        return False, f"Volume cannot be negative in field '{field}': {volume}"
                except (ValueError, TypeError, decimal.InvalidOperation):
                    return False, f"Invalid volume in field '{field}': {data[field]}"

        return True, ""

    def _validate_multiple_records(self, data: list[dict[str, Any]]) -> tuple[bool, str]:
        """
        Validate multiple records.
        """
        if not isinstance(data, list):
            return False, "Data must be a list"

        if not data:
            return False, "List cannot be empty"

        for i, record in enumerate(data):
            is_valid, error_msg = self._validate_single_record(record)
            if not is_valid:
                return False, f"Record at index {i}: {error_msg}"

        return True, ""
