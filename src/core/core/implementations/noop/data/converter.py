# ABOUTME: NoOp implementation of AbstractDataConverter that provides fake conversions
# ABOUTME: Provides minimal data conversion functionality for testing scenarios

from typing import Any, TypeVar
from decimal import Decimal
from datetime import datetime

from core.interfaces.data.converter import AbstractDataConverter
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.enum import KlineInterval, TradeSide

RawDataType = TypeVar("RawDataType")


class NoOpDataConverter(AbstractDataConverter):
    """
    No-operation implementation of AbstractDataConverter.

    This implementation provides minimal data conversion functionality that
    returns fake converted data without performing any actual conversion
    operations. It's useful for testing, performance benchmarking, and
    scenarios where data conversion is not required.

    Features:
    - Always returns fake successful conversions
    - No actual data parsing or validation
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where data conversion should be bypassed
    - Performance benchmarking without conversion overhead
    - Development environments where conversion is not needed
    - Fallback when conversion systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation data converter."""
        # No initialization needed for NoOp implementation
        pass

    def convert_trade(self, raw_trade: dict[str, Any], symbol: str) -> Trade:
        """
        Convert raw trade data - returns a fake Trade object.

        This implementation always returns a fake Trade object without
        performing any actual data conversion.

        Args:
            raw_trade: Raw trade data (ignored)
            symbol: Trading symbol

        Returns:
            A fake Trade object with default values
        """
        return Trade(
            symbol=symbol,
            price=Decimal("100.0"),
            quantity=Decimal("1.0"),
            side=TradeSide.BUY,
            timestamp=datetime.now(),
            trade_id="noop-trade-id",
            exchange="noop-exchange",
            maker_order_id="noop-maker",
            taker_order_id="noop-taker",
            received_at=datetime.now(),
            is_buyer_maker=False,
        )

    def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """
        Convert multiple raw trades - returns fake Trade objects.

        This implementation returns fake Trade objects for each input
        without performing any actual conversion.

        Args:
            raw_trades: List of raw trade data
            symbol: Trading symbol

        Returns:
            List of fake Trade objects
        """
        return [self.convert_trade(raw_trade, symbol) for raw_trade in raw_trades]

    def convert_kline(self, raw_kline: dict[str, Any], symbol: str) -> Kline:
        """
        Convert raw kline data - returns a fake Kline object.

        This implementation always returns a fake Kline object without
        performing any actual data conversion.

        Args:
            raw_kline: Raw kline data (ignored)
            symbol: Trading symbol

        Returns:
            A fake Kline object with default values
        """
        from datetime import timedelta

        now = datetime.now()
        # Ensure close_time is after open_time for validation
        open_time = now
        close_time = now + timedelta(minutes=1)

        return Kline(
            symbol=symbol,
            interval=KlineInterval.MINUTE_1,
            open_time=open_time,
            close_time=close_time,
            open_price=Decimal("100.0"),
            high_price=Decimal("105.0"),
            low_price=Decimal("95.0"),
            close_price=Decimal("102.0"),
            volume=Decimal("1000.0"),
            quote_volume=Decimal("102000.0"),
            trades_count=100,
            exchange="noop-exchange",
            taker_buy_volume=Decimal("500.0"),
            taker_buy_quote_volume=Decimal("51000.0"),
            is_closed=True,
            received_at=now,
        )

    def convert_multiple_klines(self, raw_klines: list[dict[str, Any]], symbol: str) -> list[Kline]:
        """
        Convert multiple raw klines - returns fake Kline objects.

        This implementation returns fake Kline objects for each input
        without performing any actual conversion.

        Args:
            raw_klines: List of raw kline data
            symbol: Trading symbol

        Returns:
            List of fake Kline objects
        """
        return [self.convert_kline(raw_kline, symbol) for raw_kline in raw_klines]

    def validate_raw_data(self, data: RawDataType) -> tuple[bool, str]:
        """
        Validate raw data - always returns successful validation.

        This implementation always returns True without performing
        any actual data validation.

        Args:
            data: Raw data to validate (ignored)

        Returns:
            Tuple of (True, "") indicating successful validation
        """
        return True, ""

    def to_internal_timestamp(self, timestamp: Any) -> int:
        """
        Convert timestamp to internal format - returns current time.

        This implementation returns the current timestamp in milliseconds
        without performing any actual timestamp conversion.

        Args:
            timestamp: Timestamp to convert (ignored)

        Returns:
            Current timestamp in milliseconds
        """
        return int(datetime.now().timestamp() * 1000)
