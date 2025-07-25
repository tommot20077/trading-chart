# ABOUTME: NoOp implementation of AbstractDataProvider that provides fake data
# ABOUTME: Provides minimal data provider functionality for testing scenarios

from typing import AsyncIterator, Any
from datetime import datetime, timedelta
from decimal import Decimal

from core.interfaces.data.provider import AbstractDataProvider
from core.models.data.kline import Kline
from core.models.data.trade import Trade
from core.models.data.enum import KlineInterval, TradeSide


class NoOpDataProvider(AbstractDataProvider):
    """
    No-operation implementation of AbstractDataProvider.

    This implementation provides minimal data provider functionality that
    returns fake data without connecting to any actual data sources. It's
    useful for testing, performance benchmarking, and scenarios where real
    data is not required.

    Features:
    - Always returns fake successful data
    - No actual external connections or data fetching
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where real data should be replaced with fake data
    - Performance benchmarking without external data dependencies
    - Development environments where real data is not needed
    - Fallback when data providers are unavailable
    """

    def __init__(self):
        """Initialize the no-operation data provider."""
        self._closed = False
        self._connected = False

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "NoOpDataProvider"

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and not self._closed

    async def connect(self) -> None:
        """Connect to the provider - always succeeds."""
        if self._closed:
            raise RuntimeError("Data provider is closed")
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the provider."""
        self._connected = False

    async def ping(self) -> float:
        """Ping the provider - returns fake latency."""
        if self._closed:
            raise RuntimeError("Data provider is closed")
        return 1.0  # Fake 1ms latency

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Validate config - always succeeds."""
        return True, ""

    async def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange info - returns fake info."""
        if self._closed:
            raise RuntimeError("Data provider is closed")
        return {
            "name": "NoOp Exchange",
            "status": "normal",
            "symbols": ["BTC/USDT", "ETH/USDT"],
        }

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol info - returns fake info."""
        if self._closed:
            raise RuntimeError("Data provider is closed")
        return {
            "symbol": symbol,
            "status": "trading",
            "base_asset": symbol.split("/")[0] if "/" in symbol else "BTC",
            "quote_asset": symbol.split("/")[1] if "/" in symbol else "USDT",
        }

    async def fetch_historical_trades(
        self, symbol: str, start_time: datetime, end_time: datetime, *, limit: int | None = None
    ) -> list[Trade]:
        """Fetch historical trades - returns fake trades."""
        return await self.get_latest_trades(symbol, limit or 100)

    async def fetch_historical_klines(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int | None = None,
    ) -> list[Kline]:
        """Fetch historical klines - returns fake klines."""
        return await self.get_klines(symbol, interval, start_time, end_time, limit or 100)

    async def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """Convert multiple trades - returns fake trades."""
        if self._closed:
            raise RuntimeError("Data provider is closed")
        return [self._create_fake_trade(symbol, i) for i in range(len(raw_trades))]

    async def convert_multiple_klines(self, raw_klines: list[dict[str, Any]], symbol: str) -> list[Kline]:
        """Convert multiple klines - returns fake klines."""
        if self._closed:
            raise RuntimeError("Data provider is closed")
        return [self._create_fake_kline(symbol, KlineInterval.MINUTE_1, i) for i in range(len(raw_klines))]

    def _create_fake_trade(self, symbol: str, index: int = 0) -> Trade:
        """Create a fake trade for testing."""
        return Trade(
            symbol=symbol,
            price=Decimal(f"{100 + index}.0"),
            quantity=Decimal("1.0"),
            side=TradeSide.BUY if index % 2 == 0 else TradeSide.SELL,
            timestamp=datetime.now(),
            trade_id=f"noop-trade-{index}",
            exchange="noop-exchange",
            maker_order_id=f"maker-{index}",
            taker_order_id=f"taker-{index}",
            received_at=datetime.now(),
            is_buyer_maker=index % 2 == 0,
        )

    def _create_fake_kline(self, symbol: str, interval: KlineInterval, index: int = 0) -> Kline:
        """Create a fake kline for testing."""
        now = datetime.now()
        interval_duration = KlineInterval.to_timedelta(interval)
        open_time = now + timedelta(seconds=index * interval_duration.total_seconds())
        close_time = open_time + interval_duration

        return Kline(
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            close_time=close_time,
            open_price=Decimal(f"{100 + index}.0"),
            high_price=Decimal(f"{105 + index}.0"),
            low_price=Decimal(f"{95 + index}.0"),
            close_price=Decimal(f"{102 + index}.0"),
            volume=Decimal("1000.0"),
            quote_volume=Decimal(f"{102000 + index * 1000}.0"),
            trades_count=100 + index,
            exchange="noop-exchange",
            is_closed=True,
            received_at=now,
            taker_buy_volume=Decimal("500.0"),
            taker_buy_quote_volume=Decimal("51000.0"),
        )

    async def get_latest_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get latest trades - returns fake Trade objects.

        This implementation returns fake Trade objects without connecting
        to any actual data source.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return

        Returns:
            List of fake Trade objects
        """
        if self._closed:
            raise RuntimeError("Data provider is closed")

        trades = []
        now = datetime.now()
        for i in range(min(limit, 10)):  # Return max 10 fake trades
            trades.append(
                Trade(
                    symbol=symbol,
                    price=Decimal(f"{100 + i}.0"),
                    quantity=Decimal("1.0"),
                    side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
                    timestamp=now,
                    trade_id=f"noop-trade-{i}",
                    exchange="noop-exchange",
                    maker_order_id=f"maker-{i}",
                    taker_order_id=f"taker-{i}",
                    received_at=now,
                    is_buyer_maker=i % 2 == 0,
                )
            )
        return trades

    async def get_historical_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> list[Trade]:
        """
        Get historical trades - returns fake Trade objects.

        This implementation returns fake Trade objects without connecting
        to any actual data source.

        Args:
            symbol: Trading symbol
            start_time: Start time for historical data
            end_time: End time for historical data
            limit: Maximum number of trades to return

        Returns:
            List of fake Trade objects
        """
        if self._closed:
            raise RuntimeError("Data provider is closed")

        # Return a few fake trades
        return await self.get_latest_trades(symbol, min(limit, 10))

    async def get_klines(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> list[Kline]:
        """
        Get klines - returns fake Kline objects.

        This implementation returns fake Kline objects without connecting
        to any actual data source.

        Args:
            symbol: Trading symbol
            interval: Kline interval
            start_time: Start time for kline data
            end_time: End time for kline data
            limit: Maximum number of klines to return

        Returns:
            List of fake Kline objects
        """
        if self._closed:
            raise RuntimeError("Data provider is closed")

        klines = []
        # Use start_time as base to ensure klines are within the requested range
        interval_duration = KlineInterval.to_timedelta(interval)

        for i in range(min(limit, 10)):  # Return max 10 fake klines
            open_time = start_time + timedelta(seconds=i * interval_duration.total_seconds())
            close_time = open_time + interval_duration

            # Stop if we exceed end_time
            if open_time >= end_time:
                break

            klines.append(
                Kline(
                    symbol=symbol,
                    interval=interval,
                    open_time=open_time,
                    close_time=close_time,
                    open_price=Decimal(f"{100 + i}.0"),
                    high_price=Decimal(f"{105 + i}.0"),
                    low_price=Decimal(f"{95 + i}.0"),
                    close_price=Decimal(f"{102 + i}.0"),
                    volume=Decimal("1000.0"),
                    quote_volume=Decimal(f"{102000 + i * 1000}.0"),
                    trades_count=100 + i,
                    exchange="noop-exchange",
                    is_closed=True,
                    received_at=datetime.now(),
                    taker_buy_volume=Decimal("500.0"),
                    taker_buy_quote_volume=Decimal("51000.0"),
                )
            )
        return klines

    async def stream_trades(self, symbol: str, *, start_from: datetime | None = None) -> AsyncIterator[Trade]:
        """
        Stream trades - yields fake Trade objects.

        This implementation yields a few fake Trade objects without
        connecting to any actual data stream.

        Args:
            symbol: Trading symbol

        Yields:
            Fake Trade objects
        """
        if self._closed:
            raise RuntimeError("Data provider is closed")

        # Yield a few fake trades and then stop
        for i in range(3):
            yield self._create_fake_trade(symbol, i)

    async def stream_klines(
        self, symbol: str, interval: KlineInterval, *, start_from: datetime | None = None
    ) -> AsyncIterator[Kline]:
        """
        Stream klines - yields fake Kline objects.

        This implementation yields a few fake Kline objects without
        connecting to any actual data stream.

        Args:
            symbol: Trading symbol
            interval: Kline interval

        Yields:
            Fake Kline objects
        """
        if self._closed:
            raise RuntimeError("Data provider is closed")

        # Yield a few fake klines and then stop
        for i in range(3):
            yield self._create_fake_kline(symbol, interval, i)

    async def health_check(self) -> bool:
        """
        Perform health check - always returns True.

        This implementation always returns True indicating the provider
        is healthy.

        Returns:
            True (NoOp implementation is always healthy)
        """
        return not self._closed

    async def close(self) -> None:
        """
        Close the data provider - sets closed flag.
        """
        self._closed = True
