# ABOUTME: In-memory implementation of AbstractDataProvider for testing and development
# ABOUTME: Provides mock data generation and storage without external dependencies

import asyncio
from datetime import datetime, timedelta, UTC
from typing import Any, AsyncIterator
from collections import defaultdict
import random
from decimal import Decimal

from core.interfaces.data.provider import AbstractDataProvider
from core.models import Kline, Trade, KlineInterval, TradeSide
from core.config.market_limits import get_market_limits_config
from core.exceptions.base import (
    ExternalServiceException,
    DataNotFoundException,
    ValidationException,
)


class MemoryDataProvider(AbstractDataProvider):
    """
    In-memory implementation of AbstractDataProvider.

    This implementation provides mock market data for testing and development purposes.
    It generates realistic-looking trade and kline data without requiring external connections.
    All data is stored in memory and reset when the provider is recreated.

    Features:
    - Generates realistic mock trade data
    - Creates proper kline data from trades
    - Supports all standard kline intervals
    - Simulates connection states
    - Provides configurable latency simulation
    """

    def __init__(self, name: str = "MemoryProvider", latency_ms: float = 10.0):
        """
        Initialize the memory data provider.

        Args:
            name: The name of this provider instance
            latency_ms: Simulated network latency in milliseconds
        """
        self._name = name
        self._latency_ms = latency_ms
        self._connected = False

        # In-memory storage
        self._trades: dict[str, list[Trade]] = defaultdict(list)
        self._klines: dict[tuple[str, KlineInterval], list[Kline]] = defaultdict(list)

        # Mock exchange info
        self._exchange_info = {
            "name": self._name,
            "status": "normal",
            "timezone": "UTC",
            "serverTime": datetime.now(UTC).isoformat(),
            "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],
            "rateLimits": [{"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 1200}],
        }

        # Symbol-specific info
        self._symbol_info = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "baseAssetPrecision": 8,
                "orderTypes": ["LIMIT", "MARKET"],
                "filters": [
                    {"filterType": "PRICE_FILTER", "minPrice": "0.01000000", "maxPrice": "1000000.00000000"},
                    {"filterType": "LOT_SIZE", "minQty": "0.00000100", "maxQty": "9000.00000000"},
                ],
            },
            "ETHUSDT": {
                "symbol": "ETHUSDT",
                "status": "TRADING",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "quotePrecision": 8,
                "baseAssetPrecision": 8,
                "orderTypes": ["LIMIT", "MARKET"],
                "filters": [
                    {"filterType": "PRICE_FILTER", "minPrice": "0.01000000", "maxPrice": "100000.00000000"},
                    {"filterType": "LOT_SIZE", "minQty": "0.00001000", "maxQty": "90000.00000000"},
                ],
            },
        }

        # Base prices for realistic data generation
        self._base_prices = {
            "BTCUSDT": Decimal("45000.00"),
            "ETHUSDT": Decimal("3000.00"),
            "BNBUSDT": Decimal("300.00"),
            "ADAUSDT": Decimal("0.50"),
        }

        # Keep original prices to prevent excessive drift
        self._original_prices = self._base_prices.copy()

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        return self._connected

    async def connect(self) -> None:
        """Simulate connection to the data provider."""
        await self._simulate_latency()

        if self._connected:
            return

        # Simulate connection process
        self._connected = True

        # Pre-populate some historical data for testing
        await self._generate_initial_data()

    async def disconnect(self) -> None:
        """Simulate disconnection from the data provider."""
        await self._simulate_latency()
        self._connected = False

    async def close(self) -> None:
        """Close the provider and clean up resources."""
        await self.disconnect()

        # Clear all data
        self._trades.clear()
        self._klines.clear()

    async def ping(self) -> float:
        """Simulate ping to check connectivity."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        start_time = asyncio.get_event_loop().time()
        await self._simulate_latency()
        end_time = asyncio.get_event_loop().time()

        return (end_time - start_time) * 1000  # Convert to milliseconds

    async def get_exchange_info(self) -> dict[str, Any]:
        """Return mock exchange information."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        await self._simulate_latency()

        # Update server time
        info = self._exchange_info.copy()
        info["serverTime"] = datetime.now(UTC).isoformat()
        return info

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Return mock symbol information."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        await self._simulate_latency()

        if symbol not in self._symbol_info:
            raise DataNotFoundException(
                f"Symbol {symbol} not found", code="SYMBOL_NOT_FOUND", details={"symbol": symbol}
            )

        return self._symbol_info[symbol].copy()

    async def stream_trades(self, symbol: str, *, start_from: datetime | None = None) -> AsyncIterator[Trade]:
        """Stream mock trade data for the specified symbol."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        if symbol not in self._base_prices:
            raise DataNotFoundException(
                f"Symbol {symbol} not supported",
                code="SYMBOL_NOT_SUPPORTED",
                details={"symbol": symbol, "supported_symbols": list(self._base_prices.keys())},
            )

        # If start_from is provided, yield historical trades first
        if start_from:
            historical_trades = await self.fetch_historical_trades(symbol, start_from, datetime.now(UTC))
            for trade in historical_trades:
                yield trade

        # Then stream real-time trades
        current_price = self._base_prices[symbol]

        while self._connected:
            # Generate a random trade
            trade = self._generate_random_trade(symbol, current_price)
            self._trades[symbol].append(trade)

            # Update current price slightly
            price_change = Decimal(str(random.uniform(-0.02, 0.02)))  # ±2% change
            current_price = current_price * (1 + price_change)
            self._base_prices[symbol] = current_price

            yield trade

            # Wait before next trade (simulate realistic timing)
            await asyncio.sleep(random.uniform(0.1, 2.0))

    async def stream_klines(
        self, symbol: str, interval: KlineInterval, *, start_from: datetime | None = None
    ) -> AsyncIterator[Kline]:
        """Stream mock kline data for the specified symbol and interval."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        if symbol not in self._base_prices:
            raise DataNotFoundException(
                f"Symbol {symbol} not supported",
                code="SYMBOL_NOT_SUPPORTED",
                details={"symbol": symbol, "supported_symbols": list(self._base_prices.keys())},
            )

        # If start_from is provided, yield historical klines first
        if start_from:
            historical_klines = await self.fetch_historical_klines(symbol, interval, start_from, datetime.now(UTC))
            for kline in historical_klines:
                yield kline

        # Then stream real-time klines
        interval_seconds = KlineInterval.to_seconds(interval)
        current_price = self._base_prices[symbol]

        while self._connected:
            # Generate a kline for the current interval
            now = datetime.now(UTC)
            # Align to interval boundary
            interval_start = self._align_to_interval(now, interval_seconds)

            kline = self._generate_random_kline(symbol, interval, interval_start, current_price)
            self._klines[(symbol, interval)].append(kline)

            # Update current price
            current_price = kline.close_price
            self._base_prices[symbol] = current_price

            yield kline

            # Wait for next interval
            await asyncio.sleep(interval_seconds)

    async def fetch_historical_trades(
        self, symbol: str, start_time: datetime, end_time: datetime, *, limit: int | None = None
    ) -> list[Trade]:
        """Fetch mock historical trade data."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        await self._simulate_latency()

        if symbol not in self._base_prices:
            raise DataNotFoundException(
                f"Symbol {symbol} not supported", code="SYMBOL_NOT_SUPPORTED", details={"symbol": symbol}
            )

        if start_time >= end_time:
            raise ValidationException(
                "start_time must be before end_time",
                code="INVALID_TIME_RANGE",
                details={"start_time": start_time.isoformat(), "end_time": end_time.isoformat()},
            )

        # Generate historical trades
        trades: list[Trade] = []
        current_time = start_time
        base_price = self._base_prices[symbol]

        while current_time < end_time and (limit is None or len(trades) < limit):
            trade = self._generate_random_trade(symbol, base_price, timestamp=current_time)
            trades.append(trade)

            # Move time forward randomly
            current_time += timedelta(seconds=random.uniform(1, 30))

            # Adjust price slightly
            price_change = Decimal(str(random.uniform(-0.01, 0.01)))
            base_price = base_price * (1 + price_change)

        return trades

    async def fetch_historical_klines(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int | None = None,
    ) -> list[Kline]:
        """Fetch mock historical kline data."""
        if not self._connected:
            raise ExternalServiceException("Provider not connected", code="NOT_CONNECTED")

        await self._simulate_latency()

        if symbol not in self._base_prices:
            raise DataNotFoundException(
                f"Symbol {symbol} not supported", code="SYMBOL_NOT_SUPPORTED", details={"symbol": symbol}
            )

        if start_time >= end_time:
            raise ValidationException(
                "start_time must be before end_time",
                code="INVALID_TIME_RANGE",
                details={"start_time": start_time.isoformat(), "end_time": end_time.isoformat()},
            )

        # Generate historical klines
        klines: list[Kline] = []
        interval_seconds = KlineInterval.to_seconds(interval)
        current_time = self._align_to_interval(start_time, interval_seconds)
        base_price = self._base_prices[symbol]

        while current_time < end_time and (limit is None or len(klines) < limit):
            kline = self._generate_random_kline(symbol, interval, current_time, base_price)
            klines.append(kline)

            # Move to next interval
            current_time += timedelta(seconds=interval_seconds)

            # Update base price to kline close, but prevent excessive drift
            base_price = kline.close_price
            original_price = self._original_prices[symbol]

            # Reset price if it drifts too far from original (±50%)
            if abs(base_price - original_price) / original_price > Decimal("0.5"):
                base_price = original_price

        return klines

    async def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """Convert raw trade data to Trade models."""
        await self._simulate_latency()

        trades: list[Trade] = []
        for raw_trade in raw_trades:
            try:
                trade = Trade(
                    symbol=symbol,
                    trade_id=str(raw_trade.get("id", random.randint(1000000, 9999999))),
                    price=Decimal(str(raw_trade["price"])),
                    quantity=Decimal(str(raw_trade["quantity"])),
                    side=TradeSide(raw_trade.get("side", "buy")),
                    timestamp=datetime.fromisoformat(raw_trade["timestamp"])
                    if isinstance(raw_trade["timestamp"], str)
                    else raw_trade["timestamp"],
                    exchange="MemoryProvider",
                    maker_order_id=str(raw_trade.get("maker_order_id", f"maker_{random.randint(1000, 9999)}")),
                    taker_order_id=str(raw_trade.get("taker_order_id", f"taker_{random.randint(1000, 9999)}")),
                    received_at=datetime.now(),
                    is_buyer_maker=raw_trade.get("is_buyer_maker", False),
                )
                trades.append(trade)
            except (KeyError, ValueError, TypeError) as e:
                raise ValidationException(
                    f"Invalid trade data: {e}",
                    code="INVALID_TRADE_DATA",
                    details={"raw_trade": raw_trade, "error": str(e)},
                )

        return trades

    async def convert_multiple_klines(self, raw_klines: list[dict[str, Any]], symbol: str) -> list[Kline]:
        """Convert raw kline data to Kline models."""
        await self._simulate_latency()

        klines: list[Kline] = []
        for raw_kline in raw_klines:
            try:
                kline = Kline(
                    symbol=symbol,
                    interval=KlineInterval(raw_kline["interval"]),
                    open_time=datetime.fromisoformat(raw_kline["open_time"])
                    if isinstance(raw_kline["open_time"], str)
                    else raw_kline["open_time"],
                    close_time=datetime.fromisoformat(raw_kline["close_time"])
                    if isinstance(raw_kline["close_time"], str)
                    else raw_kline["close_time"],
                    open_price=Decimal(str(raw_kline["open_price"])),
                    high_price=Decimal(str(raw_kline["high_price"])),
                    low_price=Decimal(str(raw_kline["low_price"])),
                    close_price=Decimal(str(raw_kline["close_price"])),
                    volume=Decimal(str(raw_kline["volume"])),
                    quote_volume=Decimal(str(raw_kline.get("quote_volume", "0"))),
                    trades_count=raw_kline.get("trades_count", raw_kline.get("trade_count", 0)),
                    exchange="MemoryProvider",
                    is_closed=raw_kline.get("is_closed", True),
                    received_at=datetime.now(),
                    taker_buy_volume=Decimal(str(raw_kline.get("taker_buy_base_volume", "0"))),
                    taker_buy_quote_volume=Decimal(str(raw_kline.get("taker_buy_quote_volume", "0"))),
                )
                klines.append(kline)
            except (KeyError, ValueError, TypeError) as e:
                raise ValidationException(
                    f"Invalid kline data: {e}",
                    code="INVALID_KLINE_DATA",
                    details={"raw_kline": raw_kline, "error": str(e)},
                )

        return klines

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration for the memory provider."""
        required_fields: list[str] = []  # Memory provider has no required config

        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"

        # Validate optional fields
        if "latency_ms" in config:
            try:
                latency = float(config["latency_ms"])
                if latency < 0:
                    return False, "latency_ms must be non-negative"
            except (ValueError, TypeError):
                return False, "latency_ms must be a number"

        if "name" in config:
            if not isinstance(config["name"], str) or not config["name"].strip():
                return False, "name must be a non-empty string"

        return True, ""

    # Helper methods

    async def _simulate_latency(self) -> None:
        """Simulate network latency."""
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

    async def _generate_initial_data(self) -> None:
        """Generate some initial historical data for testing."""
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)  # Only last 1 hour (reduced from 24)

        for symbol in self._base_prices.keys():
            # Generate fewer trades for faster testing
            trades = await self.fetch_historical_trades(symbol, start_time, end_time, limit=10)  # Reduced from 100
            self._trades[symbol].extend(trades)

            # Generate fewer klines for faster testing
            for interval in [KlineInterval.MINUTE_1]:  # Only 1 interval instead of 2
                klines = await self.fetch_historical_klines(symbol, interval, start_time, end_time)
                self._klines[(symbol, interval)].extend(klines)

    def _generate_random_trade(self, symbol: str, base_price: Decimal, timestamp: datetime | None = None) -> Trade:
        """Generate a random trade around the base price."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Get market limits for precision
        config = get_market_limits_config()
        limits = config.get_limits(symbol)

        # Price variation ±1% with proper precision
        price_variation = Decimal(str(random.uniform(-0.01, 0.01)))
        price = base_price * (1 + price_variation)
        # Quantize to market precision
        precision = Decimal("0.1") ** limits.price_precision
        price = price.quantize(precision)

        # Random quantity with proper precision
        quantity = Decimal(str(random.uniform(0.001, 10.0)))
        # Quantize to market precision
        quantity_precision = Decimal("0.1") ** limits.quantity_precision
        quantity = quantity.quantize(quantity_precision)

        # Random side
        side = random.choice([TradeSide.BUY, TradeSide.SELL])

        return Trade(
            symbol=symbol,
            trade_id=str(random.randint(1000000, 9999999)),
            price=price,
            quantity=quantity,
            side=side,
            timestamp=timestamp,
            exchange="memory-exchange",
            maker_order_id=f"maker_{random.randint(1000, 9999)}",
            taker_order_id=f"taker_{random.randint(1000, 9999)}",
            received_at=timestamp,
            is_buyer_maker=random.choice([True, False]),
        )

    def _generate_random_kline(
        self, symbol: str, interval: KlineInterval, open_time: datetime, base_price: Decimal
    ) -> Kline:
        """Generate a random kline starting from the base price."""
        interval_seconds = KlineInterval.to_seconds(interval)
        close_time = open_time + timedelta(seconds=interval_seconds)

        # Get market limits for precision
        config = get_market_limits_config()
        limits = config.get_limits(symbol)

        # Generate OHLC prices
        open_price = base_price

        # Random price movements within the interval
        price_changes = [random.uniform(-0.02, 0.02) for _ in range(3)]  # 3 more price points
        prices = [open_price]

        for change in price_changes:
            new_price = prices[-1] * (1 + Decimal(str(change)))
            prices.append(new_price)

        close_price = prices[-1]
        high_price = max(prices)
        low_price = min(prices)

        # Quantize all prices to market precision
        precision = Decimal("0.1") ** limits.price_precision
        open_price = open_price.quantize(precision)
        high_price = high_price.quantize(precision)
        low_price = low_price.quantize(precision)
        close_price = close_price.quantize(precision)

        # Random volume - keep it reasonable with proper precision
        volume = Decimal(str(random.uniform(100, 1000)))
        quantity_precision = Decimal("0.1") ** limits.quantity_precision
        volume = volume.quantize(quantity_precision)
        # Calculate quote volume more conservatively
        avg_price = (high_price + low_price) / 2
        quote_volume = volume * avg_price
        # Quantize quote volume to precision
        quote_volume = quote_volume.quantize(quantity_precision)

        # Ensure quote_volume doesn't exceed limits
        max_quote_volume = Decimal("500000000.00")  # 500M limit
        if quote_volume > max_quote_volume:
            quote_volume = max_quote_volume

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
            trades_count=random.randint(50, 500),
            exchange="memory-exchange",
            is_closed=True,
            received_at=close_time,
            taker_buy_volume=volume * Decimal(str(random.uniform(0.3, 0.7))),
            taker_buy_quote_volume=quote_volume * Decimal(str(random.uniform(0.3, 0.7))),
        )

    def _align_to_interval(self, timestamp: datetime, interval_seconds: int) -> datetime:
        """Align timestamp to interval boundary."""
        # Convert to Unix timestamp
        unix_timestamp = timestamp.timestamp()

        # Align to interval boundary (floor to previous boundary)
        aligned_timestamp = (unix_timestamp // interval_seconds) * interval_seconds

        # Convert back to datetime
        aligned_dt = datetime.fromtimestamp(aligned_timestamp, tz=UTC)

        # Ensure the aligned time is not after the original timestamp
        if aligned_dt > timestamp:
            aligned_dt -= timedelta(seconds=interval_seconds)

        return aligned_dt
