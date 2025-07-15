from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, AsyncIterator

from core.models import Kline, Trade, KlineInterval


class AbstractDataProvider(ABC):
    """
    [L0] Abstract interface for a data provider.

    This interface defines the contract for implementing market data providers
    that can connect to external exchanges or data sources to provide real-time
    and historical market data (trades, K-lines). Concrete implementations
    are responsible for handling specific exchange API protocols and data formats.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Retrieves the unique name of the data provider.

        This name typically identifies the exchange or data source (e.g., "Binance", "Kraken").

        Returns:
            A string representing the provider's name.
        """
        pass

    @abstractmethod
    async def connect(self) -> None:
        """
        Establishes a connection to the data provider's service.

        This asynchronous method should handle the necessary handshake, authentication,
        and initialization to prepare the provider for data exchange.

        Raises:
            ConnectionError: If the connection attempt fails.
            AuthenticationError: If authentication credentials are invalid.
            NetworkError: For general network-related issues during connection.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Closes the connection to the data provider's service.

        This asynchronous method should gracefully terminate any active connections
        and release associated network resources.

        Raises:
            NetworkError: If an error occurs during disconnection.
        """
        pass

    @abstractmethod
    def stream_trades(self, symbol: str, *, start_from: datetime | None = None) -> AsyncIterator[Trade]:
        """
        Streams real-time trade data for a specified trading symbol.

        This method is an asynchronous generator that yields `Trade` objects
        as they become available from the data provider.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT") for which to stream trades.
            start_from: Optional. A `datetime` object indicating the historical point
                        from which to start streaming trades. If `None`, streaming
                        starts from the current real-time data.

        Yields:
            `Trade`: Each standardized trade event as it occurs.

        Raises:
            ConnectionError: If the connection to the provider is lost during streaming.
            DataProviderError: For provider-specific errors during streaming.
        """
        pass

    @abstractmethod
    def stream_klines(
        self, symbol: str, interval: KlineInterval, *, start_from: datetime | None = None
    ) -> AsyncIterator[Kline]:
        """
        Streams real-time K-line (candlestick) data for a specified symbol and interval.

        This method is an asynchronous generator that yields `Kline` objects
        as new K-line candles are closed and become available from the data provider.

        Args:
            symbol: The trading symbol (e.g., "ETH/USD") for which to stream K-lines.
            interval: The `KlineInterval` enum member specifying the time granularity
                      of the K-lines (e.g., 1 minute, 1 hour).
            start_from: Optional. A `datetime` object indicating the historical point
                        from which to start streaming K-lines. If `None`, streaming
                        starts from the current real-time data.

        Yields:
            `Kline`: Each standardized K-line event as it closes.

        Raises:
            ConnectionError: If the connection to the provider is lost during streaming.
            DataProviderError: For provider-specific errors during streaming.
        """
        pass

    @abstractmethod
    async def fetch_historical_trades(
        self, symbol: str, start_time: datetime, end_time: datetime, *, limit: int | None = None
    ) -> list[Trade]:
        """
        Fetches historical trade data within a specified time range.

        This asynchronous method retrieves a list of past trade events from the provider.

        Args:
            symbol: The trading symbol for which to fetch historical trades.
            start_time: The `datetime` object marking the beginning of the historical period.
            end_time: The `datetime` object marking the end of the historical period.
            limit: Optional. The maximum number of historical trades to fetch. If `None`,
                   the provider's default or maximum limit will apply.

        Returns:
            A list of standardized `Trade` models within the specified time range.

        Raises:
            DataProviderError: If the historical data cannot be fetched (e.g., invalid time range,
                               provider error, rate limit).
            NetworkError: For underlying network issues.
            TimeoutError: If the request exceeds the allotted time.
        """
        pass

    @abstractmethod
    async def fetch_historical_klines(
        self,
        symbol: str,
        interval: KlineInterval,
        start_time: datetime,
        end_time: datetime,
        *,
        limit: int | None = None,
    ) -> list[Kline]:
        """
        Fetches historical K-line (candlestick) data within a specified time range.

        This asynchronous method retrieves a list of past K-line candles from the provider.

        Args:
            symbol: The trading symbol for which to fetch historical K-lines.
            interval: The `KlineInterval` enum member for the desired K-line granularity.
            start_time: The `datetime` object marking the beginning of the historical period.
            end_time: The `datetime` object marking the end of the historical period.
            limit: Optional. The maximum number of historical K-lines to fetch. If `None`,
                   the provider's default or maximum limit will apply.

        Returns:
            A list of standardized `Kline` models within the specified time range.

        Raises:
            DataProviderError: If the historical data cannot be fetched (e.g., invalid time range,
                               provider error, rate limit).
            NetworkError: For underlying network issues.
            TimeoutError: If the request exceeds the allotted time.
        """
        pass

    @abstractmethod
    async def get_exchange_info(self) -> dict[str, Any]:
        """
        Retrieves general information about the exchange.

        This can include supported symbols, trading rules, asset details,
        and other global exchange configurations.

        Returns:
            A dictionary containing exchange information.

        Raises:
            DataProviderError: If the information cannot be retrieved.
            NetworkError: For underlying network issues.
        """
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """
        Retrieves detailed information for a specific trading symbol.

        This can include precision rules, minimum/maximum order quantities,
        trading status, and other symbol-specific metadata.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT") for which to retrieve information.

        Returns:
            A dictionary containing detailed information for the specified symbol.

        Raises:
            DataProviderError: If the symbol information cannot be retrieved or the symbol is invalid.
            NetworkError: For underlying network issues.
        """
        pass

    @abstractmethod
    async def ping(self) -> float:
        """
        Pings the data provider to check connectivity and measure latency.

        Returns:
            The response time in milliseconds as a float.

        Raises:
            NetworkError: If the ping fails due to connectivity issues.
            TimeoutError: If the ping request times out.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Closes the data provider and cleans up any associated resources.

        This asynchronous method should release network connections, stop background
        tasks, and perform any necessary shutdown procedures to gracefully
        terminate the provider's operation.
        """
        pass

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """
        Validates the provided configuration dictionary for the data provider.

        This method ensures that the configuration contains all necessary parameters
        and that their values are valid for initializing the provider.

        Args:
            config: A dictionary containing the configuration parameters for the provider.

        Returns:
            A tuple where the first element is `True` if the configuration is valid,
            and `False` otherwise. The second element is an error message (string)
            if validation fails, or an empty string (`""`) if validation succeeds.
        """
        pass

    @abstractmethod
    async def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """
        Converts multiple raw trade records from the exchange into a list of standardized `Trade` models.

        This method is typically implemented by a data converter (e.g., `AbstractDataConverter`)
        and is included here to indicate that data providers are expected to facilitate
        this conversion for their specific raw data formats.

        Args:
            raw_trades: A list of dictionaries, where each dictionary contains raw trade data
                        from the exchange.
            symbol: The trading symbol for which the trades occurred.

        Returns:
            A list of standardized `Trade` model instances.

        Raises:
            ValueError: If any raw trade data in the list is malformed or missing critical fields.
            TypeError: If data types within any raw trade are unexpected.
        """
        pass

    @abstractmethod
    async def convert_multiple_klines(self, raw_klines: list[dict[str, Any]], symbol: str) -> list[Kline]:
        """
        Converts multiple raw K-line records from the exchange into a list of standardized `Kline` models.

        Similar to `convert_multiple_trades`, this method is typically implemented by a data converter
        and is included here to indicate that data providers are expected to facilitate
        this conversion for their specific raw K-line data formats.

        Args:
            raw_klines: A list of dictionaries, where each dictionary contains raw K-line data
                        from the exchange.
            symbol: The trading symbol for which the K-lines occurred.

        Returns:
            A list of standardized `Kline` model instances.

        Raises:
            ValueError: If any raw K-line data in the list is malformed or missing critical fields.
            TypeError: If data types within any raw K-line are unexpected.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Checks if the data provider is currently connected to its external service.

        Returns:
            `True` if the provider has an active connection, `False` otherwise.
        """
        pass

    async def __aenter__(self) -> "AbstractDataProvider":
        """
        Asynchronous context manager entry point.

        This method allows the `AbstractDataProvider` to be used with the `async with` statement.
        It ensures that the `connect()` method is called upon entering the context.

        Returns:
            The instance of the `AbstractDataProvider` itself.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Asynchronous context manager exit point.

        This method ensures that the `close()` method is called upon exiting the `async with` block,
        regardless of whether an exception occurred. It handles resource cleanup.

        Args:
            exc_type: The type of the exception that caused the context to be exited, or `None`.
            exc_val: The exception instance that caused the context to be exited, or `None`.
            exc_tb: The traceback object associated with the exception, or `None`.
        """
        await self.close()
