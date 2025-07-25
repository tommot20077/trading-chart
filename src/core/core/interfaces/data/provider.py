# ABOUTME: Abstract data provider interface for market data access and streaming
# ABOUTME: Defines the contract for components that connect to exchanges and provide real-time and historical market data

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
            str: A string representing the provider's name.
        """
        pass

    @abstractmethod
    async def connect(self) -> None:
        """
        Establishes a connection to the data provider's service.

        This asynchronous method handles the necessary handshake, authentication,
        and initialization to prepare the provider for data exchange. It ensures
        that the provider is ready to stream or fetch market data.

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

        This asynchronous method gracefully terminates any active connections
        and releases associated network resources, ensuring a clean shutdown.

        Raises:
            NetworkError: If an error occurs during disconnection.
        """
        pass

    @abstractmethod
    def stream_trades(self, symbol: str, *, start_from: datetime | None = None) -> AsyncIterator[Trade]:
        """
        Streams real-time trade data for a specified trading symbol.

        This method is an asynchronous generator that yields `Trade` objects
        as they become available from the data provider. It is designed for
        continuous, real-time consumption of trade events.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which to stream trades.
            start_from (datetime | None): Optional. A `datetime` object indicating the historical point
                                         from which to start streaming trades. If `None`, streaming
                                         starts from the current real-time data.

        Yields:
            Trade: Each standardized trade event as it occurs.

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
        It is designed for continuous, real-time consumption of candlestick data.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which to stream K-lines.
            interval (KlineInterval): The `KlineInterval` enum member specifying the time granularity
                                    of the K-lines (e.g., 1 minute, 1 hour).
            start_from (datetime | None): Optional. A `datetime` object indicating the historical point
                                         from which to start streaming K-lines. If `None`, streaming
                                         starts from the current real-time data.

        Yields:
            Kline: Each standardized K-line event as it closes.

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
        It is suitable for backtesting, analysis, or populating initial datasets.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which to fetch historical trades.
            start_time (datetime): The `datetime` object marking the beginning of the historical period (inclusive).
            end_time (datetime): The `datetime` object marking the end of the historical period (inclusive).
            limit (int | None): Optional. The maximum number of historical trades to fetch. If `None`,
                               the provider's default or maximum limit will apply.

        Returns:
            list[Trade]: A list of standardized `Trade` models within the specified time range.

        Raises:
            DataProviderError: If the historical data cannot be fetched (e.g., invalid time range,
                               provider error, or rate limit issues).
            NetworkError: For underlying network issues during the data retrieval.
            TimeoutError: If the request to the data provider exceeds the allotted time.
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
        It is useful for backtesting, charting, and historical analysis.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which to fetch historical K-lines.
            interval (KlineInterval): The `KlineInterval` enum member for the desired K-line granularity.
            start_time (datetime): The `datetime` object marking the beginning of the historical period (inclusive).
            end_time (datetime): The `datetime` object marking the end of the historical period (inclusive).
            limit (int | None): Optional. The maximum number of historical K-lines to fetch. If `None`,
                               the provider's default or maximum limit will apply.

        Returns:
            list[Kline]: A list of standardized `Kline` models within the specified time range.

        Raises:
            DataProviderError: If the historical data cannot be fetched (e.g., invalid time range,
                               provider error, or rate limit issues).
            NetworkError: For underlying network issues during the data retrieval.
            TimeoutError: If the request to the data provider exceeds the allotted time.
        """
        pass

    @abstractmethod
    async def get_exchange_info(self) -> dict[str, Any]:
        """
        Retrieves general information about the exchange.

        This asynchronous method fetches global exchange configurations, including
        supported symbols, trading rules, asset details, and other relevant metadata.

        Returns:
            dict[str, Any]: A dictionary containing comprehensive exchange information.

        Raises:
            DataProviderError: If the information cannot be retrieved due to provider-specific issues.
            NetworkError: For underlying network issues during the retrieval.
        """
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """
        Retrieves detailed information for a specific trading symbol.

        This asynchronous method fetches symbol-specific metadata, such as precision rules,
        minimum/maximum order quantities, trading status, and other relevant details.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which to retrieve information.

        Returns:
            dict[str, Any]: A dictionary containing detailed information for the specified symbol.

        Raises:
            DataProviderError: If the symbol information cannot be retrieved or the symbol is invalid.
            NetworkError: For underlying network issues during the retrieval.
        """
        pass

    @abstractmethod
    async def ping(self) -> float:
        """
        Pings the data provider to check connectivity and measure latency.

        This asynchronous method sends a lightweight request to the data provider
        to verify network connectivity and estimate the round-trip time.

        Returns:
            float: The response time in milliseconds as a float.

        Raises:
            NetworkError: If the ping fails due to connectivity issues.
            TimeoutError: If the ping request times out.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Closes the data provider and cleans up any associated resources.

        This asynchronous method releases network connections, stops background
        tasks, and performs any necessary shutdown procedures to gracefully
        terminate the provider's operation. It should be called to ensure
        proper resource management.

        Returns:
            None: This method does not return a value.

        Raises:
            DataProviderError: If cleanup fails due to provider-specific issues.
            NetworkError: If cleanup fails due to network issues.
        """
        pass

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """
        Validates the provided configuration dictionary for the data provider.

        This method ensures that the configuration contains all necessary parameters
        and that their values are valid for initializing the provider. It serves as
        an early check to prevent runtime errors due to malformed configurations.

        Args:
            config (dict[str, Any]): A dictionary containing the configuration parameters for the provider.

        Returns:
            tuple[bool, str]: A tuple where the first element is `True` if the configuration is valid,
                              and `False` otherwise. The second element is an error message (string)
                              if validation fails, or an empty string (`""`) if validation succeeds.

        Raises:
            ValueError: If the config parameter itself is malformed (e.g., not a dict).
        """
        pass

    @abstractmethod
    async def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """
        Converts multiple raw trade records from the exchange into a list of standardized `Trade` models.

        This method is typically implemented by a data converter (e.g., `AbstractDataConverter`)
        and is included here to indicate that data providers are expected to facilitate
        this conversion for their specific raw data formats. It efficiently processes
        batches of raw trade data.

        Args:
            raw_trades (list[dict[str, Any]]): A list of dictionaries, where each dictionary contains raw trade data
                                              from the exchange.
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which the trades occurred.

        Returns:
            list[Trade]: A list of standardized `Trade` model instances.

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
        this conversion for their specific raw K-line data formats. It efficiently processes
        batches of raw K-line data.

        Args:
            raw_klines (list[dict[str, Any]]): A list of dictionaries, where each dictionary contains raw K-line data
                                              from the exchange.
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which the K-lines occurred.

        Returns:
            list[Kline]: A list of standardized `Kline` model instances.

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

        This property provides a quick way to ascertain the current connection status
        without attempting to establish a new connection.

        Returns:
            bool: `True` if the provider has an active connection, `False` otherwise.
        """
        pass

    async def __aenter__(self) -> "AbstractDataProvider":
        """
        Asynchronous context manager entry point.

        This method allows the `AbstractDataProvider` to be used with the `async with` statement.
        It ensures that the `connect()` method is called upon entering the context,
        automatically establishing the connection.

        Returns:
            AbstractDataProvider: The instance of the `AbstractDataProvider` itself.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Asynchronous context manager exit point.

        This method ensures that the `close()` method is called upon exiting the `async with` block,
        regardless of whether an exception occurred. It handles resource cleanup,
        such as closing network connections and stopping background tasks.

        Args:
            exc_type (Any): The type of the exception that caused the context to be exited, or `None`.
            exc_val (Any): The exception instance that caused the context to be exited, or `None`.
            exc_tb (Any): The traceback object associated with the exception, or `None`.

        Returns:
            None: This method does not return a value.
        """
        await self.close()
