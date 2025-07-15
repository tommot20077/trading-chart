"""
ABOUTME: [L0] Data converter interface for standardizing exchange data
ABOUTME: Defines abstract methods for converting raw exchange data into internal models
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from core.models.data.kline import Kline
from core.models.data.trade import Trade

RawDataType = TypeVar("RawDataType")


class AbstractDataConverter(ABC):
    """
    [L0] Abstract base class for data converters.

    This interface defines the contract for converting raw, exchange-specific
    data formats into the standardized internal data models (`Kline`, `Trade`, etc.)
    used throughout the `asset_core` framework. Implementations of this class are
    responsible for handling the unique data structures and conventions of
    different exchange APIs and transforming their data into a consistent,
    framework-agnostic format.
    """

    @abstractmethod
    def convert_trade(self, raw_trade: dict[str, Any], symbol: str) -> Trade:
        """
        Converts a single raw trade record from an exchange into a standardized `Trade` model.

        This method is crucial for normalizing trade data, ensuring that regardless of the
        source exchange's specific JSON or dictionary structure, the output `Trade` object
        adheres to the `asset_core`'s internal data schema. It involves parsing fields
        like price, quantity, timestamp, trade ID, and buyer/seller information.

        Args:
            raw_trade (dict[str, Any]): A dictionary containing the raw trade data as received directly
                                       from the exchange's API. The exact keys and values within this
                                       dictionary are specific to each exchange.
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which the trade occurred.
                          This ensures the converted `Trade` model is correctly associated with its asset.

        Returns:
            Trade: A standardized `Trade` model instance, populated with normalized data.

        Raises:
            ValueError: If the `raw_trade` data is malformed, missing critical fields,
                        or contains invalid values that prevent proper conversion.
            TypeError: If data types within the `raw_trade` dictionary are unexpected
                       and cannot be coerced into the `Trade` model's expected types.
        """
        pass

    @abstractmethod
    def convert_multiple_trades(self, raw_trades: list[dict[str, Any]], symbol: str) -> list[Trade]:
        """
        Converts a list of raw trade records from an exchange into a list of standardized `Trade` models.

        This method provides an efficient way to process multiple trade records, applying the
        same conversion logic as `convert_trade` to each item in the input list. It is
        optimized for batch processing of historical or streaming trade data.

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
    def convert_kline(self, raw_kline: dict[str, Any], symbol: str) -> Kline:
        """
        Converts a single raw K-line (candlestick) record from an exchange into a standardized `Kline` model.

        This method is responsible for parsing the raw dictionary, extracting open, high, low, close
        prices, volume, and timestamp for a specific time interval, and mapping it to the `Kline`
        model's consistent structure.

        Args:
            raw_kline (dict[str, Any]): A dictionary containing the raw K-line data as received from the exchange.
                                       The structure of this dictionary is exchange-specific.
            symbol (str): The trading symbol (e.g., "BTCUSDT", "ETHUSD") for which the K-line occurred.

        Returns:
            Kline: A standardized `Kline` model instance.

        Raises:
            ValueError: If the `raw_kline` data is malformed, missing critical fields,
                        or contains invalid values that prevent proper conversion.
            TypeError: If data types within the `raw_kline` are unexpected.
        """
        pass

    @abstractmethod
    def convert_multiple_klines(self, raw_klines: list[dict[str, Any]], symbol: str) -> list[Kline]:
        """
        Converts multiple raw K-line records from an exchange into a list of standardized `Kline` models.

        This method efficiently processes multiple K-line records, applying the same conversion logic
        as `convert_kline` to each item in the list. It is suitable for fetching historical
        candlestick data in batches.

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

    @abstractmethod
    def validate_raw_data(self, data: RawDataType) -> tuple[bool, str]:
        """
        Validates the structure and content of raw data received from the exchange.

        This method performs preliminary checks to ensure the raw data is in an
        expected format and contains all necessary fields before attempting a full conversion.
        It helps in early detection of malformed or incomplete data, preventing
        downstream processing errors.

        Args:
            data (RawDataType): The raw data to validate. The type of this data is flexible,
                                as indicated by `RawDataType`, allowing validation of various
                                input structures (e.g., a single trade dict, a list of klines).

        Returns:
            tuple[bool, str]: A tuple where the first element is `True` if the data is considered valid
                              according to the converter's rules, and `False` otherwise. The second
                              element is an informative error message (string) if validation fails,
                              or an empty string (`""`) if validation succeeds.
        """
        pass

    @abstractmethod
    def to_internal_timestamp(self, timestamp: Any) -> int:
        """
        Converts an exchange-specific timestamp format to a standardized internal Unix timestamp in milliseconds.

        Different exchanges may provide timestamps in various formats (e.g., Unix seconds as int/float,
        Unix milliseconds as int, ISO 8601 strings, or custom date strings). This method
        standardizes them to a single, consistent integer representation (milliseconds since epoch),
        which is crucial for consistent time-series analysis and storage within `asset_core`.

        Args:
            timestamp (Any): The timestamp value from the exchange. This can be an integer,
                             float, or string, depending on the exchange's API and the raw data.

        Returns:
            int: An integer representing the Unix timestamp in milliseconds.

        Raises:
            ValueError: If the provided timestamp format is unrecognized, cannot be parsed,
                        or is otherwise invalid.
            TypeError: If the `timestamp`'s type is not convertible (e.g., an unexpected object).
        """
        pass
