from datetime import timedelta
from enum import Enum


class KlineInterval(str, Enum):
    """Enumeration for standard Kline (candlestick) intervals.

    This enum defines common timeframes for financial market data,
    ranging from 1-minute to 1-month intervals. Each member has a string
    value that is typically used in API requests or data representations.
    """

    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    @classmethod
    def to_seconds(cls, interval: "KlineInterval") -> int:
        """
        Converts a `KlineInterval` enum member to its equivalent duration in seconds.

        Args:
            interval: The `KlineInterval` enum member to convert.

        Returns:
            The duration of the interval in seconds.
        """
        mapping = {
            cls.MINUTE_1: 60,
            cls.MINUTE_3: 180,
            cls.MINUTE_5: 300,
            cls.MINUTE_15: 900,
            cls.MINUTE_30: 1800,
            cls.HOUR_1: 3600,
            cls.HOUR_2: 7200,
            cls.HOUR_4: 14400,
            cls.HOUR_6: 21600,
            cls.HOUR_8: 28800,
            cls.HOUR_12: 43200,
            cls.DAY_1: 86400,
            cls.DAY_3: 259200,
            cls.WEEK_1: 604800,
            cls.MONTH_1: 2592000,  # 30 days approximation
        }
        return mapping[interval]

    @classmethod
    def to_timedelta(cls, interval: "KlineInterval") -> timedelta:
        """
        Converts a `KlineInterval` enum member to a `datetime.timedelta` object.

        Args:
            interval: The `KlineInterval` enum member to convert.

        Returns:
            A `timedelta` object representing the duration of the interval.
        """
        return timedelta(seconds=cls.to_seconds(interval))


class AssetClass(str, Enum):
    """Abstract enumeration for asset classification types.

    This enum provides a clean interface for categorizing different types of financial
    instruments by their fundamental characteristics. The classification is intended
    to be implemented by application layers with specific business logic.

    Attributes:
        DIGITAL (str): Digital assets such as cryptocurrencies and tokens.
        TRADITIONAL (str): Traditional financial instruments like stocks, bonds, forex.
        DERIVATIVE (str): Derivative instruments including futures, options, and swaps.
        SYNTHETIC (str): Synthetic assets and tokenized instruments.
    """

    DIGITAL = "digital"
    TRADITIONAL = "traditional"
    DERIVATIVE = "derivative"
    SYNTHETIC = "synthetic"

    def __str__(self) -> str:
        """
        Returns the string representation of the asset class.

        Returns:
            The string value of the enum member.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Returns the detailed representation of the asset class.

        Returns:
            A string showing the class and member name.
        """
        return f"AssetClass.{self.name}"


class TradeSide(str, Enum):
    """
    Enumeration for the side of a trade.

    Attributes:
        BUY (str): Indicates a buy trade.
        SELL (str): Indicates a sell trade.
    """

    BUY = "buy"
    SELL = "sell"
