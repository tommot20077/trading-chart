from enum import Enum


class EventType(str, Enum):
    """
    Enumeration of predefined event types.

    Each member represents a distinct category of event that can occur within
    the system, facilitating event routing and handling.

    Attributes:
        TRADE (str): Represents a trade event.
        KLINE (str): Represents a kline (candlestick) event.
        ORDER (str): Represents an order-related event.
        CONNECTION (str): Represents a connection status change event.
        ERROR (str): Represents an error event.
        SYSTEM (str): Represents a general system event.
        ALERT (str): Represents an alert event.
        MARKET_DATA (str): Represents a market data event.
    """

    TRADE = "trade"
    KLINE = "kline"
    ORDER = "order"
    CONNECTION = "connection"
    ERROR = "error"
    SYSTEM = "system"
    ALERT = "alert"
    MARKET_DATA = "market_data"

    def __str__(self) -> str:
        return self.value
