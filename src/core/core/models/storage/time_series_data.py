from datetime import datetime
from typing import Protocol


class TimeSeriesData(Protocol):
    """
    Protocol defining the minimum interface for time-series data.

    Any data type that can be stored in a time-series repository must implement
    this protocol, providing at minimum a symbol and a primary timestamp field.
    """

    symbol: str
    timestamp: datetime  # For Trade data

    # For Kline data, we'll check for open_time in the generic methods
    @property
    def primary_timestamp(self) -> datetime:
        """Return the primary timestamp for this data point."""
        # Default implementation for Trade-like data
        return self.timestamp
