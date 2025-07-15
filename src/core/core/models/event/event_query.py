from datetime import datetime
from typing import Any

from core.models.event.event_type import EventType


class EventQuery:
    """
    [L0] Query criteria for event retrieval.

    This class encapsulates all parameters required to filter, order, and paginate
    events when querying an event storage backend. It provides a standardized way
    to define complex event retrieval requests.
    """

    def __init__(
        self,
        event_types: list[EventType] | None = None,
        symbols: list[str] | None = None,
        sources: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        correlation_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_by: str = "timestamp",
        order_desc: bool = True,
        metadata_filters: dict[str, Any] | None = None,
    ):
        """
        Initializes an EventQuery object with specified filtering and ordering criteria.

        Args:
            event_types: An optional list of `EventType` enum members. If provided,
                         only events matching one of these types will be returned.
                         Defaults to an empty list, meaning no filtering by type.
            symbols: An optional list of string symbols (e.g., "BTC/USDT"). If provided,
                     only events associated with these symbols will be returned.
                     Defaults to an empty list.
            sources: An optional list of string sources (e.g., "Binance", "Kraken"). If provided,
                     only events originating from these sources will be returned.
                     Defaults to an empty list.
            start_time: An optional `datetime` object. If provided, only events with a
                        timestamp greater than or equal to this time will be returned.
            end_time: An optional `datetime` object. If provided, only events with a
                      timestamp less than or equal to this time will be returned.
            correlation_id: An optional string. If provided, only events matching this
                            specific correlation ID will be returned. Useful for tracing
                            related events across different operations.
            limit: An optional integer. If provided, specifies the maximum number of
                   events to return in the query result. `None` means no limit.
            offset: An integer. Specifies the number of events to skip from the beginning
                    of the result set. Useful for pagination. Defaults to 0.
            order_by: A string specifying the field by which to order the results.
                      Common values include "timestamp". Defaults to "timestamp".
            order_desc: A boolean. If `True`, results are ordered in descending order
                        (e.g., newest first for timestamp). If `False`, ascending order.
                        Defaults to `True`.
            metadata_filters: An optional dictionary of key-value pairs. Only events
                              whose `metadata` dictionary contains all specified key-value
                              pairs will be returned. Defaults to an empty dictionary.
        """
        self.event_types = event_types or []
        self.symbols = symbols or []
        self.sources = sources or []
        self.start_time = start_time
        self.end_time = end_time
        self.correlation_id = correlation_id
        self.limit = limit
        self.offset = offset
        self.order_by = order_by
        self.order_desc = order_desc
        self.metadata_filters = metadata_filters or {}

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the EventQuery to a dictionary representation.

        Returns:
            A dictionary containing all query parameters with appropriate serialization
            for datetime objects and enum values.
        """
        result = {
            "event_types": [et.value for et in self.event_types],
            "symbols": self.symbols,
            "sources": self.sources,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "correlation_id": self.correlation_id,
            "limit": self.limit,
            "offset": self.offset,
            "order_by": self.order_by,
            "order_desc": self.order_desc,
            "metadata_filters": self.metadata_filters,
        }
        return result
