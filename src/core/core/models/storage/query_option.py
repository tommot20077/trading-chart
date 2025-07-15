# ABOUTME: This module defines query options for time-series data repositories
# ABOUTME: It provides configuration for pagination, ordering, and metadata inclusion


class QueryOptions:
    """
    [L0] Options for querying time-series data.

    Provides configuration options for time-series data queries including pagination,
    ordering, and metadata inclusion settings. Works with any time-series data type
    such as Kline, Trade, or other time-based data.
    """

    def __init__(
        self,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str = "timestamp",
        order_desc: bool = False,
        include_metadata: bool = False,
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if offset is not None and offset < 0:
            raise ValueError("offset must be non-negative")
        self.limit = limit
        self.offset = offset
        self.order_by = order_by
        self.order_desc = order_desc
        self.include_metadata = include_metadata
