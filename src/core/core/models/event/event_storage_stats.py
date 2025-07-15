from datetime import datetime


class EventStorageStats:
    """
    [L0] Storage statistics for monitoring and diagnostics.

    This class encapsulates various metrics about an event storage backend,
    providing insights into its current state, capacity, and event distribution.
    These statistics are valuable for monitoring the health and performance
    of the event persistence layer.
    """

    def __init__(
        self,
        total_events: int = 0,
        events_by_type: dict[str, int] | None = None,
        storage_size_bytes: int = 0,
        oldest_event_time: datetime | None = None,
        newest_event_time: datetime | None = None,
        avg_event_size_bytes: float = 0.0,
    ):
        """
        Initializes an EventStorageStats object with current storage metrics.

        Args:
            total_events: The total number of events currently stored in the backend.
                          Defaults to 0.
            events_by_type: An optional dictionary mapping event type names (strings)
                            to the count of events of that type. Defaults to an empty dictionary.
            storage_size_bytes: The total size of all stored events in bytes.
                                Defaults to 0.
            oldest_event_time: An optional `datetime` object representing the timestamp
                               of the oldest event currently in storage. `None` if no events.
            newest_event_time: An optional `datetime` object representing the timestamp
                               of the newest event currently in storage. `None` if no events.
            avg_event_size_bytes: The average size of a single event in bytes.
                                  Defaults to 0.0.
        """
        self.total_events = total_events
        self.events_by_type = events_by_type or {}
        self.storage_size_bytes = storage_size_bytes
        self.oldest_event_time = oldest_event_time
        self.newest_event_time = newest_event_time
        self.avg_event_size_bytes = avg_event_size_bytes
