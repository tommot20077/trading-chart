from typing import Any

from pydantic import Field

from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.network.enum import ConnectionStatus


class ConnectionEvent(BaseEvent[dict[str, Any]]):
    """
    Represents an event indicating a change in connection status.

    This event carries information about the current `ConnectionStatus`.
    The `event_type` is automatically set to `EventType.CONNECTION`.
    """

    event_type: EventType = Field(default=EventType.CONNECTION, frozen=True)
    status: ConnectionStatus = Field(description="The connection status")

    def __init__(self, status: ConnectionStatus, **data: Any) -> None:
        """
        Initializes a `ConnectionEvent` instance.

        Args:
            status: The current `ConnectionStatus`.
            **data: Arbitrary keyword arguments to initialize the `BaseEvent`.
                    The `event_type` is automatically set to `EventType.CONNECTION`.
        """
        event_data = data.pop("data", {})
        # Preserve original data and add status
        if isinstance(event_data, dict):
            # Create a copy to avoid modifying the original
            event_data = event_data.copy()
            event_data["status"] = status.value
        else:
            # If data is not a dict, create a new dict with status
            event_data = {"status": status.value}

        # Always force event_type to CONNECTION regardless of input
        data["event_type"] = EventType.CONNECTION
        data["data"] = event_data
        data["status"] = status

        # Provide default source if not specified
        if "source" not in data:
            data["source"] = "connection_monitor"

        super().__init__(**data)
