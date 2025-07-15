from typing import Any

from pydantic import Field

from core.models.data.base import BaseEvent
from core.models.event.event_type import EventType
from core.models.network.enum import ConnectionStatus


class ConnectionEvent(BaseEvent[dict[str, Any]]):
    """
    Represents an event indicating a change in connection status.

    This event carries information about the current `ConnectionStatus`.
    The `event_type` is automatically set to `EventType.CONNECTION`.
    """

    event_type: EventType = Field(default=EventType.CONNECTION, frozen=True)

    def __init__(self, status: ConnectionStatus, **data: Any) -> None:
        """
        Initializes a `ConnectionEvent` instance.

        Args:
            status: The current `ConnectionStatus`.
            **data: Arbitrary keyword arguments to initialize the `BaseEvent`.
                    The `event_type` is automatically set to `EventType.CONNECTION`.
        """
        event_data = data.pop("data", {})
        event_data["status"] = status.value
        # Always force event_type to CONNECTION regardless of input
        data["event_type"] = EventType.CONNECTION
        data["data"] = event_data
        super().__init__(**data)
