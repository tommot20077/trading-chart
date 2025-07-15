from typing import Any

from pydantic import Field

from core.models.data.base import BaseEvent
from core.models.event import EventPriority
from core.models.event.event_type import EventType


class ErrorEvent(BaseEvent[dict[str, Any]]):
    """
    Represents an event indicating an error has occurred.

    This event carries details about the error, including a human-readable
    message and an optional error code. The `event_type` is automatically
    set to `EventType.ERROR`, and its `priority` defaults to `EventPriority.HIGH`.
    """

    event_type: EventType = Field(default=EventType.ERROR, frozen=True)
    priority: EventPriority = Field(default=EventPriority.HIGH)

    def __init__(self, error: str, error_code: str | None = None, **data: Any) -> None:
        """
        Initializes an `ErrorEvent` instance.

        Args:
            error: A human-readable error message.
            error_code: Optional. A standardized error code string.
            **data: Arbitrary keyword arguments to initialize the `BaseEvent`.
                    The `event_type` is automatically set to `EventType.ERROR`
                    and `priority` to `EventPriority.HIGH`.
        """
        event_data = data.pop("data", {})
        event_data["error"] = error
        if error_code:
            event_data["error_code"] = error_code
        # Always force event_type to ERROR regardless of input
        data["event_type"] = EventType.ERROR
        if "priority" not in data:
            data["priority"] = EventPriority.HIGH
        data["data"] = event_data
        super().__init__(**data)
