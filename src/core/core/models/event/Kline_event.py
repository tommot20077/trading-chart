from typing import Any

from pydantic import Field

from core.models.data.event import BaseEvent
from core.models.data.kline import Kline
from core.models.event.event_type import EventType


class KlineEvent(BaseEvent[Kline]):
    """
    Represents an event containing Kline (candlestick) data.

    This event type is specifically designed to carry `Kline` objects as its payload.
    The `event_type` is automatically set to `EventType.KLINE`.
    """

    event_type: EventType = Field(default=EventType.KLINE, frozen=True)

    def __init__(self, **data: Any) -> None:
        """
        Initializes a `KlineEvent` instance.

        Args:
            **data: Arbitrary keyword arguments to initialize the `BaseEvent`.
                    The `event_type` is automatically set to `EventType.KLINE`.
        """
        # Always force event_type to KLINE regardless of input
        data["event_type"] = EventType.KLINE

        # Handle the 'kline' parameter and set it as 'data' for BaseEvent
        if "kline" in data:
            data["data"] = data.pop("kline")

        super().__init__(**data)
