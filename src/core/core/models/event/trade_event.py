from typing import Any

from pydantic import Field

from core.models.data.event import BaseEvent
from core.models.data.trade import Trade
from core.models.event.event_type import EventType


class TradeEvent(BaseEvent[Trade]):
    """
    Represents an event containing trade data.

    This event type is specifically designed to carry `Trade` objects as its payload.
    The `event_type` is automatically set to `EventType.TRADE`.
    """

    event_type: EventType = Field(default=EventType.TRADE, frozen=True)

    def __init__(self, **data: Any) -> None:
        """
        Initializes a `TradeEvent` instance.

        Args:
            **data: Arbitrary keyword arguments to initialize the `BaseEvent`.
                    The `event_type` is automatically set to `EventType.TRADE`.
        """
        # Always force event_type to TRADE regardless of input
        data["event_type"] = EventType.TRADE

        # Handle the 'trade' parameter and set it as 'data' for BaseEvent
        if "trade" in data:
            data["data"] = data.pop("trade")

        super().__init__(**data)
