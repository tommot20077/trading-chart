import asyncio
from typing import Callable, Awaitable

from core.models.data.base import BaseEvent
from core.models.event.event_type import EventType

EventHandler = Callable[[BaseEvent], None]  # Type alias for a synchronous event handler function.
AsyncEventHandler = Callable[[BaseEvent], Awaitable[None]]  # Type alias for an asynchronous event handler coroutine.


class Subscription:
    """
    Represents a single subscription to an event type within an event bus.

    This class encapsulates all the necessary details for managing a specific
    event listener, including its unique identifier, the event type it's
    interested in, the handler function to call, and optional filtering criteria.
    """

    def __init__(
        self,
        subscription_id: str,
        event_type: EventType,
        handler: EventHandler | AsyncEventHandler,
        filter_symbol: str | None = None,
    ):
        """
        Initializes a new Subscription instance.

        Args:
            subscription_id: A unique string identifier for this specific subscription.
                             This ID is used to unsubscribe later.
            event_type: The `EventType` enum member that this subscription is interested in.
                        Only events of this type will be dispatched to the handler.
            handler: The callable function (either synchronous `EventHandler` or
                     asynchronous `AsyncEventHandler`) that will be invoked when
                     a matching event is published.
            filter_symbol: An optional string representing a trading symbol. If provided,
                           the handler will only receive events of the specified `event_type`
                           that are also associated with this specific symbol. Defaults to `None`,
                           meaning no symbol filtering.
        """
        self.id = subscription_id
        self.event_type = event_type
        self.handler = handler
        self.filter_symbol = filter_symbol
        # Determine if the handler is an asynchronous coroutine function.
        self.is_async = asyncio.iscoroutinefunction(handler)

    def matches(self, event: BaseEvent) -> bool:
        """
        Checks if this subscription's criteria match the given event.

        An event matches a subscription if its `event_type` is the same as the
        subscription's `event_type`, and if a `filter_symbol` is specified,
        the event's `symbol` also matches the `filter_symbol`.

        Args:
            event: The `BaseEvent` object to check against this subscription's criteria.

        Returns:
            `True` if the event matches all criteria defined by this subscription,
            `False` otherwise.
        """
        return event.event_type == self.event_type and (
            self.filter_symbol is None or event.symbol == self.filter_symbol
        )
