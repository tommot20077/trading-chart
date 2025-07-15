from abc import abstractmethod, ABC
from typing import Callable, Awaitable

from core.models.data.base import BaseEvent
from core.models.event.event_type import EventType


EventHandler = Callable[[BaseEvent], None]  # Type alias for a synchronous event handler function.
AsyncEventHandler = Callable[[BaseEvent], Awaitable[None]]  # Type alias for an asynchronous event handler coroutine.


class AbstractEventBus(ABC):
    """
    [L0] Abstract interface for an event bus implementation.

    This abstract class defines the core contract for implementing a publish-subscribe
    event bus. It provides methods for publishing events, subscribing to specific
    event types (with optional filtering), and managing subscriptions. Concrete
    implementations should provide the underlying mechanism for event routing,
    dispatch, and delivery.

    Architecture note: This is a [L0] interface that only depends on core data models
    (`BaseEvent`, `EventType`) and provides clean, framework-agnostic abstractions
    for higher-level [L1] event implementations (e.g., in-memory bus, message queue-based bus).
    """

    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """
        Publishes an event to the event bus.

        The event bus is responsible for dispatching this event to all currently
        active and matching subscribers. This is an asynchronous operation.

        Args:
            event: The `BaseEvent` object to be published.
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | AsyncEventHandler,
        *,
        filter_symbol: str | None = None,
    ) -> str:
        """
        Subscribes a handler function to events of a specific type.

        When an event matching the `event_type` (and optional `filter_symbol`)
        is published, the provided `handler` will be invoked. This method returns
        a unique subscription ID that can be used later to unsubscribe.

        Args:
            event_type: The `EventType` enum member to subscribe to.
            handler: The callable function (synchronous or asynchronous) that will
                     process matching events.
            filter_symbol: Optional. A specific trading symbol. If provided, the handler
                           will only receive events of `event_type` that are also
                           associated with this symbol.

        Returns:
            A unique string identifier for the newly created subscription.

        Raises:
            ValueError: If the handler is invalid or subscription parameters are malformed.
            # Implementations might raise specific errors like `EventBusError`
            # if the subscription mechanism fails.
        """
        pass

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribes a handler from events using its unique subscription ID.

        Args:
            subscription_id: The unique string identifier of the subscription to remove.

        Returns:
            `True` if the subscription was found and successfully removed, `False` otherwise.
        """
        pass

    @abstractmethod
    def unsubscribe_all(self, event_type: EventType | None = None) -> int:
        """
        Unsubscribes all active handlers from the event bus.

        If `event_type` is specified, only handlers subscribed to that particular
        event type will be unsubscribed. Otherwise, all handlers for all event
        types will be removed.

        Args:
            event_type: Optional. The specific `EventType` to unsubscribe all handlers from.
                        If `None`, all subscriptions across all event types are removed.

        Returns:
            The number of handlers that were successfully unsubscribed.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Closes the event bus and cleans up any associated resources.

        This asynchronous method should release network connections, stop background
        tasks, clear internal queues, and perform any necessary shutdown procedures
        to gracefully terminate the event bus operation.
        """
        pass

    @abstractmethod
    def get_subscription_count(self, event_type: EventType | None = None) -> int:
        """
        Retrieves the number of active subscriptions currently managed by the event bus.

        Args:
            event_type: Optional. If provided, returns the count of subscriptions
                        only for this specific `EventType`. If `None`, returns the
                        total count of all active subscriptions across all types.

        Returns:
            The integer count of active subscriptions.
        """
        pass

    @abstractmethod
    async def wait_for(
        self,
        event_type: EventType,
        *,
        timeout: float | None = None,
        filter_func: Callable[[BaseEvent], bool] | None = None,
    ) -> BaseEvent | None:
        """
        Asynchronously waits for a specific event to be published on the bus.

        This method is useful for testing or for scenarios where a component
        needs to react to the *first* occurrence of a specific event that
        matches certain criteria. It can optionally include a timeout and
        a custom filtering function.

        Args:
            event_type: The `EventType` enum member of the event to wait for.
            timeout: Optional. The maximum number of seconds to wait for the event.
                     If the event does not occur within this duration, `None` is returned.
                     If `None`, it waits indefinitely.
            filter_func: Optional. A callable that takes a `BaseEvent` and returns a boolean.
                         If provided, the `wait_for` method will only consider events for which
                         `filter_func(event)` returns `True`.

        Returns:
            The `BaseEvent` object that was waited for and matched the criteria,
            or `None` if the timeout was reached without a matching event.
        """
        pass

    @property
    @abstractmethod
    def is_closed(self) -> bool:
        """
        Checks if the event bus is currently in a closed state.

        A closed event bus typically means it is no longer accepting new publications
        or dispatching events, and its resources have been cleaned up.

        Returns:
            `True` if the event bus is closed, `False` otherwise.
        """
        pass
