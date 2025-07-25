# ABOUTME: NoOp implementation of AbstractEventBus that discards all events
# ABOUTME: Provides minimal event bus functionality for testing scenarios

from typing import Callable, Optional
import uuid

from core.interfaces.event.event_bus import AbstractEventBus, EventHandler, AsyncEventHandler
from core.interfaces.middleware.pipeline import AbstractMiddlewarePipeline
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


class NoOpEventBus(AbstractEventBus):
    """
    No-operation implementation of AbstractEventBus.

    This implementation provides minimal event bus functionality that
    discards all events and maintains fake subscriptions without performing
    any actual event routing or delivery. It's useful for testing,
    performance benchmarking, and scenarios where event processing is not required.

    Features:
    - Discards all published events (no actual delivery)
    - Maintains fake subscription tracking
    - No actual event handler invocation
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where events should be silenced
    - Performance benchmarking without event processing overhead
    - Development environments where event processing is not needed
    - Fallback when event systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation event bus."""
        self._closed = False
        self._subscription_count = 0
        self._subscriptions: dict[str, tuple[EventType, str | None]] = {}
        self._middleware_pipeline: Optional[AbstractMiddlewarePipeline] = None

    async def publish(self, event: BaseEvent) -> None:
        """
        Publish an event - discards the event without delivery.

        This implementation discards the event without performing any
        actual event routing or handler invocation. If a middleware pipeline
        is configured, it will execute the pipeline but still not deliver the event.

        Args:
            event: The event to publish (discarded)
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        # Execute middleware pipeline if configured (for testing/validation purposes)
        if self._middleware_pipeline:
            from core.models.middleware import MiddlewareContext

            # Create minimal middleware context
            context = MiddlewareContext(
                id=f"noop-{uuid.uuid4()}",
                event_type=event.event_type.value if event.event_type else None,
                event_id=event.event_id,
                symbol=getattr(event, "symbol", None),
                data=event.data or {},
                metadata={
                    "processing_stage": "noop_event_bus",
                    "handler_count": 0,  # NoOp never has real handlers
                    **(event.metadata or {}),
                },
            )

            # Execute middleware pipeline but ignore result
            # This allows testing middleware behavior without actual event processing
            try:
                await self._middleware_pipeline.execute(context)
            except Exception:
                # Silently ignore middleware errors in NoOp implementation
                # This maintains the "no side effects" principle
                pass

        # Discard the event - no actual processing in NoOp implementation
        pass

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | AsyncEventHandler,
        *,
        filter_symbol: str | None = None,
    ) -> str:
        """
        Subscribe to events - returns fake subscription ID.

        This implementation creates a fake subscription without actually
        registering the handler for event delivery.

        Args:
            event_type: The event type to subscribe to
            handler: The event handler (ignored)
            filter_symbol: Optional symbol filter

        Returns:
            A fake subscription ID
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        subscription_id = f"noop-sub-{uuid.uuid4()}"
        self._subscriptions[subscription_id] = (event_type, filter_symbol)
        self._subscription_count += 1
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events - removes fake subscription.

        This implementation removes the fake subscription if it exists.

        Args:
            subscription_id: The subscription ID to remove

        Returns:
            True if subscription was found and removed, False otherwise
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            self._subscription_count -= 1
            return True
        return False

    def unsubscribe_all(self, event_type: EventType | None = None) -> int:
        """
        Unsubscribe all handlers - removes fake subscriptions.

        This implementation removes fake subscriptions matching the criteria.

        Args:
            event_type: Optional event type filter

        Returns:
            Number of subscriptions removed
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        if event_type is None:
            # Remove all subscriptions
            count = len(self._subscriptions)
            self._subscriptions.clear()
            self._subscription_count = 0
            return count
        else:
            # Remove subscriptions for specific event type
            to_remove = [
                sub_id for sub_id, (sub_event_type, _) in self._subscriptions.items() if sub_event_type == event_type
            ]
            for sub_id in to_remove:
                del self._subscriptions[sub_id]
                self._subscription_count -= 1
            return len(to_remove)

    async def close(self) -> None:
        """
        Close the event bus - sets closed flag and clears subscriptions.
        """
        self._closed = True
        self._subscriptions.clear()
        self._subscription_count = 0

    def get_subscription_count(self, event_type: EventType | None = None) -> int:
        """
        Get subscription count - returns fake subscription count.

        This implementation returns the count of fake subscriptions.

        Args:
            event_type: Optional event type filter

        Returns:
            Count of fake subscriptions
        """
        if self._closed:
            return 0

        if event_type is None:
            return int(self._subscription_count)
        else:
            return int(sum(1 for sub_event_type, _ in self._subscriptions.values() if sub_event_type == event_type))

    async def wait_for(
        self,
        event_type: EventType,
        *,
        timeout: float | None = None,
        filter_func: Callable[[BaseEvent], bool] | None = None,
    ) -> BaseEvent:
        """
        Wait for an event - always times out (no events delivered).

        This implementation always raises asyncio.TimeoutError since no events are
        actually delivered in the NoOp implementation.

        Args:
            event_type: The event type to wait for (ignored)
            timeout: Optional timeout (ignored)
            filter_func: Optional filter function (ignored)

        Returns:
            Never returns - always raises TimeoutError

        Raises:
            asyncio.TimeoutError: Always raised since no events are delivered
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        # Always raise timeout - no events are delivered in NoOp implementation
        import asyncio

        raise asyncio.TimeoutError("NoOp event bus never delivers events")

    @property
    def is_closed(self) -> bool:
        """
        Check if the event bus is closed.

        Returns:
            True if the event bus is closed, False otherwise
        """
        return bool(self._closed)

    async def set_middleware_pipeline(self, pipeline: Optional[AbstractMiddlewarePipeline]) -> None:
        """
        Set the middleware pipeline - stores reference without using it.

        This implementation stores the pipeline reference but does not
        actually use it since no events are processed in the NoOp implementation.

        Args:
            pipeline: The middleware pipeline to set (ignored in processing)
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        self._middleware_pipeline = pipeline

    async def get_middleware_pipeline(self) -> Optional[AbstractMiddlewarePipeline]:
        """
        Get the currently configured middleware pipeline.

        This implementation returns the stored pipeline reference,
        though it is not actually used for event processing.

        Returns:
            The current middleware pipeline, or None if no pipeline is set
        """
        if self._closed:
            raise RuntimeError("Event bus is closed")

        return self._middleware_pipeline
