# ABOUTME: EventMiddlewareBus implementation that integrates middleware with event processing
# ABOUTME: Decorates existing event bus to add middleware pipeline execution before event handling

import uuid
from typing import Optional, Callable, Any
from loguru import logger

from core.interfaces.event.event_bus import AbstractEventBus, EventHandler, AsyncEventHandler
from core.interfaces.middleware import AbstractMiddlewarePipeline
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType
from core.models.middleware import MiddlewareContext


class EventMiddlewareBus(AbstractEventBus):
    """
    Event bus decorator that integrates middleware pipeline execution.

    This implementation wraps an existing event bus and adds middleware
    processing before event handlers are invoked. It follows the decorator
    pattern to provide middleware functionality without changing existing
    event bus implementations.
    """

    def __init__(self, base_bus: AbstractEventBus, name: str = "EventMiddlewareBus"):
        """
        Initialize the EventMiddlewareBus.

        Args:
            base_bus: The underlying event bus to wrap with middleware.
            name: Name for identification and logging purposes.
        """
        self.base_bus = base_bus
        self.name = name
        self.middleware_pipeline: Optional[AbstractMiddlewarePipeline] = None

    async def set_middleware_pipeline(self, pipeline: Optional[AbstractMiddlewarePipeline]) -> None:
        """
        Set the middleware pipeline for event processing.

        Args:
            pipeline: The middleware pipeline to set, or None to remove it.
        """
        self.middleware_pipeline = pipeline
        logger.info(
            f"Middleware pipeline {'set' if pipeline else 'removed'} for EventMiddlewareBus",
            middleware_bus=self.name,
            pipeline_type=type(pipeline).__name__ if pipeline else None,
        )

    async def get_middleware_pipeline(self) -> Optional[AbstractMiddlewarePipeline]:
        """
        Get the currently configured middleware pipeline.

        Returns:
            The current middleware pipeline, or None if not set.
        """
        return self.middleware_pipeline

    async def publish(self, event: BaseEvent) -> None:
        """
        Publish an event through the middleware pipeline and then to handlers.

        This method first executes the middleware pipeline (if configured)
        before publishing the event to the underlying event bus. If any
        middleware in the pipeline indicates that processing should not
        continue, the event will not be published to handlers.

        Args:
            event: The event to publish.
        """
        # Execute middleware pipeline if configured
        if self.middleware_pipeline:
            try:
                # Create middleware context from event
                context = self._create_middleware_context(event)

                # Execute middleware pipeline
                pipeline_result = await self.middleware_pipeline.execute(context)

                # Check if pipeline allows continuation
                if not pipeline_result.should_continue:
                    logger.info(
                        "Event processing stopped by middleware pipeline",
                        event_id=event.event_id,
                        middleware_bus=self.name,
                        result_status=pipeline_result.status.value
                        if hasattr(pipeline_result.status, "value")
                        else str(pipeline_result.status),
                        reason=pipeline_result.data.get("reason", "No reason provided")
                        if pipeline_result.data
                        else "No reason provided",
                    )
                    return

                # Apply any context modifications back to the event
                if pipeline_result.modified_context:
                    self._apply_context_modifications(event, pipeline_result.modified_context)

                logger.debug(
                    "Event processed through middleware pipeline",
                    event_id=event.event_id,
                    middleware_bus=self.name,
                    result_status=pipeline_result.status.value
                    if hasattr(pipeline_result.status, "value")
                    else str(pipeline_result.status),
                )

            except Exception as e:
                logger.error(
                    "Error executing middleware pipeline",
                    event_id=event.event_id,
                    middleware_bus=self.name,
                    error=str(e),
                    exc_info=e,
                )
                # Re-raise the exception to prevent event processing
                raise

        # Publish the event to the underlying bus
        await self.base_bus.publish(event)

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | AsyncEventHandler,
        *,
        filter_symbol: str | None = None,
    ) -> str:
        """
        Subscribe a handler to events of a specific type.

        This method delegates to the underlying event bus.

        Args:
            event_type: The event type to subscribe to.
            handler: The handler function to call when events occur.
            filter_symbol: Optional symbol filter.

        Returns:
            Unique subscription identifier.
        """
        return self.base_bus.subscribe(event_type, handler, filter_symbol=filter_symbol)

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe a handler using its subscription ID.

        Args:
            subscription_id: The subscription ID to remove.

        Returns:
            True if successfully unsubscribed, False otherwise.
        """
        return self.base_bus.unsubscribe(subscription_id)

    def unsubscribe_all(self, event_type: EventType | None = None) -> int:
        """
        Unsubscribe all handlers for a specific event type or all types.

        Args:
            event_type: Optional event type to filter by.

        Returns:
            Number of handlers unsubscribed.
        """
        return self.base_bus.unsubscribe_all(event_type)

    async def close(self) -> None:
        """
        Close the event bus and clean up resources.

        This method closes the underlying event bus and clears the middleware pipeline.
        """
        # Clear middleware pipeline
        self.middleware_pipeline = None

        # Close the underlying bus
        await self.base_bus.close()

    def get_subscription_count(self, event_type: EventType | None = None) -> int:
        """
        Get the number of active subscriptions.

        Args:
            event_type: Optional event type to filter by.

        Returns:
            Number of active subscriptions.
        """
        return self.base_bus.get_subscription_count(event_type)

    async def wait_for(
        self,
        event_type: EventType,
        *,
        timeout: float | None = None,
        filter_func: Callable[[BaseEvent], bool] | None = None,
    ) -> BaseEvent:
        """
        Wait for a specific event to be published.

        Args:
            event_type: The event type to wait for.
            timeout: Optional timeout in seconds.
            filter_func: Optional filter function.

        Returns:
            The event that was waited for.

        Raises:
            asyncio.TimeoutError: If the timeout was reached without a matching event.
        """
        return await self.base_bus.wait_for(event_type, timeout=timeout, filter_func=filter_func)

    @property
    def is_closed(self) -> bool:
        """
        Check if the event bus is closed.

        Returns:
            True if the event bus is closed, False otherwise.
        """
        return self.base_bus.is_closed

    def _create_middleware_context(self, event: BaseEvent) -> MiddlewareContext:
        """
        Create a middleware context from an event.

        Args:
            event: The event to create context from.

        Returns:
            MiddlewareContext for middleware processing.
        """
        return MiddlewareContext(
            id=str(uuid.uuid4()),
            event_type=event.event_type.value if event.event_type else None,
            event_id=event.event_id,
            symbol=getattr(event, "symbol", None),
            data={
                # Pass the actual event data
                **(event.data or {}),
                # Also include event metadata
                "event_metadata": {
                    "event_type": event.event_type.value if event.event_type else None,
                    "priority": event.priority.value if hasattr(event, "priority") else None,
                },
            },
            metadata={
                "processing_stage": "event_middleware_processing",
                "middleware_bus_name": self.name,
                "timestamp": event.timestamp.isoformat() if hasattr(event, "timestamp") and event.timestamp else None,
                "source": getattr(event, "source", None),
                # Include event metadata
                **(event.metadata or {}),
            },
        )

    def _apply_context_modifications(self, event: BaseEvent, modifications: dict[str, Any]) -> None:
        """
        Apply middleware context modifications back to the event.

        This method allows middleware to modify event attributes.

        Args:
            event: The event to modify.
            modifications: Dictionary containing modifications to apply to the event.
        """
        # Apply data modifications if available
        data_modifications = modifications.get("data")
        if data_modifications and isinstance(data_modifications, dict):
            # Update event data if it exists
            if hasattr(event, "data") and event.data is not None:
                # Only update non-metadata fields
                for key, value in data_modifications.items():
                    if key != "event_metadata":
                        event.data[key] = value

        # Apply metadata modifications
        metadata_modifications = modifications.get("metadata")
        if metadata_modifications and isinstance(metadata_modifications, dict):
            if hasattr(event, "metadata") and event.metadata is not None:
                # Update event metadata, excluding processing stage info
                for key, value in metadata_modifications.items():
                    if key not in ["processing_stage", "middleware_bus_name"]:
                        event.metadata[key] = value

        logger.debug(
            "Applied middleware context modifications to event", event_id=event.event_id, middleware_bus=self.name
        )

    def get_bus_info(self) -> dict:
        """
        Get information about the event bus state.

        Returns:
            Dictionary containing bus information.
        """
        info = {
            "name": self.name,
            "base_bus_type": type(self.base_bus).__name__,
            "is_closed": self.is_closed,
            "has_middleware_pipeline": self.middleware_pipeline is not None,
            "total_subscriptions": self.get_subscription_count(),
        }

        # Add middleware pipeline info if available
        if self.middleware_pipeline and hasattr(self.middleware_pipeline, "get_pipeline_info"):
            info["middleware_pipeline"] = self.middleware_pipeline.get_pipeline_info()

        return info
