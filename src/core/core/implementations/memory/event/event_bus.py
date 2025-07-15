# ABOUTME: In-memory implementation of AbstractEventBus for testing and development
# ABOUTME: Provides zero-dependency event bus using Python standard library only

import asyncio
import uuid
from collections import defaultdict
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import inspect
from loguru import logger
import threading

from core.interfaces.event.event_bus import AbstractEventBus, EventHandler, AsyncEventHandler
from core.interfaces.middleware import AbstractMiddlewarePipeline
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


@dataclass
class Subscription:
    """Internal subscription data structure."""

    id: str
    event_type: EventType
    handler: EventHandler | AsyncEventHandler
    filter_symbol: str | None = None
    is_async: bool = False


class InMemoryEventBus(AbstractEventBus):
    """
    In-memory implementation of AbstractEventBus.

    This implementation provides a complete event bus solution using only Python
    standard library. It supports both synchronous and asynchronous event handlers,
    symbol filtering, and provides all required operations for pub-sub messaging.

    Features:
    - Synchronous and asynchronous event handlers
    - Symbol-based event filtering
    - Subscription management with unique IDs
    - Event waiting with timeout and custom filters
    - Thread-safe operations using asyncio locks
    - Priority-based event handling (respects event priority)
    - Graceful shutdown with resource cleanup
    - Context manager support for automatic cleanup
    - Robust error handling and event loop lifecycle management

    Thread Safety: This implementation is thread-safe when used within the same
    event loop. For multi-threaded usage, external synchronization may be required.
    """

    def __init__(self, max_queue_size: int = 10000, handler_timeout: float = 30.0, max_concurrent_handlers: int = 100):
        """Initialize the in-memory event bus.

        Args:
            max_queue_size: Maximum number of events in the priority queue
            handler_timeout: Maximum time (seconds) to wait for a handler to complete
            max_concurrent_handlers: Maximum number of handlers to run concurrently
        """
        # Store subscriptions grouped by event type for efficient lookup
        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)

        # Track all subscriptions by ID for quick unsubscribe operations
        self._subscription_by_id: Dict[str, Subscription] = {}

        # Event waiting support
        self._waiting_futures: Dict[str, Any] = {}

        # State management
        self._closed = False
        self._paused = False  # Track if processing is paused
        self._subscription_lock = threading.Lock()  # Only for subscription management

        # Priority queue for events (priority, sequence_number, event)
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._sequence_counter = 0
        self._sequence_lock = asyncio.Lock()

        # Background event processing
        self._processing_task: asyncio.Task | None = None
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_handlers)
        self._active_tasks: set[asyncio.Task] = set()  # Track active processing tasks

        # Configuration
        self._handler_timeout = handler_timeout
        self._max_queue_size = max_queue_size
        self._max_concurrent_handlers = max_concurrent_handlers

        # Statistics
        self._published_count = 0
        self._processed_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._dropped_count = 0
        self._stats_lock = asyncio.Lock()

        # Middleware pipeline
        self._middleware_pipeline: AbstractMiddlewarePipeline | None = None

        # Don't start background processing immediately
        # It will be started lazily when first needed

    def __del__(self):
        """Cleanup when object is destroyed."""
        if not self._closed and self._processing_task is not None:
            try:
                import warnings

                warnings.warn(
                    "InMemoryEventBus was not properly closed. Please call await bus.close() to avoid resource leaks.",
                    ResourceWarning,
                    stacklevel=2,
                )

                # Try to cancel processing task if it exists
                if self._processing_task and not self._processing_task.done():
                    self._processing_task.cancel()

            except (ImportError, RuntimeError, GeneratorExit):
                # Silently ignore cleanup errors during destruction
                pass

    def _start_processing_task(self) -> None:
        """Start the background event processing task."""
        if self._closed or self._paused:
            return

        if self._processing_task is None or self._processing_task.done():
            try:
                # Check if we have a running event loop
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    logger.warning("Event loop is closed, cannot start processing task")
                    return

                self._processing_task = asyncio.create_task(self._process_events_loop())
                logger.debug("Background event processing task started")
            except RuntimeError as e:
                # No running event loop, processing will be started later when needed
                logger.debug(f"Cannot start processing task: {e}")
                self._processing_task = None

    def _ensure_not_closed(self) -> None:
        """Ensure event bus is not closed."""
        if self._closed:
            raise RuntimeError("Event bus is closed")

    def _generate_subscription_id(self) -> str:
        """Generate a unique subscription ID."""
        return str(uuid.uuid4())

    def _is_async_handler(self, handler: EventHandler | AsyncEventHandler) -> bool:
        """Check if handler is async."""
        return inspect.iscoroutinefunction(handler)

    def _matches_filter(self, event: BaseEvent, subscription: Subscription) -> bool:
        """Check if event matches subscription filter."""
        # Check event type
        if event.event_type != subscription.event_type:
            return False

        # Check symbol filter if specified
        if subscription.filter_symbol is not None:
            if event.symbol != subscription.filter_symbol:
                return False

        return True

    async def _invoke_handler(self, handler: EventHandler | AsyncEventHandler, event: BaseEvent) -> None:
        """Safely invoke a handler with error handling."""
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Run synchronous handler in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, handler, event)
        except Exception as e:
            # Error counting is now handled by the caller
            logger.error("Error in event handler", handler=str(handler), error=str(e), exc_info=e)
            raise  # Re-raise for caller to handle statistics

    async def publish(self, event: BaseEvent) -> None:
        """
        Publishes an event to the event bus.

        This method is non-blocking and places the event in a priority queue
        for asynchronous processing by background handlers.

        Args:
            event: The BaseEvent object to be published.
        """
        self._ensure_not_closed()

        if not isinstance(event, BaseEvent):
            raise TypeError("Event must be an instance of BaseEvent")

        # Ensure processing task is running
        if self._processing_task is None or self._processing_task.done():
            self._start_processing_task()

        # Get next sequence number for stable sorting
        async with self._sequence_lock:
            sequence_number = self._sequence_counter
            self._sequence_counter += 1

        # Create priority queue item (priority, sequence, event)
        # Lower priority value = higher priority
        queue_item = (event.priority.value, sequence_number, event)

        try:
            # Try to put event in queue (non-blocking)
            self._event_queue.put_nowait(queue_item)

            # Update statistics
            async with self._stats_lock:
                self._published_count += 1

        except asyncio.QueueFull:
            # Queue is full, drop the event
            async with self._stats_lock:
                self._dropped_count += 1

            logger.warning(
                "Event queue full, dropping event",
                event_type=event.event_type.value,
                event_id=event.event_id,
                queue_size=self._event_queue.qsize(),
            )
            raise RuntimeError(f"Event queue full (size: {self._max_queue_size}). Event dropped.")

    async def _process_events_loop(self) -> None:
        """Background task that processes events from the priority queue."""
        logger.info("Event processing loop started")

        try:
            while not self._closed:
                queue_item = None
                try:
                    # Check if event loop is still running
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning("No running event loop, stopping event processing")
                        break

                    # Get next event from priority queue with timeout to allow checking closed state
                    try:
                        queue_item = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Timeout is normal, just continue to check closed state
                        continue

                    # Safely unpack the queue item
                    try:
                        priority, sequence, event = queue_item
                    except (ValueError, TypeError) as e:
                        logger.error("Invalid queue item format", queue_item=queue_item, error=str(e))
                        self._event_queue.task_done()
                        continue

                    # Process event in background with proper exception handling
                    task = asyncio.create_task(self._process_single_event(event))

                    # Track active tasks
                    self._active_tasks.add(task)

                    # Add done callback to handle any unhandled exceptions and cleanup
                    task.add_done_callback(self._handle_processing_task_result)

                    # Mark task as done - this should be done immediately after processing starts
                    self._event_queue.task_done()

                except asyncio.CancelledError:
                    # Processing was cancelled (likely due to shutdown)
                    if queue_item is not None:
                        try:
                            self._event_queue.task_done()
                        except ValueError:
                            pass  # task_done() called more times than get()
                    break
                except Exception as e:
                    logger.error("Error in event processing loop", error=str(e), exc_info=e)
                    # Only call task_done if we successfully got an item
                    if queue_item is not None:
                        try:
                            self._event_queue.task_done()
                        except ValueError:
                            pass  # task_done() called more times than get()
                    # Small delay to prevent tight error loops
                    try:
                        await asyncio.sleep(0.01)
                    except (asyncio.CancelledError, RuntimeError):
                        break
                    continue

        except GeneratorExit:
            # Handle GeneratorExit explicitly to avoid ignored exception warnings
            logger.info("Event processing loop terminated via GeneratorExit")
            # Re-raise to ensure proper cleanup and avoid "ignored" warnings
            raise
        except Exception as e:
            logger.error("Unexpected error in event processing loop", error=str(e), exc_info=e)
        finally:
            logger.info("Event processing loop stopped")

    def _handle_processing_task_result(self, task: asyncio.Task) -> None:
        """Handle the result of a processing task to catch any unhandled exceptions."""
        # Remove task from active tasks
        self._active_tasks.discard(task)

        try:
            # This will raise any exception that occurred in the task
            task.result()
        except asyncio.CancelledError:
            # Task was cancelled, which is expected during shutdown
            logger.debug("Processing task cancelled")
        except Exception as e:
            logger.error("Unhandled exception in processing task", error=str(e), exc_info=e)
            # Update error statistics
            try:
                asyncio.create_task(self._update_error_count())
            except RuntimeError:
                # Event loop might be closed, update synchronously
                pass

    async def _update_error_count(self) -> None:
        """Update error count in statistics."""
        async with self._stats_lock:
            self._error_count += 1

    async def _process_single_event(self, event: BaseEvent) -> None:
        """Process a single event through all matching handlers."""
        try:
            # Get subscriptions for this event type
            subscriptions = self._subscriptions.get(event.event_type, [])

            # Filter matching subscriptions
            matching_subscriptions = [sub for sub in subscriptions if self._matches_filter(event, sub)]

            # Process handlers concurrently, but each handler is limited by semaphore
            handler_tasks = []
            for subscription in matching_subscriptions:
                task = asyncio.create_task(self._invoke_handler_with_semaphore(subscription.handler, event))
                handler_tasks.append(task)

            # Wait for all handlers to complete
            if handler_tasks:
                await asyncio.gather(*handler_tasks, return_exceptions=True)

            # Notify any waiting futures
            await self._notify_waiting_futures(event)

            # Update statistics
            async with self._stats_lock:
                self._processed_count += 1

        except Exception as e:
            logger.error("Error processing event", event_id=event.event_id, error=str(e), exc_info=e)
            async with self._stats_lock:
                self._error_count += 1

    async def _invoke_handler_with_semaphore(self, handler: EventHandler | AsyncEventHandler, event: BaseEvent) -> None:
        """Invoke a handler with semaphore limit and timeout protection."""
        async with self._processing_semaphore:
            await self._invoke_handler_with_timeout(handler, event)

    async def _invoke_handler_with_timeout(self, handler: EventHandler | AsyncEventHandler, event: BaseEvent) -> None:
        """Invoke a handler with timeout protection."""
        try:
            await asyncio.wait_for(self._invoke_handler(handler, event), timeout=self._handler_timeout)
        except asyncio.TimeoutError:
            async with self._stats_lock:
                self._timeout_count += 1
            logger.warning(
                "Handler timed out", handler=str(handler), event_id=event.event_id, timeout=self._handler_timeout
            )
        except Exception as e:
            async with self._stats_lock:
                self._error_count += 1
            logger.error("Handler error", handler=str(handler), error=str(e), exc_info=e)

    async def _notify_waiting_futures(self, event: BaseEvent) -> None:
        """Notify futures waiting for specific events."""
        completed_futures = []

        for future_id, filtered_future in self._waiting_futures.items():
            if filtered_future.done():
                completed_futures.append(future_id)
                continue

            # Let the FilteredFuture handle the filtering logic
            try:
                filtered_future.set_result(event)
                if filtered_future.done():
                    completed_futures.append(future_id)
            except asyncio.InvalidStateError:
                # Future was already completed or cancelled
                completed_futures.append(future_id)

        # Clean up completed futures
        for future_id in completed_futures:
            self._waiting_futures.pop(future_id, None)

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | AsyncEventHandler,
        *,
        filter_symbol: str | None = None,
    ) -> str:
        """
        Subscribes a handler function to events of a specific type.

        Args:
            event_type: The EventType enum member to subscribe to.
            handler: The callable function that will process matching events.
            filter_symbol: Optional symbol filter.

        Returns:
            A unique string identifier for the newly created subscription.

        Raises:
            ValueError: If the handler is invalid or subscription parameters are malformed.
        """
        self._ensure_not_closed()

        if not isinstance(event_type, EventType):
            raise ValueError("event_type must be an EventType enum member")

        if not callable(handler):
            raise ValueError("handler must be callable")

        # Generate unique subscription ID
        subscription_id = self._generate_subscription_id()

        # Create subscription
        subscription = Subscription(
            id=subscription_id,
            event_type=event_type,
            handler=handler,
            filter_symbol=filter_symbol,
            is_async=self._is_async_handler(handler),
        )

        # Store subscription with lock to prevent race conditions
        with self._subscription_lock:
            self._subscriptions[event_type].append(subscription)
            self._subscription_by_id[subscription_id] = subscription

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribes a handler from events using its unique subscription ID.

        Args:
            subscription_id: The unique string identifier of the subscription to remove.

        Returns:
            True if the subscription was found and successfully removed, False otherwise.
        """
        self._ensure_not_closed()

        # Use lock to prevent race conditions during subscription removal
        with self._subscription_lock:
            subscription = self._subscription_by_id.get(subscription_id)
            if not subscription:
                return False

            # Remove from event type subscriptions
            event_subscriptions = self._subscriptions[subscription.event_type]
            try:
                event_subscriptions.remove(subscription)
            except ValueError:
                # Subscription not found in list
                pass

            # Remove from ID mapping
            del self._subscription_by_id[subscription_id]

        return True

    def unsubscribe_all(self, event_type: EventType | None = None) -> int:
        """
        Unsubscribes all active handlers from the event bus.

        Args:
            event_type: Optional EventType to unsubscribe all handlers from.
                       If None, all subscriptions across all event types are removed.

        Returns:
            The number of handlers that were successfully unsubscribed.
        """
        self._ensure_not_closed()

        removed_count = 0

        with self._subscription_lock:
            if event_type is None:
                # Remove all subscriptions
                removed_count = len(self._subscription_by_id)
                self._subscriptions.clear()
                self._subscription_by_id.clear()
            else:
                # Remove subscriptions for specific event type
                subscriptions_to_remove = self._subscriptions.get(event_type, [])
                removed_count = len(subscriptions_to_remove)

                # Remove from ID mapping
                for subscription in subscriptions_to_remove:
                    self._subscription_by_id.pop(subscription.id, None)

                # Clear event type subscriptions
                self._subscriptions[event_type].clear()

        return removed_count

    async def close(self) -> None:
        """
        Closes the event bus and cleans up any associated resources.
        """
        if self._closed:
            return

        logger.info("Starting event bus shutdown")

        # Mark as closed to stop new events from being published
        self._closed = True

        # Cancel the background processing task first
        processing_task = self._processing_task
        if processing_task and not processing_task.done():
            logger.debug("Cancelling background processing task")
            processing_task.cancel()
            try:
                # Give it a chance to handle cancellation gracefully
                await asyncio.wait_for(processing_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, GeneratorExit):
                logger.debug("Background processing task cancelled")
            except Exception as e:
                logger.warning("Error during background task cancellation", error=str(e))

        # Cancel all waiting futures early to prevent them from interfering
        waiting_futures = list(self._waiting_futures.values())
        for filtered_future in waiting_futures:
            if not filtered_future.done():
                filtered_future.cancel()
        self._waiting_futures.clear()

        # Handle active processing tasks gracefully
        active_tasks = list(self._active_tasks)
        if active_tasks:
            logger.debug(f"Waiting for {len(active_tasks)} active processing tasks to complete")

            # First, try to let tasks complete naturally (for a short time)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*active_tasks, return_exceptions=True),
                    timeout=1.5,  # Give tasks a chance to complete
                )
            except asyncio.TimeoutError:
                logger.debug("Some tasks didn't complete, cancelling remaining tasks")
                # Cancel remaining tasks if they haven't completed
                for task in active_tasks:
                    if not task.done():
                        task.cancel()

                # Wait a bit more for cancellation to take effect
                try:
                    await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=1.5)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for active tasks to be cancelled")

        # Clear active tasks
        self._active_tasks.clear()

        # Try to drain remaining events in queue (with timeout)
        try:
            # First check if there are any events left
            remaining_events = self._event_queue.qsize()
            if remaining_events > 0:
                logger.debug(f"Draining {remaining_events} remaining events from queue")
                await asyncio.wait_for(self._event_queue.join(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for event queue to drain, forcing cleanup")
        except Exception as e:
            logger.warning("Error draining event queue", error=str(e))

        # Clear all subscriptions
        with self._subscription_lock:
            subscriptions_count = len(self._subscription_by_id)
            self._subscriptions.clear()
            self._subscription_by_id.clear()
            logger.debug(f"Cleared {subscriptions_count} subscriptions")

        # Log final statistics
        logger.info(
            "Event bus closed successfully",
            published=self._published_count,
            processed=self._processed_count,
            errors=self._error_count,
            timeouts=self._timeout_count,
            dropped=self._dropped_count,
        )

    def get_subscription_count(self, event_type: EventType | None = None) -> int:
        """
        Retrieves the number of active subscriptions currently managed by the event bus.

        Args:
            event_type: Optional EventType to count subscriptions for.
                       If None, returns total count across all types.

        Returns:
            The integer count of active subscriptions.
        """
        # Allow getting subscription count even when closed for cleanup verification
        with self._subscription_lock:
            if event_type is None:
                return len(self._subscription_by_id)
            else:
                return len(self._subscriptions.get(event_type, []))

    async def wait_for(
        self,
        event_type: EventType,
        *,
        timeout: float | None = None,
        filter_func: Callable[[BaseEvent], bool] | None = None,
    ) -> BaseEvent:
        """
        Asynchronously waits for a specific event to be published on the bus.

        Args:
            event_type: The EventType enum member of the event to wait for.
            timeout: Optional timeout in seconds.
            filter_func: Optional filter function.

        Returns:
            The BaseEvent object that was waited for.

        Raises:
            asyncio.TimeoutError: If the timeout was reached without a matching event.
        """
        self._ensure_not_closed()

        if not isinstance(event_type, EventType):
            raise ValueError("event_type must be an EventType enum member")

        # Ensure processing task is running
        if self._processing_task is None or self._processing_task.done():
            self._start_processing_task()

        # Create future for this wait operation
        future_id = str(uuid.uuid4())

        # Custom future that applies filtering
        class FilteredFuture:
            def __init__(self, target_event_type: EventType, filter_func: Callable[[BaseEvent], bool] | None):
                self.target_event_type = target_event_type
                self.filter_func = filter_func
                self.future: asyncio.Future[BaseEvent] = asyncio.Future()

            def set_result(self, event: BaseEvent) -> None:
                # Only set result if event matches our criteria
                if event.event_type == self.target_event_type:
                    if self.filter_func is None or self.filter_func(event):
                        if not self.future.done():
                            self.future.set_result(event)

            def done(self) -> bool:
                return self.future.done()

            def cancel(self) -> None:
                if not self.future.done():
                    self.future.cancel()

        filtered_future = FilteredFuture(event_type, filter_func)
        self._waiting_futures[future_id] = filtered_future

        try:
            # Use default timeout of 30 seconds if none provided
            effective_timeout = timeout if timeout is not None else 30.0
            result = await asyncio.wait_for(filtered_future.future, timeout=effective_timeout)
            return result
        except asyncio.TimeoutError:
            # Re-raise TimeoutError instead of returning None
            raise
        except asyncio.CancelledError:
            # For cancelled operations, we still need to return a result
            # This shouldn't happen in normal operation, but if it does,
            # we'll raise a TimeoutError to maintain contract consistency
            raise asyncio.TimeoutError("Operation was cancelled")
        finally:
            # Clean up
            self._waiting_futures.pop(future_id, None)

    @property
    def is_closed(self) -> bool:
        """
        Checks if the event bus is currently in a closed state.

        Returns:
            True if the event bus is closed, False otherwise.
        """
        return bool(self._closed)

    # Additional utility methods for debugging and monitoring

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._subscription_lock:
            subscriptions_by_type = {
                event_type.value: len(subscriptions) for event_type, subscriptions in self._subscriptions.items()
            }
            total_subscriptions = len(self._subscription_by_id)

        return {
            "published_count": self._published_count,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "timeout_count": self._timeout_count,
            "dropped_count": self._dropped_count,
            "total_subscriptions": total_subscriptions,
            "subscriptions_by_type": subscriptions_by_type,
            "waiting_futures": len(self._waiting_futures),
            "queue_size": self._event_queue.qsize(),
            "max_queue_size": self._max_queue_size,
            "handler_timeout": self._handler_timeout,
            "max_concurrent_handlers": self._max_concurrent_handlers,
            "is_closed": self._closed,
        }

    def get_subscriptions_for_type(self, event_type: EventType) -> List[Dict[str, Any]]:
        """Get detailed subscription information for debugging."""
        with self._subscription_lock:
            subscriptions = self._subscriptions.get(event_type, [])
            return [
                {
                    "id": sub.id,
                    "event_type": sub.event_type.value,
                    "filter_symbol": sub.filter_symbol,
                    "is_async": sub.is_async,
                    "handler_name": getattr(sub.handler, "__name__", str(sub.handler)),
                }
                for sub in subscriptions
            ]

    def get_handler_timeout(self) -> float:
        """Get the current handler timeout value."""
        return self._handler_timeout

    def set_handler_timeout(self, timeout: float) -> None:
        """Set the handler timeout value.

        Args:
            timeout: New timeout value in seconds
        """
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self._handler_timeout = timeout

    async def set_middleware_pipeline(self, pipeline: AbstractMiddlewarePipeline | None) -> None:
        """
        Sets the middleware pipeline for event processing.

        Args:
            pipeline: The middleware pipeline to set, or None to remove the current pipeline.
        """
        self._middleware_pipeline = pipeline

    async def get_middleware_pipeline(self) -> AbstractMiddlewarePipeline | None:
        """
        Gets the currently configured middleware pipeline.

        Returns:
            The current middleware pipeline, or None if no pipeline is set.
        """
        return self._middleware_pipeline

    def get_queue_size(self) -> int:
        """Get the current number of events in the processing queue."""
        return self._event_queue.qsize()

    def get_queue_capacity(self) -> int:
        """Get the maximum queue capacity."""
        return self._max_queue_size

    def is_queue_full(self) -> bool:
        """Check if the event queue is full."""
        return self._event_queue.full()

    async def flush_queue(self, timeout: float = 5.0) -> bool:
        """Wait for all events in the queue to be processed.

        Args:
            timeout: Maximum time to wait for queue to empty

        Returns:
            True if queue was flushed, False if timeout occurred
        """
        try:
            # First, wait for all events to be taken from the queue
            await asyncio.wait_for(self._event_queue.join(), timeout=timeout)

            # Then, wait for all active processing tasks to complete
            if self._active_tasks:
                await asyncio.wait_for(asyncio.gather(*self._active_tasks, return_exceptions=True), timeout=timeout)

            return True
        except asyncio.TimeoutError:
            return False

    async def pause_processing(self) -> None:
        """Pause event processing by cancelling the background task."""
        self._paused = True
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    async def resume_processing(self) -> None:
        """Resume event processing by starting a new background task."""
        self._paused = False
        if not self._closed and (self._processing_task is None or self._processing_task.done()):
            self._start_processing_task()

    # Context manager support for automatic resource management
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with automatic cleanup."""
        await self.close()
        return False  # Don't suppress exceptions
