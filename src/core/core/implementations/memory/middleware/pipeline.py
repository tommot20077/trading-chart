# ABOUTME: InMemoryMiddlewarePipeline implementation for in-memory middleware management
# ABOUTME: Provides priority-based middleware execution with caching for performance

import threading
from typing import List, Optional
from datetime import datetime, UTC
from loguru import logger

from core.interfaces.middleware import AbstractMiddleware, AbstractMiddlewarePipeline
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus, PipelineResult


class InMemoryMiddlewarePipeline(AbstractMiddlewarePipeline):
    """
    In-memory implementation of middleware pipeline.

    This implementation stores middleware in memory and executes them in priority order.
    It uses caching to optimize performance by avoiding repeated sorting operations.
    """

    def __init__(self, name: str = "InMemoryMiddlewarePipeline"):
        """
        Initialize the in-memory middleware pipeline.

        Args:
            name: Name of the pipeline for identification and logging.
        """
        self.name = name
        self._middlewares: List[AbstractMiddleware] = []
        self._sorted_cache: Optional[List[AbstractMiddleware]] = None
        self._cache_dirty = False

        # Thread safety
        self._lock = threading.RLock()  # Reentrant lock for nested operations

        # Logging setup
        self._logger = logger.bind(name=f"{__name__}.{self.name}")

        # Performance statistics
        self._execution_count = 0
        self._total_execution_time_ms = 0.0
        self._average_execution_time_ms = 0.0

    async def add_middleware(self, middleware: AbstractMiddleware) -> None:
        """
        Add a middleware to the pipeline.

        The middleware will be inserted into the pipeline and the cache will be invalidated
        to ensure proper priority ordering on next execution.

        Args:
            middleware: AbstractMiddleware instance to add to the pipeline.
        """
        with self._lock:
            self._logger.debug(
                f"Adding middleware {middleware.__class__.__name__} with priority {middleware.priority.value}"
            )
            self._middlewares.append(middleware)
            self._invalidate_cache()
            self._logger.info(
                f"Middleware {middleware.__class__.__name__} added. Total count: {len(self._middlewares)}"
            )

    async def remove_middleware(self, middleware: AbstractMiddleware) -> None:
        """
        Remove a middleware from the pipeline.

        Args:
            middleware: AbstractMiddleware instance to remove from the pipeline.

        Raises:
            ValueError: If the middleware is not found in the pipeline.
        """
        with self._lock:
            try:
                self._logger.debug(f"Removing middleware {middleware.__class__.__name__}")
                self._middlewares.remove(middleware)
                self._invalidate_cache()
                self._logger.info(
                    f"Middleware {middleware.__class__.__name__} removed. Total count: {len(self._middlewares)}"
                )
            except ValueError as e:
                self._logger.error(f"Failed to remove middleware {middleware.__class__.__name__}: {str(e)}")
                raise ValueError(f"Middleware {middleware} not found in pipeline") from e

    async def execute(self, context: MiddlewareContext) -> MiddlewareResult:
        """
        Execute the entire middleware pipeline.

        This method processes the middleware chain in priority order, passing the context
        through each middleware that can process it. Execution stops if any middleware
        returns should_continue=False or if the context is cancelled.

        Args:
            context: MiddlewareContext containing event data and execution state.

        Returns:
            MiddlewareResult: Result containing the aggregated pipeline execution results.
        """
        # Thread-safe access to middleware list
        with self._lock:
            middleware_count = len(self._middlewares)
            sorted_middlewares = self._get_sorted_middlewares()

        # Create pipeline result to track execution
        pipeline_result = PipelineResult(
            pipeline_name=self.name, total_middlewares=middleware_count, status=MiddlewareStatus.SUCCESS
        )

        # Increment execution counter with thread safety
        with self._lock:
            self._execution_count += 1
            execution_id = self._execution_count

        self._logger.info(
            f"Starting pipeline execution #{execution_id} for context {context.id} with {middleware_count} middleware"
        )

        # Handle empty pipeline
        if middleware_count == 0:
            self._logger.warning(f"Execution #{execution_id}: Pipeline is empty, skipping")
            pipeline_result.mark_completed()
            return MiddlewareResult(
                middleware_name=self.name,
                status=MiddlewareStatus.SKIPPED,
                data=pipeline_result.get_summary(),
                metadata={"reason": "Empty pipeline", "execution_id": execution_id},
            )

        # Execute middleware in priority order
        middleware_executed = 0
        should_continue_execution = True  # Track if any middleware requested to stop
        for i, middleware in enumerate(sorted_middlewares):
            middleware_name = getattr(middleware, "name", middleware.__class__.__name__)
            self._logger.debug(
                f"Execution #{execution_id}: Processing middleware {i + 1}/{len(sorted_middlewares)}: {middleware_name}"
            )

            # Check if execution should continue
            if context.is_cancelled:
                self._logger.info(
                    f"Execution #{execution_id}: Context cancelled, stopping pipeline at middleware {middleware_name}"
                )
                # For cancelled contexts, the pipeline itself is successful
                # even though no middleware were executed
                pipeline_result.fix_status(MiddlewareStatus.SUCCESS)
                break

            # Check if middleware can process this context
            if not middleware.can_process(context):
                self._logger.debug(
                    f"Execution #{execution_id}: Middleware {middleware_name} cannot process context, skipping"
                )
                # Create skipped result
                skipped_result = MiddlewareResult(
                    middleware_name=middleware_name,
                    status=MiddlewareStatus.SKIPPED,
                    metadata={"reason": "Cannot process context", "execution_id": execution_id},
                )
                skipped_result.mark_completed()
                pipeline_result.add_middleware_result(skipped_result)
                continue

            # Execute middleware
            try:
                start_time = datetime.now(UTC)
                self._logger.debug(
                    f"Execution #{execution_id}: Starting middleware {middleware_name} at {start_time.isoformat()}"
                )

                # Process middleware
                middleware_result = await middleware.process(context)

                end_time = datetime.now(UTC)
                execution_time_ms = (end_time - start_time).total_seconds() * 1000

                # Ensure execution time is set
                if middleware_result.execution_time_ms is None:
                    middleware_result.execution_time_ms = execution_time_ms

                # Add execution metadata
                middleware_result.metadata.update(
                    {"execution_id": execution_id, "pipeline_name": self.name, "middleware_index": i}
                )

                # Add to pipeline result
                pipeline_result.add_middleware_result(middleware_result)
                middleware_executed += 1

                # Update context execution path
                context.add_execution_step(middleware_name)

                self._logger.info(
                    f"Execution #{execution_id}: Middleware {middleware_name} completed in {execution_time_ms:.2f}ms, "
                    f"status: {middleware_result.status.value}, should_continue: {middleware_result.should_continue}"
                )

                # Check if pipeline should continue
                if not middleware_result.should_continue:
                    self._logger.info(
                        f"Execution #{execution_id}: Middleware {middleware_name} requested to stop pipeline execution"
                    )
                    should_continue_execution = False
                    break

            except Exception as e:
                end_time = datetime.now(UTC)
                execution_time_ms = (end_time - start_time).total_seconds() * 1000

                self._logger.error(
                    f"Execution #{execution_id}: Middleware {middleware_name} failed after {execution_time_ms:.2f}ms: {str(e)}",
                    exc_info=True,
                )

                # Create error result
                error_result = MiddlewareResult(
                    middleware_name=middleware_name,
                    status=MiddlewareStatus.FAILED,
                    error=str(e),
                    error_details={
                        "exception_type": type(e).__name__,
                        "execution_id": execution_id,
                        "pipeline_name": self.name,
                        "middleware_index": i,
                    },
                    should_continue=False,
                    execution_time_ms=execution_time_ms,
                )
                error_result.mark_completed()
                pipeline_result.add_middleware_result(error_result)
                middleware_executed += 1

                # Pipeline itself is successful even if a middleware fails
                # The error is captured in the middleware result
                pipeline_result.fix_status(MiddlewareStatus.SUCCESS)
                break

        # Mark pipeline as completed and calculate total execution time
        pipeline_result.mark_completed()
        pipeline_execution_time_ms = pipeline_result.get_execution_time_ms() or 0.0

        # Update performance statistics with thread safety
        with self._lock:
            self._total_execution_time_ms += pipeline_execution_time_ms
            self._average_execution_time_ms = self._total_execution_time_ms / self._execution_count

        # Log pipeline completion
        self._logger.info(
            f"Execution #{execution_id} completed in {pipeline_execution_time_ms:.2f}ms. "
            f"Executed {middleware_executed}/{len(sorted_middlewares)} middleware, "
            f"status: {pipeline_result.status.value}"
        )

        # Return pipeline result wrapped in MiddlewareResult
        return MiddlewareResult(
            middleware_name=self.name,
            status=pipeline_result.status,
            data=pipeline_result.get_summary(),
            should_continue=should_continue_execution,  # Reflect middleware decision
            execution_time_ms=pipeline_execution_time_ms,
            metadata={
                "pipeline_results": [result.get_execution_summary() for result in pipeline_result.middleware_results],
                "execution_id": execution_id,
                "middleware_executed": middleware_executed,
                "total_middleware": len(sorted_middlewares),
                "performance_stats": {
                    "total_executions": self._execution_count,
                    "average_execution_time_ms": self._average_execution_time_ms,
                },
            },
        )

    async def get_middleware_count(self) -> int:
        """
        Get the number of middleware in the pipeline.

        Returns:
            int: Number of middleware currently in the pipeline.
        """
        with self._lock:
            return len(self._middlewares)

    async def clear(self) -> None:
        """
        Remove all middleware from the pipeline.

        This method clears the entire pipeline, removing all middleware
        and resetting the pipeline to an empty state.
        """
        with self._lock:
            middleware_count = len(self._middlewares)
            self._logger.info(f"Clearing pipeline with {middleware_count} middleware")
            self._middlewares.clear()
            self._invalidate_cache()
            # Reset performance statistics
            self._execution_count = 0
            self._total_execution_time_ms = 0.0
            self._average_execution_time_ms = 0.0
            self._logger.info("Pipeline cleared and statistics reset")

    async def get_middleware_by_priority(self) -> List[AbstractMiddleware]:
        """
        Get all middleware in the pipeline sorted by priority.

        Returns:
            List[AbstractMiddleware]: List of middleware sorted by priority
            (lowest priority value first).
        """
        with self._lock:
            return self._get_sorted_middlewares().copy()

    def _get_sorted_middlewares(self) -> List[AbstractMiddleware]:
        """
        Get middleware sorted by priority with caching.

        This method uses a cache to avoid repeated sorting operations.
        The cache is invalidated when middleware are added or removed.
        Note: This method assumes it's called within a locked context.

        Returns:
            List[AbstractMiddleware]: Sorted list of middleware.
        """
        if self._sorted_cache is None or self._cache_dirty:
            self._sorted_cache = sorted(self._middlewares, key=lambda m: m.priority.value)
            self._cache_dirty = False
            self._logger.debug(f"Middleware cache rebuilt with {len(self._sorted_cache)} items")

        return self._sorted_cache

    def _invalidate_cache(self) -> None:
        """
        Invalidate the sorted middleware cache.

        This method is called when middleware are added or removed
        to ensure the cache is refreshed on next access.
        """
        self._cache_dirty = True

    def _get_priority_name(self, priority) -> str:
        """
        Get the string name for a priority value.

        Args:
            priority: EventPriority instance

        Returns:
            str: Name of the priority level
        """
        from core.models.event.event_priority import EventPriority

        # Map common priority values to their names
        if priority == EventPriority.HIGH:
            return "HIGH"
        elif priority == EventPriority.NORMAL:
            return "NORMAL"
        elif priority == EventPriority.LOW:
            return "LOW"
        elif priority == EventPriority.CRITICAL:
            return "CRITICAL"
        elif priority == EventPriority.VERY_LOW:
            return "VERY_LOW"
        elif priority == EventPriority.HIGHEST:
            return "HIGHEST"
        elif priority == EventPriority.LOWEST:
            return "LOWEST"
        else:
            # For custom priorities, return the value as string
            return str(priority.value)

    async def is_empty(self) -> bool:
        """
        Check if the pipeline is empty.

        Returns:
            bool: True if the pipeline has no middleware, False otherwise.
        """
        with self._lock:
            return len(self._middlewares) == 0

    async def contains_middleware(self, middleware: AbstractMiddleware) -> bool:
        """
        Check if a specific middleware is in the pipeline.

        Args:
            middleware: AbstractMiddleware instance to check for.

        Returns:
            bool: True if the middleware is in the pipeline, False otherwise.
        """
        with self._lock:
            return middleware in self._middlewares

    def get_performance_stats(self) -> dict:
        """
        Get detailed performance statistics for the pipeline.

        Returns:
            dict: Performance statistics including execution counts and timing.
        """
        with self._lock:
            return {
                "pipeline_name": self.name,
                "total_executions": self._execution_count,
                "total_execution_time_ms": self._total_execution_time_ms,
                "average_execution_time_ms": self._average_execution_time_ms,
                "middleware_count": len(self._middlewares),
                "cache_status": {"cache_dirty": self._cache_dirty, "cache_exists": self._sorted_cache is not None},
            }

    def reset_performance_stats(self) -> None:
        """
        Reset performance statistics to initial state.

        This method is useful for benchmarking or when starting fresh measurements.
        """
        with self._lock:
            old_count = self._execution_count
            self._execution_count = 0
            self._total_execution_time_ms = 0.0
            self._average_execution_time_ms = 0.0
            self._logger.info(f"Performance statistics reset (previous execution count: {old_count})")

    def get_pipeline_info(self) -> dict:
        """
        Get information about the current pipeline state.

        Returns:
            dict: Pipeline information including middleware count and priority distribution.
        """
        with self._lock:
            if not self._middlewares:
                return {
                    "name": self.name,
                    "middleware_count": 0,
                    "priority_distribution": {},
                    "performance_stats": {
                        "total_executions": self._execution_count,
                        "total_execution_time_ms": self._total_execution_time_ms,
                        "average_execution_time_ms": self._average_execution_time_ms,
                    },
                }

            # Calculate priority distribution
            priority_distribution: dict[str, int] = {}
            middleware_details = []

            for middleware in self._middlewares:
                # Map priority values to their names
                priority_name = self._get_priority_name(middleware.priority)
                priority_distribution[priority_name] = priority_distribution.get(priority_name, 0) + 1

                middleware_details.append(
                    {
                        "class_name": middleware.__class__.__name__,
                        "priority_value": middleware.priority.value,
                        "priority_name": priority_name,
                    }
                )

            return {
                "name": self.name,
                "middleware_count": len(self._middlewares),
                "priority_distribution": priority_distribution,
                "middleware_names": [m.__class__.__name__ for m in self._middlewares],
                "middleware_details": middleware_details,
                "performance_stats": {
                    "total_executions": self._execution_count,
                    "total_execution_time_ms": self._total_execution_time_ms,
                    "average_execution_time_ms": self._average_execution_time_ms,
                },
                "cache_status": {"cache_dirty": self._cache_dirty, "cache_exists": self._sorted_cache is not None},
            }
