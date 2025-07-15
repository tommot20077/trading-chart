# ABOUTME: InMemoryMiddlewarePipeline implementation for in-memory middleware management
# ABOUTME: Provides priority-based middleware execution with caching for performance

from typing import List, Optional
from datetime import datetime, UTC

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

    async def add_middleware(self, middleware: AbstractMiddleware) -> None:
        """
        Add a middleware to the pipeline.

        The middleware will be inserted into the pipeline and the cache will be invalidated
        to ensure proper priority ordering on next execution.

        Args:
            middleware: AbstractMiddleware instance to add to the pipeline.
        """
        self._middlewares.append(middleware)
        self._invalidate_cache()

    async def remove_middleware(self, middleware: AbstractMiddleware) -> None:
        """
        Remove a middleware from the pipeline.

        Args:
            middleware: AbstractMiddleware instance to remove from the pipeline.

        Raises:
            ValueError: If the middleware is not found in the pipeline.
        """
        try:
            self._middlewares.remove(middleware)
            self._invalidate_cache()
        except ValueError:
            raise ValueError(f"Middleware {middleware} not found in pipeline")

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
        # Create pipeline result to track execution
        pipeline_result = PipelineResult(
            pipeline_name=self.name, total_middlewares=len(self._middlewares), status=MiddlewareStatus.SUCCESS
        )

        # Handle empty pipeline
        if not self._middlewares:
            pipeline_result.mark_completed()
            return MiddlewareResult(
                middleware_name=self.name,
                status=MiddlewareStatus.SKIPPED,
                data=pipeline_result.get_summary(),
                metadata={"reason": "Empty pipeline"},
            )

        # Get sorted middleware list
        sorted_middlewares = self._get_sorted_middlewares()

        # Execute middleware in priority order
        for middleware in sorted_middlewares:
            # Check if execution should continue
            if context.is_cancelled:
                # For cancelled contexts, the pipeline itself is successful
                # even though no middleware were executed
                pipeline_result.fix_status(MiddlewareStatus.SUCCESS)
                break

            # Check if middleware can process this context
            if not middleware.can_process(context):
                # Create skipped result
                middleware_name = getattr(middleware, "name", middleware.__class__.__name__)
                skipped_result = MiddlewareResult(
                    middleware_name=middleware_name,
                    status=MiddlewareStatus.SKIPPED,
                    metadata={"reason": "Cannot process context"},
                )
                skipped_result.mark_completed()
                pipeline_result.add_middleware_result(skipped_result)
                continue

            # Execute middleware
            try:
                start_time = datetime.now(UTC)

                # Process middleware
                middleware_result = await middleware.process(context)

                # Ensure execution time is set
                if middleware_result.execution_time_ms is None:
                    middleware_result.execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

                # Add to pipeline result
                pipeline_result.add_middleware_result(middleware_result)

                # Update context execution path
                middleware_name = getattr(middleware, "name", middleware.__class__.__name__)
                context.add_execution_step(middleware_name)

                # Check if pipeline should continue
                if not middleware_result.should_continue:
                    break

            except Exception as e:
                # Create error result
                middleware_name = getattr(middleware, "name", middleware.__class__.__name__)
                error_result = MiddlewareResult(
                    middleware_name=middleware_name,
                    status=MiddlewareStatus.FAILED,
                    error=str(e),
                    error_details={"exception_type": type(e).__name__},
                    should_continue=False,
                )
                error_result.mark_completed()
                pipeline_result.add_middleware_result(error_result)
                # Pipeline itself is successful even if a middleware fails
                # The error is captured in the middleware result
                pipeline_result.fix_status(MiddlewareStatus.SUCCESS)
                break

        # Mark pipeline as completed
        pipeline_result.mark_completed()

        # Return pipeline result wrapped in MiddlewareResult
        return MiddlewareResult(
            middleware_name=self.name,
            status=pipeline_result.status,
            data=pipeline_result.get_summary(),
            execution_time_ms=pipeline_result.get_execution_time_ms(),
            metadata={
                "pipeline_results": [result.get_execution_summary() for result in pipeline_result.middleware_results]
            },
        )

    async def get_middleware_count(self) -> int:
        """
        Get the number of middleware in the pipeline.

        Returns:
            int: Number of middleware currently in the pipeline.
        """
        return len(self._middlewares)

    async def clear(self) -> None:
        """
        Remove all middleware from the pipeline.

        This method clears the entire pipeline, removing all middleware
        and resetting the pipeline to an empty state.
        """
        self._middlewares.clear()
        self._invalidate_cache()

    async def get_middleware_by_priority(self) -> List[AbstractMiddleware]:
        """
        Get all middleware in the pipeline sorted by priority.

        Returns:
            List[AbstractMiddleware]: List of middleware sorted by priority
            (lowest priority value first).
        """
        return self._get_sorted_middlewares().copy()

    def _get_sorted_middlewares(self) -> List[AbstractMiddleware]:
        """
        Get middleware sorted by priority with caching.

        This method uses a cache to avoid repeated sorting operations.
        The cache is invalidated when middleware are added or removed.

        Returns:
            List[AbstractMiddleware]: Sorted list of middleware.
        """
        if self._sorted_cache is None or self._cache_dirty:
            self._sorted_cache = sorted(self._middlewares, key=lambda m: m.priority.value)
            self._cache_dirty = False

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

    def get_pipeline_info(self) -> dict:
        """
        Get information about the current pipeline state.

        Returns:
            dict: Pipeline information including middleware count and priority distribution.
        """
        if not self._middlewares:
            return {"name": self.name, "middleware_count": 0, "priority_distribution": {}}

        # Calculate priority distribution
        priority_distribution: dict[str, int] = {}
        for middleware in self._middlewares:
            # Map priority values to their names
            priority_name = self._get_priority_name(middleware.priority)
            priority_distribution[priority_name] = priority_distribution.get(priority_name, 0) + 1

        return {
            "name": self.name,
            "middleware_count": len(self._middlewares),
            "priority_distribution": priority_distribution,
            "middleware_names": [m.__class__.__name__ for m in self._middlewares],
        }
