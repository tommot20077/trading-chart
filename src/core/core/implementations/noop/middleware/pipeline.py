# ABOUTME: NoOp middleware pipeline implementation for testing and minimal scenarios
# ABOUTME: Provides no-operation middleware pipeline that satisfies interface contracts without processing

from typing import List
from loguru import logger

from core.interfaces.middleware import AbstractMiddleware, AbstractMiddlewarePipeline
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus


class NoOpMiddlewarePipeline(AbstractMiddlewarePipeline):
    """
    No-operation implementation of middleware pipeline.

    This implementation provides a middleware pipeline that satisfies the interface
    contract but performs no actual processing. It's designed for:
    - Testing scenarios where middleware logic should be bypassed
    - Minimal system configurations
    - Performance testing baselines
    - Development environments where middleware is not needed

    All methods return appropriate success responses without executing any logic.
    """

    def __init__(self, name: str = "NoOpMiddlewarePipeline"):
        """
        Initialize the no-operation middleware pipeline.

        Args:
            name: Name of the pipeline for identification and logging.
        """
        self.name = name
        self._middleware_count = 0
        self._logger = logger.bind(name=f"{__name__}.{self.name}")

        self._logger.debug(f"NoOp middleware pipeline '{name}' initialized")

    async def add_middleware(self, middleware: AbstractMiddleware) -> None:
        """
        Add a middleware to the pipeline (no-op).

        Records the middleware addition but does not store or process it.

        Args:
            middleware: AbstractMiddleware instance (ignored in no-op implementation).
        """
        self._middleware_count += 1
        self._logger.debug(f"NoOp: Middleware {middleware.__class__.__name__} added (count: {self._middleware_count})")

    async def remove_middleware(self, middleware: AbstractMiddleware) -> None:
        """
        Remove a middleware from the pipeline (no-op).

        Records the middleware removal but does not perform actual removal logic.

        Args:
            middleware: AbstractMiddleware instance (ignored in no-op implementation).
        """
        if self._middleware_count > 0:
            self._middleware_count -= 1
        self._logger.debug(
            f"NoOp: Middleware {middleware.__class__.__name__} removed (count: {self._middleware_count})"
        )

    async def execute(self, context: MiddlewareContext) -> MiddlewareResult:
        """
        Execute the middleware pipeline (no-op).

        Returns a successful result without processing any middleware.
        This allows the system to continue normal operation while bypassing
        all middleware logic.

        Args:
            context: MiddlewareContext containing event data and execution state.

        Returns:
            MiddlewareResult indicating successful no-op execution.
        """
        self._logger.debug(f"NoOp: Pipeline execution for context {context.id} (no processing performed)")

        return MiddlewareResult(
            middleware_name=self.name,
            status=MiddlewareStatus.SUCCESS,
            data={"message": "No-op execution completed"},
            metadata={
                "pipeline_type": "NoOp",
                "middleware_count": self._middleware_count,
                "context_id": context.id,
                "execution_mode": "bypass",
            },
            should_continue=True,
            execution_time_ms=0.0,  # No actual processing time
        )

    async def get_middleware_count(self) -> int:
        """
        Get the number of middleware in the pipeline.

        Returns:
            int: Number of middleware that would be in the pipeline (tracked count).
        """
        return self._middleware_count

    async def clear(self) -> None:
        """
        Remove all middleware from the pipeline (no-op).

        Resets the middleware count but doesn't perform actual cleanup since
        no middleware are actually stored.
        """
        old_count = self._middleware_count
        self._middleware_count = 0
        self._logger.debug(f"NoOp: Pipeline cleared (previous count: {old_count})")

    async def get_middleware_by_priority(self) -> List[AbstractMiddleware]:
        """
        Get all middleware in the pipeline sorted by priority (no-op).

        Returns:
            List[AbstractMiddleware]: Empty list since no middleware are stored.
        """
        self._logger.debug("NoOp: Returning empty middleware list")
        return []

    async def is_empty(self) -> bool:
        """
        Check if the pipeline is empty.

        Returns:
            bool: True if no middleware have been added, False otherwise.
        """
        return self._middleware_count == 0

    async def contains_middleware(self, middleware: AbstractMiddleware) -> bool:
        """
        Check if a specific middleware is in the pipeline.

        Args:
            middleware: AbstractMiddleware instance to check for.

        Returns:
            bool: Always False since no middleware are actually stored.
        """
        return False

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for the pipeline.

        Returns:
            dict: Performance statistics showing no-op characteristics.
        """
        return {
            "pipeline_name": self.name,
            "pipeline_type": "NoOp",
            "total_executions": 0,  # No actual executions tracked
            "total_execution_time_ms": 0.0,
            "average_execution_time_ms": 0.0,
            "middleware_count": self._middleware_count,
            "performance_impact": "none",
        }

    def reset_performance_stats(self) -> None:
        """
        Reset performance statistics (no-op).

        Since no actual statistics are maintained, this is effectively a no-op.
        """
        self._logger.debug("NoOp: Performance statistics reset (no actual stats maintained)")

    def get_pipeline_info(self) -> dict:
        """
        Get information about the current pipeline state.

        Returns:
            dict: Pipeline information for the no-op implementation.
        """
        return {
            "name": self.name,
            "type": "NoOp",
            "middleware_count": self._middleware_count,
            "description": "No-operation middleware pipeline for testing and minimal scenarios",
            "features": [
                "Interface compliant",
                "Zero processing overhead",
                "Always successful execution",
                "Suitable for testing and development",
            ],
            "performance_characteristics": {
                "execution_time": "0ms (no processing)",
                "memory_usage": "minimal",
                "cpu_impact": "none",
            },
        }

    def __str__(self) -> str:
        """
        String representation of the NoOp pipeline.

        Returns:
            str: Human-readable description.
        """
        return f"NoOpMiddlewarePipeline(name='{self.name}', count={self._middleware_count})"

    def __repr__(self) -> str:
        """
        Developer representation of the NoOp pipeline.

        Returns:
            str: Detailed representation for debugging.
        """
        return f"NoOpMiddlewarePipeline(name='{self.name}', middleware_count={self._middleware_count})"
