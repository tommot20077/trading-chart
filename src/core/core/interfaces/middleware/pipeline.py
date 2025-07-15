# ABOUTME: Abstract middleware pipeline interface for managing middleware chains
# ABOUTME: Defines the contract for middleware pipeline implementations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.middleware.context import MiddlewareContext
    from core.models.middleware.result import MiddlewareResult
    from .middleware import AbstractMiddleware


class AbstractMiddlewarePipeline(ABC):
    """
    Abstract base class for middleware pipeline implementations.

    This class defines the contract for managing and executing a chain of
    middleware components in priority order.
    """

    @abstractmethod
    async def add_middleware(self, middleware: "AbstractMiddleware") -> None:
        """
        Add a middleware to the pipeline.

        The middleware will be inserted into the pipeline according to its
        priority value. Lower priority values will be executed first.

        Args:
            middleware: AbstractMiddleware instance to add to the pipeline.
        """
        pass

    @abstractmethod
    async def remove_middleware(self, middleware: "AbstractMiddleware") -> None:
        """
        Remove a middleware from the pipeline.

        Args:
            middleware: AbstractMiddleware instance to remove from the pipeline.

        Raises:
            ValueError: If the middleware is not found in the pipeline.
        """
        pass

    @abstractmethod
    async def execute(self, context: "MiddlewareContext") -> "MiddlewareResult":
        """
        Execute the entire middleware pipeline.

        This method processes the middleware chain in priority order,
        passing the context through each middleware that can process it.
        Execution stops if any middleware returns should_continue=False.

        Args:
            context: MiddlewareContext containing event data and execution state.

        Returns:
            MiddlewareResult containing the aggregated execution results from
            all middleware in the pipeline.
        """
        pass

    @abstractmethod
    async def get_middleware_count(self) -> int:
        """
        Get the number of middleware in the pipeline.

        Returns:
            int: Number of middleware currently in the pipeline.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Remove all middleware from the pipeline.

        This method clears the entire pipeline, removing all middleware
        and resetting the pipeline to an empty state.
        """
        pass

    @abstractmethod
    async def get_middleware_by_priority(self) -> list["AbstractMiddleware"]:
        """
        Get all middleware in the pipeline sorted by priority.

        Returns:
            list[AbstractMiddleware]: List of middleware sorted by priority
            (lowest priority value first).
        """
        pass
