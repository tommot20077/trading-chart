# ABOUTME: Abstract middleware interface defining the core middleware contract
# ABOUTME: All middleware implementations must inherit from AbstractMiddleware

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.models.event.event_priority import EventPriority

if TYPE_CHECKING:
    from core.models.middleware.context import MiddlewareContext
    from core.models.middleware.result import MiddlewareResult


class AbstractMiddleware(ABC):
    """
    Abstract base class for all middleware implementations.

    This class defines the contract that all middleware must follow, providing
    a unified interface for middleware execution within the event pipeline.
    """

    def __init__(self, priority: EventPriority = EventPriority.NORMAL):
        """
        Initialize middleware with priority.

        Args:
            priority: EventPriority for middleware execution order.
                     Lower values have higher priority (Spring Security pattern).
        """
        self.priority = priority

    @abstractmethod
    async def process(self, context: "MiddlewareContext") -> "MiddlewareResult":
        """
        Process the middleware logic.

        This method contains the core business logic of the middleware.
        It receives a context containing all relevant information and
        returns a result indicating the processing outcome.

        Args:
            context: MiddlewareContext containing event data, user info,
                    metadata, and execution state.

        Returns:
            MiddlewareResult containing the processing outcome, timing
            information, and any data modifications.

        Raises:
            Exception: Implementation-specific exceptions should be caught
                      and returned as failed MiddlewareResult.
        """
        pass

    @abstractmethod
    def can_process(self, context: "MiddlewareContext") -> bool:
        """
        Determine if this middleware can process the given context.

        This method allows middleware to selectively process only certain
        types of events or contexts based on their content or metadata.

        Args:
            context: MiddlewareContext to evaluate for processing capability.

        Returns:
            bool: True if this middleware can process the context, False otherwise.
        """
        pass

    def __lt__(self, other: "AbstractMiddleware") -> bool:
        """Less than comparison for priority ordering."""
        return self.priority < other.priority

    def __le__(self, other: "AbstractMiddleware") -> bool:
        """Less than or equal comparison for priority ordering."""
        return self.priority <= other.priority

    def __gt__(self, other: "AbstractMiddleware") -> bool:
        """Greater than comparison for priority ordering."""
        return self.priority > other.priority

    def __ge__(self, other: "AbstractMiddleware") -> bool:
        """Greater than or equal comparison for priority ordering."""
        return self.priority >= other.priority

    def __eq__(self, other: object) -> bool:
        """Equality comparison for middleware."""
        if not isinstance(other, AbstractMiddleware):
            return NotImplemented
        return self.priority == other.priority

    def __hash__(self) -> int:
        """Hash implementation for middleware."""
        return hash((self.__class__.__name__, self.priority))

    def __repr__(self) -> str:
        """String representation of middleware."""
        priority_name = self._get_priority_name()
        return f"{self.__class__.__name__}(priority={priority_name})"

    def _get_priority_name(self) -> str:
        """Get the string name for the priority value."""
        # Map common priority values to their names
        if self.priority == EventPriority.HIGH:
            return "HIGH"
        elif self.priority == EventPriority.NORMAL:
            return "NORMAL"
        elif self.priority == EventPriority.LOW:
            return "LOW"
        elif self.priority == EventPriority.CRITICAL:
            return "CRITICAL"
        elif self.priority == EventPriority.VERY_LOW:
            return "VERY_LOW"
        elif self.priority == EventPriority.HIGHEST:
            return "HIGHEST"
        elif self.priority == EventPriority.LOWEST:
            return "LOWEST"
        else:
            # For custom priorities, return the value as string
            return str(self.priority.value)
