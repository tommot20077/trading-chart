# ABOUTME: MiddlewareContext model for storing middleware execution context
# ABOUTME: Contains event data, user information, metadata, and execution state

from datetime import datetime, UTC
from typing import Any, Dict, Optional, TypeVar, Generic, List
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")


class MiddlewareContext(BaseModel, Generic[T]):
    """
    Middleware execution context.

    This model contains all the information needed for middleware execution,
    including event data, user information, metadata, and execution state.
    """

    # Context identification
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique context identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Context creation timestamp")

    # Request/Event related information
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracing")
    event_id: Optional[str] = Field(default=None, description="Event identifier")
    event_type: Optional[str] = Field(default=None, description="Type of event being processed")
    symbol: Optional[str] = Field(default=None, description="Trading symbol if applicable")

    # Data payload
    data: Optional[T] = Field(default=None, description="Event or request data payload")

    # Metadata and configuration
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Distributed tracing information
    trace_id: Optional[str] = Field(default=None, description="Distributed trace identifier")
    span_id: Optional[str] = Field(default=None, description="Span identifier for tracing")

    # User and session information
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

    # Execution state
    is_cancelled: bool = Field(default=False, description="Whether execution has been cancelled")
    execution_path: List[str] = Field(default_factory=list, description="List of middleware names executed")

    # Configuration flags
    enable_logging: bool = Field(default=True, description="Whether to enable logging for this context")
    enable_metrics: bool = Field(default=True, description="Whether to collect metrics for this context")
    enable_tracing: bool = Field(default=True, description="Whether to enable distributed tracing")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # Note: json_encoders is deprecated in Pydantic v2
        # datetime serialization is handled automatically via isoformat()
    )

    def add_execution_step(self, middleware_name: str) -> None:
        """
        Add a middleware to the execution path.

        Args:
            middleware_name: Name of the middleware that was executed.
        """
        self.execution_path.append(middleware_name)

    def cancel(self) -> None:
        """Cancel the middleware execution."""
        self.is_cancelled = True

    def get_execution_duration(self) -> float:
        """
        Get the duration since context creation in milliseconds.

        Returns:
            float: Duration in milliseconds.
        """
        now = datetime.now(UTC)
        return (now - self.timestamp).total_seconds() * 1000

    def clone(self) -> "MiddlewareContext[T]":
        """
        Create a copy of this context with a new ID.

        Returns:
            MiddlewareContext: A new context with the same data but new ID.
        """
        data = self.model_dump()
        data["id"] = str(uuid4())
        data["timestamp"] = datetime.now(UTC)
        return MiddlewareContext[T](**data)

    # Data manipulation methods
    def set_data(self, key: str, value: Any) -> None:
        """
        Set a data value in the context.

        Args:
            key: The data key
            value: The data value
        """
        if not hasattr(self, "_context_data"):
            self._context_data = {}
        self._context_data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get a data value from the context.

        Args:
            key: The data key
            default: Default value if key not found

        Returns:
            The data value or default
        """
        if not hasattr(self, "_context_data"):
            self._context_data = {}
        return self._context_data.get(key, default)

    def get_all_data(self) -> Dict[str, Any]:
        """
        Get all data from the context.

        Returns:
            Dictionary of all context data
        """
        if not hasattr(self, "_context_data"):
            self._context_data = {}
        return self._context_data.copy()

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value in the context.

        Args:
            key: The metadata key
            value: The metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value from the context.

        Args:
            key: The metadata key
            default: Default value if key not found

        Returns:
            The metadata value or default
        """
        return self.metadata.get(key, default)

    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get all metadata from the context.

        Returns:
            Dictionary of all context metadata
        """
        return self.metadata.copy()

    def get_execution_path(self) -> List[str]:
        """
        Get the execution path of middleware.

        Returns:
            List of middleware names that have been executed
        """
        return self.execution_path.copy()
