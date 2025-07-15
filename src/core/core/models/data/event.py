from datetime import datetime, UTC
from typing import Any, TypeVar, Generic
from uuid import uuid4

from pydantic import Field, field_validator, BaseModel

from core.models.event.event_priority import EventPriority
from core.models.event.event_type import EventType

T = TypeVar("T")


class BaseEvent(BaseModel, Generic[T]):
    """
    Base class for all events in the system.

    This Pydantic model provides a common structure for all events, including
    metadata such as event ID, type, timestamp, source, and an optional symbol.
    It also includes a generic `data` field to hold the event-specific payload.

    Type Parameters:
        T: The type of the `data` payload for the event.

    Attributes:
        event_id (str): A unique identifier for the event, automatically generated.
        event_type (EventType): The type of the event, indicating its category.
        timestamp (datetime): The UTC timestamp when the event occurred, automatically set.
        source (str): The origin of the event (e.g., exchange name, service name).
        symbol (str | None): The trading symbol associated with the event, if applicable.
        data (T): The actual payload of the event, typed generically.
        priority (EventPriority): The priority level of the event. Defaults to `NORMAL`.
        correlation_id (str | None): An optional ID for correlating related events across different systems.
        metadata (dict[str, Any]): A dictionary for any additional, unstructured metadata.
    """

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        """
        Validates and ensures the timestamp is timezone-aware UTC.

        Args:
            v: The datetime object to validate.

        Returns:
            The validated datetime object, converted to UTC if necessary.

        Raises:
            ValueError: If the value is not a datetime object.
        """
        if not isinstance(v, datetime):
            raise ValueError("Value must be a datetime object")

        if v.tzinfo is None:
            # Assume UTC if no timezone
            return v.replace(tzinfo=UTC)

        # Convert to UTC if different timezone
        return v.astimezone(UTC)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str | None) -> str | None:
        """
        Validates and normalizes the trading symbol.

        The symbol is stripped of whitespace and converted to uppercase.

        Args:
            v: The symbol string to validate.

        Returns:
            The normalized symbol string, or `None` if the input was `None`.

        Raises:
            ValueError: If the symbol is not a string or becomes empty after stripping.
        """
        if v is None:
            return None

        # Handle non-string values
        if not isinstance(v, str):
            raise ValueError("Symbol must be a string")

        # Strip whitespace and convert to uppercase
        v = v.strip().upper()

        # Check for empty symbol after stripping
        if not v:
            raise ValueError("Symbol cannot be empty")

        return v

    # Model fields
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the event")
    event_type: EventType = Field(description="The type of the event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="UTC timestamp when the event occurred"
    )
    source: str = Field(description="The origin of the event")
    symbol: str | None = Field(default=None, description="Trading symbol associated with the event")
    data: T = Field(description="The actual payload of the event")
    priority: EventPriority = Field(default=EventPriority.NORMAL, description="Priority level of the event")
    correlation_id: str | None = Field(default=None, description="ID for correlating related events")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional unstructured metadata")

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the `BaseEvent` instance to a dictionary representation.

        The timestamp is converted to ISO 8601 format.

        Returns:
            A dictionary representing the event.
        """
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def __str__(self) -> str:
        """
        Returns a string representation of the `BaseEvent` instance.

        Returns:
            A string in the format `ClassName(id=..., type=..., source=...)`.
        """
        return f"{self.__class__.__name__}(id={self.event_id}, type={self.event_type}, source={self.source})"
