# ABOUTME: Event priority system similar to Spring Security filter ordering
# ABOUTME: Lower numbers indicate higher priority (executed first)

from __future__ import annotations

import sys
from typing import Any, ClassVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema
from pydantic_core import core_schema


class EventPriority:
    """
    Event priority system inspired by Spring Security filter ordering.

    Lower integer values indicate higher priority (executed first).
    This allows for fine-grained control over event processing order
    and enables insertion of custom priorities between predefined values.
    """

    # Define priority constants as class variables (initialized after class definition)
    HIGHEST: ClassVar[EventPriority]
    CRITICAL: ClassVar[EventPriority]
    HIGH: ClassVar[EventPriority]
    NORMAL: ClassVar[EventPriority]
    LOW: ClassVar[EventPriority]
    VERY_LOW: ClassVar[EventPriority]
    LOWEST: ClassVar[EventPriority]

    def __init__(self, value: int):
        """
        Initialize with a custom priority value.

        Args:
            value: Priority value (lower = higher priority)
        """
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EventPriority):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, EventPriority):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, EventPriority):
            return self.value <= other.value
        elif isinstance(other, int):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, EventPriority):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, EventPriority):
            return self.value >= other.value
        elif isinstance(other, int):
            return self.value >= other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"EventPriority({self.value})"

    @classmethod
    def custom(cls, value: int) -> EventPriority:
        """
        Create a custom priority value.

        Args:
            value: Priority value (lower = higher priority)

        Returns:
            EventPriority instance with the specified value
        """
        return cls(value)

    @classmethod
    def before(cls, priority: int | EventPriority, offset: int = 10) -> EventPriority:
        """
        Create a priority that executes before the given priority.

        Args:
            priority: Base priority value (int or EventPriority)
            offset: How much earlier to execute (default: 10)

        Returns:
            EventPriority instance with higher priority
        """
        base_value = priority.value if isinstance(priority, EventPriority) else priority
        return cls(base_value - offset)

    @classmethod
    def after(cls, priority: int | EventPriority, offset: int = 10) -> EventPriority:
        """
        Create a priority that executes after the given priority.

        Args:
            priority: Base priority value (int or EventPriority)
            offset: How much later to execute (default: 10)

        Returns:
            EventPriority instance with lower priority
        """
        base_value = priority.value if isinstance(priority, EventPriority) else priority
        return cls(base_value + offset)

    @classmethod
    def after_priority(cls, priority: int | EventPriority, offset: int = 10) -> EventPriority:
        """
        Create a priority that executes after the given priority.

        Args:
            priority: Base priority value (int or EventPriority)
            offset: How much later to execute (default: 10)

        Returns:
            EventPriority instance with lower priority
        """
        base_value = priority.value if isinstance(priority, EventPriority) else priority
        return cls(base_value + offset)

    @classmethod
    def before_priority(cls, priority: int | EventPriority, offset: int = 10) -> EventPriority:
        """
        Create a priority that executes before the given priority.

        Args:
            priority: Base priority value (int or EventPriority)
            offset: How much earlier to execute (default: 10)

        Returns:
            EventPriority instance with higher priority
        """
        base_value = priority.value if isinstance(priority, EventPriority) else priority
        return cls(base_value - offset)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: "GetCoreSchemaHandler") -> "CoreSchema":
        """
        Provide Pydantic core schema for EventPriority.

        This allows EventPriority to be used in Pydantic models by treating it
        as an integer with custom validation and serialization.
        """

        def validate_priority(value: Any, _info, _context) -> EventPriority:
            if isinstance(value, EventPriority):
                return value
            elif isinstance(value, int):
                return cls(value)
            else:
                raise ValueError(f"Expected EventPriority or int, got {type(value)}")

        return core_schema.with_info_wrap_validator_function(
            validate_priority,
            core_schema.int_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.value if isinstance(value, cls) else value,
                return_schema=core_schema.int_schema(),
            ),
        )


# Initialize class attributes after class definition
EventPriority.HIGHEST = EventPriority(-sys.maxsize - 1)
EventPriority.CRITICAL = EventPriority(0)
EventPriority.HIGH = EventPriority(100)
EventPriority.NORMAL = EventPriority(200)
EventPriority.LOW = EventPriority(300)
EventPriority.VERY_LOW = EventPriority(400)
EventPriority.LOWEST = EventPriority(sys.maxsize)
