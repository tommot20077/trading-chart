# ABOUTME: In-memory implementation of AbstractEventSerializer using JSON serialization
# ABOUTME: Provides JSON-based event serialization with zero external dependencies

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Type
from uuid import UUID

from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.models.data.event import BaseEvent
from core.models.event.trade_event import TradeEvent
from core.models.event.Kline_event import KlineEvent
from core.models.event.connection_event import ConnectionEvent
from core.models.event.error_event import ErrorEvent
from core.models.event.event_type import EventType
from core.models.event.event_priority import EventPriority
from core.models.data.enum import KlineInterval, TradeSide, AssetClass
from core.models.network.enum import ConnectionStatus
from core.exceptions.base import EventSerializationError, EventDeserializationError


class MemoryEventSerializer(AbstractEventSerializer):
    """
    In-memory implementation of AbstractEventSerializer using JSON.

    This implementation provides JSON-based serialization and deserialization
    of BaseEvent objects. It handles all standard event types and their payloads,
    including proper conversion of complex types like Decimal, datetime, and enums.

    Features:
    - JSON-based serialization (human-readable and debuggable)
    - Support for all core event types (Trade, Kline, Connection, Error)
    - Proper handling of Decimal, datetime, UUID, and enum types
    - Automatic event type detection and reconstruction
    - Comprehensive error handling with detailed context
    - Zero external dependencies (uses only Python standard library)
    """

    def __init__(self, pretty_print: bool = False, ensure_ascii: bool = False):
        """
        Initialize the memory event serializer.

        Args:
            pretty_print: If True, format JSON with indentation for readability
            ensure_ascii: If True, escape non-ASCII characters in JSON output
        """
        self._pretty_print = pretty_print
        self._ensure_ascii = ensure_ascii

        # Mapping of event types to their corresponding classes
        self._event_type_mapping: Dict[EventType, Type[BaseEvent]] = {
            EventType.TRADE: TradeEvent,
            EventType.KLINE: KlineEvent,
            EventType.CONNECTION: ConnectionEvent,
            EventType.ERROR: ErrorEvent,
        }

    def serialize(self, event: BaseEvent) -> bytes:
        """
        Serializes a BaseEvent object into JSON bytes.

        Args:
            event: The BaseEvent object to be serialized

        Returns:
            A bytes object containing the JSON representation of the event

        Raises:
            EventSerializationError: If the event cannot be serialized
        """
        try:
            # Convert event to dictionary with proper type handling
            event_dict = self._event_to_dict(event)

            # Add metadata for deserialization
            event_dict["__serializer_version__"] = "1.0"
            event_dict["__event_class__"] = event.__class__.__name__

            # Serialize to JSON
            json_kwargs: dict[str, Any] = {
                "ensure_ascii": self._ensure_ascii,
            }

            if self._pretty_print:
                json_kwargs["indent"] = 2
            else:
                json_kwargs["separators"] = (",", ":")

            json_str = json.dumps(event_dict, **json_kwargs)
            return json_str.encode("utf-8")

        except (TypeError, ValueError, AttributeError) as e:
            raise EventSerializationError(
                f"Failed to serialize event: {e}",
                code="SERIALIZATION_FAILED",
                details={
                    "event_type": str(event.event_type),
                    "event_id": event.event_id,
                    "error": str(e),
                },
            )
        except Exception as e:
            raise EventSerializationError(
                f"Unexpected error during serialization: {e}",
                code="UNEXPECTED_SERIALIZATION_ERROR",
                details={
                    "event_type": str(event.event_type),
                    "event_id": event.event_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def deserialize(self, data: bytes) -> BaseEvent:
        """
        Deserializes JSON bytes back into a BaseEvent object.

        Args:
            data: The bytes object containing the JSON serialized event data

        Returns:
            A reconstructed BaseEvent object

        Raises:
            EventDeserializationError: If the byte data cannot be deserialized
        """
        try:
            # Decode bytes to string
            json_str = data.decode("utf-8")

            # Parse JSON
            event_dict = json.loads(json_str)

            # Validate serializer metadata
            if not isinstance(event_dict, dict):
                raise EventDeserializationError("Invalid event data: expected dictionary", code="INVALID_EVENT_FORMAT")

            # Check serializer version compatibility
            serializer_version = event_dict.get("__serializer_version__")
            if serializer_version != "1.0":
                raise EventDeserializationError(
                    f"Unsupported serializer version: {serializer_version}",
                    code="UNSUPPORTED_VERSION",
                    details={"version": serializer_version},
                )

            # Remove metadata
            event_class_name = event_dict.pop("__event_class__", None)
            event_dict.pop("__serializer_version__", None)

            # Convert dictionary back to event object
            return self._dict_to_event(event_dict, event_class_name)

        except json.JSONDecodeError as e:
            raise EventDeserializationError(
                f"Invalid JSON data: {e}",
                code="INVALID_JSON",
                details={"error": str(e), "position": getattr(e, "pos", None)},
            )
        except UnicodeDecodeError as e:
            raise EventDeserializationError(
                f"Invalid UTF-8 encoding: {e}", code="INVALID_ENCODING", details={"error": str(e)}
            )
        except EventDeserializationError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            raise EventDeserializationError(
                f"Unexpected error during deserialization: {e}",
                code="UNEXPECTED_DESERIALIZATION_ERROR",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _event_to_dict(self, event: BaseEvent) -> Dict[str, Any]:
        """
        Convert a BaseEvent to a JSON-serializable dictionary.

        Args:
            event: The BaseEvent to convert

        Returns:
            A dictionary representation of the event
        """
        # Start with the basic event data
        event_dict = event.model_dump()

        # Handle special types that need custom serialization
        event_dict = self._serialize_special_types(event_dict)

        return dict(event_dict)

    def _dict_to_event(self, event_dict: Dict[str, Any], event_class_name: str | None = None) -> BaseEvent:
        """
        Convert a dictionary back to a BaseEvent object.

        Args:
            event_dict: The dictionary representation of the event
            event_class_name: Optional class name hint for reconstruction

        Returns:
            A reconstructed BaseEvent object
        """
        # Deserialize special types
        event_dict = self._deserialize_special_types(event_dict)

        # Determine the event type
        event_type_str = event_dict.get("event_type")
        if not event_type_str:
            raise EventDeserializationError("Missing event_type field", code="MISSING_EVENT_TYPE")

        try:
            event_type = EventType(event_type_str)
        except ValueError:
            raise EventDeserializationError(
                f"Unknown event type: {event_type_str}",
                code="UNKNOWN_EVENT_TYPE",
                details={"event_type": event_type_str},
            )

        # Get the appropriate event class
        event_class = self._event_type_mapping.get(event_type)
        if not event_class:
            # Fallback to generic BaseEvent for unknown types
            event_class = BaseEvent

        # Reconstruct the event object
        try:
            # Special handling for events with custom constructors
            if event_type == EventType.CONNECTION:
                # ConnectionEvent requires status parameter
                status_value = event_dict.get("status")
                if not status_value:
                    # Try to get status from data field
                    data_dict = event_dict.get("data", {})
                    if isinstance(data_dict, dict):
                        status_value = data_dict.get("status")

                if status_value:
                    # Convert string back to ConnectionStatus enum if needed
                    from core.models.network.enum import ConnectionStatus

                    if isinstance(status_value, str):
                        try:
                            status = ConnectionStatus(status_value)
                        except ValueError:
                            # If invalid status string, default to DISCONNECTED
                            status = ConnectionStatus.DISCONNECTED
                    else:
                        status = status_value

                    # Keep the original data and pass status separately
                    event_dict_copy = event_dict.copy()
                    event_dict_copy["status"] = status
                    return event_class(**event_dict_copy)
                else:
                    # If no status found anywhere, default to DISCONNECTED
                    from core.models.network.enum import ConnectionStatus

                    status = ConnectionStatus.DISCONNECTED
                    event_dict_copy = event_dict.copy()
                    event_dict_copy["status"] = status
                    return event_class(**event_dict_copy)
            elif event_type == EventType.ERROR:
                # ErrorEvent requires error parameter from data
                error_data = event_dict.get("data", {})
                error_message = error_data.get("error")
                error_code = error_data.get("error_code")
                if error_message:
                    # Pass error info as constructor parameters
                    event_dict_copy = event_dict.copy()
                    # Remove error info from data since it will be passed as constructor params
                    data_copy = error_data.copy()
                    data_copy.pop("error", None)
                    data_copy.pop("error_code", None)
                    event_dict_copy["data"] = data_copy
                    return ErrorEvent(error=error_message, error_code=error_code, **event_dict_copy)

            # Default reconstruction for other event types
            return event_class(**event_dict)
        except Exception as e:
            raise EventDeserializationError(
                f"Failed to reconstruct {event_class.__name__}: {e}",
                code="RECONSTRUCTION_FAILED",
                details={
                    "event_class": event_class.__name__,
                    "event_type": event_type_str,
                    "error": str(e),
                },
            )

    def _serialize_special_types(self, obj: Any) -> Any:
        """
        Recursively serialize special types to JSON-compatible formats.

        Args:
            obj: The object to serialize

        Returns:
            A JSON-compatible representation
        """
        if isinstance(obj, dict):
            return {key: self._serialize_special_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_special_types(item) for item in obj]
        elif isinstance(obj, Decimal):
            return {"__type__": "Decimal", "__value__": str(obj)}
        elif isinstance(obj, datetime):
            return {"__type__": "datetime", "__value__": obj.isoformat()}
        elif isinstance(obj, UUID):
            return {"__type__": "UUID", "__value__": str(obj)}
        elif isinstance(obj, EventPriority):
            return {"__type__": "EventPriority", "__value__": obj.value}
        elif isinstance(obj, (EventType, KlineInterval, TradeSide, AssetClass, ConnectionStatus)):
            return {"__type__": type(obj).__name__, "__value__": obj.value}
        elif hasattr(obj, "model_dump"):  # Pydantic models
            return self._serialize_special_types(obj.model_dump())
        else:
            return obj

    def _deserialize_special_types(self, obj: Any) -> Any:
        """
        Recursively deserialize special types from JSON-compatible formats.

        Args:
            obj: The object to deserialize

        Returns:
            The original type representation
        """
        if isinstance(obj, dict):
            # Check if this is a special type marker
            if "__type__" in obj and "__value__" in obj:
                type_name = obj["__type__"]
                value = obj["__value__"]

                if type_name == "Decimal":
                    return Decimal(value)
                elif type_name == "datetime":
                    return datetime.fromisoformat(value)
                elif type_name == "UUID":
                    return UUID(value)
                elif type_name == "EventPriority":
                    return EventPriority(value)
                elif type_name == "EventType":
                    return EventType(value)
                elif type_name == "KlineInterval":
                    return KlineInterval(value)
                elif type_name == "TradeSide":
                    return TradeSide(value)
                elif type_name == "AssetClass":
                    return AssetClass(value)
                elif type_name == "ConnectionStatus":
                    return ConnectionStatus(value)
                else:
                    # Unknown special type, return as-is
                    return obj
            else:
                # Regular dictionary, recurse into values
                return {key: self._deserialize_special_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_special_types(item) for item in obj]
        else:
            return obj

    def get_supported_event_types(self) -> list[EventType]:
        """
        Get the list of event types supported by this serializer.

        Returns:
            A list of supported EventType values
        """
        return list(self._event_type_mapping.keys())

    def register_event_type(self, event_type: EventType, event_class: Type[BaseEvent]) -> None:
        """
        Register a new event type with its corresponding class.

        Args:
            event_type: The EventType enum value
            event_class: The BaseEvent subclass for this event type
        """
        self._event_type_mapping[event_type] = event_class

    def validate_event(self, event: BaseEvent) -> bool:
        """
        Validate that an event can be serialized and deserialized correctly.

        Args:
            event: The event to validate

        Returns:
            True if the event can be round-trip serialized, False otherwise
        """
        try:
            # Attempt round-trip serialization
            serialized = self.serialize(event)
            deserialized = self.deserialize(serialized)

            # Basic validation - check that key fields match
            return (
                event.event_id == deserialized.event_id
                and event.event_type == deserialized.event_type
                and event.source == deserialized.source
            )
        except (EventSerializationError, EventDeserializationError):
            return False

    def get_serialization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the serializer configuration.

        Returns:
            A dictionary containing serializer statistics and configuration
        """
        return {
            "serializer_type": "JSON",
            "pretty_print": self._pretty_print,
            "ensure_ascii": self._ensure_ascii,
            "supported_event_types": [str(et) for et in self.get_supported_event_types()],
            "version": "1.0",
        }
