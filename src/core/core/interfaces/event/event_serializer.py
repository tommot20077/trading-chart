# ABOUTME: Abstract event serializer interface for event data serialization and deserialization
# ABOUTME: Defines the contract for components that convert events between object and byte representations

from abc import ABC, abstractmethod

from core.models.data.event import BaseEvent


class AbstractEventSerializer(ABC):
    """
    [L0] Abstract base class for event serialization.

    This interface defines the contract for converting `BaseEvent` objects
    into a bytes stream (serialization) and converting bytes streams back into
    `BaseEvent` objects (deserialization). This is essential for persisting
    events to storage or transmitting them over a network.
    """

    @abstractmethod
    def serialize(self, event: BaseEvent) -> bytes:
        """
        Serializes a `BaseEvent` object into a byte representation.

        This method converts a structured `BaseEvent` object into a byte stream,
        making it suitable for storage or network transmission. The specific byte
        format (e.g., JSON, MessagePack, Protobuf) depends on the concrete implementation
        of this abstract class.

        Args:
            event (BaseEvent): The `BaseEvent` object to be serialized.

        Returns:
            bytes: A `bytes` object representing the serialized event.

        Raises:
            EventSerializationError: If the event cannot be serialized (e.g., due to
                                     unsupported data types, malformed event data, or
                                     issues with the underlying serialization library).
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> BaseEvent:
        """
        Deserializes a byte representation back into a `BaseEvent` object.

        This method reconstructs the original `BaseEvent` object from the byte stream.
        It assumes the byte data was serialized by a compatible serializer. This is crucial
        for consuming events from storage or network.

        Args:
            data (bytes): The `bytes` object containing the serialized event data.

        Returns:
            BaseEvent: A reconstructed `BaseEvent` object.

        Raises:
            EventDeserializationError: If the byte data is malformed, corrupted,
                                       or cannot be deserialized into a valid event (e.g., due to
                                       schema mismatch or invalid data format).
        """
        pass
