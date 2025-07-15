from abc import ABC, abstractmethod

from core.models.data.base import BaseEvent


class AbstractEventSerializer(ABC):
    """
    [L0] Abstract base class for event serialization.

    This interface defines the contract for converting `BaseEvent` objects
    into a byte stream (serialization) and converting byte streams back into
    `BaseEvent` objects (deserialization). This is essential for persisting
    events to storage or transmitting them over a network.
    """

    @abstractmethod
    def serialize(self, event: BaseEvent) -> bytes:
        """
        Serializes a `BaseEvent` object into a byte representation.

        The specific byte format (e.g., JSON, MessagePack, Protobuf) depends
        on the concrete implementation of this abstract class.

        Args:
            event: The `BaseEvent` object to be serialized.

        Returns:
            A `bytes` object representing the serialized event.

        Raises:
            EventSerializationError: If the event cannot be serialized (e.g., due to
                                     unsupported data types, malformed event data).
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> BaseEvent:
        """
        Deserializes a byte representation back into a `BaseEvent` object.

        This method must be able to reconstruct the original `BaseEvent` object
        from the byte stream, assuming it was serialized by a compatible serializer.

        Args:
            data: The `bytes` object containing the serialized event data.

        Returns:
            A reconstructed `BaseEvent` object.

        Raises:
            EventDeserializationError: If the byte data is malformed, corrupted,
                                       or cannot be deserialized into a valid event.
        """
        pass
