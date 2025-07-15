# ABOUTME: Defines the abstract interface for event storage operations.
# ABOUTME: This module specifies the contract for storing, retrieving, and managing events.

from abc import abstractmethod, ABC
from typing import AsyncIterator

from core.interfaces.event.event_serializer import AbstractEventSerializer
from core.models.data.event import BaseEvent
from core.models.event.event_query import EventQuery
from core.models.event.event_storage_stats import EventStorageStats


class AbstractEventStorage(ABC):
    """
    Abstract base class for event storage backends.

    This interface defines the contract for implementing event persistence and
    retrieval functionality. Concrete implementations are responsible for
    interacting with specific storage technologies (e.g., relational databases,
    NoSQL databases, file systems, in-memory stores) to store and query events.

    Architecture Note: This is a [L0] interface that only depends on core data models
    (`BaseEvent`, `EventQuery`, `EventStorageStats`) and provides clean abstractions
    for higher-level [L1] storage implementations.
    """

    def __init__(self, serializer: "AbstractEventSerializer"):
        """
        Initializes the AbstractEventStorage with a serializer.

        All concrete event storage implementations will need a serializer to
        convert events to/from a storable format.

        Args:
            serializer (AbstractEventSerializer): An instance of `AbstractEventSerializer` used for
                serializing events before storage and deserializing them after retrieval.
        """
        self.serializer = serializer

    @abstractmethod
    async def store_event(self, event: BaseEvent) -> str:
        """
        Stores a single event in the storage backend.

        The event is typically serialized before storage. This method should
        return a unique identifier for the stored event within the storage system.

        Args:
            event (BaseEvent): The `BaseEvent` object to be stored.

        Returns:
            str: A string representing the unique storage ID of the stored event.

        Raises:
            EventStorageError: If the event cannot be stored (e.g., connection issues,
                permission problems, storage full).
        """
        pass

    @abstractmethod
    async def store_events(self, events: list[BaseEvent]) -> list[str]:
        """
        Stores multiple events in the storage backend as a batch operation.

        This method should optimize for storing multiple events efficiently.

        Args:
            events (list[BaseEvent]): A list of `BaseEvent` objects to be stored.

        Returns:
            list[str]: A list of strings, where each string is the unique storage ID for the
                corresponding stored event in the input list.

        Raises:
            EventStorageError: If any event in the batch cannot be stored.
        """
        pass

    @abstractmethod
    async def retrieve_event(self, storage_id: str) -> BaseEvent | None:
        """
        Retrieves a single event from the storage backend using its unique storage ID.

        The retrieved data is typically deserialized back into a `BaseEvent` object.

        Args:
            storage_id (str): The unique string identifier of the event to retrieve.

        Returns:
            BaseEvent | None: The deserialized `BaseEvent` object if found, otherwise `None`.

        Raises:
            EventStorageError: If the event cannot be retrieved (e.g., data corruption,
                connection issues, permissions).
        """
        pass

    @abstractmethod
    async def query_events(self, query: EventQuery) -> list[BaseEvent]:
        """
        Queries events from the storage backend based on the provided `EventQuery` criteria.

        This method should apply filtering, ordering, and pagination as specified
        in the `EventQuery` object.

        Args:
            query (EventQuery): An `EventQuery` object specifying the filtering, ordering,
                and pagination criteria for event retrieval.

        Returns:
            list[BaseEvent]: A list of `BaseEvent` objects that match the query criteria,
                sorted and paginated as specified.

        Raises:
            EventStorageError: If the query operation fails (e.g., invalid query,
                connection issues).
        """
        pass

    @abstractmethod
    def stream_events(self, query: EventQuery) -> AsyncIterator[BaseEvent]:
        """
        Streams events from the storage backend based on the provided `EventQuery` criteria.

        This method is an asynchronous generator, yielding events one by one
        as they are retrieved and filtered. This is efficient for processing
        large datasets without loading all events into memory simultaneously.

        Args:
            query (EventQuery): An `EventQuery` object specifying the filtering, ordering,
                and pagination criteria for event streaming.

        Yields:
            BaseEvent: Each event that matches the query criteria.

        Raises:
            EventStorageError: If the streaming operation encounters an error.
        """
        pass

    @abstractmethod
    async def delete_event(self, storage_id: str) -> bool:
        """
        Deletes a single event from the storage backend using its unique storage ID.

        Removes a specific event from the storage system based on its unique identifier.

        Args:
            storage_id (str): The unique string identifier of the event to delete.

        Returns:
            bool: `True` if the event was found and successfully deleted, `False` otherwise.

        Raises:
            EventStorageError: If the deletion operation fails (e.g., connection issues,
                permissions).
        """
        pass

    @abstractmethod
    async def delete_events(self, query: EventQuery) -> int:
        """
        Deletes multiple events from the storage backend based on the provided `EventQuery` criteria.

        This method should efficiently remove all events that match the specified query.

        Args:
            query (EventQuery): An `EventQuery` object specifying the criteria for events to be deleted.

        Returns:
            int: The number of events that were found and successfully deleted.

        Raises:
            EventStorageError: If the batch deletion operation fails.
        """
        pass

    @abstractmethod
    async def get_stats(self) -> EventStorageStats:
        """
        Retrieves comprehensive statistics about the event storage backend.

        This method provides metrics such as total event count, event counts by type,
        total storage size, and temporal information (oldest/newest event timestamps).
        These statistics are vital for monitoring and capacity planning.

        Returns:
            EventStorageStats: An `EventStorageStats` object containing various statistics about the storage.

        Raises:
            EventStorageError: If statistics cannot be retrieved (e.g., connection issues).
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Performs a health check on the event storage backend.

        This method should verify the connectivity and basic functionality of the
        underlying storage system.

        Returns:
            bool: `True` if the storage backend is healthy and operational, `False` otherwise.

        Raises:
            EventStorageError: If the health check itself encounters an unrecoverable error.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Closes the event storage backend and cleans up any associated resources.

        This asynchronous method should release database connections, close file handles,
        stop background processes, and perform any necessary shutdown procedures
        to gracefully terminate the storage component.

        Returns:
            None: This method does not return any value.
        """
        pass
