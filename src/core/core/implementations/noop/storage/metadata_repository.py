# ABOUTME: NoOp implementation of AbstractMetadataRepository that provides fake storage
# ABOUTME: Provides minimal metadata storage functionality for testing scenarios

from typing import Any
from datetime import datetime

from core.interfaces.storage.metadata_repository import AbstractMetadataRepository


class NoOpMetadataRepository(AbstractMetadataRepository):
    """
    No-operation implementation of AbstractMetadataRepository.

    This implementation provides minimal metadata storage functionality that
    simulates storage operations without actually persisting data. It's useful
    for testing, performance benchmarking, and scenarios where metadata storage
    is not required.

    Features:
    - Simulates successful storage operations
    - No actual data persistence
    - Minimal resource usage
    - Fast execution
    - No side effects

    Use Cases:
    - Testing environments where storage should be bypassed
    - Performance benchmarking without storage overhead
    - Development environments where storage is not needed
    - Fallback when storage systems are unavailable
    """

    def __init__(self):
        """Initialize the no-operation metadata repository."""
        self._closed = False
        # Track "stored" keys for simulation purposes
        self._fake_keys: set[str] = set()

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set a key-value pair - simulates storage without persistence.

        This implementation simulates successful storage without actually
        persisting the data.

        Args:
            key: The key to store
            value: The value to store (ignored)
            ttl: Optional time-to-live in seconds (ignored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        # Simulate storage by tracking the key
        self._fake_keys.add(key)

    async def get(self, key: str) -> Any | None:
        """
        Get a value by key - always returns None (no data stored).

        This implementation always returns None since no data is actually
        stored in the NoOp implementation.

        Args:
            key: The key to retrieve

        Returns:
            None (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        # Always return None - no data is actually stored
        return None

    async def delete(self, key: str) -> bool:
        """
        Delete a key - simulates deletion.

        This implementation simulates successful deletion if the key
        was previously "stored".

        Args:
            key: The key to delete

        Returns:
            True if key was "stored", False otherwise
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        if key in self._fake_keys:
            self._fake_keys.remove(key)
            return True
        return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists - checks fake storage.

        This implementation checks if the key was previously "stored"
        in the fake storage.

        Args:
            key: The key to check

        Returns:
            True if key was "stored", False otherwise
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        return key in self._fake_keys

    async def get_keys(self, pattern: str | None = None) -> list[str]:
        """
        Get all keys matching pattern - returns fake stored keys.

        This implementation returns the list of fake stored keys.

        Args:
            pattern: Optional pattern to match (ignored in NoOp)

        Returns:
            List of fake stored keys
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        return list(self._fake_keys)

    async def set_with_expiry(self, key: str, value: Any, expire_at: datetime) -> None:
        """
        Set a key-value pair with expiry - simulates storage.

        This implementation simulates storage with expiry without actually
        implementing expiry logic.

        Args:
            key: The key to store
            value: The value to store (ignored)
            expire_at: Expiry datetime (ignored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        # Simulate storage by tracking the key
        self._fake_keys.add(key)

    async def get_ttl(self, key: str) -> int | None:
        """
        Get time-to-live for a key - returns fake TTL.

        This implementation returns a fake TTL if the key exists.

        Args:
            key: The key to check

        Returns:
            Fake TTL in seconds if key exists, None otherwise
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        if key in self._fake_keys:
            return 3600  # Fake 1 hour TTL
        return None

    async def increment(self, key: str, delta: int = 1) -> int:
        """
        Increment a numeric value - returns fake incremented value.

        This implementation returns a fake incremented value and tracks
        the key as stored.

        Args:
            key: The key to increment
            delta: The increment value

        Returns:
            Fake incremented value
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        self._fake_keys.add(key)
        return delta  # Return the delta as the fake new value

    async def get_multiple(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values - returns empty dict (no data stored).

        This implementation returns an empty dict since no data is
        actually stored.

        Args:
            keys: List of keys to retrieve

        Returns:
            Empty dict (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        # Return empty dict - no data is actually stored
        return {}

    async def set_multiple(self, data: dict[str, Any], ttl: int | None = None) -> None:
        """
        Set multiple key-value pairs - simulates storage.

        This implementation simulates storage of multiple keys.

        Args:
            data: Dictionary of key-value pairs to store
            ttl: Optional time-to-live in seconds (ignored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        # Simulate storage by tracking the keys
        self._fake_keys.update(data.keys())

    async def delete_multiple(self, keys: list[str]) -> int:
        """
        Delete multiple keys - simulates deletion.

        This implementation simulates deletion of multiple keys.

        Args:
            keys: List of keys to delete

        Returns:
            Number of keys that were "stored" and deleted
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        deleted_count = 0
        for key in keys:
            if key in self._fake_keys:
                self._fake_keys.remove(key)
                deleted_count += 1
        return deleted_count

    async def clear_all(self) -> int:
        """
        Clear all data - removes all fake stored keys.

        This implementation clears all fake stored keys.

        Returns:
            Number of keys that were cleared
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        count = len(self._fake_keys)
        self._fake_keys.clear()
        return count

    async def health_check(self) -> bool:
        """
        Perform health check - always returns True.

        This implementation always returns True indicating the repository
        is healthy.

        Returns:
            True (NoOp implementation is always healthy)
        """
        return not self._closed

    async def set_with_ttl(self, key: str, value: Any, ttl_seconds: int) -> None:
        """
        Set a key-value pair with TTL - simulates storage.

        Args:
            key: The key to store
            value: The value to store (ignored)
            ttl_seconds: TTL in seconds (ignored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")
        self._fake_keys.add(key)

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        """
        List keys with optional prefix - returns fake stored keys.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of fake stored keys matching prefix
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")

        if prefix is None:
            return list(self._fake_keys)
        return [key for key in self._fake_keys if key.startswith(prefix)]

    async def set_last_sync_time(self, symbol: str, data_type: str, timestamp: datetime) -> None:
        """
        Set last sync time for a symbol and data type - simulates storage.

        Args:
            symbol: Trading symbol
            data_type: The type of data being synced
            timestamp: Last sync timestamp (ignored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")
        self._fake_keys.add(f"last_sync:{symbol}:{data_type}")

    async def get_last_sync_time(self, symbol: str, data_type: str) -> datetime | None:
        """
        Get last sync time for a symbol and data type - returns None (no data stored).

        Args:
            symbol: Trading symbol
            data_type: The type of data being synced

        Returns:
            None (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")
        return None

    async def set_backfill_status(self, symbol: str, data_type: str, status: dict[str, Any]) -> None:
        """
        Set backfill status for a symbol and data type - simulates storage.

        Args:
            symbol: Trading symbol
            data_type: The type of data being backfilled
            status: Backfill status dictionary (ignored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")
        self._fake_keys.add(f"backfill_status:{symbol}:{data_type}")

    async def get_backfill_status(self, symbol: str, data_type: str) -> dict[str, Any] | None:
        """
        Get backfill status for a symbol and data type - returns None (no data stored).

        Args:
            symbol: Trading symbol
            data_type: The type of data being backfilled

        Returns:
            None (no data is actually stored)
        """
        if self._closed:
            raise RuntimeError("Metadata repository is closed")
        return None

    async def close(self) -> None:
        """
        Close the repository - sets closed flag and clears fake data.
        """
        self._closed = True
        self._fake_keys.clear()
