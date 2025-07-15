# ABOUTME: In-memory implementation of AbstractMetadataRepository
# ABOUTME: Provides key-value storage with TTL support for metadata and configuration

from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import threading
import time
import asyncio


from core.interfaces.storage.metadata_repository import AbstractMetadataRepository
from core.exceptions import StorageError


class InMemoryMetadataRepository(AbstractMetadataRepository):
    """
    In-memory implementation of AbstractMetadataRepository.

    This repository provides efficient key-value storage for metadata with
    TTL (Time-To-Live) support and specialized methods for synchronization
    and backfill status tracking.

    Features:
    - Key-value storage with dictionary-based data
    - TTL support with automatic expiration
    - Prefix-based key filtering
    - Specialized sync time and backfill status methods
    - Thread-safe operations
    - Automatic cleanup of expired entries
    """

    def __init__(self, cleanup_interval: float = 60.0, max_keys: int = 10000):
        """
        Initialize the in-memory metadata repository.

        Args:
            cleanup_interval: Interval in seconds for automatic cleanup of expired entries
            max_keys: Maximum number of keys to store (to prevent memory issues)
        """
        self.cleanup_interval = cleanup_interval
        self.max_keys = max_keys

        # Main storage: key -> {value, created_at}
        self._data: Dict[str, Dict[str, Any]] = {}

        # TTL storage: key -> expiration_timestamp
        self._ttl: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Repository state
        self._closed = False

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        # Don't start cleanup task during initialization - it will be started on first operation

    def _start_cleanup_task(self) -> None:
        """Start the automatic cleanup task."""
        if self._closed:
            return

        try:
            # Only start if there's no active task and we have an event loop
            if (self._cleanup_task is None or self._cleanup_task.done()) and not self._closed:
                self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
        except RuntimeError:
            # No event loop available - cleanup task will be started on first operation
            pass

    async def _cleanup_expired_entries(self) -> None:
        """Periodically clean up expired entries."""
        while not self._closed:
            try:
                await asyncio.sleep(self.cleanup_interval)

                if self._closed:
                    break

                current_time = time.time()
                expired_keys = []

                with self._lock:
                    for key, expiry_time in self._ttl.items():
                        if current_time > expiry_time:
                            expired_keys.append(key)

                    # Remove expired entries
                    for key in expired_keys:
                        self._data.pop(key, None)
                        self._ttl.pop(key, None)

            except asyncio.CancelledError:
                break
            except Exception:
                # Continue cleanup even if there's an error
                continue

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Sets or updates a value for a given key.

        Args:
            key: The unique key for the metadata entry.
            value: The JSON-serializable dictionary to store.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        # Start cleanup task if not already running
        self._start_cleanup_task()

        if not isinstance(key, str) or not key:
            raise StorageError("Key must be a non-empty string", code="INVALID_KEY")

        if not isinstance(value, dict):
            raise StorageError("Value must be a dictionary", code="INVALID_VALUE")

        with self._lock:
            # Check storage limits
            if key not in self._data and len(self._data) >= self.max_keys:
                raise StorageError(f"Maximum keys limit ({self.max_keys}) exceeded", code="STORAGE_LIMIT_EXCEEDED")

            # Store the value with metadata
            self._data[key] = {"value": value, "created_at": datetime.now(UTC), "updated_at": datetime.now(UTC)}

            # Remove from TTL if it exists (regular set doesn't have TTL)
            self._ttl.pop(key, None)

    async def get(self, key: str) -> Dict[str, Any] | None:
        """
        Retrieves the value associated with a given key.

        Args:
            key: The key of the metadata to retrieve.

        Returns:
            The deserialized dictionary value, or None if the key is not found.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        # Start cleanup task if not already running
        self._start_cleanup_task()

        if not isinstance(key, str) or not key:
            raise StorageError("Key must be a non-empty string", code="INVALID_KEY")

        with self._lock:
            # Check if key exists
            if key not in self._data:
                return None

            # Check TTL expiration
            if key in self._ttl:
                current_time = time.time()
                if current_time > self._ttl[key]:
                    # Remove expired entry
                    self._data.pop(key, None)
                    self._ttl.pop(key, None)
                    return None

            value = self._data[key]["value"]
            return value if value is not None else None

    async def exists(self, key: str) -> bool:
        """
        Checks if a key exists in the repository.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if not isinstance(key, str) or not key:
            raise StorageError("Key must be a non-empty string", code="INVALID_KEY")

        with self._lock:
            # Check if key exists
            if key not in self._data:
                return False

            # Check TTL expiration
            if key in self._ttl:
                current_time = time.time()
                if current_time > self._ttl[key]:
                    # Remove expired entry
                    self._data.pop(key, None)
                    self._ttl.pop(key, None)
                    return False

            return True

    async def delete(self, key: str) -> bool:
        """
        Deletes a key-value pair from the repository.

        Args:
            key: The key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.

        Raises:
            StorageError: If the deletion fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if not isinstance(key, str) or not key:
            raise StorageError("Key must be a non-empty string", code="INVALID_KEY")

        with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            self._ttl.pop(key, None)
            return existed

    async def list_keys(self, prefix: str | None = None) -> List[str]:
        """
        Lists all keys, optionally filtering by a prefix.

        Args:
            prefix: If provided, only keys starting with this prefix will be returned.

        Returns:
            A list of keys matching the prefix.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        with self._lock:
            current_time = time.time()
            valid_keys = []

            for key in self._data.keys():
                # Check TTL expiration
                if key in self._ttl and current_time > self._ttl[key]:
                    continue

                # Apply prefix filter
                if prefix is None or key.startswith(prefix):
                    valid_keys.append(key)

            return sorted(valid_keys)

    async def set_with_ttl(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """
        Sets a key-value pair with a Time-To-Live (TTL).

        Args:
            key: The unique key for the metadata entry.
            value: The JSON-serializable dictionary to store.
            ttl_seconds: The time-to-live for the entry, in seconds.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        if not isinstance(key, str) or not key:
            raise StorageError("Key must be a non-empty string", code="INVALID_KEY")

        if not isinstance(value, dict):
            raise StorageError("Value must be a dictionary", code="INVALID_VALUE")

        if ttl_seconds <= 0:
            raise StorageError("TTL must be positive", code="INVALID_TTL")

        with self._lock:
            # Check storage limits
            if key not in self._data and len(self._data) >= self.max_keys:
                raise StorageError(f"Maximum keys limit ({self.max_keys}) exceeded", code="STORAGE_LIMIT_EXCEEDED")

            # Store the value with metadata
            self._data[key] = {"value": value, "created_at": datetime.now(UTC), "updated_at": datetime.now(UTC)}

            # Set TTL
            self._ttl[key] = time.time() + ttl_seconds

    async def get_last_sync_time(
        self,
        symbol: str,
        data_type: str,
    ) -> datetime | None:
        """
        Retrieves the last synchronization timestamp for a specific data type and symbol.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").
            data_type: The type of data that was synced (e.g., "trades", "klines_1m").

        Returns:
            The datetime of the last sync, or None if no sync has occurred.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        key = f"sync_time:{symbol}:{data_type}"
        data = await self.get(key)

        if data is None:
            return None

        timestamp_str = data.get("timestamp")
        if timestamp_str is None:
            return None

        try:
            # Parse ISO format timestamp
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    async def set_last_sync_time(
        self,
        symbol: str,
        data_type: str,
        timestamp: datetime,
    ) -> None:
        """
        Sets the last synchronization timestamp for a specific data type and symbol.

        Args:
            symbol: The trading symbol.
            data_type: The type of data being synced.
            timestamp: The timestamp of the last successful sync.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        key = f"sync_time:{symbol}:{data_type}"
        value = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "data_type": data_type,
            "set_at": datetime.now(UTC).isoformat(),
        }

        await self.set(key, value)

    async def get_backfill_status(
        self,
        symbol: str,
        data_type: str,
    ) -> Dict[str, Any] | None:
        """
        Retrieves the backfill status for a given data type and symbol.

        Args:
            symbol: The trading symbol.
            data_type: The type of data being backfilled.

        Returns:
            A dictionary containing the backfill status, or None if no status exists.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        key = f"backfill_status:{symbol}:{data_type}"
        result = await self.get(key)
        return result if result is not None else None

    async def set_backfill_status(
        self,
        symbol: str,
        data_type: str,
        status: Dict[str, Any],
    ) -> None:
        """
        Sets or updates the backfill status.

        Args:
            symbol: The trading symbol.
            data_type: The type of data being backfilled.
            status: A dictionary containing the current backfill status.

        Raises:
            StorageError: If the operation fails.
        """
        if self._closed:
            raise StorageError("Repository is closed", code="REPOSITORY_CLOSED")

        key = f"backfill_status:{symbol}:{data_type}"

        # Add metadata to the status
        enhanced_status = {
            **status,
            "symbol": symbol,
            "data_type": data_type,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        await self.set(key, enhanced_status)

    async def close(self) -> None:
        """
        Closes the repository and releases any underlying resources.
        """
        self._closed = True

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear all data
        with self._lock:
            self._data.clear()
            self._ttl.clear()

    # Additional helper methods for testing and debugging

    async def get_all_keys(self) -> List[str]:
        """Get all keys in the repository (including expired ones)."""
        with self._lock:
            return list(self._data.keys())

    async def get_ttl_keys(self) -> List[str]:
        """Get all keys that have TTL set."""
        with self._lock:
            return list(self._ttl.keys())

    async def get_key_info(self, key: str) -> Dict[str, Any] | None:
        """Get detailed information about a key."""
        with self._lock:
            if key not in self._data:
                return None

            info = {
                "key": key,
                "exists": True,
                "has_ttl": key in self._ttl,
                "created_at": self._data[key]["created_at"].isoformat(),
                "updated_at": self._data[key]["updated_at"].isoformat(),
            }

            if key in self._ttl:
                current_time = time.time()
                expiry_time = self._ttl[key]
                info.update(
                    {
                        "expires_at": datetime.fromtimestamp(expiry_time, UTC).isoformat(),
                        "ttl_seconds": max(0, expiry_time - current_time),
                        "is_expired": current_time > expiry_time,
                    }
                )

            return info

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = 0

            for key, expiry_time in self._ttl.items():
                if current_time > expiry_time:
                    expired_count += 1

            return {
                "total_keys": len(self._data),
                "ttl_keys": len(self._ttl),
                "expired_keys": expired_count,
                "max_keys": self.max_keys,
                "cleanup_interval": self.cleanup_interval,
                "is_closed": self._closed,
            }

    async def clear_all(self) -> None:
        """Clear all data from the repository."""
        with self._lock:
            self._data.clear()
            self._ttl.clear()

    async def force_cleanup(self) -> int:
        """Force immediate cleanup of expired entries and return count of removed entries."""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, expiry_time in self._ttl.items():
                if current_time > expiry_time:
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                self._data.pop(key, None)
                self._ttl.pop(key, None)

        return len(expired_keys)

    async def get_sync_times(self, symbol: str | None = None) -> Dict[str, Any]:
        """Get all sync times, optionally filtered by symbol."""
        sync_keys = await self.list_keys("sync_time:")
        sync_times = {}

        for key in sync_keys:
            data = await self.get(key)
            if data:
                # Parse key to extract symbol and data_type
                parts = key.split(":", 2)
                if len(parts) == 3:
                    _, key_symbol, data_type = parts
                    if symbol is None or key_symbol == symbol:
                        sync_times[f"{key_symbol}:{data_type}"] = data

        return sync_times

    async def get_backfill_statuses(self, symbol: str | None = None) -> Dict[str, Any]:
        """Get all backfill statuses, optionally filtered by symbol."""
        backfill_keys = await self.list_keys("backfill_status:")
        backfill_statuses = {}

        for key in backfill_keys:
            data = await self.get(key)
            if data:
                # Parse key to extract symbol and data_type
                parts = key.split(":", 2)
                if len(parts) == 3:
                    _, key_symbol, data_type = parts
                    if symbol is None or key_symbol == symbol:
                        backfill_statuses[f"{key_symbol}:{data_type}"] = data

        return backfill_statuses
