# ABOUTME: Unit tests for InMemoryMetadataRepository implementation
# ABOUTME: Tests all functionality including TTL, sync times, backfill status, and cleanup

import pytest
import asyncio
import time_machine
from datetime import datetime, UTC

from core.implementations.memory.storage.metadata_repository import InMemoryMetadataRepository
from core.exceptions import StorageError


@pytest.fixture
def repository():
    """Create a repository instance for testing."""
    return InMemoryMetadataRepository(cleanup_interval=0.1, max_keys=100)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "version": "1.0.0",
        "config": {"enabled": True, "timeout": 30},
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": {"source": "test", "priority": "high"},
    }


class TestInMemoryMetadataRepository:
    """Test suite for InMemoryMetadataRepository."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repository_initialization(self, repository):
        """Test repository initialization."""
        assert repository.cleanup_interval == 0.1
        assert repository.max_keys == 100
        assert not repository._closed
        assert len(repository._data) == 0
        assert len(repository._ttl) == 0

        # Test memory usage
        memory_usage = await repository.get_memory_usage()
        assert memory_usage["total_keys"] == 0
        assert memory_usage["ttl_keys"] == 0
        assert memory_usage["expired_keys"] == 0
        assert memory_usage["max_keys"] == 100
        assert memory_usage["cleanup_interval"] == 0.1
        assert not memory_usage["is_closed"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_and_get_basic(self, repository, sample_metadata):
        """Test basic set and get operations."""
        key = "test_key"

        await repository.set(key, sample_metadata)

        retrieved = await repository.get(key)
        assert retrieved == sample_metadata

        # Check exists
        exists = await repository.exists(key)
        assert exists is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, repository):
        """Test getting a non-existent key."""
        result = await repository.get("nonexistent_key")
        assert result is None

        exists = await repository.exists("nonexistent_key")
        assert exists is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_update_existing_key(self, repository, sample_metadata):
        """Test updating an existing key."""
        key = "test_key"

        # Set initial value
        await repository.set(key, sample_metadata)

        # Update with new value
        updated_metadata = {"version": "2.0.0", "config": {"enabled": False, "timeout": 60}}

        await repository.set(key, updated_metadata)

        # Should return updated value
        retrieved = await repository.get(key)
        assert retrieved == updated_metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_existing_key(self, repository, sample_metadata):
        """Test deleting an existing key."""
        key = "test_key"

        # Set and verify
        await repository.set(key, sample_metadata)
        assert await repository.exists(key) is True

        # Delete and verify
        deleted = await repository.delete(key)
        assert deleted is True
        assert await repository.exists(key) is False
        assert await repository.get(key) is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, repository):
        """Test deleting a non-existent key."""
        deleted = await repository.delete("nonexistent_key")
        assert deleted is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_keys_all(self, repository, sample_metadata):
        """Test listing all keys."""
        keys = ["key1", "key2", "key3"]

        for key in keys:
            await repository.set(key, sample_metadata)

        listed_keys = await repository.list_keys()
        assert sorted(listed_keys) == sorted(keys)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_keys_with_prefix(self, repository, sample_metadata):
        """Test listing keys with prefix filter."""
        keys = ["app_config", "app_status", "user_data", "user_preferences"]

        for key in keys:
            await repository.set(key, sample_metadata)

        # Test app_ prefix
        app_keys = await repository.list_keys("app_")
        assert sorted(app_keys) == ["app_config", "app_status"]

        # Test user_ prefix
        user_keys = await repository.list_keys("user_")
        assert sorted(user_keys) == ["user_data", "user_preferences"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_keys_empty(self, repository):
        """Test listing keys when repository is empty."""
        keys = await repository.list_keys()
        assert keys == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_with_ttl(self, repository, sample_metadata):
        """Test setting key with TTL."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            key = "ttl_key"
            ttl_seconds = 1

            await repository.set_with_ttl(key, sample_metadata, ttl_seconds)

            # Should exist immediately
            assert await repository.exists(key) is True
            assert await repository.get(key) == sample_metadata

            # Wait for expiration
            traveller.shift(ttl_seconds + 0.1)

            # Should be expired
            assert await repository.exists(key) is False
            assert await repository.get(key) is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_with_ttl_overwrites_normal_key(self, repository, sample_metadata):
        """Test that set_with_ttl overwrites normal keys."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            key = "test_key"

            # Set normal key
            await repository.set(key, sample_metadata)
            assert await repository.exists(key) is True

            # Set with TTL
            ttl_data = {"ttl": "test"}
            await repository.set_with_ttl(key, ttl_data, 1)

            # Should have TTL data
            assert await repository.get(key) == ttl_data

            # Wait for expiration
            traveller.shift(1.1)
            assert await repository.exists(key) is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_overwrites_ttl_key(self, repository, sample_metadata):
        """Test that normal set removes TTL."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            key = "test_key"

            # Set with TTL
            await repository.set_with_ttl(key, {"ttl": "test"}, 1)

            # Overwrite with normal set
            await repository.set(key, sample_metadata)

            # Should not expire
            traveller.shift(1.1)
            assert await repository.exists(key) is True
            assert await repository.get(key) == sample_metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_last_sync_time_nonexistent(self, repository):
        """Test getting sync time for non-existent symbol/data_type."""
        sync_time = await repository.get_last_sync_time("BTCUSDT", "trades")
        assert sync_time is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_and_get_last_sync_time(self, repository):
        """Test setting and getting last sync time."""
        symbol = "BTCUSDT"
        data_type = "trades"
        timestamp = datetime(2024, 1, 1, 12, 30, 0, tzinfo=UTC)

        await repository.set_last_sync_time(symbol, data_type, timestamp)

        retrieved_time = await repository.get_last_sync_time(symbol, data_type)
        assert retrieved_time == timestamp

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_and_get_last_sync_time_multiple(self, repository):
        """Test setting and getting multiple sync times."""
        test_cases = [
            ("BTCUSDT", "trades", datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)),
            ("BTCUSDT", "klines_1m", datetime(2024, 1, 1, 12, 5, 0, tzinfo=UTC)),
            ("ETHUSDT", "trades", datetime(2024, 1, 1, 12, 10, 0, tzinfo=UTC)),
        ]

        # Set all sync times
        for symbol, data_type, timestamp in test_cases:
            await repository.set_last_sync_time(symbol, data_type, timestamp)

        # Verify all sync times
        for symbol, data_type, expected_timestamp in test_cases:
            retrieved_time = await repository.get_last_sync_time(symbol, data_type)
            assert retrieved_time == expected_timestamp

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_backfill_status_nonexistent(self, repository):
        """Test getting backfill status for non-existent symbol/data_type."""
        status = await repository.get_backfill_status("BTCUSDT", "trades")
        assert status is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_and_get_backfill_status(self, repository):
        """Test setting and getting backfill status."""
        symbol = "BTCUSDT"
        data_type = "trades"
        status = {"progress": 0.75, "last_processed": "2024-01-01T12:00:00Z", "errors": 0, "state": "running"}

        await repository.set_backfill_status(symbol, data_type, status)

        retrieved_status = await repository.get_backfill_status(symbol, data_type)

        # Should contain original status plus metadata
        assert retrieved_status["progress"] == status["progress"]
        assert retrieved_status["last_processed"] == status["last_processed"]
        assert retrieved_status["errors"] == status["errors"]
        assert retrieved_status["state"] == status["state"]
        assert retrieved_status["symbol"] == symbol
        assert retrieved_status["data_type"] == data_type
        assert "updated_at" in retrieved_status

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_keys_limit(self, repository):
        """Test maximum keys limit."""
        # Fill up to the limit
        for i in range(repository.max_keys):
            await repository.set(f"key_{i}", {"value": i})

        # Should work fine
        assert len(repository._data) == repository.max_keys

        # Try to add one more
        with pytest.raises(StorageError) as exc_info:
            await repository.set("overflow_key", {"value": "overflow"})

        assert exc_info.value.code == "STORAGE_LIMIT_EXCEEDED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_keys_limit_update_existing(self, repository):
        """Test that updating existing keys doesn't count against limit."""
        # Fill up to the limit
        for i in range(repository.max_keys):
            await repository.set(f"key_{i}", {"value": i})

        # Update existing key should work
        await repository.set("key_0", {"value": "updated"})

        # Verify update worked
        result = await repository.get("key_0")
        assert result["value"] == "updated"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_automatic_cleanup(self, repository):
        """Test automatic cleanup of expired entries."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Set a key with very short TTL
            await repository.set_with_ttl("short_ttl", {"test": "data"}, 1)

            # Should exist initially
            assert await repository.exists("short_ttl") is True

            # Wait for cleanup (cleanup_interval is 0.1 seconds)
            traveller.shift(1.2)

            # Should be cleaned up
            assert await repository.exists("short_ttl") is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_force_cleanup(self, repository):
        """Test force cleanup of expired entries."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Set multiple keys with TTL
            await repository.set_with_ttl("key1", {"test": "data1"}, 1)
            await repository.set_with_ttl("key2", {"test": "data2"}, 1)
            await repository.set_with_ttl("key3", {"test": "data3"}, 1)

            # Wait for expiration
            traveller.shift(1.1)

            # Force cleanup
            cleaned_count = await repository.force_cleanup()
            assert cleaned_count == 3

            # Keys should be gone
            assert await repository.exists("key1") is False
            assert await repository.exists("key2") is False
            assert await repository.exists("key3") is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_repository(self, repository, sample_metadata):
        """Test closing the repository."""
        # Set some data
        await repository.set("test_key", sample_metadata)

        # Close repository
        await repository.close()

        # Should be closed
        assert repository._closed is True

        # All data should be cleared
        assert len(repository._data) == 0
        assert len(repository._ttl) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_operations_on_closed_repository(self, repository, sample_metadata):
        """Test operations on closed repository."""
        await repository.close()

        # All operations should raise StorageError
        with pytest.raises(StorageError) as exc_info:
            await repository.set("key", sample_metadata)
        assert exc_info.value.code == "REPOSITORY_CLOSED"

        with pytest.raises(StorageError) as exc_info:
            await repository.get("key")
        assert exc_info.value.code == "REPOSITORY_CLOSED"

        with pytest.raises(StorageError) as exc_info:
            await repository.exists("key")
        assert exc_info.value.code == "REPOSITORY_CLOSED"

        with pytest.raises(StorageError) as exc_info:
            await repository.delete("key")
        assert exc_info.value.code == "REPOSITORY_CLOSED"

        with pytest.raises(StorageError) as exc_info:
            await repository.list_keys()
        assert exc_info.value.code == "REPOSITORY_CLOSED"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self, repository, sample_metadata):
        """Test repository as async context manager."""
        async with repository as repo:
            await repo.set("test_key", sample_metadata)
            result = await repo.get("test_key")
            assert result == sample_metadata

        # Repository should be closed
        assert repository._closed is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_key_validation(self, repository, sample_metadata):
        """Test invalid key validation."""
        # Test empty key
        with pytest.raises(StorageError) as exc_info:
            await repository.set("", sample_metadata)
        assert exc_info.value.code == "INVALID_KEY"

        # Test non-string key
        with pytest.raises(StorageError) as exc_info:
            await repository.set(123, sample_metadata)
        assert exc_info.value.code == "INVALID_KEY"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_value_validation(self, repository):
        """Test invalid value validation."""
        # Test non-dict value
        with pytest.raises(StorageError) as exc_info:
            await repository.set("key", "not_a_dict")
        assert exc_info.value.code == "INVALID_VALUE"

        with pytest.raises(StorageError) as exc_info:
            await repository.set("key", 123)
        assert exc_info.value.code == "INVALID_VALUE"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_ttl_validation(self, repository, sample_metadata):
        """Test invalid TTL validation."""
        # Test negative TTL
        with pytest.raises(StorageError) as exc_info:
            await repository.set_with_ttl("key", sample_metadata, -1)
        assert exc_info.value.code == "INVALID_TTL"

        # Test zero TTL
        with pytest.raises(StorageError) as exc_info:
            await repository.set_with_ttl("key", sample_metadata, 0)
        assert exc_info.value.code == "INVALID_TTL"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_all_keys(self, repository, sample_metadata):
        """Test getting all keys including expired ones."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Set normal keys
            await repository.set("key1", sample_metadata)
            await repository.set("key2", sample_metadata)

            # Set TTL key
            await repository.set_with_ttl("ttl_key", sample_metadata, 1)

            # Get all keys
            all_keys = await repository.get_all_keys()
            assert sorted(all_keys) == ["key1", "key2", "ttl_key"]

            # Wait for TTL expiration
            traveller.shift(1.1)

            # get_all_keys may or may not show expired key depending on cleanup timing
            all_keys = await repository.get_all_keys()
            # Either it's been cleaned up by automatic cleanup or it's still there
            assert len(all_keys) >= 2  # At least the two normal keys

            # But list_keys should not show expired keys
            active_keys = await repository.list_keys()
            assert "ttl_key" not in active_keys

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_ttl_keys(self, repository, sample_metadata):
        """Test getting keys with TTL."""
        # Set normal key
        await repository.set("normal_key", sample_metadata)

        # Set TTL keys
        await repository.set_with_ttl("ttl_key1", sample_metadata, 60)
        await repository.set_with_ttl("ttl_key2", sample_metadata, 120)

        ttl_keys = await repository.get_ttl_keys()
        assert sorted(ttl_keys) == ["ttl_key1", "ttl_key2"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_key_info(self, repository, sample_metadata):
        """Test getting detailed key information."""
        # Test non-existent key
        info = await repository.get_key_info("nonexistent")
        assert info is None

        # Test normal key
        await repository.set("normal_key", sample_metadata)
        info = await repository.get_key_info("normal_key")

        assert info["key"] == "normal_key"
        assert info["exists"] is True
        assert info["has_ttl"] is False
        assert "created_at" in info
        assert "updated_at" in info

        # Test TTL key
        await repository.set_with_ttl("ttl_key", sample_metadata, 60)
        info = await repository.get_key_info("ttl_key")

        assert info["key"] == "ttl_key"
        assert info["exists"] is True
        assert info["has_ttl"] is True
        assert "expires_at" in info
        assert "ttl_seconds" in info
        assert info["is_expired"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_memory_usage(self, repository, sample_metadata):
        """Test getting memory usage statistics."""
        # Initially empty
        usage = await repository.get_memory_usage()
        assert usage["total_keys"] == 0
        assert usage["ttl_keys"] == 0
        assert usage["expired_keys"] == 0

        # Add some data
        await repository.set("key1", sample_metadata)
        await repository.set("key2", sample_metadata)
        await repository.set_with_ttl("ttl_key", sample_metadata, 60)

        usage = await repository.get_memory_usage()
        assert usage["total_keys"] == 3
        assert usage["ttl_keys"] == 1
        assert usage["expired_keys"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_all(self, repository, sample_metadata):
        """Test clearing all data."""
        # Add some data
        await repository.set("key1", sample_metadata)
        await repository.set_with_ttl("ttl_key", sample_metadata, 60)

        # Clear all
        await repository.clear_all()

        # Should be empty
        assert len(repository._data) == 0
        assert len(repository._ttl) == 0
        assert await repository.list_keys() == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_sync_times(self, repository):
        """Test getting all sync times."""
        # Set multiple sync times
        await repository.set_last_sync_time("BTCUSDT", "trades", datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC))
        await repository.set_last_sync_time("BTCUSDT", "klines_1m", datetime(2024, 1, 1, 12, 5, 0, tzinfo=UTC))
        await repository.set_last_sync_time("ETHUSDT", "trades", datetime(2024, 1, 1, 12, 10, 0, tzinfo=UTC))

        # Get all sync times
        all_sync_times = await repository.get_sync_times()
        assert len(all_sync_times) == 3
        assert "BTCUSDT:trades" in all_sync_times
        assert "BTCUSDT:klines_1m" in all_sync_times
        assert "ETHUSDT:trades" in all_sync_times

        # Filter by symbol
        btc_sync_times = await repository.get_sync_times("BTCUSDT")
        assert len(btc_sync_times) == 2
        assert "BTCUSDT:trades" in btc_sync_times
        assert "BTCUSDT:klines_1m" in btc_sync_times

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_backfill_statuses(self, repository):
        """Test getting all backfill statuses."""
        # Set multiple backfill statuses
        await repository.set_backfill_status("BTCUSDT", "trades", {"progress": 0.5})
        await repository.set_backfill_status("BTCUSDT", "klines_1m", {"progress": 0.8})
        await repository.set_backfill_status("ETHUSDT", "trades", {"progress": 0.3})

        # Get all backfill statuses
        all_statuses = await repository.get_backfill_statuses()
        assert len(all_statuses) == 3
        assert "BTCUSDT:trades" in all_statuses
        assert "BTCUSDT:klines_1m" in all_statuses
        assert "ETHUSDT:trades" in all_statuses

        # Filter by symbol
        btc_statuses = await repository.get_backfill_statuses("BTCUSDT")
        assert len(btc_statuses) == 2
        assert "BTCUSDT:trades" in btc_statuses
        assert "BTCUSDT:klines_1m" in btc_statuses

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ttl_with_list_keys(self, repository, sample_metadata):
        """Test that expired TTL keys are filtered from list_keys."""
        with time_machine.travel("2024-01-01 12:00:00", tick=False) as traveller:
            # Set keys with different TTL
            await repository.set("normal_key", sample_metadata)
            await repository.set_with_ttl("short_ttl", sample_metadata, 1)
            await repository.set_with_ttl("long_ttl", sample_metadata, 60)

            # Initially all should be listed
            keys = await repository.list_keys()
            assert len(keys) == 3

            # Wait for short TTL to expire
            traveller.shift(1.1)

            # Only normal_key and long_ttl should be listed
            keys = await repository.list_keys()
            assert len(keys) == 2
            assert "normal_key" in keys
            assert "long_ttl" in keys
            assert "short_ttl" not in keys

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_time_iso_format_parsing(self, repository):
        """Test that sync times are correctly parsed from ISO format."""
        symbol = "BTCUSDT"
        data_type = "trades"

        # Test with different timezone formats
        test_timestamp = datetime(2024, 1, 1, 12, 30, 45, tzinfo=UTC)

        await repository.set_last_sync_time(symbol, data_type, test_timestamp)
        retrieved_time = await repository.get_last_sync_time(symbol, data_type)

        assert retrieved_time == test_timestamp
        assert retrieved_time.tzinfo == UTC

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_time_invalid_format(self, repository):
        """Test handling of invalid sync time format."""
        # Manually set invalid sync time data
        key = "sync_time:BTCUSDT:trades"
        await repository.set(key, {"timestamp": "invalid_timestamp"})

        # Should return None for invalid format
        sync_time = await repository.get_last_sync_time("BTCUSDT", "trades")
        assert sync_time is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, repository, sample_metadata):
        """Test concurrent operations on the repository."""
        # Create multiple concurrent operations
        tasks = []

        for i in range(10):
            task = asyncio.create_task(repository.set(f"key_{i}", {**sample_metadata, "id": i}))
            tasks.append(task)

        # Wait for all operations to complete
        await asyncio.gather(*tasks)

        # Verify all keys were set
        keys = await repository.list_keys()
        assert len(keys) == 10

        # Verify all data is correct
        for i in range(10):
            data = await repository.get(f"key_{i}")
            assert data["id"] == i
