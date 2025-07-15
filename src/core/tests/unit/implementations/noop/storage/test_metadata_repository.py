# ABOUTME: Unit tests for NoOpMetadataRepository
# ABOUTME: Tests for no-operation metadata storage implementation

import pytest
from datetime import datetime

from core.implementations.noop.storage.metadata_repository import NoOpMetadataRepository


class TestNoOpMetadataRepository:
    """Test cases for NoOpMetadataRepository."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test repository initialization."""
        repo = NoOpMetadataRepository()
        assert repo is not None
        assert not repo._closed
        assert len(repo._fake_keys) == 0

    @pytest.mark.asyncio
    async def test_set_simulates_storage(self):
        """Test that set simulates storage by tracking keys."""
        repo = NoOpMetadataRepository()

        await repo.set("test_key", {"value": "test"})

        assert "test_key" in repo._fake_keys

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        """Test that get always returns None (no actual storage)."""
        repo = NoOpMetadataRepository()

        await repo.set("test_key", {"value": "test"})
        result = await repo.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_exists_checks_fake_storage(self):
        """Test that exists checks fake storage tracking."""
        repo = NoOpMetadataRepository()

        assert not await repo.exists("test_key")

        await repo.set("test_key", {"value": "test"})
        assert await repo.exists("test_key")

    @pytest.mark.asyncio
    async def test_delete_removes_from_fake_storage(self):
        """Test that delete removes from fake storage tracking."""
        repo = NoOpMetadataRepository()

        await repo.set("test_key", {"value": "test"})
        assert await repo.exists("test_key")

        result = await repo.delete("test_key")
        assert result is True
        assert not await repo.exists("test_key")

        # Delete non-existent key
        result = await repo.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_keys_returns_fake_keys(self):
        """Test that get_keys returns tracked fake keys."""
        repo = NoOpMetadataRepository()

        await repo.set("key1", {"value": "test1"})
        await repo.set("key2", {"value": "test2"})

        keys = await repo.get_keys()
        assert "key1" in keys
        assert "key2" in keys
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_set_multiple_tracks_all_keys(self):
        """Test that set_multiple tracks all keys."""
        repo = NoOpMetadataRepository()

        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        await repo.set_multiple(data)

        keys = await repo.get_keys()
        assert all(key in keys for key in data.keys())

    @pytest.mark.asyncio
    async def test_delete_multiple_removes_tracked_keys(self):
        """Test that delete_multiple removes tracked keys."""
        repo = NoOpMetadataRepository()

        await repo.set("key1", {"value": "test1"})
        await repo.set("key2", {"value": "test2"})
        await repo.set("key3", {"value": "test3"})

        count = await repo.delete_multiple(["key1", "key3", "nonexistent"])
        assert count == 2  # Only key1 and key3 existed

        keys = await repo.get_keys()
        assert "key2" in keys
        assert "key1" not in keys
        assert "key3" not in keys

    @pytest.mark.asyncio
    async def test_clear_all_removes_all_keys(self):
        """Test that clear_all removes all tracked keys."""
        repo = NoOpMetadataRepository()

        await repo.set("key1", {"value": "test1"})
        await repo.set("key2", {"value": "test2"})

        count = await repo.clear_all()
        assert count == 2

        keys = await repo.get_keys()
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_open(self):
        """Test that health_check returns True when repository is open."""
        repo = NoOpMetadataRepository()

        assert await repo.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_closed(self):
        """Test that health_check returns False when repository is closed."""
        repo = NoOpMetadataRepository()

        await repo.close()
        assert await repo.health_check() is False

    @pytest.mark.asyncio
    async def test_operations_after_close_raise_error(self):
        """Test that operations after close raise RuntimeError."""
        repo = NoOpMetadataRepository()
        await repo.close()

        with pytest.raises(RuntimeError):
            await repo.set("key", {"value": "test"})

        with pytest.raises(RuntimeError):
            await repo.get("key")

        with pytest.raises(RuntimeError):
            await repo.delete("key")

        with pytest.raises(RuntimeError):
            await repo.exists("key")

    @pytest.mark.asyncio
    async def test_backfill_status_methods(self):
        """Test backfill status tracking methods."""
        repo = NoOpMetadataRepository()

        # Set backfill status
        await repo.set_backfill_status("BTC/USDT", "klines_1m", {"status": "completed"})
        assert "backfill_status:BTC/USDT:klines_1m" in repo._fake_keys

        # Get backfill status (always returns None)
        status = await repo.get_backfill_status("BTC/USDT", "klines_1m")
        assert status is None

    @pytest.mark.asyncio
    async def test_last_sync_time_methods(self):
        """Test last sync time tracking methods."""
        repo = NoOpMetadataRepository()

        # Set last sync time
        now = datetime.now()
        await repo.set_last_sync_time("ETH/USDT", "klines_1m", now)
        assert "last_sync:ETH/USDT:klines_1m" in repo._fake_keys

        # Get last sync time (always returns None)
        sync_time = await repo.get_last_sync_time("ETH/USDT", "klines_1m")
        assert sync_time is None
