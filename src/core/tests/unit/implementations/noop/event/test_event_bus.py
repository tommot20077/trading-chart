# ABOUTME: Unit tests for NoOpEventBus
# ABOUTME: Tests for no-operation event bus implementation

import pytest
import asyncio
from unittest.mock import Mock

from core.implementations.noop.event.event_bus import NoOpEventBus
from core.models.event.event_type import EventType
from core.models.data.event import BaseEvent


class TestNoOpEventBus:
    """Test cases for NoOpEventBus."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test event bus initialization."""
        bus = NoOpEventBus()
        assert bus is not None
        assert not bus.is_closed
        assert bus.get_subscription_count() == 0

    @pytest.mark.asyncio
    async def test_publish_discards_events(self):
        """Test that publish discards events without error."""
        bus = NoOpEventBus()
        event = Mock(spec=BaseEvent)

        # Should not raise exception
        await bus.publish(event)
        await bus.publish(None)

    @pytest.mark.unit
    def test_subscribe_returns_fake_id(self):
        """Test that subscribe returns fake subscription ID."""
        bus = NoOpEventBus()
        handler = Mock()

        sub_id = bus.subscribe(EventType.TRADE, handler)

        assert isinstance(sub_id, str)
        assert sub_id.startswith("noop-sub-")
        assert bus.get_subscription_count() == 1
        assert bus.get_subscription_count(EventType.TRADE) == 1

    @pytest.mark.unit
    def test_subscribe_with_filter(self):
        """Test subscribing with symbol filter."""
        bus = NoOpEventBus()
        handler = Mock()

        sub_id = bus.subscribe(EventType.TRADE, handler, filter_symbol="BTC/USDT")

        assert isinstance(sub_id, str)
        assert bus.get_subscription_count() == 1

    @pytest.mark.unit
    def test_unsubscribe_existing(self):
        """Test unsubscribing existing subscription."""
        bus = NoOpEventBus()
        handler = Mock()

        sub_id = bus.subscribe(EventType.TRADE, handler)
        assert bus.get_subscription_count() == 1

        result = bus.unsubscribe(sub_id)
        assert result is True
        assert bus.get_subscription_count() == 0

    @pytest.mark.unit
    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing non-existent subscription."""
        bus = NoOpEventBus()

        result = bus.unsubscribe("nonexistent-id")
        assert result is False

    @pytest.mark.unit
    def test_unsubscribe_all(self):
        """Test unsubscribing all handlers."""
        bus = NoOpEventBus()

        bus.subscribe(EventType.TRADE, Mock())
        bus.subscribe(EventType.KLINE, Mock())
        bus.subscribe(EventType.TRADE, Mock())

        assert bus.get_subscription_count() == 3

        # Unsubscribe all TRADE handlers
        count = bus.unsubscribe_all(EventType.TRADE)
        assert count == 2
        assert bus.get_subscription_count() == 1

        # Unsubscribe all remaining handlers
        count = bus.unsubscribe_all()
        assert count == 1
        assert bus.get_subscription_count() == 0

    @pytest.mark.asyncio
    async def test_wait_for_raises_timeout(self):
        """Test that wait_for always raises TimeoutError."""
        bus = NoOpEventBus()

        # Should raise TimeoutError immediately (no events delivered)
        with pytest.raises(asyncio.TimeoutError):
            await bus.wait_for(EventType.TRADE, timeout=0.1)

    @pytest.mark.asyncio
    async def test_close_sets_closed_flag(self):
        """Test that close sets closed flag and clears subscriptions."""
        bus = NoOpEventBus()

        bus.subscribe(EventType.TRADE, Mock())
        assert bus.get_subscription_count() == 1
        assert not bus.is_closed

        await bus.close()

        assert bus.is_closed
        assert bus.get_subscription_count() == 0

    @pytest.mark.asyncio
    async def test_operations_after_close_raise_error(self):
        """Test that operations after close raise errors."""
        bus = NoOpEventBus()
        await bus.close()

        with pytest.raises(RuntimeError):
            await bus.publish(Mock())

        with pytest.raises(RuntimeError):
            bus.subscribe(EventType.TRADE, Mock())

        with pytest.raises(RuntimeError):
            bus.unsubscribe("some-id")

        with pytest.raises(RuntimeError):
            await bus.wait_for(EventType.TRADE)
