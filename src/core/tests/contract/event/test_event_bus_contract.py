# ABOUTME: Contract tests for AbstractEventBus implementations
# ABOUTME: Ensures all event bus implementations conform to the interface specification

import asyncio
import pytest
from abc import ABC, abstractmethod
from unittest.mock import Mock

from core.interfaces.event.event_bus import AbstractEventBus
from core.implementations.memory.event.event_bus import InMemoryEventBus
from core.models.data.event import BaseEvent
from core.models.event.event_type import EventType


class AbstractEventBusContract(ABC):
    """
    Contract test base class for AbstractEventBus implementations.

    This class defines the contract that all AbstractEventBus implementations
    must satisfy. Concrete test classes should inherit from this and implement
    the get_event_bus method to provide their specific implementation.
    """

    @abstractmethod
    def get_event_bus(self) -> AbstractEventBus:
        """Get an instance of the event bus implementation to test."""
        pass

    @pytest.fixture
    def event_bus(self):
        """Fixture providing the event bus implementation."""
        return self.get_event_bus()

    @pytest.fixture
    def sample_event(self):
        """Sample event for testing."""
        return BaseEvent(event_type=EventType.TRADE, source="test_source", data={"test": "data"})

    # Contract Tests

    @pytest.mark.contract
    def test_implements_abstract_event_bus(self, event_bus):
        """Test that implementation properly implements AbstractEventBus."""
        assert isinstance(event_bus, AbstractEventBus)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_publish_accepts_base_event(self, event_bus, sample_event):
        """Test that publish method accepts BaseEvent instances."""
        # Should not raise an exception
        await event_bus.publish(sample_event)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_publish_rejects_invalid_input(self, event_bus):
        """Test that publish method rejects invalid input."""
        with pytest.raises((TypeError, ValueError)):
            await event_bus.publish(None)

        with pytest.raises((TypeError, ValueError)):
            await event_bus.publish("not an event")

        with pytest.raises((TypeError, ValueError)):
            await event_bus.publish({"not": "an event"})

    @pytest.mark.contract
    def test_subscribe_returns_string_id(self, event_bus):
        """Test that subscribe returns a string subscription ID."""
        handler = Mock()
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        assert isinstance(subscription_id, str)
        assert len(subscription_id) > 0

    @pytest.mark.contract
    def test_subscribe_validates_event_type(self, event_bus):
        """Test that subscribe validates event_type parameter."""
        handler = Mock()

        with pytest.raises((TypeError, ValueError)):
            event_bus.subscribe("invalid", handler)

        with pytest.raises((TypeError, ValueError)):
            event_bus.subscribe(None, handler)

    @pytest.mark.contract
    def test_subscribe_validates_handler(self, event_bus):
        """Test that subscribe validates handler parameter."""
        with pytest.raises((TypeError, ValueError)):
            event_bus.subscribe(EventType.TRADE, "not callable")

        with pytest.raises((TypeError, ValueError)):
            event_bus.subscribe(EventType.TRADE, None)

    @pytest.mark.contract
    def test_unsubscribe_returns_boolean(self, event_bus):
        """Test that unsubscribe returns a boolean."""
        handler = Mock()
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        result = event_bus.unsubscribe(subscription_id)
        assert isinstance(result, bool)
        assert result is True

        # Second unsubscribe should return False
        result = event_bus.unsubscribe(subscription_id)
        assert isinstance(result, bool)
        assert result is False

    @pytest.mark.contract
    def test_unsubscribe_handles_invalid_id(self, event_bus):
        """Test that unsubscribe handles invalid subscription IDs gracefully."""
        result = event_bus.unsubscribe("invalid-id")
        assert isinstance(result, bool)
        assert result is False

    @pytest.mark.contract
    def test_unsubscribe_all_returns_count(self, event_bus):
        """Test that unsubscribe_all returns the count of removed subscriptions."""
        handler = Mock()

        # Subscribe multiple handlers
        event_bus.subscribe(EventType.TRADE, handler)
        event_bus.subscribe(EventType.KLINE, handler)

        # Unsubscribe all
        count = event_bus.unsubscribe_all()
        assert isinstance(count, int)
        assert count == 2

        # Second call should return 0
        count = event_bus.unsubscribe_all()
        assert isinstance(count, int)
        assert count == 0

    @pytest.mark.contract
    @pytest.mark.timeout(30)
    def test_unsubscribe_all_with_event_type(self, event_bus):
        """Test unsubscribe_all with specific event type."""
        handler = Mock()

        # Subscribe to different event types
        event_bus.subscribe(EventType.TRADE, handler)
        event_bus.subscribe(EventType.TRADE, handler)
        event_bus.subscribe(EventType.KLINE, handler)

        # Unsubscribe only TRADE events
        count = event_bus.unsubscribe_all(EventType.TRADE)
        assert isinstance(count, int)
        assert count == 2

        # Should still have KLINE subscription
        assert event_bus.get_subscription_count() == 1

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_close_is_async(self, event_bus):
        """Test that close method is async and can be awaited."""
        # Should not raise an exception
        await event_bus.close()

    @pytest.mark.contract
    @pytest.mark.timeout(30)
    def test_get_subscription_count_returns_int(self, event_bus):
        """Test that get_subscription_count returns an integer."""
        count = event_bus.get_subscription_count()
        assert isinstance(count, int)
        assert count >= 0

        # With specific event type
        count = event_bus.get_subscription_count(EventType.TRADE)
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_wait_for_returns_event_or_none(self, event_bus):
        """Test that wait_for returns BaseEvent or raises TimeoutError."""
        # Test with timeout (should raise TimeoutError)
        with pytest.raises(asyncio.TimeoutError):
            await event_bus.wait_for(EventType.TRADE, timeout=0.01)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_wait_for_validates_event_type(self, event_bus):
        """Test that wait_for validates event_type parameter."""
        with pytest.raises((TypeError, ValueError)):
            await event_bus.wait_for("invalid")

        with pytest.raises((TypeError, ValueError)):
            await event_bus.wait_for(None)

    @pytest.mark.contract
    def test_is_closed_property_returns_bool(self, event_bus):
        """Test that is_closed property returns a boolean."""
        result = event_bus.is_closed
        assert isinstance(result, bool)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_operations_after_close(self, event_bus):
        """Test that operations after close raise appropriate errors."""
        handler = Mock()
        sample_event = BaseEvent(event_type=EventType.TRADE, source="test", data={})

        # Close the event bus
        await event_bus.close()
        assert event_bus.is_closed

        # Operations should fail or handle gracefully
        with pytest.raises(RuntimeError):
            event_bus.subscribe(EventType.TRADE, handler)

        with pytest.raises(RuntimeError):
            await event_bus.publish(sample_event)

        with pytest.raises(RuntimeError):
            event_bus.unsubscribe("any-id")

        with pytest.raises(RuntimeError):
            event_bus.unsubscribe_all()

        with pytest.raises(RuntimeError):
            await event_bus.wait_for(EventType.TRADE)


class TestInMemoryEventBusContract(AbstractEventBusContract):
    """Contract tests for InMemoryEventBus implementation."""

    def get_event_bus(self) -> AbstractEventBus:
        """Get InMemoryEventBus instance for testing."""
        return InMemoryEventBus()

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_memory_specific_functionality(self):
        """Test InMemoryEventBus specific functionality."""
        event_bus = self.get_event_bus()

        # Test statistics method (if available)
        if hasattr(event_bus, "get_statistics"):
            stats = event_bus.get_statistics()
            assert isinstance(stats, dict)
            assert "published_count" in stats
            assert "error_count" in stats

        # Clean up
        await event_bus.close()


# Additional integration-style contract tests


@pytest.mark.contract
@pytest.mark.asyncio
async def test_publish_subscribe_integration():
    """Integration test for publish-subscribe functionality."""
    event_bus = InMemoryEventBus()
    received_events = []

    def handler(event):
        received_events.append(event)

    try:
        # Subscribe
        subscription_id = event_bus.subscribe(EventType.TRADE, handler)

        # Publish
        test_event = BaseEvent(event_type=EventType.TRADE, source="integration_test", data={"test": "integration"})
        await event_bus.publish(test_event)

        # Wait for event processing to complete
        await asyncio.sleep(0.1)

        # Verify
        assert len(received_events) == 1
        assert received_events[0].data["test"] == "integration"

        # Unsubscribe
        result = event_bus.unsubscribe(subscription_id)
        assert result is True

    finally:
        await event_bus.close()


@pytest.mark.contract
@pytest.mark.asyncio
async def test_symbol_filtering_contract():
    """Contract test for symbol filtering functionality."""
    event_bus = InMemoryEventBus()
    filtered_events = []
    all_events = []

    def filtered_handler(event):
        filtered_events.append(event)

    def all_handler(event):
        all_events.append(event)

    try:
        # Subscribe with and without filter
        event_bus.subscribe(EventType.TRADE, filtered_handler, filter_symbol="BTC")
        event_bus.subscribe(EventType.TRADE, all_handler)

        # Publish events with different symbols
        btc_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="BTC", data={})

        eth_event = BaseEvent(event_type=EventType.TRADE, source="test", symbol="ETH", data={})

        await event_bus.publish(btc_event)
        await event_bus.publish(eth_event)

        # Wait for event processing to complete
        await asyncio.sleep(0.1)

        # Verify filtering
        assert len(filtered_events) == 1  # Only BTC event
        assert filtered_events[0].symbol == "BTC"

        assert len(all_events) == 2  # Both events

    finally:
        await event_bus.close()
