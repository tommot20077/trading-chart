# ABOUTME: Unit tests for event enumeration classes including EventType
# ABOUTME: Tests cover normal cases, exception cases, and boundary cases following TDD principles

import pytest

from core.models.event.event_type import EventType


class TestEventType:
    """Test cases for EventType enum."""

    @pytest.mark.unit
    def test_event_type_values_normal_case(self):
        """Test that EventType enum has correct string values."""
        assert EventType.TRADE == "trade"
        assert EventType.KLINE == "kline"
        assert EventType.ORDER == "order"
        assert EventType.CONNECTION == "connection"
        assert EventType.ERROR == "error"
        assert EventType.SYSTEM == "system"
        assert EventType.ALERT == "alert"
        assert EventType.MARKET_DATA == "market_data"

    @pytest.mark.unit
    def test_event_type_string_representation(self):
        """Test __str__ method returns correct string representation."""
        assert str(EventType.TRADE) == "trade"
        assert str(EventType.KLINE) == "kline"
        assert str(EventType.ORDER) == "order"
        assert str(EventType.CONNECTION) == "connection"
        assert str(EventType.ERROR) == "error"
        assert str(EventType.SYSTEM) == "system"
        assert str(EventType.ALERT) == "alert"
        assert str(EventType.MARKET_DATA) == "market_data"

    @pytest.mark.unit
    def test_event_type_equality(self):
        """Test equality comparison of EventType enum members."""
        assert EventType.TRADE == EventType.TRADE
        assert EventType.KLINE == EventType.KLINE
        assert EventType.TRADE != EventType.KLINE
        assert EventType.ORDER != EventType.CONNECTION
        assert EventType.ERROR != EventType.SYSTEM

    @pytest.mark.unit
    def test_event_type_membership(self):
        """Test membership checking for EventType enum."""
        event_values = [event.value for event in EventType]
        assert "trade" in event_values
        assert "kline" in event_values
        assert "order" in event_values
        assert "connection" in event_values
        assert "error" in event_values
        assert "system" in event_values
        assert "alert" in event_values
        assert "market_data" in event_values
        assert "invalid_event" not in event_values

    @pytest.mark.unit
    def test_event_type_iteration(self):
        """Test iteration over EventType enum members."""
        events = list(EventType)
        assert len(events) == 8
        assert EventType.TRADE in events
        assert EventType.KLINE in events
        assert EventType.ORDER in events
        assert EventType.CONNECTION in events
        assert EventType.ERROR in events
        assert EventType.SYSTEM in events
        assert EventType.ALERT in events
        assert EventType.MARKET_DATA in events

    @pytest.mark.unit
    def test_event_type_from_string(self):
        """Test creating EventType from string value."""
        assert EventType("trade") == EventType.TRADE
        assert EventType("kline") == EventType.KLINE
        assert EventType("order") == EventType.ORDER
        assert EventType("connection") == EventType.CONNECTION
        assert EventType("error") == EventType.ERROR
        assert EventType("system") == EventType.SYSTEM
        assert EventType("alert") == EventType.ALERT
        assert EventType("market_data") == EventType.MARKET_DATA

    @pytest.mark.unit
    def test_event_type_invalid_value_raises_exception(self):
        """Test that invalid event type value raises ValueError."""
        with pytest.raises(ValueError):
            EventType("invalid_event")

        with pytest.raises(ValueError):
            EventType("TRADE")  # Case sensitive

        with pytest.raises(ValueError):
            EventType("")

    @pytest.mark.unit
    def test_event_type_completeness(self):
        """Test that all expected event types are present."""
        expected_types = {"trade", "kline", "order", "connection", "error", "system", "alert", "market_data"}
        actual_types = {event.value for event in EventType}
        assert actual_types == expected_types

    @pytest.mark.unit
    def test_event_type_business_logic_categories(self):
        """Test logical categorization of event types."""
        # Data events
        data_events = [EventType.TRADE, EventType.KLINE, EventType.MARKET_DATA]
        for event in data_events:
            assert event.value in ["trade", "kline", "market_data"]

        # System events
        system_events = [EventType.CONNECTION, EventType.ERROR, EventType.SYSTEM]
        for event in system_events:
            assert event.value in ["connection", "error", "system"]

        # Trading events
        trading_events = [EventType.ORDER, EventType.TRADE]
        for event in trading_events:
            assert event.value in ["order", "trade"]

    @pytest.mark.unit
    def test_event_type_repr_representation(self):
        """Test repr representation of EventType enum members."""
        assert "EventType.TRADE" in repr(EventType.TRADE)
        assert "EventType.KLINE" in repr(EventType.KLINE)
        assert "EventType.ORDER" in repr(EventType.ORDER)
        assert "EventType.CONNECTION" in repr(EventType.CONNECTION)
        assert "EventType.ERROR" in repr(EventType.ERROR)
        assert "EventType.SYSTEM" in repr(EventType.SYSTEM)
        assert "EventType.ALERT" in repr(EventType.ALERT)
        assert "EventType.MARKET_DATA" in repr(EventType.MARKET_DATA)

    @pytest.mark.unit
    def test_event_type_hash_consistency(self):
        """Test that EventType enum members are hashable and consistent."""
        # Test that enum members can be used as dictionary keys
        event_dict = {
            EventType.TRADE: "trade_handler",
            EventType.KLINE: "kline_handler",
            EventType.ORDER: "order_handler",
            EventType.CONNECTION: "connection_handler",
            EventType.ERROR: "error_handler",
            EventType.SYSTEM: "system_handler",
            EventType.ALERT: "alert_handler",
            EventType.MARKET_DATA: "market_data_handler",
        }

        assert len(event_dict) == 8
        assert event_dict[EventType.TRADE] == "trade_handler"
        assert event_dict[EventType.KLINE] == "kline_handler"

        # Test that hash is consistent
        assert hash(EventType.TRADE) == hash(EventType.TRADE)
        assert hash(EventType.TRADE) != hash(EventType.KLINE)

    @pytest.mark.unit
    def test_event_type_set_operations(self):
        """Test EventType enum members in set operations."""
        event_set = {EventType.TRADE, EventType.KLINE, EventType.TRADE}
        assert len(event_set) == 2  # Duplicates removed
        assert EventType.TRADE in event_set
        assert EventType.KLINE in event_set

        # Test set intersection
        data_events = {EventType.TRADE, EventType.KLINE}
        system_events = {EventType.CONNECTION, EventType.ERROR, EventType.SYSTEM}
        assert data_events.intersection(system_events) == set()

        # Test set union
        all_events = data_events.union(system_events).union({EventType.ORDER, EventType.ALERT, EventType.MARKET_DATA})
        assert len(all_events) == 8
