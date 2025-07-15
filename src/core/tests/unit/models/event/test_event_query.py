# ABOUTME: Unit tests for EventQuery model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for EventQuery

import pytest
from datetime import datetime, UTC

from core.models.event.event_query import EventQuery
from core.models.event.event_type import EventType


class TestEventQuery:
    """
    Comprehensive unit tests for EventQuery model.

    Tests cover:
    - Normal case: Valid EventQuery creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    @pytest.mark.unit
    def test_event_query_creation_with_defaults(self):
        """
        Test EventQuery creation with default values.

        Verifies:
        - All default values are properly set
        - Empty lists are created for optional list parameters
        - Default ordering and pagination values are correct
        """
        # Act
        query = EventQuery()

        # Assert
        assert query.event_types == []
        assert query.symbols == []
        assert query.sources == []
        assert query.start_time is None
        assert query.end_time is None
        assert query.correlation_id is None
        assert query.limit is None
        assert query.offset == 0
        assert query.order_by == "timestamp"
        assert query.order_desc is True
        assert query.metadata_filters == {}

    @pytest.mark.unit
    def test_event_query_creation_with_all_parameters(self):
        """
        Test EventQuery creation with all parameters specified.

        Verifies:
        - All parameters are properly stored
        - Complex parameter combinations work correctly
        """
        # Arrange
        event_types = [EventType.TRADE, EventType.KLINE]
        symbols = ["BTCUSDT", "ETHUSDT"]
        sources = ["binance", "coinbase"]
        start_time = datetime(2024, 1, 1, tzinfo=UTC)
        end_time = datetime(2024, 1, 2, tzinfo=UTC)
        correlation_id = "query_001"
        limit = 100
        offset = 50
        order_by = "event_id"
        order_desc = False
        metadata_filters = {"exchange": "binance", "market": "spot"}

        # Act
        query = EventQuery(
            event_types=event_types,
            symbols=symbols,
            sources=sources,
            start_time=start_time,
            end_time=end_time,
            correlation_id=correlation_id,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc,
            metadata_filters=metadata_filters,
        )

        # Assert
        assert query.event_types == event_types
        assert query.symbols == symbols
        assert query.sources == sources
        assert query.start_time == start_time
        assert query.end_time == end_time
        assert query.correlation_id == correlation_id
        assert query.limit == limit
        assert query.offset == offset
        assert query.order_by == order_by
        assert query.order_desc == order_desc
        assert query.metadata_filters == metadata_filters

    @pytest.mark.unit
    def test_event_query_with_single_event_type(self):
        """
        Test EventQuery with single event type.

        Verifies:
        - Single event type filtering works
        - Other parameters remain at defaults
        """
        # Act
        query = EventQuery(event_types=[EventType.ERROR])

        # Assert
        assert query.event_types == [EventType.ERROR]
        assert len(query.event_types) == 1

    @pytest.mark.unit
    def test_event_query_with_multiple_event_types(self):
        """
        Test EventQuery with multiple event types.

        Verifies:
        - Multiple event types are properly stored
        - All EventType enum values work
        """
        # Arrange
        all_event_types = [
            EventType.TRADE,
            EventType.KLINE,
            EventType.ORDER,
            EventType.CONNECTION,
            EventType.ERROR,
            EventType.SYSTEM,
        ]

        # Act
        query = EventQuery(event_types=all_event_types)

        # Assert
        assert query.event_types == all_event_types
        assert len(query.event_types) == 6

    @pytest.mark.unit
    def test_event_query_time_range_filtering(self):
        """
        Test EventQuery with time range filtering.

        Verifies:
        - Start and end times are properly stored
        - Time range logic is correct
        """
        # Arrange
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 1, 18, 0, 0, tzinfo=UTC)

        # Act
        query = EventQuery(start_time=start_time, end_time=end_time)

        # Assert
        assert query.start_time == start_time
        assert query.end_time == end_time
        assert query.start_time < query.end_time

    @pytest.mark.unit
    def test_event_query_pagination_parameters(self):
        """
        Test EventQuery with pagination parameters.

        Verifies:
        - Limit and offset work correctly
        - Boundary values are handled properly
        """
        # Test normal pagination
        query1 = EventQuery(limit=50, offset=100)
        assert query1.limit == 50
        assert query1.offset == 100

        # Test zero offset
        query2 = EventQuery(limit=25, offset=0)
        assert query2.limit == 25
        assert query2.offset == 0

        # Test no limit
        query3 = EventQuery(offset=10)
        assert query3.limit is None
        assert query3.offset == 10

    @pytest.mark.unit
    def test_event_query_ordering_parameters(self):
        """
        Test EventQuery with different ordering parameters.

        Verifies:
        - Different order_by fields work
        - Both ascending and descending order work
        """
        # Test ascending order by event_id
        query1 = EventQuery(order_by="event_id", order_desc=False)
        assert query1.order_by == "event_id"
        assert query1.order_desc is False

        # Test descending order by timestamp (default)
        query2 = EventQuery(order_by="timestamp", order_desc=True)
        assert query2.order_by == "timestamp"
        assert query2.order_desc is True

        # Test custom field ordering
        query3 = EventQuery(order_by="priority", order_desc=True)
        assert query3.order_by == "priority"
        assert query3.order_desc is True

    @pytest.mark.unit
    def test_event_query_metadata_filtering(self):
        """
        Test EventQuery with metadata filtering.

        Verifies:
        - Metadata filters are properly stored
        - Complex metadata structures work
        """
        # Arrange
        metadata_filters = {
            "exchange": "binance",
            "market_type": "spot",
            "user_id": "12345",
            "session_id": "sess_abc123",
        }

        # Act
        query = EventQuery(metadata_filters=metadata_filters)

        # Assert
        assert query.metadata_filters == metadata_filters
        assert len(query.metadata_filters) == 4

    @pytest.mark.unit
    def test_event_query_to_dict_method(self):
        """
        Test EventQuery to_dict method.

        Verifies:
        - Dictionary contains all fields
        - Datetime objects are properly serialized
        - Enum values are converted to strings
        - Structure is correct for serialization
        """
        # Arrange
        start_time = datetime(2024, 1, 1, tzinfo=UTC)
        end_time = datetime(2024, 1, 2, tzinfo=UTC)

        query = EventQuery(
            event_types=[EventType.TRADE, EventType.KLINE],
            symbols=["BTCUSDT"],
            sources=["binance"],
            start_time=start_time,
            end_time=end_time,
            correlation_id="test_001",
            limit=100,
            offset=50,
            order_by="timestamp",
            order_desc=True,
            metadata_filters={"test": "value"},
        )

        # Act
        query_dict = query.to_dict()

        # Assert
        assert "event_types" in query_dict
        assert "symbols" in query_dict
        assert "sources" in query_dict
        assert "start_time" in query_dict
        assert "end_time" in query_dict
        assert "correlation_id" in query_dict
        assert "limit" in query_dict
        assert "offset" in query_dict
        assert "order_by" in query_dict
        assert "order_desc" in query_dict
        assert "metadata_filters" in query_dict

        # Check enum serialization
        assert query_dict["event_types"] == ["trade", "kline"]

        # Check datetime serialization
        assert query_dict["start_time"] == start_time.isoformat()
        assert query_dict["end_time"] == end_time.isoformat()

        # Check other values
        assert query_dict["symbols"] == ["BTCUSDT"]
        assert query_dict["correlation_id"] == "test_001"
        assert query_dict["limit"] == 100

    @pytest.mark.unit
    def test_event_query_to_dict_with_none_values(self):
        """
        Test EventQuery to_dict method with None values.

        Verifies:
        - None values are properly serialized
        - Optional fields handle None correctly
        """
        # Act
        query = EventQuery()
        query_dict = query.to_dict()

        # Assert
        assert query_dict["start_time"] is None
        assert query_dict["end_time"] is None
        assert query_dict["correlation_id"] is None
        assert query_dict["limit"] is None
        assert query_dict["event_types"] == []
        assert query_dict["symbols"] == []

    @pytest.mark.unit
    def test_event_query_boundary_values(self):
        """
        Test EventQuery with boundary values.

        Verifies:
        - Very large limits work
        - Very large offsets work
        - Edge case datetime values work
        """
        # Test large pagination values
        query1 = EventQuery(limit=999999, offset=999999)
        assert query1.limit == 999999
        assert query1.offset == 999999

        # Test minimum datetime
        min_time = datetime.min.replace(tzinfo=UTC)
        query2 = EventQuery(start_time=min_time)
        assert query2.start_time == min_time

        # Test maximum datetime
        max_time = datetime.max.replace(tzinfo=UTC)
        query3 = EventQuery(end_time=max_time)
        assert query3.end_time == max_time

    @pytest.mark.unit
    def test_event_query_empty_lists_vs_none(self):
        """
        Test EventQuery behavior with empty lists vs None.

        Verifies:
        - Empty lists are used instead of None for list parameters
        - Behavior is consistent
        """
        # Test with explicit empty lists
        query1 = EventQuery(event_types=[], symbols=[], sources=[])

        assert query1.event_types == []
        assert query1.symbols == []
        assert query1.sources == []

        # Test with None (should convert to empty lists)
        query2 = EventQuery(event_types=None, symbols=None, sources=None)

        assert query2.event_types == []
        assert query2.symbols == []
        assert query2.sources == []

    @pytest.mark.unit
    def test_event_query_complex_metadata_filters(self):
        """
        Test EventQuery with complex metadata filter structures.

        Verifies:
        - Nested dictionaries work
        - Various data types in metadata work
        - Complex filter combinations work
        """
        # Arrange
        complex_metadata = {
            "string_field": "test_value",
            "numeric_field": 12345,
            "boolean_field": True,
            "list_field": ["item1", "item2"],
            "nested_dict": {"inner_field": "inner_value", "inner_number": 67890},
        }

        # Act
        query = EventQuery(metadata_filters=complex_metadata)

        # Assert
        assert query.metadata_filters == complex_metadata
        assert query.metadata_filters["string_field"] == "test_value"
        assert query.metadata_filters["numeric_field"] == 12345
        assert query.metadata_filters["nested_dict"]["inner_field"] == "inner_value"

    @pytest.mark.unit
    def test_event_query_symbol_case_sensitivity(self):
        """
        Test EventQuery symbol handling and case sensitivity.

        Verifies:
        - Symbols are stored as provided
        - Case sensitivity is preserved
        """
        # Arrange
        symbols = ["BTCUSDT", "btcusdt", "BtcUsdt"]

        # Act
        query = EventQuery(symbols=symbols)

        # Assert
        assert query.symbols == symbols
        assert len(set(query.symbols)) == 3  # All different due to case

    @pytest.mark.unit
    def test_event_query_correlation_id_formats(self):
        """
        Test EventQuery with different correlation ID formats.

        Verifies:
        - Various correlation ID formats work
        - Special characters are handled
        """
        # Test UUID format
        uuid_correlation = "550e8400-e29b-41d4-a716-446655440000"
        query1 = EventQuery(correlation_id=uuid_correlation)
        assert query1.correlation_id == uuid_correlation

        # Test custom format
        custom_correlation = "batch_2024_001_trade_processing"
        query2 = EventQuery(correlation_id=custom_correlation)
        assert query2.correlation_id == custom_correlation

        # Test with special characters
        special_correlation = "corr-id_!@#$%^&*()"
        query3 = EventQuery(correlation_id=special_correlation)
        assert query3.correlation_id == special_correlation
