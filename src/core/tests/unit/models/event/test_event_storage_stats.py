# ABOUTME: Unit tests for EventStorageStats model
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for EventStorageStats

import pytest
from datetime import datetime, UTC

from core.models.event.event_storage_stats import EventStorageStats


class TestEventStorageStats:
    """
    Comprehensive unit tests for EventStorageStats model.
    
    Tests cover:
    - Normal case: Valid EventStorageStats creation and behavior
    - Exception case: Invalid inputs and error handling
    - Boundary case: Edge cases and extreme values
    """

    def test_event_storage_stats_creation_with_defaults(self):
        """
        Test EventStorageStats creation with default values.
        
        Verifies:
        - All default values are properly set
        - Empty dictionary is created for events_by_type
        - Numeric defaults are zero
        - Time defaults are None
        """
        # Act
        stats = EventStorageStats()
        
        # Assert
        assert stats.total_events == 0
        assert stats.events_by_type == {}
        assert stats.storage_size_bytes == 0
        assert stats.oldest_event_time is None
        assert stats.newest_event_time is None
        assert stats.avg_event_size_bytes == 0.0

    def test_event_storage_stats_creation_with_all_parameters(self):
        """
        Test EventStorageStats creation with all parameters specified.
        
        Verifies:
        - All parameters are properly stored
        - Complex parameter combinations work correctly
        """
        # Arrange
        total_events = 10000
        events_by_type = {
            "trade": 5000,
            "kline": 3000,
            "error": 100,
            "connection": 50,
            "system": 1850
        }
        storage_size_bytes = 1024 * 1024 * 50  # 50MB
        oldest_event_time = datetime(2024, 1, 1, tzinfo=UTC)
        newest_event_time = datetime(2024, 1, 31, tzinfo=UTC)
        avg_event_size_bytes = 5120.5
        
        # Act
        stats = EventStorageStats(
            total_events=total_events,
            events_by_type=events_by_type,
            storage_size_bytes=storage_size_bytes,
            oldest_event_time=oldest_event_time,
            newest_event_time=newest_event_time,
            avg_event_size_bytes=avg_event_size_bytes
        )
        
        # Assert
        assert stats.total_events == total_events
        assert stats.events_by_type == events_by_type
        assert stats.storage_size_bytes == storage_size_bytes
        assert stats.oldest_event_time == oldest_event_time
        assert stats.newest_event_time == newest_event_time
        assert stats.avg_event_size_bytes == avg_event_size_bytes

    def test_event_storage_stats_events_by_type_structure(self):
        """
        Test EventStorageStats with various events_by_type structures.
        
        Verifies:
        - Different event type names work
        - Numeric values are properly stored
        - Dictionary structure is preserved
        """
        # Test with standard event types
        standard_events = {
            "trade": 1000,
            "kline": 500,
            "order": 200,
            "connection": 10,
            "error": 5,
            "system": 50
        }
        
        stats1 = EventStorageStats(events_by_type=standard_events)
        assert stats1.events_by_type == standard_events
        assert len(stats1.events_by_type) == 6
        
        # Test with custom event types
        custom_events = {
            "custom_event_type_1": 100,
            "custom_event_type_2": 200,
            "special-event": 50
        }
        
        stats2 = EventStorageStats(events_by_type=custom_events)
        assert stats2.events_by_type == custom_events

    def test_event_storage_stats_time_range_validation(self):
        """
        Test EventStorageStats with time range validation.
        
        Verifies:
        - Oldest and newest times work correctly
        - Time ordering is logical (oldest <= newest)
        """
        # Arrange
        oldest_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        newest_time = datetime(2024, 1, 31, 18, 30, 0, tzinfo=UTC)
        
        # Act
        stats = EventStorageStats(
            oldest_event_time=oldest_time,
            newest_event_time=newest_time
        )
        
        # Assert
        assert stats.oldest_event_time == oldest_time
        assert stats.newest_event_time == newest_time
        assert stats.oldest_event_time <= stats.newest_event_time

    def test_event_storage_stats_same_oldest_newest_time(self):
        """
        Test EventStorageStats when oldest and newest times are the same.
        
        Verifies:
        - Same time for both oldest and newest is valid
        - Represents single event or events at same time
        """
        # Arrange
        same_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        
        # Act
        stats = EventStorageStats(
            oldest_event_time=same_time,
            newest_event_time=same_time
        )
        
        # Assert
        assert stats.oldest_event_time == same_time
        assert stats.newest_event_time == same_time
        assert stats.oldest_event_time == stats.newest_event_time

    def test_event_storage_stats_large_numbers(self):
        """
        Test EventStorageStats with large numeric values.
        
        Verifies:
        - Large event counts work
        - Large storage sizes work
        - Large average sizes work
        """
        # Arrange - Large values
        large_total = 1_000_000_000  # 1 billion events
        large_storage = 1024 * 1024 * 1024 * 100  # 100GB
        large_avg = 10240.75  # ~10KB average
        
        # Act
        stats = EventStorageStats(
            total_events=large_total,
            storage_size_bytes=large_storage,
            avg_event_size_bytes=large_avg
        )
        
        # Assert
        assert stats.total_events == large_total
        assert stats.storage_size_bytes == large_storage
        assert stats.avg_event_size_bytes == large_avg

    def test_event_storage_stats_zero_values(self):
        """
        Test EventStorageStats with zero values.
        
        Verifies:
        - Zero values are valid for all numeric fields
        - Represents empty storage correctly
        """
        # Act
        stats = EventStorageStats(
            total_events=0,
            storage_size_bytes=0,
            avg_event_size_bytes=0.0
        )
        
        # Assert
        assert stats.total_events == 0
        assert stats.storage_size_bytes == 0
        assert stats.avg_event_size_bytes == 0.0

    def test_event_storage_stats_fractional_average_size(self):
        """
        Test EventStorageStats with fractional average event size.
        
        Verifies:
        - Decimal values work for average size
        - Precision is preserved
        """
        # Test various fractional values
        fractional_values = [0.1, 123.456789, 999.99, 0.000001]
        
        for avg_size in fractional_values:
            stats = EventStorageStats(avg_event_size_bytes=avg_size)
            assert stats.avg_event_size_bytes == avg_size

    def test_event_storage_stats_empty_events_by_type(self):
        """
        Test EventStorageStats with empty events_by_type dictionary.
        
        Verifies:
        - Empty dictionary is valid
        - Represents no event type breakdown
        """
        # Test explicit empty dictionary
        stats1 = EventStorageStats(events_by_type={})
        assert stats1.events_by_type == {}
        
        # Test None (should convert to empty dict)
        stats2 = EventStorageStats(events_by_type=None)
        assert stats2.events_by_type == {}

    def test_event_storage_stats_boundary_datetime_values(self):
        """
        Test EventStorageStats with boundary datetime values.
        
        Verifies:
        - Minimum and maximum datetime values work
        - Edge cases are handled properly
        """
        # Test minimum datetime
        min_time = datetime.min.replace(tzinfo=UTC)
        stats1 = EventStorageStats(oldest_event_time=min_time)
        assert stats1.oldest_event_time == min_time
        
        # Test maximum datetime
        max_time = datetime.max.replace(tzinfo=UTC)
        stats2 = EventStorageStats(newest_event_time=max_time)
        assert stats2.newest_event_time == max_time

    def test_event_storage_stats_realistic_scenario(self):
        """
        Test EventStorageStats with realistic production scenario.
        
        Verifies:
        - Realistic values work together
        - Calculations make sense
        """
        # Arrange - Realistic production scenario
        total_events = 50_000_000  # 50M events
        events_by_type = {
            "trade": 30_000_000,     # 60% trades
            "kline": 15_000_000,     # 30% klines
            "order": 3_000_000,      # 6% orders
            "connection": 1_000_000, # 2% connection events
            "error": 500_000,        # 1% errors
            "system": 500_000        # 1% system events
        }
        storage_size_bytes = 1024 * 1024 * 1024 * 25  # 25GB
        oldest_time = datetime(2024, 1, 1, tzinfo=UTC)
        newest_time = datetime(2024, 12, 31, tzinfo=UTC)
        avg_size = storage_size_bytes / total_events  # Calculate realistic average
        
        # Act
        stats = EventStorageStats(
            total_events=total_events,
            events_by_type=events_by_type,
            storage_size_bytes=storage_size_bytes,
            oldest_event_time=oldest_time,
            newest_event_time=newest_time,
            avg_event_size_bytes=avg_size
        )
        
        # Assert
        assert stats.total_events == total_events
        assert sum(stats.events_by_type.values()) == total_events
        assert stats.storage_size_bytes == storage_size_bytes
        assert stats.oldest_event_time < stats.newest_event_time
        assert stats.avg_event_size_bytes > 0

    def test_event_storage_stats_inconsistent_data(self):
        """
        Test EventStorageStats with potentially inconsistent data.
        
        Verifies:
        - Model accepts inconsistent data (validation is external concern)
        - All values are stored as provided
        """
        # Arrange - Inconsistent data (events_by_type sum != total_events)
        total_events = 1000
        events_by_type = {
            "trade": 500,
            "kline": 600  # Sum = 1100, which is > total_events
        }
        
        # Act - Should not raise error (validation is external)
        stats = EventStorageStats(
            total_events=total_events,
            events_by_type=events_by_type
        )
        
        # Assert
        assert stats.total_events == total_events
        assert stats.events_by_type == events_by_type
        assert sum(stats.events_by_type.values()) != stats.total_events

    def test_event_storage_stats_special_characters_in_event_types(self):
        """
        Test EventStorageStats with special characters in event type names.
        
        Verifies:
        - Special characters in keys are handled
        - Unicode characters work
        """
        # Arrange
        special_events = {
            "event-with-dashes": 100,
            "event_with_underscores": 200,
            "event.with.dots": 50,
            "event with spaces": 25,
            "事件类型": 10,  # Chinese characters
            "événement": 5   # French characters
        }
        
        # Act
        stats = EventStorageStats(events_by_type=special_events)
        
        # Assert
        assert stats.events_by_type == special_events
        assert stats.events_by_type["event-with-dashes"] == 100
        assert stats.events_by_type["事件类型"] == 10

    def test_event_storage_stats_negative_values(self):
        """
        Test EventStorageStats with negative values.
        
        Verifies:
        - Negative values are accepted (validation is external)
        - Model stores values as provided
        """
        # Act - Negative values (might be invalid but model should accept)
        stats = EventStorageStats(
            total_events=-100,
            storage_size_bytes=-1024,
            avg_event_size_bytes=-10.5
        )
        
        # Assert
        assert stats.total_events == -100
        assert stats.storage_size_bytes == -1024
        assert stats.avg_event_size_bytes == -10.5