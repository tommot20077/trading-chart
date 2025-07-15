# ABOUTME: Unit tests for EventPriority system with comprehensive validation testing
# ABOUTME: Tests normal cases, exception cases, and boundary conditions for priority ordering system

import pytest
import sys

from core.models.event.event_priority import EventPriority


class TestEventPriority:
    """Test suite for EventPriority system validation and behavior."""

    def test_event_priority_predefined_constants(self):
        """Test predefined priority constants and their values."""
        # Act & Assert - predefined constants
        assert EventPriority.HIGHEST.value == -sys.maxsize - 1
        assert EventPriority.CRITICAL.value == 0
        assert EventPriority.HIGH.value == 100
        assert EventPriority.NORMAL.value == 200
        assert EventPriority.LOW.value == 300
        assert EventPriority.VERY_LOW.value == 400
        assert EventPriority.LOWEST.value == sys.maxsize

    def test_event_priority_ordering_logic(self):
        """Test priority ordering (lower numbers = higher priority)."""
        # Arrange
        highest = EventPriority(EventPriority.HIGHEST.value)
        critical = EventPriority(EventPriority.CRITICAL.value)
        high = EventPriority(EventPriority.HIGH.value)
        normal = EventPriority(EventPriority.NORMAL.value)
        low = EventPriority(EventPriority.LOW.value)
        lowest = EventPriority(EventPriority.LOWEST.value)
        
        # Act & Assert - ordering relationships
        assert highest < critical < high < normal < low < lowest
        assert lowest > low > normal > high > critical > highest

    def test_event_priority_custom_creation(self):
        """Test creating custom priority values."""
        # Act
        custom_priority = EventPriority(150)
        
        # Assert
        assert custom_priority.value == 150
        assert EventPriority.HIGH < custom_priority < EventPriority.NORMAL

    def test_event_priority_custom_class_method(self):
        """Test custom class method for creating priorities."""
        # Act
        custom_priority = EventPriority.custom(250)
        
        # Assert
        assert custom_priority.value == 250
        assert isinstance(custom_priority, EventPriority)
        assert EventPriority.NORMAL < custom_priority < EventPriority.LOW

    def test_event_priority_before_method(self):
        """Test before method for creating higher priority values."""
        # Act
        before_normal = EventPriority.before(EventPriority.NORMAL)
        before_normal_custom = EventPriority.before(EventPriority.NORMAL, offset=20)
        
        # Assert
        assert before_normal.value == EventPriority.NORMAL.value - 10  # Default offset
        assert before_normal_custom.value == EventPriority.NORMAL.value - 20
        assert before_normal < EventPriority.NORMAL
        assert before_normal_custom < EventPriority.NORMAL

    def test_event_priority_after_method(self):
        """Test after method for creating lower priority values."""
        # Act
        after_normal = EventPriority.after(EventPriority.NORMAL)
        after_normal_custom = EventPriority.after(EventPriority.NORMAL, offset=30)
        
        # Assert
        assert after_normal.value == EventPriority.NORMAL.value + 10  # Default offset
        assert after_normal_custom.value == EventPriority.NORMAL.value + 30
        assert after_normal > EventPriority.NORMAL
        assert after_normal_custom > EventPriority.NORMAL

    def test_event_priority_before_priority_method(self):
        """Test before_priority method with EventPriority instances."""
        # Arrange
        normal_priority = EventPriority(EventPriority.NORMAL.value)
        
        # Act
        before_priority = EventPriority.before_priority(normal_priority)
        before_priority_custom = EventPriority.before_priority(normal_priority, offset=25)
        
        # Assert
        assert before_priority.value == normal_priority.value - 10
        assert before_priority_custom.value == normal_priority.value - 25
        assert before_priority < normal_priority
        assert before_priority_custom < normal_priority

    def test_event_priority_after_priority_method(self):
        """Test after_priority method with EventPriority instances."""
        # Arrange
        high_priority = EventPriority(EventPriority.HIGH.value)
        
        # Act
        after_priority = EventPriority.after_priority(high_priority)
        after_priority_custom = EventPriority.after_priority(high_priority, offset=15)
        
        # Assert
        assert after_priority.value == high_priority.value + 10
        assert after_priority_custom.value == high_priority.value + 15
        assert after_priority > high_priority
        assert after_priority_custom > high_priority

    def test_event_priority_equality_comparison(self):
        """Test equality comparison between EventPriority instances and integers."""
        # Arrange
        priority1 = EventPriority(100)
        priority2 = EventPriority(100)
        priority3 = EventPriority(200)
        
        # Act & Assert - EventPriority equality
        assert priority1 == priority2
        assert priority1 != priority3
        
        # Act & Assert - integer equality
        assert priority1 == 100
        assert priority1 != 200
        
        # Act & Assert - inequality with other types
        assert priority1 != "100"
        assert priority1 != None
        assert priority1 != []

    def test_event_priority_less_than_comparison(self):
        """Test less than comparison operations."""
        # Arrange
        high_priority = EventPriority(EventPriority.HIGH.value)
        normal_priority = EventPriority(EventPriority.NORMAL.value)
        
        # Act & Assert - EventPriority comparison
        assert high_priority < normal_priority
        assert not normal_priority < high_priority
        
        # Act & Assert - integer comparison
        assert high_priority < 150
        assert not high_priority < 50
        
        # Act & Assert - invalid comparison should raise TypeError
        with pytest.raises(TypeError):
            high_priority < "invalid"

    def test_event_priority_less_than_or_equal_comparison(self):
        """Test less than or equal comparison operations."""
        # Arrange
        priority1 = EventPriority(100)
        priority2 = EventPriority(100)
        priority3 = EventPriority(200)
        
        # Act & Assert - EventPriority comparison
        assert priority1 <= priority2  # Equal
        assert priority1 <= priority3  # Less than
        assert not priority3 <= priority1
        
        # Act & Assert - integer comparison
        assert priority1 <= 100  # Equal
        assert priority1 <= 150  # Less than
        assert not priority1 <= 50

    def test_event_priority_greater_than_comparison(self):
        """Test greater than comparison operations."""
        # Arrange
        low_priority = EventPriority(EventPriority.LOW.value)
        high_priority = EventPriority(EventPriority.HIGH.value)
        
        # Act & Assert - EventPriority comparison
        assert low_priority > high_priority
        assert not high_priority > low_priority
        
        # Act & Assert - integer comparison
        assert low_priority > 250
        assert not low_priority > 350

    def test_event_priority_greater_than_or_equal_comparison(self):
        """Test greater than or equal comparison operations."""
        # Arrange
        priority1 = EventPriority(200)
        priority2 = EventPriority(200)
        priority3 = EventPriority(100)
        
        # Act & Assert - EventPriority comparison
        assert priority1 >= priority2  # Equal
        assert priority1 >= priority3  # Greater than
        assert not priority3 >= priority1
        
        # Act & Assert - integer comparison
        assert priority1 >= 200  # Equal
        assert priority1 >= 150  # Greater than
        assert not priority1 >= 250

    def test_event_priority_hash_functionality(self):
        """Test hash functionality for use in sets and dictionaries."""
        # Arrange
        priority1 = EventPriority(100)
        priority2 = EventPriority(100)
        priority3 = EventPriority(200)
        
        # Act
        priority_set = {priority1, priority2, priority3}
        priority_dict = {priority1: "high", priority3: "low"}
        
        # Assert
        assert len(priority_set) == 2  # priority1 and priority2 are equal
        assert priority_dict[priority1] == "high"
        assert priority_dict[priority3] == "low"
        assert hash(priority1) == hash(priority2)
        assert hash(priority1) != hash(priority3)

    def test_event_priority_string_representation(self):
        """Test string representation methods."""
        # Arrange
        priority = EventPriority(150)
        
        # Act & Assert - __str__ method
        assert str(priority) == "150"
        
        # Act & Assert - __repr__ method
        assert repr(priority) == "EventPriority(150)"

    def test_event_priority_boundary_values(self):
        """Test boundary cases with extreme values."""
        # Act & Assert - maximum values
        max_priority = EventPriority(sys.maxsize)
        assert max_priority.value == sys.maxsize
        
        # Act & Assert - minimum values
        min_priority = EventPriority(-sys.maxsize - 1)
        assert min_priority.value == -sys.maxsize - 1
        
        # Act & Assert - zero value
        zero_priority = EventPriority(0)
        assert zero_priority.value == 0
        assert zero_priority == EventPriority.CRITICAL

    def test_event_priority_sorting_behavior(self):
        """Test sorting behavior in collections."""
        # Arrange
        priorities = [
            EventPriority(EventPriority.LOW.value),
            EventPriority(EventPriority.CRITICAL.value),
            EventPriority(EventPriority.HIGH.value),
            EventPriority(EventPriority.NORMAL.value),
            EventPriority(EventPriority.HIGHEST.value)
        ]
        
        # Act
        sorted_priorities = sorted(priorities)
        
        # Assert - should be sorted by priority (lowest value first = highest priority)
        expected_order = [
            EventPriority.HIGHEST.value,
            EventPriority.CRITICAL.value,
            EventPriority.HIGH.value,
            EventPriority.NORMAL.value,
            EventPriority.LOW.value
        ]
        for i, priority in enumerate(sorted_priorities):
            assert priority.value == expected_order[i]

    def test_event_priority_chaining_operations(self):
        """Test chaining multiple priority operations."""
        # Arrange
        base_priority = EventPriority.NORMAL
        
        # Act
        higher_priority = EventPriority.before(base_priority, 50)
        even_higher = EventPriority.before_priority(higher_priority, 25)
        
        # Assert
        assert even_higher < higher_priority < base_priority
        assert even_higher.value == base_priority.value - 50 - 25
        assert even_higher.value == EventPriority.NORMAL.value - 75

    def test_event_priority_edge_case_offsets(self):
        """Test edge cases with large offsets."""
        # Act & Assert - large positive offset
        large_offset_priority = EventPriority.after(EventPriority.CRITICAL, 1000000)
        assert large_offset_priority.value == EventPriority.CRITICAL.value + 1000000
        
        # Act & Assert - large negative offset (via before)
        large_negative_priority = EventPriority.before(EventPriority.LOWEST, 1000000)
        assert large_negative_priority.value == EventPriority.LOWEST.value - 1000000