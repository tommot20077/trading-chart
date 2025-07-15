# ABOUTME: Unit tests for QueryOptions class
# ABOUTME: Tests cover normal cases, validation, boundary conditions, and error handling

import pytest

from core.models.storage.query_option import QueryOptions


class TestQueryOptions:
    """Test cases for QueryOptions class."""

    @pytest.mark.unit
    def test_query_options_default_values(self):
        """Test QueryOptions creation with default values."""
        options = QueryOptions()

        assert options.limit is None
        assert options.offset is None
        assert options.order_by == "timestamp"
        assert options.order_desc is False
        assert options.include_metadata is False

    @pytest.mark.unit
    def test_query_options_with_all_parameters(self):
        """Test QueryOptions creation with all parameters specified."""
        options = QueryOptions(limit=100, offset=50, order_by="created_at", order_desc=True, include_metadata=True)

        assert options.limit == 100
        assert options.offset == 50
        assert options.order_by == "created_at"
        assert options.order_desc is True
        assert options.include_metadata is True

    @pytest.mark.unit
    def test_query_options_with_partial_parameters(self):
        """Test QueryOptions creation with some parameters specified."""
        options = QueryOptions(limit=25, order_desc=True)

        assert options.limit == 25
        assert options.offset is None
        assert options.order_by == "timestamp"
        assert options.order_desc is True
        assert options.include_metadata is False

    @pytest.mark.unit
    def test_query_options_limit_validation_positive(self):
        """Test QueryOptions with valid positive limit."""
        options = QueryOptions(limit=1)
        assert options.limit == 1

        options = QueryOptions(limit=1000)
        assert options.limit == 1000

    @pytest.mark.unit
    def test_query_options_limit_validation_zero(self):
        """Test QueryOptions with zero limit (valid)."""
        options = QueryOptions(limit=0)
        assert options.limit == 0

    @pytest.mark.unit
    def test_query_options_limit_validation_negative(self):
        """Test QueryOptions with negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be non-negative"):
            QueryOptions(limit=-1)

        with pytest.raises(ValueError, match="limit must be non-negative"):
            QueryOptions(limit=-100)

    @pytest.mark.unit
    def test_query_options_offset_validation_positive(self):
        """Test QueryOptions with valid positive offset."""
        options = QueryOptions(offset=1)
        assert options.offset == 1

        options = QueryOptions(offset=500)
        assert options.offset == 500

    @pytest.mark.unit
    def test_query_options_offset_validation_zero(self):
        """Test QueryOptions with zero offset (valid)."""
        options = QueryOptions(offset=0)
        assert options.offset == 0

    @pytest.mark.unit
    def test_query_options_offset_validation_negative(self):
        """Test QueryOptions with negative offset raises ValueError."""
        with pytest.raises(ValueError, match="offset must be non-negative"):
            QueryOptions(offset=-1)

        with pytest.raises(ValueError, match="offset must be non-negative"):
            QueryOptions(offset=-50)

    @pytest.mark.unit
    def test_query_options_order_by_variations(self):
        """Test QueryOptions with different order_by values."""
        order_by_values = ["timestamp", "created_at", "updated_at", "id", "price", "volume", "custom_field"]

        for order_by in order_by_values:
            options = QueryOptions(order_by=order_by)
            assert options.order_by == order_by

    @pytest.mark.unit
    def test_query_options_order_by_empty_string(self):
        """Test QueryOptions with empty string order_by."""
        options = QueryOptions(order_by="")
        assert options.order_by == ""

    @pytest.mark.unit
    def test_query_options_order_desc_boolean_values(self):
        """Test QueryOptions with different order_desc boolean values."""
        # Test True
        options = QueryOptions(order_desc=True)
        assert options.order_desc is True

        # Test False
        options = QueryOptions(order_desc=False)
        assert options.order_desc is False

    @pytest.mark.unit
    def test_query_options_include_metadata_boolean_values(self):
        """Test QueryOptions with different include_metadata boolean values."""
        # Test True
        options = QueryOptions(include_metadata=True)
        assert options.include_metadata is True

        # Test False
        options = QueryOptions(include_metadata=False)
        assert options.include_metadata is False

    @pytest.mark.unit
    @pytest.mark.external
    def test_query_options_boundary_values(self):
        """Test QueryOptions with boundary values."""
        # Maximum reasonable values
        options = QueryOptions(
            limit=999999999,
            offset=999999999,
            order_by="very_long_field_name_that_might_exist_in_some_database_schema",
            order_desc=True,
            include_metadata=True,
        )

        assert options.limit == 999999999
        assert options.offset == 999999999
        assert options.order_by == "very_long_field_name_that_might_exist_in_some_database_schema"
        assert options.order_desc is True
        assert options.include_metadata is True

    @pytest.mark.unit
    def test_query_options_none_values_explicit(self):
        """Test QueryOptions with explicitly set None values."""
        options = QueryOptions(limit=None, offset=None, order_by="timestamp", order_desc=False, include_metadata=False)

        assert options.limit is None
        assert options.offset is None
        assert options.order_by == "timestamp"
        assert options.order_desc is False
        assert options.include_metadata is False

    @pytest.mark.unit
    def test_query_options_mixed_valid_invalid_parameters(self):
        """Test QueryOptions with mix of valid and invalid parameters."""
        # Valid limit, invalid offset
        with pytest.raises(ValueError, match="offset must be non-negative"):
            QueryOptions(limit=10, offset=-5)

        # Invalid limit, valid offset
        with pytest.raises(ValueError, match="limit must be non-negative"):
            QueryOptions(limit=-10, offset=5)

        # Both invalid
        with pytest.raises(ValueError, match="limit must be non-negative"):
            QueryOptions(limit=-10, offset=-5)

    @pytest.mark.unit
    def test_query_options_attribute_access(self):
        """Test that all attributes are accessible after creation."""
        options = QueryOptions(limit=42, offset=21, order_by="custom_timestamp", order_desc=True, include_metadata=True)

        # Test attribute access
        assert hasattr(options, "limit")
        assert hasattr(options, "offset")
        assert hasattr(options, "order_by")
        assert hasattr(options, "order_desc")
        assert hasattr(options, "include_metadata")

        # Test values
        assert options.limit == 42
        assert options.offset == 21
        assert options.order_by == "custom_timestamp"
        assert options.order_desc is True
        assert options.include_metadata is True

    @pytest.mark.unit
    def test_query_options_attribute_modification(self):
        """Test that attributes can be modified after creation."""
        options = QueryOptions()

        # Modify attributes
        options.limit = 100
        options.offset = 50
        options.order_by = "modified_field"
        options.order_desc = True
        options.include_metadata = True

        # Verify modifications
        assert options.limit == 100
        assert options.offset == 50
        assert options.order_by == "modified_field"
        assert options.order_desc is True
        assert options.include_metadata is True

    @pytest.mark.unit
    def test_query_options_string_representation(self):
        """Test string representation of QueryOptions (if implemented)."""
        options = QueryOptions(limit=10, offset=5, order_by="timestamp", order_desc=True, include_metadata=False)

        # Test that str() doesn't raise an error
        str_repr = str(options)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
