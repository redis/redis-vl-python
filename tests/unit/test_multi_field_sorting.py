"""Unit tests for multiple field sorting functionality (issue #373)."""

import pytest

from redisvl.query import CountQuery, FilterQuery, TextQuery, VectorQuery
from redisvl.query.filter import Tag


class TestMultiFieldSortingFilterQuery:
    """Test multiple field sorting in FilterQuery."""

    def test_sort_by_single_field_string_backward_compat(self):
        """Test backward compatibility with single string sort_by."""
        query = FilterQuery(sort_by="price")

        # Should work as before - single field, default ASC
        assert query._sortby is not None
        # The _sortby object should have the field set
        assert "price" in str(query._sortby.args)

    def test_sort_by_single_field_with_direction(self):
        """Test single field with explicit direction as tuple."""
        query = FilterQuery(sort_by=("price", "DESC"))

        assert query._sortby is not None
        args_str = str(query._sortby.args)
        assert "price" in args_str
        assert "DESC" in args_str

    def test_sort_by_multiple_fields_list_of_strings(self):
        """Test multiple fields as list of strings (all ASC by default).

        Note: Redis Search only supports single-field sorting. Only the first field is used.
        """
        query = FilterQuery(sort_by=["price", "rating", "age"])

        assert query._sortby is not None
        # Only the first field should be in the sortby
        args_str = str(query._sortby.args)
        assert "price" in args_str
        assert "ASC" in args_str
        # Additional fields are not included in Redis SORTBY
        assert "rating" not in args_str
        assert "age" not in args_str

    def test_sort_by_multiple_fields_list_of_tuples(self):
        """Test multiple fields as list of tuples with directions.

        Note: Only the first field is actually used in Redis sorting.
        """
        query = FilterQuery(
            sort_by=[("price", "DESC"), ("rating", "ASC"), ("age", "DESC")]
        )

        assert query._sortby is not None
        args_str = str(query._sortby.args)
        # Only the first field should be in the sortby
        assert "price" in args_str
        assert "DESC" in args_str
        # Additional fields are not included
        assert "rating" not in args_str
        assert "age" not in args_str

    def test_sort_by_multiple_fields_mixed_formats(self):
        """Test multiple fields with mixed formats (string and tuple).

        Note: Only the first field is actually used in Redis sorting.
        """
        query = FilterQuery(sort_by=["price", ("rating", "DESC")])

        assert query._sortby is not None
        args_str = str(query._sortby.args)
        # Only the first field (price with default ASC) should be in sortby
        assert "price" in args_str
        assert "ASC" in args_str
        # Second field is not included
        assert "rating" not in args_str

    def test_sort_by_invalid_direction(self):
        """Test that invalid sort direction raises ValueError."""
        with pytest.raises(ValueError, match="Sort direction must be 'ASC' or 'DESC'"):
            FilterQuery(sort_by=[("price", "INVALID")])

    def test_sort_by_invalid_type(self):
        """Test that invalid sort_by type raises TypeError."""
        with pytest.raises(TypeError):
            FilterQuery(sort_by=123)

    def test_sort_by_empty_list(self):
        """Test that empty list is handled gracefully."""
        query = FilterQuery(sort_by=[])
        # Should not set sortby
        assert query._sortby is None

    def test_sort_by_none(self):
        """Test that None is handled gracefully (no sorting)."""
        query = FilterQuery(sort_by=None)
        assert query._sortby is None


class TestMultiFieldSortingVectorQuery:
    """Test multiple field sorting in VectorQuery."""

    def test_vector_query_default_sorts_by_distance(self):
        """Test that VectorQuery defaults to sorting by vector_distance."""
        query = VectorQuery(vector=[0.1, 0.2, 0.3], vector_field_name="embedding")

        assert query._sortby is not None
        assert "vector_distance" in str(query._sortby.args)

    def test_vector_query_custom_sort_single_field(self):
        """Test VectorQuery with custom single field sort."""
        query = VectorQuery(
            vector=[0.1, 0.2, 0.3], vector_field_name="embedding", sort_by="price"
        )

        assert query._sortby is not None
        assert "price" in str(query._sortby.args)

    def test_vector_query_custom_sort_multiple_fields(self):
        """Test VectorQuery with multiple field sort.

        Note: Only the first field is actually used in Redis sorting.
        """
        query = VectorQuery(
            vector=[0.1, 0.2, 0.3],
            vector_field_name="embedding",
            sort_by=[("price", "DESC"), "rating"],
        )

        assert query._sortby is not None
        args_str = str(query._sortby.args)
        # Only the first field should be used
        assert "price" in args_str
        assert "DESC" in args_str
        assert "rating" not in args_str


class TestMultiFieldSortingTextQuery:
    """Test multiple field sorting in TextQuery."""

    def test_text_query_single_field_sort(self):
        """Test TextQuery with single field sort."""
        query = TextQuery(
            text="example", text_field_name="content", sort_by="timestamp"
        )

        assert query._sortby is not None
        assert "timestamp" in str(query._sortby.args)

    def test_text_query_multiple_field_sort(self):
        """Test TextQuery with multiple field sort.

        Note: Only the first field is actually used in Redis sorting.
        """
        query = TextQuery(
            text="example",
            text_field_name="content",
            sort_by=[("relevance", "DESC"), ("timestamp", "ASC")],
        )

        assert query._sortby is not None
        args_str = str(query._sortby.args)
        # Only the first field should be used
        assert "relevance" in args_str
        assert "DESC" in args_str
        assert "timestamp" not in args_str


class TestSortByMethodChaining:
    """Test sort_by method can be called after initialization."""

    def test_filter_query_sort_by_chaining(self):
        """Test calling sort_by() method after FilterQuery initialization.

        Note: Only the first field is actually used in Redis sorting.
        """
        query = FilterQuery()
        query.sort_by([("price", "DESC"), "rating"])

        assert query._sortby is not None
        args_str = str(query._sortby.args)
        # Only the first field should be used
        assert "price" in args_str
        assert "DESC" in args_str
        assert "rating" not in args_str

    def test_sort_by_replacement(self):
        """Test that calling sort_by() replaces previous sort."""
        query = FilterQuery(sort_by="price")

        # Replace with new sort
        query.sort_by([("rating", "DESC")])

        # Should only have rating now (price replaced)
        args_str = str(query._sortby.args)
        assert "rating" in args_str
        assert "DESC" in args_str
        assert "price" not in args_str
