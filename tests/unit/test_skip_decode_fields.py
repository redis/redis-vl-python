"""Unit tests for skip_decode parameter in query return_fields (issue #252)."""

import pytest

from redisvl.query import FilterQuery, RangeQuery, VectorQuery


class TestSkipDecodeFields:
    """Test the skip_decode parameter for return_fields method."""

    def test_filter_query_skip_decode_single_field(self):
        """Test FilterQuery with skip_decode for a single field."""
        query = FilterQuery(num_results=10)

        # Use the new skip_decode parameter
        query.return_fields("title", "year", "embedding", skip_decode=["embedding"])

        # Check that fields are added correctly
        assert hasattr(query, "_return_fields")
        assert "title" in query._return_fields
        assert "year" in query._return_fields
        assert "embedding" in query._return_fields

        # Check that decode settings are tracked
        assert hasattr(query, "_skip_decode_fields")
        assert "embedding" in query._skip_decode_fields

    def test_filter_query_skip_decode_multiple_fields(self):
        """Test FilterQuery with skip_decode for multiple fields."""
        query = FilterQuery(num_results=10)

        # Use skip_decode with multiple fields
        query.return_fields(
            "title",
            "year",
            "embedding",
            "image_data",
            skip_decode=["embedding", "image_data"],
        )

        # Check that all fields are added
        assert len(query._return_fields) == 4

        # Check that both fields are in skip_decode
        assert "embedding" in query._skip_decode_fields
        assert "image_data" in query._skip_decode_fields

    def test_vector_query_skip_decode_single_field(self):
        """Test VectorQuery with skip_decode parameter."""
        query = VectorQuery(
            vector=[0.1, 0.2, 0.3],
            vector_field_name="vector_field",
            return_fields=None,  # Will set with method
            num_results=5,
        )

        # Use the new API
        query.return_fields(
            "id", "vector_field", "metadata", skip_decode=["vector_field"]
        )

        # Check fields
        assert "id" in query._return_fields
        assert "vector_field" in query._return_fields
        assert "metadata" in query._return_fields

        # Check skip_decode
        assert hasattr(query, "_skip_decode_fields")
        assert "vector_field" in query._skip_decode_fields

    def test_range_query_skip_decode(self):
        """Test RangeQuery with skip_decode parameter."""
        query = RangeQuery(
            vector=[0.1, 0.2, 0.3],
            vector_field_name="embedding",
            distance_threshold=0.5,
            return_fields=None,
            num_results=10,
        )

        # Use skip_decode
        query.return_fields("doc_id", "text", "embedding", skip_decode=["embedding"])

        # Verify
        assert "doc_id" in query._return_fields
        assert "text" in query._return_fields
        assert "embedding" in query._return_fields
        assert "embedding" in query._skip_decode_fields

    def test_skip_decode_empty_list(self):
        """Test skip_decode with empty list (all fields should be decoded)."""
        query = FilterQuery(num_results=10)

        # Empty skip_decode list
        query.return_fields("field1", "field2", "field3", skip_decode=[])

        # All fields present but none skipped
        assert len(query._return_fields) == 3
        assert len(query._skip_decode_fields) == 0

    def test_skip_decode_none_default(self):
        """Test that skip_decode defaults to None (backwards compatible)."""
        query = FilterQuery(num_results=10)

        # No skip_decode parameter (backwards compatibility)
        query.return_fields("field1", "field2", "field3")

        # Fields should be added normally
        assert len(query._return_fields) == 3

        # No skip_decode_fields should be set
        if hasattr(query, "_skip_decode_fields"):
            assert len(query._skip_decode_fields) == 0

    def test_skip_decode_field_not_in_return_fields(self):
        """Test skip_decode with field not in return_fields (should be ignored)."""
        query = FilterQuery(num_results=10)

        # Skip decode for field not being returned
        query.return_fields("field1", "field2", skip_decode=["field3"])

        # Only requested fields should be present
        assert len(query._return_fields) == 2
        assert "field1" in query._return_fields
        assert "field2" in query._return_fields

        # Skip decode should be tracked even if field not returned
        # (implementation may choose to ignore or track it)
        assert hasattr(query, "_skip_decode_fields")

    def test_multiple_return_fields_calls_with_skip_decode(self):
        """Test calling return_fields multiple times with skip_decode."""
        query = FilterQuery(num_results=10)

        # First call
        query.return_fields("field1", skip_decode=["field1"])

        # Second call should replace, not append
        query.return_fields("field2", "field3", skip_decode=["field3"])

        # Should only have fields from second call
        assert "field1" not in query._return_fields
        assert "field2" in query._return_fields
        assert "field3" in query._return_fields

        # Skip decode should also be replaced
        assert "field1" not in query._skip_decode_fields
        assert "field3" in query._skip_decode_fields

    def test_skip_decode_with_string_input(self):
        """Test skip_decode accepts single string as well as list."""
        query = FilterQuery(num_results=10)

        # Single string for skip_decode
        query.return_fields("field1", "field2", skip_decode="field1")

        # Should work same as list with single element
        assert "field1" in query._skip_decode_fields
        assert "field2" not in query._skip_decode_fields

    def test_skip_decode_type_validation(self):
        """Test that skip_decode validates input types."""
        query = FilterQuery(num_results=10)

        # Invalid type should raise error
        with pytest.raises(TypeError, match="skip_decode must be"):
            query.return_fields("field1", skip_decode=123)

        # Dict should also fail
        with pytest.raises(TypeError, match="skip_decode must be"):
            query.return_fields("field1", skip_decode={"field": True})
