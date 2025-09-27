"""Integration tests for skip_decode parameter in query return_fields (issue #252)."""

import numpy as np
import pytest
from redis import Redis

from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, RangeQuery, VectorQuery
from redisvl.schema import IndexSchema


@pytest.fixture
def sample_schema():
    """Create a sample schema with various field types."""
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "test_skip_decode",
                "prefix": "doc",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text"},
                {"name": "year", "type": "numeric"},
                {"name": "description", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 128,
                        "algorithm": "flat",
                        "distance_metric": "cosine",
                    },
                },
                {
                    "name": "image_data",
                    "type": "tag",
                },  # Will store binary data as tag
            ],
        }
    )


@pytest.fixture
def search_index(redis_url, sample_schema):
    """Create and populate a test index."""
    index = SearchIndex(sample_schema, redis_url=redis_url)

    # Clear any existing data
    try:
        index.delete(drop=True)
    except RedisSearchError:
        pass  # Index may not exist, which is fine

    # Create the index
    index.create(overwrite=True)

    # Populate with test data
    data = []
    for i in range(5):
        embedding_vector = np.random.rand(128).astype(np.float32)
        doc = {
            "title": f"Document {i}",
            "year": 2020 + i,
            "description": f"This is document number {i}",
            "embedding": embedding_vector.tobytes(),  # Store as binary
            "image_data": f"binary_image_{i}".encode("utf-8"),  # Store as binary
        }
        data.append(doc)

    # Load data into Redis
    index.load(data, id_field="title")

    yield index

    # Cleanup
    try:
        index.delete(drop=True)
    except RedisSearchError:
        pass  # Ignore cleanup errors


class TestSkipDecodeIntegration:
    """Integration tests for skip_decode functionality with real Redis."""

    def test_filter_query_skip_decode_single_field(self, search_index):
        """Test FilterQuery with skip_decode for embedding field."""
        query = FilterQuery(num_results=10)
        query.return_fields("title", "year", "embedding", skip_decode=["embedding"])

        results = search_index.query(query)

        # Verify we got results
        assert len(results) > 0

        # Check first result
        first_result = results[0]
        assert "title" in first_result
        assert "year" in first_result
        assert "embedding" in first_result

        # Title and year should be decoded strings
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["year"], str)  # Redis returns as string

        # Embedding should remain as bytes (not decoded)
        assert isinstance(first_result["embedding"], bytes)

    def test_filter_query_skip_decode_multiple_fields(self, search_index):
        """Test FilterQuery with skip_decode for multiple binary fields."""
        query = FilterQuery(num_results=10)
        query.return_fields(
            "title",
            "year",
            "embedding",
            "image_data",
            skip_decode=["embedding", "image_data"],
        )

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # Decoded fields
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["year"], str)

        # Non-decoded fields (should be bytes)
        assert isinstance(first_result["embedding"], bytes)
        assert isinstance(first_result["image_data"], bytes)

    def test_filter_query_no_skip_decode_default(self, search_index):
        """Test FilterQuery without skip_decode (default behavior)."""
        query = FilterQuery(num_results=10)
        query.return_fields("title", "year", "description")

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # All fields should be decoded to strings
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["year"], str)
        assert isinstance(first_result["description"], str)

    def test_vector_query_skip_decode(self, search_index):
        """Test VectorQuery with skip_decode for embedding field."""
        # Create a random query vector
        query_vector = np.random.rand(128).astype(np.float32)

        query = VectorQuery(
            vector=query_vector.tolist(),
            vector_field_name="embedding",
            return_fields=None,  # Will set with method
            num_results=3,
            return_score=True,  # Explicitly request distance score
        )

        # Use skip_decode for embedding
        query.return_fields("title", "embedding", skip_decode=["embedding"])

        results = search_index.query(query)

        assert len(results) > 0

        for result in results:
            assert isinstance(result["title"], str)
            # Embedding should be bytes (not decoded)
            assert isinstance(result["embedding"], bytes)
            # Distance score is added automatically by VectorQuery when return_score=True
            # but may not be in the result dict, just check the fields we requested

    def test_range_query_skip_decode(self, search_index):
        """Test RangeQuery with skip_decode for binary fields."""
        # Create a random query vector
        query_vector = np.random.rand(128).astype(np.float32)

        query = RangeQuery(
            vector=query_vector.tolist(),
            vector_field_name="embedding",
            distance_threshold=1.0,
            return_fields=None,
            num_results=10,
        )

        query.return_fields("title", "year", "embedding", skip_decode=["embedding"])

        results = search_index.query(query)

        if len(results) > 0:  # Range query might not return results
            first_result = results[0]
            assert isinstance(first_result["title"], str)
            assert isinstance(first_result["year"], str)
            assert isinstance(first_result["embedding"], bytes)

    def test_backward_compat_return_field_decode_false(self, search_index):
        """Test backward compatibility with return_field(decode_field=False)."""
        query = FilterQuery(num_results=10)

        # Use old API - return_field with decode_field=False
        query.return_field("embedding", decode_field=False)
        query.return_field("image_data", decode_field=False)
        query.return_fields("title", "year")  # These should be decoded

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # Decoded fields
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["year"], str)

        # Non-decoded fields (using old API)
        assert isinstance(first_result["embedding"], bytes)
        assert isinstance(first_result["image_data"], bytes)

    def test_mixed_api_usage(self, search_index):
        """Test mixing old and new API calls."""
        query = FilterQuery(num_results=10)

        # First use old API
        query.return_field("image_data", decode_field=False)

        # Then use new API with skip_decode
        query.return_fields("title", "year", "embedding", skip_decode=["embedding"])

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # The new API call should have replaced everything
        # (when skip_decode is provided, it clears previous fields)
        assert "title" in first_result
        assert "year" in first_result
        assert "embedding" in first_result

        # image_data should not be in results since return_fields
        # with skip_decode clears previous fields
        assert "image_data" not in first_result

    def test_skip_decode_with_empty_list(self, search_index):
        """Test skip_decode with empty list (all fields decoded)."""
        query = FilterQuery(num_results=10)
        query.return_fields("title", "year", "description", skip_decode=[])

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # All fields should be decoded
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["year"], str)
        assert isinstance(first_result["description"], str)

    def test_skip_decode_with_string_parameter(self, search_index):
        """Test skip_decode accepts a single string instead of list."""
        query = FilterQuery(num_results=10)

        # Pass a single string instead of list
        query.return_fields("title", "embedding", skip_decode="embedding")

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        assert isinstance(first_result["title"], str)
        # Embedding should be bytes (not decoded)
        assert isinstance(first_result["embedding"], bytes)

    def test_multiple_calls_without_skip_decode(self, search_index):
        """Test multiple return_fields calls without skip_decode (additive behavior)."""
        query = FilterQuery(num_results=10)

        # Multiple calls without skip_decode should be additive
        query.return_fields("title")
        query.return_fields("year")
        query.return_field("embedding", decode_field=False)

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # All fields should be present (additive behavior)
        assert "title" in first_result
        assert "year" in first_result
        assert "embedding" in first_result

        # Check types
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["year"], str)
        assert isinstance(first_result["embedding"], bytes)

    def test_replacement_behavior_with_skip_decode(self, search_index):
        """Test that skip_decode parameter triggers replacement behavior."""
        query = FilterQuery(num_results=10)

        # First set some fields
        query.return_fields("title", "description")

        # Then call with skip_decode - should replace, not add
        query.return_fields("year", "embedding", skip_decode=["embedding"])

        results = search_index.query(query)

        assert len(results) > 0

        first_result = results[0]
        # Only fields from second call should be present
        assert "year" in first_result
        assert "embedding" in first_result

        # Fields from first call should NOT be present (replaced)
        assert "title" not in first_result
        assert "description" not in first_result

        # Check embedding is not decoded
        assert isinstance(first_result["embedding"], bytes)
