"""Integration tests for multi-prefix index support.

Tests that queries return results from all configured prefixes when using
multi-prefix indexes.
"""

import uuid

import pytest

from redisvl.index import SearchIndex
from redisvl.query import (
    CountQuery,
    FilterQuery,
    TextQuery,
    VectorQuery,
    VectorRangeQuery,
)
from redisvl.query.filter import Num, Tag
from redisvl.redis.utils import array_to_buffer


@pytest.fixture
def multi_prefix_index(redis_url, worker_id):
    """Create a multi-prefix index with data loaded under different prefixes."""
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"multi_prefix_test_{worker_id}_{unique_id}"

    # Connect to get raw client for manual index creation
    from redisvl.redis.connection import RedisConnectionFactory

    client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)

    # Clean up any existing index
    try:
        client.ft(index_name).dropindex(delete_documents=True)
    except Exception:
        pass

    # Create index with multiple prefixes using raw command
    client.execute_command(
        "FT.CREATE",
        index_name,
        "ON",
        "HASH",
        "PREFIX",
        "2",
        f"prefix_a_{unique_id}:",
        f"prefix_b_{unique_id}:",
        "SCHEMA",
        "user",
        "TAG",
        "credit_score",
        "TAG",
        "job",
        "TEXT",
        "age",
        "NUMERIC",
        "user_embedding",
        "VECTOR",
        "FLAT",
        "6",
        "TYPE",
        "FLOAT32",
        "DIM",
        "3",
        "DISTANCE_METRIC",
        "COSINE",
    )

    # Connect using from_existing
    index = SearchIndex.from_existing(index_name, redis_url=redis_url)

    # Verify multi-prefix was preserved
    assert isinstance(index.schema.index.prefix, list)
    assert len(index.schema.index.prefix) == 2

    # Prepare test data for prefix_a (2 docs)
    data_prefix_a = [
        {
            "user": "john",
            "credit_score": "high",
            "job": "engineer at tech company",
            "age": 30,
            "user_embedding": array_to_buffer([0.1, 0.2, 0.3], dtype="float32"),
        },
        {
            "user": "jane",
            "credit_score": "medium",
            "job": "doctor at hospital",
            "age": 35,
            "user_embedding": array_to_buffer([0.2, 0.3, 0.4], dtype="float32"),
        },
    ]

    # Prepare test data for prefix_b (2 docs)
    data_prefix_b = [
        {
            "user": "bob",
            "credit_score": "low",
            "job": "teacher at school",
            "age": 40,
            "user_embedding": array_to_buffer([0.3, 0.4, 0.5], dtype="float32"),
        },
        {
            "user": "alice",
            "credit_score": "high",
            "job": "lawyer at firm",
            "age": 45,
            "user_embedding": array_to_buffer([0.4, 0.5, 0.6], dtype="float32"),
        },
    ]

    # Load data with explicit keys for each prefix
    keys_a = [
        f"prefix_a_{unique_id}:doc1",
        f"prefix_a_{unique_id}:doc2",
    ]
    keys_b = [
        f"prefix_b_{unique_id}:doc1",
        f"prefix_b_{unique_id}:doc2",
    ]

    index.load(data_prefix_a, keys=keys_a)
    index.load(data_prefix_b, keys=keys_b)

    yield index, unique_id

    # Cleanup
    try:
        index.delete(drop=True)
    except Exception:
        pass


def _count_prefixes(results, unique_id):
    """Count how many results come from each prefix."""
    prefix_a_count = sum(
        1 for r in results if r.get("id", "").startswith(f"prefix_a_{unique_id}:")
    )
    prefix_b_count = sum(
        1 for r in results if r.get("id", "").startswith(f"prefix_b_{unique_id}:")
    )
    return prefix_a_count, prefix_b_count


class TestMultiPrefixVectorQuery:
    """Test VectorQuery with multi-prefix indexes."""

    def test_vector_query_returns_both_prefixes(self, multi_prefix_index):
        """VectorQuery should return results from all prefixes."""
        index, unique_id = multi_prefix_index

        query = VectorQuery(
            vector=[0.25, 0.35, 0.45],
            vector_field_name="user_embedding",
            return_fields=["user", "credit_score", "job", "age"],
            num_results=10,
        )

        results = index.query(query)

        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        prefix_a_count, prefix_b_count = _count_prefixes(results, unique_id)
        assert prefix_a_count == 2, f"Expected 2 from prefix_a, got {prefix_a_count}"
        assert prefix_b_count == 2, f"Expected 2 from prefix_b, got {prefix_b_count}"

    def test_vector_query_with_filter_both_prefixes(self, multi_prefix_index):
        """VectorQuery with filter should return results from all prefixes."""
        index, unique_id = multi_prefix_index

        # Filter for credit_score == "high" (john from prefix_a, alice from prefix_b)
        query = VectorQuery(
            vector=[0.25, 0.35, 0.45],
            vector_field_name="user_embedding",
            return_fields=["user", "credit_score"],
            filter_expression=Tag("credit_score") == "high",
            num_results=10,
        )

        results = index.query(query)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        prefix_a_count, prefix_b_count = _count_prefixes(results, unique_id)
        assert prefix_a_count == 1, f"Expected 1 from prefix_a, got {prefix_a_count}"
        assert prefix_b_count == 1, f"Expected 1 from prefix_b, got {prefix_b_count}"


class TestMultiPrefixVectorRangeQuery:
    """Test VectorRangeQuery with multi-prefix indexes."""

    def test_range_query_returns_both_prefixes(self, multi_prefix_index):
        """VectorRangeQuery should return results from all prefixes within range."""
        index, unique_id = multi_prefix_index

        query = VectorRangeQuery(
            vector=[0.25, 0.35, 0.45],
            vector_field_name="user_embedding",
            return_fields=["user", "credit_score"],
            distance_threshold=0.5,  # Wide threshold to get all docs
            num_results=10,
        )

        results = index.query(query)

        # Should get results from both prefixes
        prefix_a_count, prefix_b_count = _count_prefixes(results, unique_id)
        assert prefix_a_count > 0, "Expected results from prefix_a"
        assert prefix_b_count > 0, "Expected results from prefix_b"


class TestMultiPrefixFilterQuery:
    """Test FilterQuery with multi-prefix indexes."""

    def test_filter_query_returns_both_prefixes(self, multi_prefix_index):
        """FilterQuery should return results from all prefixes."""
        index, unique_id = multi_prefix_index

        query = FilterQuery(
            return_fields=["user", "credit_score", "age"],
            filter_expression=Num("age") >= 30,
            num_results=10,
        )

        results = index.query(query)

        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        prefix_a_count, prefix_b_count = _count_prefixes(results, unique_id)
        assert prefix_a_count == 2, f"Expected 2 from prefix_a, got {prefix_a_count}"
        assert prefix_b_count == 2, f"Expected 2 from prefix_b, got {prefix_b_count}"

    def test_filter_query_tag_both_prefixes(self, multi_prefix_index):
        """FilterQuery with tag filter should return results from all prefixes."""
        index, unique_id = multi_prefix_index

        # Filter for credit_score == "high" (john from prefix_a, alice from prefix_b)
        query = FilterQuery(
            return_fields=["user", "credit_score"],
            filter_expression=Tag("credit_score") == "high",
            num_results=10,
        )

        results = index.query(query)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        prefix_a_count, prefix_b_count = _count_prefixes(results, unique_id)
        assert prefix_a_count == 1, f"Expected 1 from prefix_a, got {prefix_a_count}"
        assert prefix_b_count == 1, f"Expected 1 from prefix_b, got {prefix_b_count}"


class TestMultiPrefixCountQuery:
    """Test CountQuery with multi-prefix indexes."""

    def test_count_query_counts_all_prefixes(self, multi_prefix_index):
        """CountQuery should count documents from all prefixes."""
        index, unique_id = multi_prefix_index

        query = CountQuery(filter_expression=Tag("credit_score") == "high")
        count = index.query(query)

        assert count == 2, f"Expected count of 2, got {count}"

    def test_count_query_all_docs(self, multi_prefix_index):
        """CountQuery with wildcard should count all documents from all prefixes."""
        index, unique_id = multi_prefix_index

        query = CountQuery(filter_expression="*")
        count = index.query(query)

        assert count == 4, f"Expected count of 4, got {count}"


class TestMultiPrefixTextQuery:
    """Test TextQuery with multi-prefix indexes."""

    def test_text_query_returns_both_prefixes(self, multi_prefix_index):
        """TextQuery should return results from all prefixes."""
        index, unique_id = multi_prefix_index

        # Search for terms that appear in jobs from both prefixes
        # prefix_a: "engineer", "doctor" | prefix_b: "teacher", "lawyer"
        # Use wildcard to match partial terms
        query = TextQuery(
            text="engineer|doctor|teacher|lawyer",
            text_field_name="job",
            return_fields=["user", "job"],
            num_results=10,
        )

        results = index.query(query)

        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

        prefix_a_count, prefix_b_count = _count_prefixes(results, unique_id)
        assert prefix_a_count == 2, f"Expected 2 from prefix_a, got {prefix_a_count}"
        assert prefix_b_count == 2, f"Expected 2 from prefix_b, got {prefix_b_count}"

    def test_text_query_specific_term(self, multi_prefix_index):
        """TextQuery for specific term should return matching docs from any prefix."""
        index, unique_id = multi_prefix_index

        # Search for "engineer" (only john from prefix_a)
        query = TextQuery(
            text="engineer",
            text_field_name="job",
            return_fields=["user", "job"],
            num_results=10,
        )

        results = index.query(query)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0]["user"] == "john"


class TestMultiPrefixIndexCreation:
    """Test creating multi-prefix indexes via redisvl (not raw commands)."""

    def test_create_index_with_prefix_list(self, redis_url, worker_id):
        """Test that SearchIndex.create() works with a list of prefixes."""
        unique_id = str(uuid.uuid4())[:8]
        index_name = f"create_multi_prefix_{worker_id}_{unique_id}"

        schema = {
            "index": {
                "name": index_name,
                "prefix": [f"pfx_a_{unique_id}", f"pfx_b_{unique_id}"],
                "storage_type": "hash",
            },
            "fields": [
                {"name": "user", "type": "tag"},
                {"name": "age", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        }

        index = SearchIndex.from_dict(schema, redis_url=redis_url)

        try:
            # This should work with our fix
            index.create(overwrite=True)

            # Verify index was created
            assert index.exists()

            # Load data with explicit keys to different prefixes
            data = [
                {
                    "user": "test_user",
                    "age": 25,
                    "embedding": array_to_buffer([0.1, 0.2, 0.3], dtype="float32"),
                }
            ]

            # Load to first prefix
            keys_a = [f"pfx_a_{unique_id}:doc1"]
            index.load(data, keys=keys_a)

            # Load to second prefix
            keys_b = [f"pfx_b_{unique_id}:doc1"]
            index.load(data, keys=keys_b)

            # Query should find both
            query = CountQuery(filter_expression="*")
            count = index.query(query)
            assert count == 2, f"Expected 2 docs, got {count}"

        finally:
            try:
                index.delete(drop=True)
            except Exception:
                pass
