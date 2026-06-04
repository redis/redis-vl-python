"""Integration tests for UNF and NOINDEX field attributes."""

import numpy as np
import pytest

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery


def _index_config(redis_test_name, base: str):
    name = redis_test_name(base)
    return {"name": name, "prefix": f"{name}:"}


def _delete_index(index: SearchIndex):
    try:
        index.delete(drop=True)
    except Exception:
        pass


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _result_field(client, doc: dict, field: str):
    if field in doc:
        return _decode(doc[field])
    return _decode(client.hget(doc["id"], field))


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {
            "id": "doc1",
            "title": "First Document",
            "content": "This is searchable content",
            "score": 95.5,
            "price": 100,
            "tags": "red,blue",
            "location": "0,0",
            "vector": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
        },
        {
            "id": "doc2",
            "title": "Second Document",
            "content": "More searchable text here",
            "score": 87.3,
            "price": 200,
            "tags": "green,yellow",
            "location": "1,1",
            "vector": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32).tobytes(),
        },
        {
            "id": "doc3",
            "title": "Third Document",
            "content": "Additional content for search",
            "score": 92.1,
            "price": 150,
            "tags": "blue,yellow",
            "location": "2,2",
            "vector": np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32).tobytes(),
        },
    ]


class TestNoIndexIntegration:
    """Test NOINDEX functionality with real Redis."""

    def test_text_field_with_noindex_not_searchable(
        self, client, sample_data, redis_test_name
    ):
        """Test that TEXT field with NOINDEX cannot be searched."""
        index_config = _index_config(redis_test_name, "test_noindex_text")
        schema = {
            "index": index_config,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"no_index": True, "sortable": True},
                },
                {"name": "content", "type": "text"},
            ],
        }

        index = None
        try:
            index = SearchIndex.from_dict(schema, redis_client=client)
            index.create(overwrite=True, drop=True)
            index.load(sample_data)

            # Should NOT find documents when searching by title (NOINDEX field)
            query = FilterQuery(
                return_fields=["id", "title"],
                filter_expression="@title:(First)",
            )

            # NOINDEX fields return empty results, not an error
            results = index.query(query)
            assert len(results) == 0  # No results because field is not indexed

            # Should find documents when searching by content (indexed field)
            query2 = FilterQuery(
                return_fields=["id", "content"],
                filter_expression="@content:(searchable)",
            )
            results2 = index.query(query2)
            assert len(results2) > 0

            # But title should still be sortable
            query3 = FilterQuery(
                return_fields=["id", "title"],
                filter_expression="*",
                sort_by="title",
            )
            results3 = index.query(query3)
            assert len(results3) == 3
            # Verify sorting worked
            titles = [_result_field(client, doc, "title") for doc in results3]
            assert titles == sorted(titles)

        finally:
            if index is not None:
                _delete_index(index)

    def test_numeric_field_with_noindex_not_searchable(
        self, client, sample_data, redis_test_name
    ):
        """Test that NUMERIC field with NOINDEX cannot be searched."""
        index_config = _index_config(redis_test_name, "test_noindex_numeric")
        schema = {
            "index": index_config,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "score",
                    "type": "numeric",
                    "attrs": {"no_index": True, "sortable": True},
                },
                {"name": "price", "type": "numeric"},
            ],
        }

        index = None
        try:
            index = SearchIndex.from_dict(schema, redis_client=client)
            index.create(overwrite=True, drop=True)
            index.load(sample_data)

            # Should NOT find documents when filtering by score (NOINDEX field)
            query = FilterQuery(
                return_fields=["id", "score"],
                filter_expression="@score:[90 100]",
            )

            # NOINDEX fields return empty results, not an error
            results = index.query(query)
            assert len(results) == 0  # No results because field is not indexed

            # Should find documents when filtering by price (indexed field)
            query2 = FilterQuery(
                return_fields=["id", "price"],
                filter_expression="@price:[100 200]",
            )
            results2 = index.query(query2)
            assert len(results2) >= 2

            # But score should still be sortable
            query3 = FilterQuery(
                return_fields=["id", "score"],
                filter_expression="*",
                sort_by="score",
            )
            results3 = index.query(query3)
            assert len(results3) == 3
            # Verify sorting worked
            scores = [float(_result_field(client, doc, "score")) for doc in results3]
            assert scores == sorted(scores)

        finally:
            if index is not None:
                _delete_index(index)

    def test_tag_field_with_noindex_not_searchable(
        self, client, sample_data, redis_test_name
    ):
        """Test that TAG field with NOINDEX cannot be searched."""
        index_config = _index_config(redis_test_name, "test_noindex_tag")
        schema = {
            "index": index_config,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "tags",
                    "type": "tag",
                    "attrs": {"no_index": True, "sortable": True},
                },
            ],
        }

        index = None
        try:
            index = SearchIndex.from_dict(schema, redis_client=client)
            index.create(overwrite=True, drop=True)
            index.load(sample_data)

            # Should NOT find documents when filtering by tags (NOINDEX field)
            query = FilterQuery(
                return_fields=["id", "tags"],
                filter_expression="@tags:{blue}",
            )

            # NOINDEX fields return empty results, not an error
            results = index.query(query)
            assert len(results) == 0  # No results because field is not indexed

            # But tags should still be sortable and retrievable
            query2 = FilterQuery(
                return_fields=["id", "tags"],
                filter_expression="*",
                sort_by="tags",
            )
            results2 = index.query(query2)
            assert len(results2) == 3
            # Verify we can retrieve the field values
            assert all(
                _result_field(client, doc, "tags") is not None for doc in results2
            )

        finally:
            if index is not None:
                _delete_index(index)

    def test_mixed_index_and_noindex_fields(self, client, sample_data, redis_test_name):
        """Test index with mix of indexed and non-indexed fields."""
        index_config = _index_config(redis_test_name, "test_mixed_index")
        schema = {
            "index": index_config,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"no_index": True, "sortable": True},
                },
                {"name": "content", "type": "text"},
                {
                    "name": "score",
                    "type": "numeric",
                    "attrs": {"no_index": True, "sortable": True},
                },
                {"name": "price", "type": "numeric"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": 4,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        }

        index = None
        try:
            index = SearchIndex.from_dict(schema, redis_client=client)
            index.create(overwrite=True, drop=True)
            index.load(sample_data)

            # Complex query using only indexed fields
            query = VectorQuery(
                vector=[0.15, 0.25, 0.35, 0.45],
                vector_field_name="vector",
                return_fields=["id", "title", "content", "score", "price"],
                num_results=3,
            )
            results = index.query(query)
            assert len(results) >= 1

            # Verify NOINDEX fields are still stored and retrievable. Redis
            # latest can omit projected NOINDEX/SORTABLE fields from FT.SEARCH
            # results, so fall back to the backing hash for value checks.
            for doc in results:
                assert _result_field(client, doc, "title") is not None
                assert _result_field(client, doc, "score") is not None
                assert _result_field(client, doc, "content") is not None
                assert _result_field(client, doc, "price") is not None

        finally:
            if index is not None:
                _delete_index(index)


class TestUnfIntegration:
    """Test UNF functionality with real Redis."""

    def test_text_field_unf_sortable_unnormalized(self, client, redis_test_name):
        """Test that TEXT field with UNF and SORTABLE preserves original case."""
        # Create two indices - one with UNF, one without
        index_config_with_unf = _index_config(redis_test_name, "test_unf_text")
        index_config_without_unf = _index_config(redis_test_name, "test_no_unf_text")
        schema_with_unf = {
            "index": index_config_with_unf,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"unf": True, "sortable": True},
                },
            ],
        }

        schema_without_unf = {
            "index": index_config_without_unf,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"sortable": True},  # No UNF
                },
            ],
        }

        # Test data with mixed case
        test_data = [
            {"id": "1", "title": "ZEBRA"},
            {"id": "2", "title": "apple"},
            {"id": "3", "title": "Banana"},
        ]

        index_unf = None
        index_no_unf = None
        try:
            # Test with UNF (preserves case for sorting)
            index_unf = SearchIndex.from_dict(schema_with_unf, redis_client=client)
            index_unf.create(overwrite=True, drop=True)
            index_unf.load(test_data)

            query = FilterQuery(
                return_fields=["id", "title"],
                filter_expression="*",
                sort_by="title",
            )
            results_unf = index_unf.query(query)
            titles_unf = [_result_field(client, doc, "title") for doc in results_unf]

            # With UNF, uppercase comes before lowercase in ASCII order
            # Expected order: Banana, ZEBRA, apple (B=66, Z=90, a=97)
            assert titles_unf == ["Banana", "ZEBRA", "apple"]

            # Test without UNF (normalizes to lowercase for sorting)
            index_no_unf = SearchIndex.from_dict(
                schema_without_unf, redis_client=client
            )
            index_no_unf.create(overwrite=True, drop=True)
            index_no_unf.load(test_data)

            query_no_unf = FilterQuery(
                return_fields=["id", "title"],
                filter_expression="*",
                sort_by="title",
            )
            results_no_unf = index_no_unf.query(query_no_unf)
            titles_no_unf = [
                _result_field(client, doc, "title") for doc in results_no_unf
            ]

            # Without UNF, all normalized to lowercase for sorting
            # Expected order: apple, Banana, ZEBRA (alphabetical)
            assert titles_no_unf == ["apple", "Banana", "ZEBRA"]

        finally:
            if index_unf is not None:
                _delete_index(index_unf)
            if index_no_unf is not None:
                _delete_index(index_no_unf)

    def test_numeric_field_unf_behavior(self, client, redis_test_name):
        """Test NUMERIC field UNF behavior - Redis always applies UNF to sortable numeric."""
        index_config = _index_config(redis_test_name, "test_numeric_unf")
        schema = {
            "index": index_config,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "score",
                    "type": "numeric",
                    "attrs": {"sortable": True},  # UNF is implicit for numeric
                },
            ],
        }

        test_data = [
            {"id": "1", "score": 100.5},
            {"id": "2", "score": 50.2},
            {"id": "3", "score": 75.8},
        ]

        index = None
        try:
            index = SearchIndex.from_dict(schema, redis_client=client)
            index.create(overwrite=True, drop=True)
            index.load(test_data)

            query = FilterQuery(
                return_fields=["id", "score"],
                filter_expression="*",
                sort_by="score",
            )
            results = index.query(query)
            scores = [float(_result_field(client, doc, "score")) for doc in results]

            # Numeric sorting should work correctly
            assert scores == [50.2, 75.8, 100.5]

        finally:
            if index is not None:
                _delete_index(index)


class TestSchemaRoundtrip:
    """Test that schemas with UNF/NOINDEX can be saved and loaded correctly."""

    def test_schema_persistence_with_new_attributes(
        self, client, sample_data, redis_test_name
    ):
        """Test that index with UNF/NOINDEX can be created and retrieved."""
        index_config = _index_config(redis_test_name, "test_persistence")
        schema = {
            "index": index_config,
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "title",
                    "type": "text",
                    "attrs": {"no_index": True, "sortable": True, "unf": True},
                },
                {
                    "name": "score",
                    "type": "numeric",
                    "attrs": {"no_index": True, "sortable": True},
                },
            ],
        }

        index = None
        try:
            # Create index
            index = SearchIndex.from_dict(schema, redis_client=client)
            index.create(overwrite=True, drop=True)
            index.load(sample_data)

            # Load index from Redis
            index2 = SearchIndex.from_existing(
                index_config["name"], redis_client=client
            )

            # Verify fields have correct attributes
            title_field = index2.schema.fields["title"]
            assert title_field.attrs.no_index is True
            assert title_field.attrs.sortable is True
            assert title_field.attrs.unf is True  # Should be preserved for TEXT field

            score_field = index2.schema.fields["score"]
            assert score_field.attrs.no_index is True
            assert score_field.attrs.sortable is True
            # Note: unf for numeric is not preserved as Redis always applies it

            # Verify the index still works
            query = FilterQuery(
                return_fields=["id", "title", "score"],
                filter_expression="*",
                sort_by="title",
            )
            results = index2.query(query)
            assert len(results) == 3

        finally:
            if index is not None:
                _delete_index(index)
