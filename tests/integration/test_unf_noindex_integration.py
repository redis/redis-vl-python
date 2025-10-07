"""Integration tests for UNF and NOINDEX field attributes."""

import numpy as np
import pytest

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery


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

    def test_text_field_with_noindex_not_searchable(self, client, sample_data):
        """Test that TEXT field with NOINDEX cannot be searched."""
        schema = {
            "index": {"name": "test_noindex_text", "prefix": "noindex:"},
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

        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create(overwrite=True)
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
        titles = [doc["title"] for doc in results3]
        assert titles == sorted(titles)

        index.delete()

    def test_numeric_field_with_noindex_not_searchable(self, client, sample_data):
        """Test that NUMERIC field with NOINDEX cannot be searched."""
        schema = {
            "index": {"name": "test_noindex_numeric", "prefix": "noindex_num:"},
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

        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create(overwrite=True)
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
        scores = [float(doc["score"]) for doc in results3]
        assert scores == sorted(scores)

        index.delete()

    def test_tag_field_with_noindex_not_searchable(self, client, sample_data):
        """Test that TAG field with NOINDEX cannot be searched."""
        schema = {
            "index": {"name": "test_noindex_tag", "prefix": "noindex_tag:"},
            "fields": [
                {"name": "id", "type": "tag"},
                {
                    "name": "tags",
                    "type": "tag",
                    "attrs": {"no_index": True, "sortable": True},
                },
            ],
        }

        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create(overwrite=True)
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
        assert all("tags" in doc for doc in results2)

        index.delete()

    def test_mixed_index_and_noindex_fields(self, client, sample_data):
        """Test index with mix of indexed and non-indexed fields."""
        schema = {
            "index": {"name": "test_mixed_index", "prefix": "mixed:"},
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

        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create(overwrite=True)
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

        # Verify NOINDEX fields are still returned
        for doc in results:
            assert "title" in doc  # NOINDEX field should still be retrievable
            assert "score" in doc  # NOINDEX field should still be retrievable
            assert "content" in doc
            assert "price" in doc

        index.delete()


class TestUnfIntegration:
    """Test UNF functionality with real Redis."""

    def test_text_field_unf_sortable_unnormalized(self, client):
        """Test that TEXT field with UNF and SORTABLE preserves original case."""
        # Create two indices - one with UNF, one without
        schema_with_unf = {
            "index": {"name": "test_unf_text", "prefix": "unf:"},
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
            "index": {"name": "test_no_unf_text", "prefix": "no_unf:"},
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

        # Test with UNF (preserves case for sorting)
        index_unf = SearchIndex.from_dict(schema_with_unf, redis_client=client)
        index_unf.create(overwrite=True)
        index_unf.load(test_data)

        query = FilterQuery(
            return_fields=["id", "title"],
            filter_expression="*",
            sort_by="title",
        )
        results_unf = index_unf.query(query)
        titles_unf = [doc["title"] for doc in results_unf]

        # With UNF, uppercase comes before lowercase in ASCII order
        # Expected order: Banana, ZEBRA, apple (B=66, Z=90, a=97)
        assert titles_unf == ["Banana", "ZEBRA", "apple"]

        # Test without UNF (normalizes to lowercase for sorting)
        index_no_unf = SearchIndex.from_dict(schema_without_unf, redis_client=client)
        index_no_unf.create(overwrite=True)
        index_no_unf.load(test_data)

        query_no_unf = FilterQuery(
            return_fields=["id", "title"],
            filter_expression="*",
            sort_by="title",
        )
        results_no_unf = index_no_unf.query(query_no_unf)
        titles_no_unf = [doc["title"] for doc in results_no_unf]

        # Without UNF, all normalized to lowercase for sorting
        # Expected order: apple, Banana, ZEBRA (alphabetical)
        assert titles_no_unf == ["apple", "Banana", "ZEBRA"]

        index_unf.delete()
        index_no_unf.delete()

    def test_numeric_field_unf_behavior(self, client):
        """Test NUMERIC field UNF behavior - Redis always applies UNF to sortable numeric."""
        schema = {
            "index": {"name": "test_numeric_unf", "prefix": "num_unf:"},
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

        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create(overwrite=True)
        index.load(test_data)

        query = FilterQuery(
            return_fields=["id", "score"],
            filter_expression="*",
            sort_by="score",
        )
        results = index.query(query)
        scores = [float(doc["score"]) for doc in results]

        # Numeric sorting should work correctly
        assert scores == [50.2, 75.8, 100.5]

        index.delete()


class TestSchemaRoundtrip:
    """Test that schemas with UNF/NOINDEX can be saved and loaded correctly."""

    def test_schema_persistence_with_new_attributes(self, client, sample_data):
        """Test that index with UNF/NOINDEX can be created and retrieved."""
        schema = {
            "index": {"name": "test_persistence", "prefix": "persist:"},
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

        # Create index
        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create(overwrite=True)
        index.load(sample_data)

        # Load index from Redis
        index2 = SearchIndex.from_existing("test_persistence", redis_client=client)

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

        index.delete()
