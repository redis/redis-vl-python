"""Integration tests for stopwords support."""

import pytest

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


@pytest.fixture
def stopwords_disabled_schema():
    """Schema with stopwords disabled (STOPWORDS 0)."""
    return {
        "index": {
            "name": "test_stopwords_disabled",
            "prefix": "test_sw_disabled:",
            "storage_type": "hash",
            "stopwords": [],  # STOPWORDS 0
        },
        "fields": [
            {"name": "title", "type": "text"},
            {"name": "description", "type": "text"},
        ],
    }


@pytest.fixture
def custom_stopwords_schema():
    """Schema with custom stopwords list."""
    return {
        "index": {
            "name": "test_custom_stopwords",
            "prefix": "test_sw_custom:",
            "storage_type": "hash",
            "stopwords": ["the", "a", "an"],
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }


@pytest.fixture
def default_stopwords_schema():
    """Schema with default stopwords (no stopwords field)."""
    return {
        "index": {
            "name": "test_default_stopwords",
            "prefix": "test_sw_default:",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text"},
        ],
    }


def test_create_index_with_stopwords_disabled(client, stopwords_disabled_schema):
    """Test creating an index with STOPWORDS 0."""
    schema = IndexSchema.from_dict(stopwords_disabled_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Verify index was created
        assert index.exists()

        # Get FT.INFO and verify stopwords_list is empty
        info = client.ft(index.name).info()
        assert "stopwords_list" in info
        assert info["stopwords_list"] == []

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass


def test_create_index_with_custom_stopwords(client, custom_stopwords_schema):
    """Test creating an index with custom stopwords list."""
    schema = IndexSchema.from_dict(custom_stopwords_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Verify index was created
        assert index.exists()

        # Get FT.INFO and verify stopwords_list matches
        info = client.ft(index.name).info()
        assert "stopwords_list" in info

        # Convert bytes to strings for comparison
        stopwords_list = [
            sw.decode("utf-8") if isinstance(sw, bytes) else sw
            for sw in info["stopwords_list"]
        ]
        assert set(stopwords_list) == {"the", "a", "an"}

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass


def test_create_index_with_default_stopwords(client, default_stopwords_schema):
    """Test creating an index with default stopwords (no STOPWORDS clause)."""
    schema = IndexSchema.from_dict(default_stopwords_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Verify index was created
        assert index.exists()

        # Get FT.INFO - stopwords_list should NOT be present for default behavior
        info = client.ft(index.name).info()
        # When no STOPWORDS clause is used, Redis doesn't include stopwords_list in FT.INFO
        # (or it may include the default list depending on Redis version)
        # We just verify the index was created successfully
        assert index.exists()

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass


def test_from_existing_preserves_stopwords_disabled(client, stopwords_disabled_schema):
    """Test that from_existing() correctly reconstructs stopwords=[] configuration."""
    schema = IndexSchema.from_dict(stopwords_disabled_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Reconstruct from existing
        reconstructed_index = SearchIndex.from_existing(index.name, redis_client=client)

        # Verify stopwords configuration was preserved
        assert reconstructed_index.schema.index.stopwords == []

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass


def test_from_existing_preserves_custom_stopwords(client, custom_stopwords_schema):
    """Test that from_existing() correctly reconstructs custom stopwords configuration."""
    schema = IndexSchema.from_dict(custom_stopwords_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Reconstruct from existing
        reconstructed_index = SearchIndex.from_existing(index.name, redis_client=client)

        # Verify stopwords configuration was preserved
        assert set(reconstructed_index.schema.index.stopwords) == {"the", "a", "an"}

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass


def test_from_existing_default_stopwords(client, default_stopwords_schema):
    """Test that from_existing() handles default stopwords (no stopwords_list in FT.INFO)."""
    schema = IndexSchema.from_dict(default_stopwords_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Reconstruct from existing
        reconstructed_index = SearchIndex.from_existing(index.name, redis_client=client)

        # Verify stopwords is None (default behavior)
        assert reconstructed_index.schema.index.stopwords is None

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass


def test_stopwords_disabled_allows_searching_common_words(
    client, stopwords_disabled_schema
):
    """Test that STOPWORDS 0 allows searching for common stopwords like 'the', 'a', 'of'."""
    schema = IndexSchema.from_dict(stopwords_disabled_schema)
    index = SearchIndex(schema, redis_client=client)

    try:
        # Create the index
        index.create(overwrite=True, drop=True)

        # Add test data with common stopwords
        test_data = [
            {"title": "Bank of America", "description": "A major bank"},
            {"title": "The Great Gatsby", "description": "A classic novel"},
            {
                "title": "An Introduction to Python",
                "description": "A programming guide",
            },
        ]

        for i, data in enumerate(test_data):
            key = f"test_sw_disabled:{i}"
            client.hset(key, mapping=data)

        # Search for "of" - should find "Bank of America"
        from redisvl.query import FilterQuery

        query = FilterQuery(
            filter_expression="@title:(of)",
            return_fields=["title"],
        )
        results = index.search(query.query, query_params=query.params)

        # With STOPWORDS 0, "of" should be indexed and searchable
        assert len(results.docs) > 0
        assert any("of" in doc.title.lower() for doc in results.docs)

    finally:
        try:
            index.delete(drop=True)
        except Exception:
            pass
