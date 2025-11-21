"""Integration tests for stopwords support."""

import pytest

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
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


@pytest.fixture
def stopwords_disabled_index(client, stopwords_disabled_schema):
    """Index fixture with stopwords disabled."""
    schema = IndexSchema.from_dict(stopwords_disabled_schema)
    index = SearchIndex(schema, redis_client=client)
    index.create(overwrite=True, drop=True)

    yield index

    index.delete(drop=True)


@pytest.fixture
def custom_stopwords_index(client, custom_stopwords_schema):
    """Index fixture with custom stopwords."""
    schema = IndexSchema.from_dict(custom_stopwords_schema)
    index = SearchIndex(schema, redis_client=client)
    index.create(overwrite=True, drop=True)

    yield index

    index.delete(drop=True)


@pytest.fixture
def default_stopwords_index(client, default_stopwords_schema):
    """Index fixture with default stopwords."""
    schema = IndexSchema.from_dict(default_stopwords_schema)
    index = SearchIndex(schema, redis_client=client)
    index.create(overwrite=True, drop=True)

    yield index

    index.delete(drop=True)


def test_create_index_with_stopwords_disabled(client, stopwords_disabled_index):
    """Test creating an index with STOPWORDS 0."""
    # Verify index was created
    assert stopwords_disabled_index.exists()

    # Get FT.INFO and verify stopwords_list is empty
    info = client.ft(stopwords_disabled_index.name).info()
    assert "stopwords_list" in info
    assert info["stopwords_list"] == []


def test_create_index_with_custom_stopwords(client, custom_stopwords_index):
    """Test creating an index with custom stopwords list."""
    # Verify index was created
    assert custom_stopwords_index.exists()

    # Get FT.INFO and verify stopwords_list matches
    info = client.ft(custom_stopwords_index.name).info()
    assert "stopwords_list" in info

    # Convert bytes to strings for comparison
    stopwords_list = [
        sw.decode("utf-8") if isinstance(sw, bytes) else sw
        for sw in info["stopwords_list"]
    ]
    assert set(stopwords_list) == {"the", "a", "an"}


def test_create_index_with_default_stopwords(default_stopwords_index):
    """Test creating an index with default stopwords (no STOPWORDS clause)."""
    # Verify index was created
    assert default_stopwords_index.exists()

    # When no STOPWORDS clause is used, Redis doesn't include stopwords_list in FT.INFO
    # (or it may include the default list depending on Redis version)
    # We just verify the index was created successfully with default behavior


def test_from_existing_preserves_stopwords_disabled(client, stopwords_disabled_index):
    """Test that from_existing() correctly reconstructs stopwords=[] configuration."""
    # Reconstruct from existing
    reconstructed_index = SearchIndex.from_existing(
        stopwords_disabled_index.name, redis_client=client
    )

    # Verify stopwords configuration was preserved
    assert reconstructed_index.schema.index.stopwords == []


def test_from_existing_preserves_custom_stopwords(client, custom_stopwords_index):
    """Test that from_existing() correctly reconstructs custom stopwords configuration."""
    # Reconstruct from existing
    reconstructed_index = SearchIndex.from_existing(
        custom_stopwords_index.name, redis_client=client
    )

    # Verify stopwords configuration was preserved
    assert set(reconstructed_index.schema.index.stopwords) == {"the", "a", "an"}


def test_from_existing_default_stopwords(client, default_stopwords_index):
    """Test that from_existing() handles default stopwords (no stopwords_list in FT.INFO)."""
    # Reconstruct from existing
    reconstructed_index = SearchIndex.from_existing(
        default_stopwords_index.name, redis_client=client
    )

    # Verify stopwords is None (default behavior)
    assert reconstructed_index.schema.index.stopwords is None


def test_stopwords_disabled_allows_searching_common_words(
    client, stopwords_disabled_index
):
    """Test that STOPWORDS 0 allows searching for common stopwords like 'the', 'a', 'of'."""
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
    query = FilterQuery(
        filter_expression="@title:(of)",
        return_fields=["title"],
    )
    results = stopwords_disabled_index.search(query.query, query_params=query.params)

    # With STOPWORDS 0, "of" should be indexed and searchable
    assert len(results.docs) > 0
    assert any("of" in doc.title.lower() for doc in results.docs)
