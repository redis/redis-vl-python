"""Integration tests for TextQuery with field weights."""

import uuid

import pytest

from redisvl.index import SearchIndex
from redisvl.query import TextQuery
from redisvl.query.filter import Tag
from tests.conftest import skip_if_redis_version_below


@pytest.fixture
def weighted_index(client, redis_url, worker_id):
    # BM25 scorer requires Redis Stack 7.2.0 or higher
    skip_if_redis_version_below(client, "7.2.0", "BM25 scorer not available")
    """Create an index with multiple text fields for testing weights."""
    unique_id = str(uuid.uuid4())[:8]
    schema_dict = {
        "index": {
            "name": f"weighted_test_idx_{worker_id}_{unique_id}",
            "prefix": f"weighted_doc_{worker_id}_{unique_id}",
            "storage_type": "json",
        },
        "fields": [
            {"name": "title", "type": "text"},
            {"name": "content", "type": "text"},
            {"name": "tags", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "score", "type": "numeric"},
        ],
    }

    index = SearchIndex.from_dict(schema_dict, redis_url=redis_url)
    index.create(overwrite=True)

    # Load test data
    data = [
        {
            "id": "1",
            "title": "Redis database introduction",
            "content": "A comprehensive guide to getting started with Redis",
            "tags": "tutorial beginner",
            "category": "database",
            "score": 95,
        },
        {
            "id": "2",
            "title": "Advanced caching strategies",
            "content": "Learn about Redis caching patterns and best practices",
            "tags": "redis cache performance",
            "category": "optimization",
            "score": 88,
        },
        {
            "id": "3",
            "title": "Python programming basics",
            "content": "Introduction to Python with examples using Redis client",
            "tags": "python redis programming",
            "category": "programming",
            "score": 90,
        },
        {
            "id": "4",
            "title": "Data structures overview",
            "content": "Understanding Redis data structures and their applications",
            "tags": "redis structures",
            "category": "database",
            "score": 85,
        },
    ]

    index.load(data)
    yield index
    index.delete(drop=True)


def test_text_query_with_single_weighted_field(weighted_index):
    """Test TextQuery with a single weighted field."""
    text = "redis"

    # Query with higher weight on title
    query = TextQuery(
        text=text,
        text_field_name={"title": 5.0},
        return_fields=["title", "content"],
        num_results=4,
    )

    results = weighted_index.query(query)
    assert len(results) > 0

    # The document with "Redis" in the title should rank high
    top_result = results[0]
    assert "redis" in top_result["title"].lower()


def test_text_query_with_multiple_weighted_fields(weighted_index):
    """Test TextQuery with multiple weighted fields."""
    text = "redis"

    # Query across multiple fields with different weights
    query = TextQuery(
        text=text,
        text_field_name={"title": 3.0, "content": 2.0, "tags": 1.0},
        return_fields=["title", "content", "tags"],
        num_results=4,
    )

    results = weighted_index.query(query)
    assert len(results) > 0

    # Check that results contain the search term in at least one field
    for result in results:
        text_found = (
            "redis" in result.get("title", "").lower()
            or "redis" in result.get("content", "").lower()
            or "redis" in result.get("tags", "").lower()
        )
        assert text_found


def test_text_query_weights_with_filter(weighted_index):
    """Test TextQuery with weights and filter expression."""
    text = "redis"

    # Query with weights and filter
    filter_expr = Tag("category") == "database"
    query = TextQuery(
        text=text,
        text_field_name={"title": 5.0, "content": 1.0},
        filter_expression=filter_expr,
        return_fields=["title", "content", "category"],
        num_results=4,
    )

    results = weighted_index.query(query)

    # Should only get database category results
    for result in results:
        assert result["category"] == "database"


def test_dynamic_weight_update(weighted_index):
    """Test updating field weights dynamically."""
    text = "redis"

    # Start with equal weights
    query = TextQuery(
        text=text,
        text_field_name={"title": 1.0, "content": 1.0},
        return_fields=["title", "content"],
        num_results=4,
    )

    results1 = weighted_index.query(query)

    # Update to prioritize title
    query.set_field_weights({"title": 10.0, "content": 1.0})

    results2 = weighted_index.query(query)

    # Results might be reordered based on new weights
    # At minimum, both queries should return results
    assert len(results1) > 0
    assert len(results2) > 0


def test_backward_compatibility_single_field(weighted_index):
    """Test that the original single field API still works."""
    text = "redis"

    # Original API with single field name
    query = TextQuery(
        text=text,
        text_field_name="content",
        return_fields=["title", "content"],
        num_results=4,
    )

    results = weighted_index.query(query)
    assert len(results) > 0

    # Check results are from content field
    for result in results:
        if "redis" in result.get("content", "").lower():
            break
    else:
        # At least one result should have redis in content
        assert False, "No results with 'redis' in content field"
