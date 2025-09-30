"""Integration tests for multiple field sorting feature (issue #373)."""

import pytest

from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, TextQuery, VectorQuery
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from tests.conftest import skip_if_redis_version_below


def assert_warning_logged(caplog, message_fragment):
    """Helper to assert that a warning containing the message fragment was logged.

    Args:
        caplog: pytest caplog fixture
        message_fragment: String fragment to search for in log messages
    """
    assert any(
        message_fragment in record.message for record in caplog.records
    ), f"Expected warning containing '{message_fragment}' to be logged"


@pytest.fixture
def products_schema():
    """Create a schema for product data with multiple sortable fields."""
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "products_test",
                "prefix": "product:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "name", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "price", "type": "numeric"},
                {"name": "rating", "type": "numeric"},
                {"name": "stock", "type": "numeric"},
                {"name": "brand", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "algorithm": "flat",
                        "distance_metric": "cosine",
                    },
                },
            ],
        }
    )


@pytest.fixture
def products_index(products_schema, client):
    """Create and populate a products index with test data."""
    index = SearchIndex(schema=products_schema, redis_client=client)
    index.create(overwrite=True, drop=True)

    # Load test product data with varied values for sorting
    products = [
        {
            "name": "Laptop Pro",
            "category": "electronics",
            "price": 1299.99,
            "rating": 4.5,
            "stock": 15,
            "brand": "TechCorp",
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "name": "Wireless Mouse",
            "category": "electronics",
            "price": 29.99,
            "rating": 4.8,
            "stock": 100,
            "brand": "TechCorp",
            "embedding": [0.2, 0.3, 0.4],
        },
        {
            "name": "USB Cable",
            "category": "electronics",
            "price": 9.99,
            "rating": 4.2,
            "stock": 250,
            "brand": "GenericBrand",
            "embedding": [0.3, 0.4, 0.5],
        },
        {
            "name": "Gaming Keyboard",
            "category": "electronics",
            "price": 149.99,
            "rating": 4.9,
            "stock": 45,
            "brand": "TechCorp",
            "embedding": [0.1, 0.3, 0.5],
        },
        {
            "name": "Monitor 27inch",
            "category": "electronics",
            "price": 349.99,
            "rating": 4.6,
            "stock": 30,
            "brand": "DisplayCo",
            "embedding": [0.2, 0.2, 0.4],
        },
        {
            "name": "Desk Chair",
            "category": "furniture",
            "price": 199.99,
            "rating": 4.3,
            "stock": 20,
            "brand": "FurniturePlus",
            "embedding": [0.5, 0.1, 0.2],
        },
        {
            "name": "Standing Desk",
            "category": "furniture",
            "price": 499.99,
            "rating": 4.7,
            "stock": 12,
            "brand": "FurniturePlus",
            "embedding": [0.4, 0.2, 0.1],
        },
        {
            "name": "Office Lamp",
            "category": "furniture",
            "price": 39.99,
            "rating": 4.4,
            "stock": 75,
            "brand": "LightCo",
            "embedding": [0.3, 0.3, 0.3],
        },
    ]

    # Preprocess function to convert embeddings to bytes for HASH storage
    def preprocess(item):
        """Convert embedding list to bytes for HASH storage."""
        item_copy = item.copy()
        item_copy["embedding"] = array_to_buffer(item["embedding"], "float32")
        return item_copy

    # Insert products with preprocessing
    for i, product in enumerate(products):
        index.load([product], keys=[f"product:{i}"], preprocess=preprocess)

    yield index

    # Cleanup
    index.delete(drop=True)


class TestMultiFieldSortingFilterQuery:
    """Integration tests for FilterQuery with multiple field sorting."""

    def test_single_field_sort_ascending(self, products_index):
        """Test sorting by single field in ascending order."""
        query = FilterQuery(sort_by="price", num_results=10)
        results = products_index.query(query)

        assert len(results) > 0
        # Verify results are sorted by price ascending
        prices = [float(doc["price"]) for doc in results]
        assert prices == sorted(prices), "Results should be sorted by price ascending"
        assert prices[0] == 9.99  # USB Cable (cheapest)
        assert prices[-1] == 1299.99  # Laptop Pro (most expensive)

    def test_single_field_sort_descending_tuple(self, products_index):
        """Test sorting by single field in descending order using tuple format."""
        query = FilterQuery(sort_by=("rating", "DESC"), num_results=10)
        results = products_index.query(query)

        assert len(results) > 0
        # Verify results are sorted by rating descending
        ratings = [float(doc["rating"]) for doc in results]
        assert ratings == sorted(
            ratings, reverse=True
        ), "Results should be sorted by rating descending"
        assert ratings[0] == 4.9  # Gaming Keyboard (highest rating)

    def test_multiple_fields_uses_first_only(self, products_index, caplog):
        """Test that multiple fields logs warning and uses only first field."""
        import logging

        caplog.set_level(logging.WARNING)

        query = FilterQuery(
            sort_by=[("price", "ASC"), ("rating", "DESC"), "stock"], num_results=10
        )
        results = products_index.query(query)

        # Check that warning was logged
        assert_warning_logged(caplog, "Multiple sort fields specified")
        assert_warning_logged(caplog, "Using first field: 'price'")

        # Verify only first field (price) is used for sorting
        assert len(results) > 0
        prices = [float(doc["price"]) for doc in results]
        assert prices == sorted(
            prices
        ), "Results should be sorted by price (first field)"

    def test_multiple_fields_mixed_format(self, products_index, caplog):
        """Test multiple fields with mixed string and tuple format."""
        import logging

        caplog.set_level(logging.WARNING)

        query = FilterQuery(sort_by=["stock", ("price", "DESC")], num_results=10)
        results = products_index.query(query)

        # Check warning
        assert_warning_logged(caplog, "Multiple sort fields specified")

        # Verify first field (stock) is used - ascending order
        assert len(results) > 0
        stock = [int(doc["stock"]) for doc in results]
        assert stock == sorted(stock), "Results should be sorted by stock ascending"

    def test_sort_with_filter_expression(self, products_index):
        """Test sorting combined with filter expression."""
        from redisvl.query.filter import Tag

        category_filter = Tag("category") == "electronics"
        query = FilterQuery(
            filter_expression=category_filter, sort_by=("price", "DESC"), num_results=10
        )
        results = products_index.query(query)

        # All results should be electronics
        assert all(doc["category"] == "electronics" for doc in results)

        # Should be sorted by price descending
        prices = [float(doc["price"]) for doc in results]
        assert prices == sorted(prices, reverse=True)


class TestMultiFieldSortingVectorQuery:
    """Integration tests for VectorQuery with multiple field sorting."""

    def test_vector_query_default_sort(self, products_index):
        """Test that VectorQuery defaults to sorting by vector distance."""
        query = VectorQuery(
            vector=[0.1, 0.2, 0.3],
            vector_field_name="embedding",
            num_results=5,
        )
        results = products_index.query(query)

        assert len(results) > 0
        # Should be sorted by vector_distance (ascending)
        distances = [float(doc["vector_distance"]) for doc in results]
        assert distances == sorted(
            distances
        ), "Results should be sorted by vector distance"

    def test_vector_query_custom_sort_single_field(self, products_index):
        """Test VectorQuery with custom single field sort."""
        query = VectorQuery(
            vector=[0.1, 0.2, 0.3],
            vector_field_name="embedding",
            return_fields=["name", "price", "rating"],
            sort_by=("price", "ASC"),
            num_results=5,
        )
        results = products_index.query(query)

        assert len(results) > 0
        # Should be sorted by price, not vector distance
        prices = [float(doc["price"]) for doc in results]
        assert prices == sorted(prices), "Results should be sorted by price"

    def test_vector_query_multiple_fields_warning(self, products_index, caplog):
        """Test that VectorQuery with multiple sort fields logs warning."""
        import logging

        caplog.set_level(logging.WARNING)

        query = VectorQuery(
            vector=[0.1, 0.2, 0.3],
            vector_field_name="embedding",
            return_fields=["name", "price", "rating"],
            sort_by=[("rating", "DESC"), "price"],
            num_results=5,
        )
        results = products_index.query(query)

        # Check warning
        assert_warning_logged(caplog, "Multiple sort fields specified")
        assert_warning_logged(caplog, "Using first field: 'rating'")

        # Verify first field (rating) is used
        assert len(results) > 0
        ratings = [float(doc["rating"]) for doc in results]
        assert ratings == sorted(ratings, reverse=True)


class TestMultiFieldSortingTextQuery:
    """Integration tests for TextQuery with multiple field sorting."""

    def test_text_query_with_custom_sort(self, products_index):
        """Test TextQuery with custom sort field."""
        skip_if_redis_version_below(
            products_index.client, "7.0.0", "BM25STD scorer requires Redis 7.0+"
        )

        query = TextQuery(
            text="tech",
            text_field_name="name",
            return_fields=["name", "price", "rating"],
            sort_by=("price", "DESC"),
            num_results=10,
        )
        results = products_index.query(query)

        # Results should contain "tech" in the name and be sorted by price descending
        if len(results) > 0:
            prices = [float(doc["price"]) for doc in results]
            assert prices == sorted(prices, reverse=True)

    def test_text_query_multiple_fields(self, products_index, caplog):
        """Test TextQuery with multiple sort fields."""
        skip_if_redis_version_below(
            products_index.client, "7.0.0", "BM25STD scorer requires Redis 7.0+"
        )

        import logging

        caplog.set_level(logging.WARNING)

        query = TextQuery(
            text="desk",
            text_field_name="name",
            return_fields=["name", "price", "rating"],
            sort_by=[("price", "ASC"), ("rating", "DESC")],
            num_results=10,
        )
        results = products_index.query(query)

        # Check warning
        assert_warning_logged(caplog, "Multiple sort fields specified")

        # Should use first field (price ASC)
        if len(results) > 0:
            prices = [float(doc["price"]) for doc in results]
            assert prices == sorted(prices)


class TestSortingEdgeCases:
    """Test edge cases and error handling for sorting."""

    def test_invalid_sort_direction(self):
        """Test that invalid sort direction raises ValueError."""
        with pytest.raises(ValueError, match="Sort direction must be 'ASC' or 'DESC'"):
            FilterQuery(sort_by=[("price", "INVALID")])

    def test_invalid_sort_type(self):
        """Test that invalid sort type raises TypeError."""
        with pytest.raises(TypeError):
            FilterQuery(sort_by=123)

    def test_empty_sort_list(self, products_index):
        """Test that empty sort list is handled gracefully."""
        query = FilterQuery(sort_by=[], num_results=10)
        results = products_index.query(query)

        # Should work, just no specific sort order (Redis default)
        assert len(results) > 0

    def test_sort_by_none(self, products_index):
        """Test that sort_by=None is handled gracefully."""
        query = FilterQuery(sort_by=None, num_results=10)
        results = products_index.query(query)

        # Should work, just no specific sort order
        assert len(results) > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing sort_by usage."""

    def test_old_style_sort_by_field_only(self, products_index):
        """Test old style sort_by with just field name (backward compatible)."""
        query = FilterQuery(sort_by="price", num_results=10)
        results = products_index.query(query)

        assert len(results) > 0
        prices = [float(doc["price"]) for doc in results]
        assert prices == sorted(prices)

    def test_old_style_sort_by_method_with_asc_param(self, products_index):
        """Test old style query.sort_by(field, asc=False) still works."""
        query = FilterQuery()
        query.sort_by("price", asc=False)

        results = products_index.query(query)

        assert len(results) > 0
        prices = [float(doc["price"]) for doc in results]
        assert prices == sorted(prices, reverse=True)

    def test_method_chaining_replaces_sort(self, products_index):
        """Test that calling sort_by() again replaces previous sort."""
        query = FilterQuery(sort_by="price")
        query.sort_by(("rating", "DESC"))

        results = products_index.query(query)

        # Should be sorted by rating (the replacement), not price
        assert len(results) > 0
        ratings = [float(doc["rating"]) for doc in results]
        assert ratings == sorted(ratings, reverse=True)
