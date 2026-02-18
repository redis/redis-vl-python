"""Integration tests for SQLQuery class.

These tests verify that SQLQuery can translate SQL-like syntax
into proper Redis queries and return expected results.
"""

import uuid

import pytest

from redisvl.index import SearchIndex
from redisvl.query import SQLQuery


@pytest.fixture
def sql_index(redis_url, worker_id):
    """Create a products index for SQL query testing."""
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"sql_products_{worker_id}_{unique_id}"

    index = SearchIndex.from_dict(
        {
            "index": {
                "name": index_name,
                "prefix": f"product_{worker_id}_{unique_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text", "attrs": {"sortable": True}},
                {"name": "name", "type": "text", "attrs": {"sortable": True}},
                {"name": "price", "type": "numeric", "attrs": {"sortable": True}},
                {"name": "stock", "type": "numeric", "attrs": {"sortable": True}},
                {"name": "rating", "type": "numeric", "attrs": {"sortable": True}},
                {"name": "category", "type": "tag", "attrs": {"sortable": True}},
                {"name": "tags", "type": "tag"},
            ],
        },
        redis_url=redis_url,
    )

    index.create(overwrite=True)

    # Load test data
    products = [
        {
            "title": "Gaming laptop Pro",
            "name": "Gaming Laptop",
            "price": 899,
            "stock": 10,
            "rating": 4.5,
            "category": "electronics",
            "tags": "sale,featured",
        },
        {
            "title": "Budget laptop Basic",
            "name": "Budget Laptop",
            "price": 499,
            "stock": 25,
            "rating": 3.8,
            "category": "electronics",
            "tags": "sale",
        },
        {
            "title": "Premium laptop Ultra",
            "name": "Premium Laptop",
            "price": 1299,
            "stock": 5,
            "rating": 4.9,
            "category": "electronics",
            "tags": "featured",
        },
        {
            "title": "Python Programming",
            "name": "Python Book",
            "price": 45,
            "stock": 100,
            "rating": 4.7,
            "category": "books",
            "tags": "bestseller",
        },
        {
            "title": "Redis in Action",
            "name": "Redis Book",
            "price": 55,
            "stock": 50,
            "rating": 4.6,
            "category": "books",
            "tags": "featured",
        },
        {
            "title": "Data Science Guide",
            "name": "DS Book",
            "price": 65,
            "stock": 30,
            "rating": 4.4,
            "category": "books",
            "tags": "sale",
        },
        {
            "title": "Wireless Mouse",
            "name": "Mouse",
            "price": 29,
            "stock": 200,
            "rating": 4.2,
            "category": "electronics",
            "tags": "sale",
        },
        {
            "title": "Mechanical Keyboard",
            "name": "Keyboard",
            "price": 149,
            "stock": 75,
            "rating": 4.6,
            "category": "electronics",
            "tags": "featured",
        },
        {
            "title": "USB Hub",
            "name": "Hub",
            "price": 25,
            "stock": 150,
            "rating": 3.9,
            "category": "electronics",
            "tags": "sale",
        },
        {
            "title": "Monitor Stand",
            "name": "Stand",
            "price": 89,
            "stock": 40,
            "rating": 4.1,
            "category": "accessories",
            "tags": "sale,featured",
        },
        {
            "title": "Desk Lamp",
            "name": "Lamp",
            "price": 35,
            "stock": 80,
            "rating": 4.0,
            "category": "accessories",
            "tags": "sale",
        },
        {
            "title": "Notebook Set",
            "name": "Notebooks",
            "price": 15,
            "stock": 300,
            "rating": 4.3,
            "category": "stationery",
            "tags": "bestseller",
        },
        {
            "title": "Laptop and Keyboard Bundle",
            "name": "Bundle Pack",
            "price": 999,
            "stock": 15,
            "rating": 4.7,
            "category": "electronics",
            "tags": "featured,sale",
        },
    ]

    index.load(products)

    yield index

    # Cleanup
    index.delete(drop=True)


class TestSQLQueryBasic:
    """Tests for basic SQL SELECT queries."""

    def test_import_sql_query(self):
        """Verify SQLQuery can be imported from redisvl.query."""
        from redisvl.query import SQLQuery

        assert SQLQuery is not None

    def test_select_all_fields(self, sql_index):
        """Test SELECT * returns all fields."""
        sql_query = SQLQuery(f"SELECT * FROM {sql_index.name}")
        results = sql_index.query(sql_query)

        assert len(results) > 0
        # Verify results contain expected fields
        assert "title" in results[0]
        assert "price" in results[0]

    def test_select_specific_fields(self, sql_index):
        """Test SELECT with specific field list."""
        sql_query = SQLQuery(f"SELECT title, price FROM {sql_index.name}")
        results = sql_index.query(sql_query)

        assert len(results) > 0
        # Results should contain requested fields
        assert "title" in results[0]
        assert "price" in results[0]

    def test_redis_query_string_with_client(self, sql_index):
        """Test redis_query_string() with redis_client returns the Redis command string."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE category = 'electronics'
        """
        )

        # Get the Redis command string using redis_client
        redis_cmd = sql_query.redis_query_string(redis_client=sql_index._redis_client)

        # Verify it's a valid FT.SEARCH command
        assert redis_cmd.startswith("FT.SEARCH")
        assert sql_index.name in redis_cmd
        assert "electronics" in redis_cmd

    def test_redis_query_string_with_url(self, sql_index, redis_url):
        """Test redis_query_string() with redis_url returns the Redis command string."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE category = 'electronics'
        """
        )

        # Get the Redis command string using redis_url
        redis_cmd = sql_query.redis_query_string(redis_url=redis_url)

        # Verify it's a valid FT.SEARCH command
        assert redis_cmd.startswith("FT.SEARCH")
        assert sql_index.name in redis_cmd
        assert "electronics" in redis_cmd

    def test_redis_query_string_aggregate(self, sql_index):
        """Test redis_query_string() returns FT.AGGREGATE for aggregation queries."""
        sql_query = SQLQuery(
            f"""
            SELECT category, COUNT(*) as count
            FROM {sql_index.name}
            GROUP BY category
        """
        )

        redis_cmd = sql_query.redis_query_string(redis_client=sql_index._redis_client)

        # Verify it's a valid FT.AGGREGATE command
        assert redis_cmd.startswith("FT.AGGREGATE")
        assert sql_index.name in redis_cmd
        assert "GROUPBY" in redis_cmd


class TestSQLQueryWhere:
    """Tests for SQL WHERE clause filtering."""

    def test_where_tag_equals(self, sql_index):
        """Test WHERE with tag field equality."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price, category
            FROM {sql_index.name}
            WHERE category = 'electronics'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert result["category"] == "electronics"

    def test_where_numeric_comparison(self, sql_index):
        """Test WHERE with numeric field comparison."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price < 50
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert float(result["price"]) < 50

    def test_where_combined_and(self, sql_index):
        """Test WHERE with AND combining multiple conditions."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price, category
            FROM {sql_index.name}
            WHERE category = 'electronics' AND price < 100
        """
        )
        results = sql_index.query(sql_query)

        for result in results:
            assert result["category"] == "electronics"
            assert float(result["price"]) < 100

    def test_where_numeric_range(self, sql_index):
        """Test WHERE with numeric range (BETWEEN equivalent)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price >= 25 AND price <= 50
        """
        )
        results = sql_index.query(sql_query)

        for result in results:
            price = float(result["price"])
            assert 25 <= price <= 50


class TestSQLQueryTagOperators:
    """Tests for SQL tag field operators."""

    def test_tag_not_equals(self, sql_index):
        """Test tag != operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, category
            FROM {sql_index.name}
            WHERE category != 'electronics'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert result["category"] != "electronics"

    def test_tag_in(self, sql_index):
        """Test tag IN operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, category
            FROM {sql_index.name}
            WHERE category IN ('books', 'accessories')
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert result["category"] in ("books", "accessories")


class TestSQLQueryNumericOperators:
    """Tests for SQL numeric field operators."""

    def test_numeric_greater_than(self, sql_index):
        """Test numeric > operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price > 100
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert float(result["price"]) > 100

    def test_numeric_equals(self, sql_index):
        """Test numeric = operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price = 45
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            assert float(result["price"]) == 45

    def test_numeric_not_equals(self, sql_index):
        """Test numeric != operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price != 45
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert float(result["price"]) != 45

    @pytest.mark.xfail(reason="Numeric IN operator not yet supported in sql-redis")
    def test_numeric_in(self, sql_index):
        """Test numeric IN operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price IN (45, 55, 65)
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            assert float(result["price"]) in (45, 55, 65)

    def test_numeric_between(self, sql_index):
        """Test numeric BETWEEN operator."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            WHERE price BETWEEN 40 AND 60
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            price = float(result["price"])
            assert 40 <= price <= 60


class TestSQLQueryTextOperators:
    """Tests for SQL text field operators."""

    def test_text_equals(self, sql_index):
        """Test text = operator (full-text search)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE title = 'laptop'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            assert "laptop" in result["title"].lower()

    def test_text_not_equals(self, sql_index):
        """Test text != operator (negated full-text search)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE title != 'laptop'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            # Results should not contain 'laptop' as a primary match
            assert "laptop" not in result["title"].lower()

    def test_text_prefix(self, sql_index):
        """Test text prefix search with wildcard (term*)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE title = 'lap*'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            # Should match titles starting with "lap" (e.g., "laptop")
            assert "lap" in result["title"].lower()

    def test_text_suffix(self, sql_index):
        """Test text suffix search with wildcard (*term)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE name = '*book'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            # Should match names ending with "book" (e.g., "Python Book")
            assert "book" in result["name"].lower()

    def test_text_fuzzy(self, sql_index):
        """Test text fuzzy search with Levenshtein distance (%term%)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE title = '%laptap%'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            # Should fuzzy match "laptop" even with typo "laptap"
            assert "laptop" in result["title"].lower()

    def test_text_phrase(self, sql_index):
        """Test text phrase search (multi-word exact phrase)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE title = 'gaming laptop'
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            # Should match exact phrase "gaming laptop"
            title_lower = result["title"].lower()
            assert "gaming" in title_lower and "laptop" in title_lower

    def test_text_phrase_with_stopword(self, sql_index):
        """Test text phrase search containing stop words.

        Redis does not index stop words (like 'and', 'the', 'is') by default.
        The sql-redis library works around this by automatically stripping
        stop words from phrase searches and emitting a warning.
        See: https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/stopwords/
        """
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            sql_query = SQLQuery(
                f"""
                SELECT title, name
                FROM {sql_index.name}
                WHERE title = 'laptop and keyboard'
            """
            )
            results = sql_index.query(sql_query)

            # Should find the "Laptop and Keyboard Bundle" product
            assert len(results) >= 1
            # Verify at least one result contains both "laptop" and "keyboard"
            found_match = False
            for result in results:
                title_lower = result["title"].lower()
                if "laptop" in title_lower and "keyboard" in title_lower:
                    found_match = True
                    break
            assert found_match, "Expected to find a result with 'laptop' and 'keyboard'"

            # Verify a warning was emitted about stopword removal
            stopword_warnings = [
                warning
                for warning in w
                if "Stopwords" in str(warning.message)
                and "and" in str(warning.message).lower()
            ]
            assert (
                len(stopword_warnings) >= 1
            ), "Expected a warning about stopword removal"

    @pytest.mark.xfail(reason="Text IN operator not yet supported in sql-redis")
    def test_text_in(self, sql_index):
        """Test text IN operator (multiple term search)."""
        sql_query = SQLQuery(
            f"""
            SELECT title, name
            FROM {sql_index.name}
            WHERE title IN ('Python', 'Redis')
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) >= 1
        for result in results:
            title_lower = result["title"].lower()
            assert "python" in title_lower or "redis" in title_lower


class TestSQLQueryOrderBy:
    """Tests for SQL ORDER BY clause."""

    def test_order_by_asc(self, sql_index):
        """Test ORDER BY ascending."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            ORDER BY price ASC
        """
        )
        results = sql_index.query(sql_query)

        prices = [float(r["price"]) for r in results]
        assert prices == sorted(prices)

    def test_order_by_desc(self, sql_index):
        """Test ORDER BY descending."""
        sql_query = SQLQuery(
            f"""
            SELECT title, price
            FROM {sql_index.name}
            ORDER BY price DESC
        """
        )
        results = sql_index.query(sql_query)

        prices = [float(r["price"]) for r in results]
        assert prices == sorted(prices, reverse=True)


class TestSQLQueryLimit:
    """Tests for SQL LIMIT and OFFSET clauses."""

    def test_limit(self, sql_index):
        """Test LIMIT clause."""
        sql_query = SQLQuery(f"SELECT title FROM {sql_index.name} LIMIT 3")
        results = sql_index.query(sql_query)

        assert len(results) == 3

    def test_limit_with_offset(self, sql_index):
        """Test LIMIT with OFFSET for pagination."""
        # First page
        sql_query1 = SQLQuery(
            f"SELECT title FROM {sql_index.name} ORDER BY price ASC LIMIT 3 OFFSET 0"
        )
        results1 = sql_index.query(sql_query1)

        # Second page
        sql_query2 = SQLQuery(
            f"SELECT title FROM {sql_index.name} ORDER BY price ASC LIMIT 3 OFFSET 3"
        )
        results2 = sql_index.query(sql_query2)

        assert len(results1) == 3
        assert len(results2) == 3
        # Pages should have different results
        titles1 = {r["title"] for r in results1}
        titles2 = {r["title"] for r in results2}
        assert titles1.isdisjoint(titles2)


class TestSQLQueryAggregation:
    """Tests for SQL aggregation (GROUP BY, COUNT, AVG, etc.)."""

    def test_count_all(self, sql_index):
        """Test COUNT(*) aggregation."""
        sql_query = SQLQuery(f"SELECT COUNT(*) as total FROM {sql_index.name}")
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert int(results[0]["total"]) == 13  # 13 products in test data

    def test_group_by_with_count(self, sql_index):
        """Test GROUP BY with COUNT."""
        sql_query = SQLQuery(
            f"""
            SELECT category, COUNT(*) as count
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        # Should have groups for electronics, books, accessories, stationery
        categories = {r["category"] for r in results}
        assert "electronics" in categories
        assert "books" in categories

    def test_group_by_with_avg(self, sql_index):
        """Test GROUP BY with AVG."""
        sql_query = SQLQuery(
            f"""
            SELECT category, AVG(price) as avg_price
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        # All results should have category and avg_price
        for result in results:
            assert "category" in result
            assert "avg_price" in result
            assert float(result["avg_price"]) > 0

    def test_group_by_with_filter(self, sql_index):
        """Test GROUP BY with WHERE filter."""
        sql_query = SQLQuery(
            f"""
            SELECT category, AVG(price) as avg_price
            FROM {sql_index.name}
            WHERE stock > 50
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "category" in result
            assert "avg_price" in result

    def test_group_by_with_sum(self, sql_index):
        """Test GROUP BY with SUM reducer."""
        sql_query = SQLQuery(
            f"""
            SELECT category, SUM(price) as total_price
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "category" in result
            assert "total_price" in result
            assert float(result["total_price"]) > 0

    def test_group_by_with_min(self, sql_index):
        """Test GROUP BY with MIN reducer."""
        sql_query = SQLQuery(
            f"""
            SELECT category, MIN(price) as min_price
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "category" in result
            assert "min_price" in result
            assert float(result["min_price"]) > 0

    def test_group_by_with_max(self, sql_index):
        """Test GROUP BY with MAX reducer."""
        sql_query = SQLQuery(
            f"""
            SELECT category, MAX(price) as max_price
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "category" in result
            assert "max_price" in result
            assert float(result["max_price"]) > 0

    def test_global_sum(self, sql_index):
        """Test global SUM aggregation (no GROUP BY)."""
        sql_query = SQLQuery(
            f"""
            SELECT SUM(price) as total
            FROM {sql_index.name}
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert "total" in results[0]
        assert float(results[0]["total"]) > 0

    def test_global_min(self, sql_index):
        """Test global MIN aggregation (no GROUP BY)."""
        sql_query = SQLQuery(
            f"""
            SELECT MIN(price) as min_price
            FROM {sql_index.name}
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert "min_price" in results[0]
        assert float(results[0]["min_price"]) > 0

    def test_global_max(self, sql_index):
        """Test global MAX aggregation (no GROUP BY)."""
        sql_query = SQLQuery(
            f"""
            SELECT MAX(price) as max_price
            FROM {sql_index.name}
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert "max_price" in results[0]
        assert float(results[0]["max_price"]) > 0

    def test_multiple_reducers(self, sql_index):
        """Test multiple reducers in a single query."""
        sql_query = SQLQuery(
            f"""
            SELECT category, COUNT(*) as count, SUM(price) as total, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "category" in result
            assert "count" in result
            assert "total" in result
            assert "avg_price" in result
            assert "min_price" in result
            assert "max_price" in result

    def test_count_distinct(self, sql_index):
        """Test COUNT_DISTINCT reducer using Redis-specific syntax."""
        sql_query = SQLQuery(
            f"""
            SELECT COUNT_DISTINCT(category) as unique_categories
            FROM {sql_index.name}
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert "unique_categories" in results[0]
        # Should have 4 unique categories: electronics, books, accessories, stationery
        assert int(results[0]["unique_categories"]) == 4

    def test_stddev(self, sql_index):
        """Test STDDEV reducer."""
        sql_query = SQLQuery(
            f"""
            SELECT STDDEV(price) as price_stddev
            FROM {sql_index.name}
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert "price_stddev" in results[0]
        # Verify it's a valid numeric value
        stddev_value = float(results[0]["price_stddev"])
        assert stddev_value >= 0  # Standard deviation is always non-negative

    def test_quantile(self, sql_index):
        """Test QUANTILE reducer."""
        sql_query = SQLQuery(
            f"""
            SELECT QUANTILE(price, 0.5) as median_price
            FROM {sql_index.name}
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) == 1
        assert "median_price" in results[0]
        # Verify it's a valid numeric value
        median_value = float(results[0]["median_price"])
        assert median_value >= 0

    def test_tolist(self, sql_index):
        """Test TOLIST reducer via ARRAY_AGG SQL function."""
        sql_query = SQLQuery(
            f"""
            SELECT category, ARRAY_AGG(title) as titles
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "titles" in result
            # TOLIST returns a comma-separated string or list of values
            assert result["titles"] is not None

    def test_first_value(self, sql_index):
        """Test FIRST_VALUE reducer."""
        sql_query = SQLQuery(
            f"""
            SELECT category, FIRST_VALUE(title) as first_title
            FROM {sql_index.name}
            GROUP BY category
        """
        )
        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "first_title" in result
            # Verify it's a non-empty string
            assert isinstance(result["first_title"], str)
            assert len(result["first_title"]) > 0


class TestSQLQueryIntegration:
    """End-to-end integration tests matching proposal examples."""

    def test_proposal_example_basic(self, sql_index):
        """Test the basic example from the MLP proposal."""
        # Example from proposal doc (adapted for our test data)
        sql_query = SQLQuery(
            f"""
            SELECT title, price, category
            FROM {sql_index.name}
            WHERE category = 'books'
        """
        )

        results = sql_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert result["category"] == "books"
            assert "title" in result
            assert "price" in result


@pytest.fixture
def vector_index(redis_url, worker_id):
    """Create a books index with vector embeddings for SQL query testing."""
    import numpy as np

    unique_id = str(uuid.uuid4())[:8]
    index_name = f"sql_books_{worker_id}_{unique_id}"

    index = SearchIndex.from_dict(
        {
            "index": {
                "name": index_name,
                "prefix": f"book_{worker_id}_{unique_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text", "attrs": {"sortable": True}},
                {"name": "genre", "type": "tag", "attrs": {"sortable": True}},
                {"name": "price", "type": "numeric", "attrs": {"sortable": True}},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 4,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        },
        redis_url=redis_url,
    )

    index.create(overwrite=True)

    # Create test books with embeddings
    books = [
        {
            "title": "Dune",
            "genre": "Science Fiction",
            "price": 15,
            "embedding": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
        },
        {
            "title": "Foundation",
            "genre": "Science Fiction",
            "price": 18,
            "embedding": np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32).tobytes(),
        },
        {
            "title": "Neuromancer",
            "genre": "Science Fiction",
            "price": 12,
            "embedding": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32).tobytes(),
        },
        {
            "title": "The Hobbit",
            "genre": "Fantasy",
            "price": 14,
            "embedding": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32).tobytes(),
        },
        {
            "title": "1984",
            "genre": "Dystopian",
            "price": 25,
            "embedding": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32).tobytes(),
        },
    ]

    index.load(books)

    yield index

    # Cleanup
    index.delete(drop=True)


class TestSQLQueryVectorSearch:
    """Tests for SQL vector similarity search using cosine_distance() and vector_distance()."""

    def test_vector_distance_function(self, vector_index):
        """Test vector search with vector_distance() function."""
        import numpy as np

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title, vector_distance(embedding, :vec) AS score
            FROM {vector_index.name}
            LIMIT 3
            """,
            params={"vec": query_vector},
        )

        results = vector_index.query(sql_query)

        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert "title" in result
            assert "score" in result
            # Score should be a valid non-negative distance value
            score = float(result["score"])
            assert score >= 0

    def test_vector_cosine_similarity(self, vector_index):
        """Test vector search with cosine_distance() function - pgvector style."""
        import numpy as np

        # Query vector similar to Science Fiction books
        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT
                title,
                genre,
                price,
                cosine_distance(embedding, :query_vector) AS vector_distance
            FROM {vector_index.name}
            WHERE genre = 'Science Fiction'
                AND price <= 20
            ORDER BY cosine_distance(embedding, :query_vector)
            LIMIT 3
            """,
            params={"query_vector": query_vector},
        )

        results = vector_index.query(sql_query)

        # Should return Science Fiction books under $20
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert result["genre"] == "Science Fiction"
            assert float(result["price"]) <= 20
            # Verify vector_distance is returned (like VectorQuery with return_score=True)
            assert "vector_distance" in result
            # Distance should be a valid non-negative value
            distance = float(result["vector_distance"])
            assert distance >= 0

    def test_vector_redis_query_string(self, vector_index, redis_url):
        """Test redis_query_string() returns correct KNN query for vector search."""
        import numpy as np

        # Query vector
        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title, cosine_distance(embedding, :vec) AS vector_distance
            FROM {vector_index.name}
            LIMIT 3
            """,
            params={"vec": query_vector},
        )

        # Get the Redis command string
        redis_cmd = sql_query.redis_query_string(redis_url=redis_url)

        # Verify it's a valid FT.SEARCH with KNN syntax
        assert redis_cmd.startswith("FT.SEARCH")
        assert vector_index.name in redis_cmd
        assert "KNN 3" in redis_cmd
        assert "@embedding" in redis_cmd
        assert "$vector" in redis_cmd
        assert "vector_distance" in redis_cmd

    def test_vector_search_with_prefilter_redis_query_string(
        self, vector_index, redis_url
    ):
        """Test redis_query_string() returns correct prefiltered KNN query."""
        import numpy as np

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title, genre, cosine_distance(embedding, :vec) AS vector_distance
            FROM {vector_index.name}
            WHERE genre = 'Science Fiction'
            LIMIT 3
            """,
            params={"vec": query_vector},
        )

        redis_cmd = sql_query.redis_query_string(redis_url=redis_url)

        # Verify prefilter syntax: (filter)=>[KNN ...]
        assert redis_cmd.startswith("FT.SEARCH")
        assert "Science Fiction" in redis_cmd or "Science\\ Fiction" in redis_cmd
        assert "=>[KNN" in redis_cmd
        assert "KNN 3" in redis_cmd
