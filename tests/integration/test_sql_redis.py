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
        assert int(results[0]["total"]) == 12  # 12 products in test data

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
    """Tests for SQL vector similarity search using cosine_distance()."""

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
