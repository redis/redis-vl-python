---
myst:
  html_meta:
    "description lang=en": |
      RedisVL query types - vector search, filtering, text search, and hybrid queries.
---

# Query Types

RedisVL provides several query types, each optimized for different search patterns. Understanding when to use each helps you build efficient search applications.

## Vector Queries

Vector queries find documents by semantic similarity. You provide a vector (typically an embedding of text or images), and Redis returns documents whose vectors are closest to yours.

### VectorQuery

The most common query type. Returns the top K most similar documents using KNN (K-Nearest Neighbors) search.

```python
from redisvl.query import VectorQuery

query = VectorQuery(
    vector=embedding,           # Your query embedding
    vector_field_name="embedding",
    num_results=10
)
results = index.query(query)
```

Use when you want to find the N most similar items regardless of how similar they actually are. Good for "find me things like this" searches.

### VectorRangeQuery

Returns all documents within a specified distance threshold. Unlike VectorQuery, this doesn't limit results to a fixed K—it returns everything within the radius.

```python
from redisvl.query import VectorRangeQuery

query = VectorRangeQuery(
    vector=embedding,
    vector_field_name="embedding",
    distance_threshold=0.3      # Return all within this distance
)
results = index.query(query)
```

Use when similarity threshold matters more than result count. Good for "find everything similar enough" searches, like deduplication or clustering.

## Filter Queries

Filter queries find documents by exact field matching without vector similarity.

### FilterQuery

Searches using filter expressions on indexed fields. Supports tag matching, numeric ranges, text search, and geographic filters.

```python
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag, Num

query = FilterQuery(
    filter_expression=(Tag("category") == "electronics") & (Num("price") < 100),
    return_fields=["title", "price"],
    num_results=20
)
results = index.query(query)
```

Use when you need precise filtering without semantic similarity—finding all products in a category, all users in a region, or all records within a date range.

### CountQuery

Returns only the count of matching documents, not the documents themselves. More efficient than FilterQuery when you only need the count.

```python
from redisvl.query import CountQuery
from redisvl.query.filter import Tag

query = CountQuery(filter_expression=Tag("status") == "active")
count = index.query(query)
```

Use for analytics, pagination totals, or checking if matches exist before running a full query.

## Text Queries

Text queries perform full-text search with relevance scoring.

### TextQuery

Searches text fields using Redis's full-text search capabilities. Supports multiple scoring algorithms (BM25, TF-IDF), stopword handling, and field weighting.

```python
from redisvl.query import TextQuery

query = TextQuery(
    text="machine learning",
    text_field_name="content",
    text_scorer="BM25STD",
    num_results=10
)
results = index.query(query)
```

Use when you need keyword-based search with relevance ranking—traditional search engine behavior where exact word matches matter.

## Hybrid Queries

Hybrid queries combine multiple search strategies for better results than either alone.

### HybridQuery

Combines text search and vector search in a single query using Redis's native hybrid search. Supports multiple fusion methods:

- **RRF (Reciprocal Rank Fusion)**: Combines rankings from both searches. Good when you trust both signals equally.
- **Linear**: Weighted combination of scores. Good when you want to tune the balance between text and semantic relevance.

```python
from redisvl.query import HybridQuery

query = HybridQuery(
    text="machine learning frameworks",
    text_field_name="content",
    vector=embedding,
    vector_field_name="embedding",
    combination_method="RRF",
    num_results=10
)
results = index.query(query)
```

Use when neither pure keyword search nor pure semantic search gives good enough results. Common in RAG applications where you want both exact matches and semantic understanding.

```{note}
HybridQuery requires Redis >= 8.4.0 and redis-py >= 7.1.0.
```

### AggregateHybridQuery

Similar to HybridQuery but uses Redis aggregation pipelines. Provides more control over score combination and result processing.

Use when you need custom score normalization or complex result transformations that HybridQuery doesn't support.

## Multi-Vector Queries

### MultiVectorQuery

Searches across multiple vector fields simultaneously with configurable weights per field.

```python
from redisvl.query import MultiVectorQuery, Vector

query = MultiVectorQuery(
    vectors=[
        Vector(vector=text_embedding, field_name="text_vector", weight=0.7),
        Vector(vector=image_embedding, field_name="image_vector", weight=0.3),
    ],
    num_results=10
)
results = index.query(query)
```

Use for multimodal search—finding documents that match across text embeddings, image embeddings, and other vector representations. Each vector field can have different importance weights.

## SQL Queries

### SQLQuery

Translates SQL SELECT statements into Redis queries. Provides a familiar interface for developers coming from relational databases.

```python
from redisvl.query import SQLQuery

query = SQLQuery("""
    SELECT title, price, category
    FROM products
    WHERE category = 'electronics' AND price < 100
""")
results = index.query(query)
```

`SQLQuery` also accepts `sql_redis_options`, which are forwarded to the
underlying `sql-redis` executor. This is mainly useful for tuning schema
caching behavior.

```python
query = SQLQuery(
    """
    SELECT title, price, category
    FROM products
    WHERE category = 'electronics' AND price < 100
    """,
    sql_redis_options={"schema_cache_strategy": "lazy"},
)
```

- `"lazy"` (default) loads schemas only when a query touches an index, which
  keeps startup and one-off queries cheaper.
- `"load_all"` preloads all schemas up front, which can help repeated query
  workloads that span many indexes.

**Aggregations and grouping:**

```python
query = SQLQuery("""
    SELECT category, COUNT(*) as count, AVG(price) as avg_price
    FROM products
    GROUP BY category
    ORDER BY count DESC
""")
```

**Geographic queries** with `geo_distance()`:

```python
# Find stores within 50km of San Francisco
query = SQLQuery("""
    SELECT name, category
    FROM stores
    WHERE geo_distance(location, POINT(-122.4194, 37.7749), 'km') < 50
""")

# Calculate distances
query = SQLQuery("""
    SELECT name, geo_distance(location, POINT(-122.4194, 37.7749)) AS distance
    FROM stores
""")
```

**Date queries** with ISO date literals and date functions:

```python
# Filter by date range
query = SQLQuery("""
    SELECT name FROM events
    WHERE created_at BETWEEN '2024-01-01' AND '2024-03-31'
""")

# Extract date parts
query = SQLQuery("""
    SELECT YEAR(created_at) AS year, COUNT(*) AS count
    FROM events
    GROUP BY year
""")
```

**Vector similarity search** with parameters:

```python
query = SQLQuery("""
    SELECT title, vector_distance(embedding, :vec) AS score
    FROM products
    LIMIT 5
""", params={"vec": embedding_bytes})
```

Use when your team is more comfortable with SQL syntax, or when integrating with tools that generate SQL.

```{note}
SQLQuery requires the optional `sql-redis` package. Install with: `pip install redisvl[sql-redis]`
```

For comprehensive examples including geographic filtering, date functions, and vector search, see the [SQL to Redis Queries guide](../user_guide/12_sql_to_redis_queries.ipynb).

## Choosing the Right Query

| Use Case | Query Type |
|----------|------------|
| Semantic similarity search | VectorQuery |
| Find all items within similarity threshold | VectorRangeQuery |
| Exact field matching | FilterQuery |
| Count matching records | CountQuery |
| Keyword search with relevance | TextQuery |
| Combined keyword + semantic | HybridQuery |
| Multimodal search | MultiVectorQuery |
| SQL-familiar interface | SQLQuery |

## Common Patterns

### Vector Search with Filters

All vector queries support filter expressions. Combine semantic search with metadata filtering:

```python
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Num

query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    filter_expression=(Tag("category") == "electronics") & (Num("price") < 100),
    num_results=10
)
```

### Hybrid Search for RAG

For retrieval-augmented generation, hybrid search often outperforms pure vector search:

```python
from redisvl.query import HybridQuery

query = HybridQuery(
    text="machine learning frameworks",
    text_field_name="content",
    vector=embedding,
    vector_field_name="embedding",
    combination_method="RRF",
    num_results=5
)
```

**Learn more:** {doc}`/user_guide/11_advanced_queries` demonstrates these query types in detail.

