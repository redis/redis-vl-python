---
myst:
  html_meta:
    "description lang=en": |
      RedisVL search and indexing - schemas, field types, storage, and query patterns.
---

# Search & Indexing

Vector search in Redis works differently from traditional databases. Understanding the underlying model helps you design better schemas and write more effective queries.

## How Redis Indexes Work

Redis search indexes are secondary structures that sit alongside your data. When you create an index, you're telling Redis: "Watch all keys that match this prefix, and build a searchable structure from these specific fields."

The index doesn't store your data—it references it. Your documents live as Redis Hash or JSON objects, and the index maintains pointers and optimized structures for fast lookup. When you write a document, Redis automatically updates the index. When you delete a document, the index entry is removed.

This design means searches are fast (the index is optimized for queries) while writes remain efficient (only the affected index entries are updated). It also means you can have multiple indexes over the same data with different field configurations.

## Field Types and Their Purpose

Each field type serves a different search use case.

**Text fields** enable full-text search. Redis tokenizes the content, applies stemming (so "running" matches "run"), and builds an inverted index. Text search finds documents containing specific words or phrases, ranked by relevance.

**Tag fields** are for exact-match filtering. Unlike text fields, tags are not tokenized or stemmed. A tag value is treated as an atomic unit. This is ideal for categories, statuses, IDs, and other discrete values where you want exact matches.

**Numeric fields** support range queries and sorting. You can filter for values greater than, less than, or between bounds. Numeric fields are also used for sorting results.

**Geo fields** enable location-based queries. You can find documents within a radius of a point or within a bounding box.

**Vector fields** enable similarity search. Each document has an embedding vector, and queries find documents whose vectors are closest to a query vector. This is the foundation of semantic search.

## Vector Indexing Algorithms

Vector similarity search requires specialized data structures. Redis offers three algorithms, each with different trade-offs.

**Flat indexing** performs exact nearest-neighbor search by comparing the query vector against every indexed vector. This guarantees finding the true closest matches, but search time grows linearly with dataset size. Use flat indexing for small datasets (under ~100K vectors) where exact results matter.

**HNSW (Hierarchical Navigable Small World)** is an approximate algorithm that builds a multi-layer graph structure. Queries navigate this graph to find approximate nearest neighbors in logarithmic time. HNSW typically achieves 95-99% recall (meaning it finds 95-99% of the true nearest neighbors) while being orders of magnitude faster than flat search on large datasets. This is the default choice for most applications.

**SVS (Scalable Vector Search)** is designed for very large datasets with memory constraints. It supports vector compression techniques that reduce memory footprint at the cost of some recall. SVS is useful when you have millions of vectors and memory is a limiting factor.

The algorithm choice is made at index creation and cannot be changed without rebuilding the index.

## Distance Metrics

When comparing vectors, you need a way to measure how "close" two vectors are. Redis supports three distance metrics.

**Cosine distance** measures the angle between vectors, ignoring their magnitude. Two vectors pointing in the same direction have distance 0; opposite directions have distance 2. Cosine is widely used because most embedding models produce vectors where direction encodes meaning and magnitude is less important. Similarity equals 1 minus distance.

**Euclidean distance (L2)** measures the straight-line distance between vector endpoints. Unlike cosine, it considers magnitude. Euclidean distance ranges from 0 to infinity.

**Inner product (IP)** is the dot product of two vectors. It combines both direction and magnitude. When vectors are normalized (magnitude 1), inner product equals cosine similarity. Inner product can be negative and ranges from negative infinity to positive infinity.

Choose your metric based on how your embedding model was trained. Most text embedding models use cosine.

## Storage: Hash vs JSON

Redis offers two storage formats for documents.

**Hash storage** is a flat key-value structure where each field is a top-level key. It's simple, fast, and works well when your documents don't have nested structures. Field names in your schema map directly to hash field names.

**JSON storage** supports nested documents. You can store complex objects and use JSONPath expressions to index nested fields. This is useful when your data is naturally hierarchical or when you want to store the original document structure without flattening it.

The choice affects how you structure data and how you reference fields in schemas. Hash is simpler; JSON is more flexible.

## Index Lifecycle

Indexes have a straightforward lifecycle: create, use, and eventually delete.

**Creating an index** registers the schema with Redis. The `create()` method sends the schema definition to Redis, which builds the necessary data structures. If an index with the same name exists, you can choose to overwrite it (optionally dropping existing data) or raise an error.

**Checking existence** with `exists()` tells you whether an index is registered in Redis. This is useful before creating (to avoid errors) or before querying (to ensure the index is ready).

**Connecting to existing indexes** is possible with `from_existing()`. If an index was created elsewhere—by another application, a previous deployment, or the CLI—you can connect to it by name. RedisVL fetches the index metadata from Redis and reconstructs the schema automatically.

```python
# Connect to an index that already exists in Redis
index = SearchIndex.from_existing("my-index", redis_url="redis://localhost:6379")
```

**Deleting an index** with `delete()` removes the index definition from Redis. By default, this also deletes all documents associated with the index. Pass `drop=False` to keep the documents while removing only the index structure.

**Clearing data** with `clear()` removes all documents from the index without deleting the index itself. The schema remains intact, ready for new data.

## Data Validation

RedisVL can validate data against your schema before loading it to Redis. This catches type mismatches, missing required fields, and invalid values early—before they cause problems in production.

Enable validation by setting `validate_on_load=True` when creating the index:

```python
index = SearchIndex.from_dict(schema, redis_url="redis://localhost:6379", validate_on_load=True)
```

When validation is enabled, each object is checked against the schema during `load()`. If validation fails, a `SchemaValidationError` is raised with details about which field failed and why.

Validation adds overhead, so it's typically enabled during development and testing, then disabled in production once you're confident in your data pipeline.

## Schema Evolution

Redis doesn't support modifying an existing index schema. Once an index is created, its field definitions are fixed.

To change a schema, you create a new index with the updated configuration, reindex your data into it, update your application to use the new index, and then delete the old index. This pattern—create new, migrate, switch, drop old—is the standard approach for schema changes in production.

Planning your schema carefully upfront reduces the need for migrations, but the capability exists when requirements evolve.

---

**Related concepts:** {doc}`field-attributes` explains how to configure field options like `sortable` and `index_missing`. {doc}`queries` covers the different query types available.

**Learn more:** {doc}`/user_guide/01_getting_started` walks through building your first index. {doc}`/user_guide/05_hash_vs_json` compares storage options in depth. {doc}`/user_guide/02_complex_filtering` covers query composition.

