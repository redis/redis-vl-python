---
myst:
  html_meta:
    "description lang=en": |
      RedisVL architecture - how the library structures vector search on Redis.
---

# Architecture

RedisVL sits between your application and Redis, providing a structured way to define, populate, and query vector search indexes.

```{image} /_static/redisvl-architecture.svg
:alt: RedisVL Architecture
:align: center
:width: 100%
```

## The Core Pattern

Every RedisVL application follows a consistent workflow: **define → create → load → query**.

First, you define a **schema** that describes your data. The schema specifies which fields exist, what types they are, and how they should be indexed. This includes declaring vector fields with their dimensionality and the algorithm Redis should use for similarity search.

Next, you create an **index** in Redis based on that schema. The index is a persistent structure that Redis uses to make searches fast. Creating an index tells Redis how to organize and access your data.

Then you **load** your data. Documents are stored as Redis Hash or JSON objects. As documents are written, Redis automatically indexes them according to your schema—no separate indexing step required.

Finally, you **query** the index. RedisVL provides query builders that construct Redis search commands for you. You can search by vector similarity, filter by metadata, combine multiple criteria, or mix full-text search with semantic search.

This pattern applies whether you're building a simple semantic search or a complex multi-modal retrieval system.

## Schemas as Contracts

The schema is the source of truth for your index. It defines the contract between your data and Redis.

A schema includes the index name, a key prefix (so Redis knows which keys belong to this index), the storage type (Hash or JSON), and a list of field definitions. Each field has a name, a type, and type-specific configuration.

For vector fields, you specify the dimensionality (which must match your embedding model's output), the distance metric (cosine, Euclidean, or inner product), and the indexing algorithm. These choices are locked in when the index is created—changing them requires building a new index.

The schema can be defined programmatically in Python or loaded from a YAML file. YAML schemas are useful for version control, sharing between environments, and keeping configuration separate from code.

## Query Composition

RedisVL's query builders let you compose search operations without writing raw Redis commands.

**Vector queries** find the K most similar items to a query vector. You provide an embedding, and Redis returns the nearest neighbors according to your configured distance metric.

**Range queries** find all vectors within a distance threshold. Instead of asking for the top K, you're asking for everything "close enough" to a query point.

**Filter queries** narrow results by metadata. You can filter on text fields, tags, numeric ranges, and geographic areas. Filters apply before the vector search, reducing the candidate set.

**Hybrid queries** combine keyword search with semantic search. This is useful when you want to match on specific terms while also considering semantic relevance.

These query types can be combined. A typical pattern is vector search with metadata filters—for example, finding similar products but only in a specific category or price range.

## Extensions as Patterns

Extensions are higher-level abstractions built on RedisVL's core primitives. Each extension encapsulates a common AI workflow pattern.

**Semantic caching** stores LLM responses and retrieves them when similar prompts are seen again. This reduces API costs and latency without requiring exact-match caching.

**Message history** stores conversation turns and retrieves context for LLM prompts. It can retrieve by recency (most recent messages) or by relevance (semantically similar messages).

**Semantic routing** classifies queries into predefined categories based on similarity to reference phrases. This enables intent detection, topic routing, and guardrails.

Each extension manages its own Redis index internally. You interact with a clean, purpose-specific API rather than managing schemas and queries yourself.

---

**Learn more:** {doc}`/user_guide/01_getting_started` covers the core workflow. {doc}`extensions` explains each extension pattern in detail.

