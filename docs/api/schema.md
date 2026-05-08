---
description: IndexSchema and field type API reference for RedisVL.
---

# Schema

Schemas describe an index: which fields exist, their types, attributes, and the
storage backing them. Build them in YAML, in Python dictionaries, or directly
from objects.

A schema is composed of three top-level pieces:

| Component | Description |
|-----------|-------------|
| `version` | Schema spec version. Current supported version is `0.1.0`. |
| `index`   | Index settings: name, key prefix, key separator, storage type. |
| `fields`  | Subset of fields to index, with type-specific attributes. |

## IndexSchema

::: redisvl.schema.IndexSchema
    options:
      members_order: source
      filters:
        - "!^_"
        - "!^generate_fields$"
        - "!^validate_and_create_fields$"
        - "!^redis_fields$"

## Index-Level Stopwords Configuration

The `IndexInfo` class supports index-level stopwords configuration through the
`stopwords` field. This controls which words are filtered during indexing
(server-side), as opposed to query-time filtering (client-side).

**Configuration Options:**

- `None` (default): Use Redis default stopwords (~300 common words)
- `[]` (empty list): Disable stopwords completely (`STOPWORDS 0`)
- Custom list: Specify your own stopwords (e.g. `["the", "a", "an"]`)

```python
from redisvl.schema import IndexSchema

# Disable stopwords to search for phrases like "Bank of Glasberliner"
schema = IndexSchema.from_dict({
    "index": {
        "name": "company-idx",
        "prefix": "company",
        "stopwords": [],  # STOPWORDS 0
    },
    "fields": [
        {"name": "name", "type": "text"},
    ],
})
```

For detailed information about stopwords configuration and best practices, see
the [Advanced Queries](../user_guide/11_advanced_queries.ipynb) guide.

## Field Types

### Text Fields

::: redisvl.schema.fields.TextField
    options:
      show_root_heading: true

::: redisvl.schema.fields.TextFieldAttributes
    options:
      show_root_heading: true
      members_order: source

### Tag Fields

::: redisvl.schema.fields.TagField
    options:
      show_root_heading: true

::: redisvl.schema.fields.TagFieldAttributes
    options:
      show_root_heading: true
      members_order: source

### Numeric Fields

::: redisvl.schema.fields.NumericField
    options:
      show_root_heading: true

::: redisvl.schema.fields.NumericFieldAttributes
    options:
      show_root_heading: true
      members_order: source

### Geo Fields

::: redisvl.schema.fields.GeoField
    options:
      show_root_heading: true

::: redisvl.schema.fields.GeoFieldAttributes
    options:
      show_root_heading: true
      members_order: source

## Vector Field Types

All vector fields share a base set of attributes (`dims`, `algorithm`,
`datatype`, `distance_metric`, `initial_cap`, `index_missing`) and add
algorithm-specific configuration on top.

### Common Vector Attributes

::: redisvl.schema.fields.BaseVectorFieldAttributes
    options:
      show_root_heading: true
      members_order: source

### HNSW

Graph-based approximate search with excellent recall. Best for general-purpose
vector search (10K–1M+ vectors).

::: redisvl.schema.fields.HNSWVectorField
    options:
      show_root_heading: true

::: redisvl.schema.fields.HNSWVectorFieldAttributes
    options:
      show_root_heading: true
      members_order: source

### SVS-VAMANA

Fast approximate nearest neighbor search with optional compression. Best for
large datasets (>100K vectors) on Intel hardware with memory constraints.
Requires Redis >= 8.2.0 with Redis Search >= 2.8.10.

::: redisvl.schema.fields.SVSVectorField
    options:
      show_root_heading: true

::: redisvl.schema.fields.SVSVectorFieldAttributes
    options:
      show_root_heading: true
      members_order: source

### FLAT

Brute-force exact search. Best for small datasets (<10K vectors) requiring
100% accuracy.

::: redisvl.schema.fields.FlatVectorField
    options:
      show_root_heading: true

::: redisvl.schema.fields.FlatVectorFieldAttributes
    options:
      show_root_heading: true

## SVS-VAMANA Configuration Utilities

For SVS-VAMANA indices, RedisVL provides utilities to help configure
compression settings and estimate memory savings.

### CompressionAdvisor

::: redisvl.utils.compression.CompressionAdvisor
    options:
      show_root_heading: true

### SVSConfig

::: redisvl.utils.compression.SVSConfig
    options:
      show_root_heading: true

## Vector Algorithm Comparison

| Algorithm | Best for | Performance | Memory | Trade-offs |
|-----------|----------|-------------|--------|------------|
| **FLAT** | <100K vectors | 100% recall, O(n) | Minimal | Exact, slow at scale |
| **HNSW** | 100K–1M+ vectors | 95–99% recall, O(log n) | Moderate | Fast approximate |
| **SVS-VAMANA** | Memory-constrained, large | 90–95% recall, O(log n) | Low (with compression) | Intel-optimized, requires Redis >= 8.2 |

For complete Redis field documentation, see the official
[FT.CREATE reference](https://redis.io/commands/ft.create/).
