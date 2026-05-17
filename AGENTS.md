# AGENTS.md

Quick reference for AI agents using `redisvl`. For agents working *on* the
codebase itself, see [docs/for-ais-only/](docs/for-ais-only/index.md).

## What redisvl is

A Python library for using Redis as a vector database. It wraps Redis Search
(`FT.CREATE`, `FT.SEARCH`, `FT.AGGREGATE`, vector index types) behind:

- **`SearchIndex` / `AsyncSearchIndex`**: schema-driven index management.
- **Query classes**: `VectorQuery`, `VectorRangeQuery`, `FilterQuery`,
  `HybridQuery`, `MultiVectorQuery`, `TextQuery`, `CountQuery`, `SQLQuery`.
- **Filter expressions**: `Tag`, `Text`, `Num`, `Geo`, `GeoRadius`.
- **Extensions**: `SemanticCache`, `LangCacheSemanticCache`, `EmbeddingsCache`,
  `MessageHistory` / `SemanticMessageHistory`, `SemanticRouter`.
- **Vectorizers**: OpenAI, Azure OpenAI, Cohere, HuggingFace
  (sentence-transformers), Mistral, Vertex AI, Bedrock, VoyageAI, custom.
- **Rerankers**: Cohere, HuggingFace cross-encoder, VoyageAI.
- **CLI**: `rvl index`, `rvl stats`, `rvl mcp`, `rvl version`.
- **MCP server**: serves an existing Redis index over stdio / HTTP / SSE.

## Install

```bash
pip install redisvl
# common provider extras
pip install redisvl[openai,cohere,sentence-transformers]
# everything (heavy)
pip install redisvl[all]
```

Requires Python 3.10+ and a Redis 8.x instance with the search module
(`docker run -d -p 6379:6379 redis:8.4`).

## Minimum viable use

```python
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

schema = IndexSchema.from_dict({
    "index": {"name": "docs", "prefix": "doc:", "storage_type": "hash"},
    "fields": [
        {"name": "title", "type": "text"},
        {"name": "category", "type": "tag"},
        {"name": "embedding", "type": "vector",
         "attrs": {"dims": 1536, "algorithm": "hnsw",
                   "distance_metric": "cosine", "datatype": "float32"}},
    ],
})

index = SearchIndex(schema, redis_url="redis://localhost:6379")
index.create(overwrite=True)

index.load([
    {"title": "intro", "category": "guide", "embedding": vector_bytes},
])

results = index.query(VectorQuery(
    vector=query_embedding,
    vector_field_name="embedding",
    return_fields=["title", "category"],
    num_results=10,
))
```

## Public import paths (stable)

Use the **subpackage**, not the module:

```python
from redisvl.index import SearchIndex, AsyncSearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import (
    VectorQuery, VectorRangeQuery, FilterQuery, CountQuery, TextQuery,
    HybridQuery, MultiVectorQuery, AggregateHybridQuery, SQLQuery, Vector,
)
from redisvl.query.filter import Tag, Text, Num, Geo, GeoRadius
from redisvl.extensions.cache.llm import SemanticCache, LangCacheSemanticCache
from redisvl.extensions.message_history import (
    MessageHistory, SemanticMessageHistory,
)
from redisvl.extensions.router import SemanticRouter, Route, RoutingConfig
from redisvl.utils.vectorize import (
    HFTextVectorizer, OpenAITextVectorizer, AzureOpenAITextVectorizer,
    CohereTextVectorizer, MistralAITextVectorizer, VoyageAIVectorizer,
    VertexAIVectorizer, BedrockVectorizer, CustomVectorizer,
)
from redisvl.utils.rerank import (
    CohereReranker, HFCrossEncoderReranker, VoyageAIReranker,
)
```

## What docs to read

- Concepts → [docs/concepts/](docs/concepts/index.md): how indexes, schemas,
  queries, and extensions fit together.
- User Guide → [docs/user_guide/](docs/user_guide/index.md): notebooks for
  every common task.
- API Reference → [docs/api/](docs/api/index.md): the generated reference.
- Examples → [docs/examples/](docs/examples/index.md): links to
  redis-ai-resources for end-to-end recipes.
- For AI Agents (codebase contributors) →
  [docs/for-ais-only/](docs/for-ais-only/index.md).

## Machine-readable indexes

When the docs are built, they emit:

- [`llms.txt`](https://docs.redisvl.com/llms.txt) — flat index of every doc.
- [`llms-full.txt`](https://docs.redisvl.com/llms-full.txt) — concatenated
  full content for one-shot loading.

## Things to know before suggesting code

- **Always combine schema + algorithm changes.** Bundling datatype and
  algorithm changes into a single index patch produces one drop/rebuild cycle
  instead of two.
- **`MessageHistory` / `SemanticMessageHistory`** replace the deprecated
  `SessionManager` / `SemanticSessionManager`. The old names still import but
  emit a `DeprecationWarning` and will be removed.
- **SVS-VAMANA** requires Redis ≥ 8.2.0 with Redis Search ≥ 2.8.10 and only
  supports `float16` / `float32` datatypes.
- **`SQLQuery`** requires the `redisvl[sql-redis]` extra and translates SQL
  `SELECT` into `FT.SEARCH` / `FT.AGGREGATE` via the
  [sql-redis](https://github.com/redis-developer/sql-redis) project.
- **`HybridQuery` vs `AggregateHybridQuery`** weight scores differently:
  `HybridQuery.linear_alpha` weights *text*, `AggregateHybridQuery.alpha`
  weights *vector*. Recheck `alpha` when switching.
