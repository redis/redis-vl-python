# How-To Guides

How-to guides are **task-oriented** recipes that help you accomplish specific goals. Each guide focuses on solving a particular problem and can be completed independently.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🤖 LLM Extensions

- [Cache LLM Responses](../03_llmcache.ipynb) -- semantic caching to reduce costs and latency
- [Manage LLM Message History](../07_message_history.ipynb) -- persistent chat history with relevancy retrieval
- [Route Queries with SemanticRouter](../08_semantic_router.ipynb) -- classify intents and route queries
:::

:::{grid-item-card} 🔍 Querying

- [Query and Filter Data](../02_complex_filtering.ipynb) -- combine tag, numeric, geo, and text filters
- [Use Advanced Query Types](../11_advanced_queries.ipynb) -- hybrid, multi-vector, range, and text queries
- [Write SQL Queries for Redis](../12_sql_to_redis_queries.ipynb) -- translate SQL to Redis query syntax
:::

:::{grid-item-card} 🧮 Embeddings

- [Create Embeddings with Vectorizers](../04_vectorizers.ipynb) -- OpenAI, Cohere, HuggingFace, and more
- [Cache Embeddings](../10_embeddings_cache.ipynb) -- reduce costs by caching embedding vectors
:::

:::{grid-item-card} ⚡ Optimization

- [Rerank Search Results](../06_rerankers.ipynb) -- improve relevance with cross-encoders and rerankers
- [Optimize Indexes with SVS-VAMANA](../09_svs_vamana.ipynb) -- graph-based vector search with compression
:::

:::{grid-item-card} 💾 Storage

- [Choose a Storage Type](../05_hash_vs_json.ipynb) -- Hash vs JSON formats and nested data
:::

::::

## Quick Reference

| I want to... | Guide |
|--------------|-------|
| Cache LLM responses | [Cache LLM Responses](../03_llmcache.ipynb) |
| Store chat history | [Manage LLM Message History](../07_message_history.ipynb) |
| Route queries by intent | [Route Queries with SemanticRouter](../08_semantic_router.ipynb) |
| Filter results by multiple criteria | [Query and Filter Data](../02_complex_filtering.ipynb) |
| Use hybrid or multi-vector queries | [Use Advanced Query Types](../11_advanced_queries.ipynb) |
| Translate SQL to Redis | [Write SQL Queries for Redis](../12_sql_to_redis_queries.ipynb) |
| Choose an embedding model | [Create Embeddings with Vectorizers](../04_vectorizers.ipynb) |
| Speed up embedding generation | [Cache Embeddings](../10_embeddings_cache.ipynb) |
| Improve search accuracy | [Rerank Search Results](../06_rerankers.ipynb) |
| Optimize index performance | [Optimize Indexes with SVS-VAMANA](../09_svs_vamana.ipynb) |
| Decide on storage format | [Choose a Storage Type](../05_hash_vs_json.ipynb) |

```{toctree}
:hidden:

Cache LLM Responses <../03_llmcache>
Manage LLM Message History <../07_message_history>
Route Queries with SemanticRouter <../08_semantic_router>
Query and Filter Data <../02_complex_filtering>
Create Embeddings with Vectorizers <../04_vectorizers>
Choose a Storage Type <../05_hash_vs_json>
Rerank Search Results <../06_rerankers>
Optimize Indexes with SVS-VAMANA <../09_svs_vamana>
Cache Embeddings <../10_embeddings_cache>
Use Advanced Query Types <../11_advanced_queries>
Write SQL Queries for Redis <../12_sql_to_redis_queries>
```
