# Use Cases

RedisVL powers a wide range of AI applications. Here's how to apply its features to common use cases.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🧠 Agent Context
Provide agents with the right information at the right time.

- **RAG** -- Retrieval-Augmented Generation with [vector search](../01_getting_started.ipynb) and [hybrid queries](../11_advanced_queries.ipynb)
- **Memory** -- Persistent [message history](../07_message_history.ipynb) across sessions
- **Context Engineering** -- Combine [filtering](../02_complex_filtering.ipynb), [reranking](../06_rerankers.ipynb), and [embeddings](../04_vectorizers.ipynb) to curate the optimal context window
:::

:::{grid-item-card} ⚡ Agent Optimization
Reduce latency and cost for AI workloads.

- **Semantic Caching** -- Cache LLM responses by meaning with [SemanticCache](../03_llmcache.ipynb)
- **Embeddings Caching** -- Avoid redundant embedding calls with [EmbeddingsCache](../10_embeddings_cache.ipynb)
- **Semantic Routing** -- Route queries to the right handler with [SemanticRouter](../08_semantic_router.ipynb)
:::

:::{grid-item-card} 🔍 General Search
Build search experiences that understand meaning, not just keywords.

- **Semantic Search** -- [Vector queries](../01_getting_started.ipynb) with [complex filtering](../02_complex_filtering.ipynb)
- **Hybrid Search** -- Combine keyword and vector search with [advanced query types](../11_advanced_queries.ipynb)
- **SQL Translation** -- Use familiar SQL syntax with [SQLQuery](../12_sql_to_redis_queries.ipynb)
:::

:::{grid-item-card} 🎯 Personalization & RecSys
Drive engagement with personalized recommendations.

- **User Similarity** -- Find similar users or items using [vector search](../01_getting_started.ipynb)
- **Real-Time Ranking** -- Combine vector similarity with [metadata filtering](../02_complex_filtering.ipynb) and [reranking](../06_rerankers.ipynb)
- **Multi-Signal Matching** -- Search across multiple embedding fields with [MultiVectorQuery](../11_advanced_queries.ipynb)
:::

::::
