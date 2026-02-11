---
myst:
  html_meta:
    "description lang=en": |
      User guides for RedisVL - Learn how to build AI applications with Redis as your vector database
---

# User Guides

Welcome to the RedisVL user guides! Whether you're just getting started or building advanced AI applications, these guides will help you make the most of Redis as your vector database.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Getting Started
:link: getting_started/index
:link-type: doc

**New to RedisVL?** Start here to learn the basics and build your first vector search application in minutes.

+++
Installation → Schema → Index → Query
:::

:::{grid-item-card} 📚 Tutorials
:link: tutorials/index
:link-type: doc

**Learn by doing.** Step-by-step guides that teach you key features through hands-on examples.

+++
Semantic Caching • Chatbots • Query Routing
:::

:::{grid-item-card} 🛠️ How-To Guides
:link: how_to_guides/index
:link-type: doc

**Solve specific problems.** Task-oriented recipes for querying, embeddings, optimization, and storage.

+++
Filtering • Advanced Queries • Reranking
:::

:::{grid-item-card} 💡 Use Cases
:link: use_cases/index
:link-type: doc

**Build real applications.** Complete end-to-end examples showing production-ready architectures.

+++
Semantic Search • RAG • Chatbots
:::

::::

## Quick Navigation

### By Experience Level

| Level | Recommended Path |
|-------|------------------|
| **Beginner** | [Getting Started](getting_started/index) → [Semantic Caching Tutorial](03_llmcache) → [Vectorizers Guide](04_vectorizers) |
| **Intermediate** | [Hybrid Queries](02_hybrid_queries) → [Message History](07_message_history) → [Rerankers](06_rerankers) |
| **Advanced** | [Advanced Queries](11_advanced_queries) → [SVS-VAMANA](09_svs_vamana) → [Use Cases](use_cases/index) |

### By Use Case

| I want to... | Start here |
|--------------|------------|
| Build semantic search | [Getting Started](01_getting_started) → [Hybrid Queries](02_hybrid_queries) |
| Cache LLM responses | [Semantic Caching Tutorial](03_llmcache) → [Embeddings Cache](10_embeddings_cache) |
| Build a chatbot | [Message History Tutorial](07_message_history) |
| Route queries by intent | [Semantic Router Tutorial](08_semantic_router) |
| Improve search quality | [Rerankers Guide](06_rerankers) → [Advanced Queries](11_advanced_queries) |
| Optimize performance | [SVS-VAMANA](09_svs_vamana) → [Hash vs JSON](05_hash_vs_json) |

## 🚀 Getting Started

New to RedisVL? Start here to learn the basics and build your first application.

```{toctree}
:caption: Getting Started
:maxdepth: 1

getting_started/index
01_getting_started
```

## 📚 Tutorials

Step-by-step tutorials to learn key RedisVL features through hands-on examples.

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/index
03_llmcache
07_message_history
08_semantic_router
```

## 🛠️ How-To Guides

Task-oriented guides for specific use cases and features.

```{toctree}
:caption: How-To Guides
:maxdepth: 2

how_to_guides/index
how_to_guides/querying/index
how_to_guides/embeddings/index
how_to_guides/optimization/index
how_to_guides/storage/index
```

## 💡 Use Cases

Complete examples showing how to build real-world applications with RedisVL.

```{toctree}
:caption: Use Cases
:maxdepth: 1

use_cases/index
```

## 📖 All Guides

Complete reference list of all user guides.

```{toctree}
:caption: All Guides
:maxdepth: 1

01_getting_started
02_complex_filtering
03_llmcache
04_vectorizers
05_hash_vs_json
06_rerankers
07_message_history
08_semantic_router
09_svs_vamana
10_embeddings_cache
11_advanced_queries
12_sql_to_redis_queries
```