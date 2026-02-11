# How-To Guides

How-to guides are **task-oriented** recipes that help you accomplish specific goals. Unlike tutorials, they assume you have basic knowledge and focus on solving particular problems.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🔍 Querying
:link: querying/index
:link-type: doc

Learn different ways to query your vector data effectively.

- Complex filtering with multiple criteria
- Advanced queries (hybrid, multi-vector, range)
- SQL to Redis query translation
:::

:::{grid-item-card} 🧮 Embeddings
:link: embeddings/index
:link-type: doc

Work with vector embeddings effectively.

- Choosing the right vectorizer
- Caching embeddings for performance
- Comparing embedding models
:::

:::{grid-item-card} ⚡ Optimization
:link: optimization/index
:link-type: doc

Optimize your RedisVL applications for production.

- Reranking for better relevance
- SVS-VAMANA indexing algorithm
- Performance tuning strategies
:::

:::{grid-item-card} 💾 Storage
:link: storage/index
:link-type: doc

Configure data storage options for your use case.

- Hash vs JSON storage formats
- Memory optimization
- Data migration strategies
:::

::::

## Quick Reference

| I want to... | Go to |
|--------------|-------|
| Filter results by multiple criteria | [Complex Filtering](../02_complex_filtering.ipynb) |
| Use hybrid or multi-vector queries | [Advanced Queries](../11_advanced_queries.ipynb) |
| Translate SQL to Redis | [SQL Translation](../12_sql_to_redis_queries.ipynb) |
| Choose an embedding model | [Choosing Vectorizers](../04_vectorizers.ipynb) |
| Speed up embedding generation | [Caching Embeddings](../10_embeddings_cache.ipynb) |
| Improve search accuracy | [Reranking Results](../06_rerankers.ipynb) |
| Optimize index performance | [SVS-VAMANA Index](../09_svs_vamana.ipynb) |
| Decide on storage format | [Hash vs JSON](../05_hash_vs_json.ipynb) |

## How to Use These Guides

Each how-to guide:

| Aspect | What You Get |
|--------|--------------|
| **Focus** | Solves one specific problem |
| **Code** | Working examples ready to adapt |
| **Context** | Explains key decisions and trade-offs |
| **Links** | Points to related guides and resources |

## Contributing

Have a how-to guide you'd like to see? [Open an issue](https://github.com/redis/redis-vl-python/issues) or submit a PR!

