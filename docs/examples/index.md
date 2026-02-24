---
myst:
  html_meta:
    "description lang=en": |
        Examples for RedisVL users
---


# Example Gallery

Explore community examples of RedisVL in the wild.

```{tip}
For a comprehensive collection of Redis AI examples, tutorials, and recipes, visit the
**[Redis AI Resources](https://github.com/redis-developer/redis-ai-resources)** repository.
It includes notebooks for RAG, agents, semantic caching, recommendation systems, and more.
```

## Demo Applications

Full-stack applications showcasing RedisVL and Redis vector search capabilities.

```{gallery-grid} ../_static/gallery.yaml
:grid-columns: "1 1 2 2"
```

```{note}
If you are using RedisVL, please consider adding your example to this page by
opening a Pull Request on [GitHub](https://github.com/redis/redis-vl-python)
```

---

## Code Recipes

Runnable Jupyter notebooks from [Redis AI Resources](https://github.com/redis-developer/redis-ai-resources) covering real-world use cases. Each recipe includes a Google Colab link for easy execution.

### RAG with Frameworks

Build retrieval-augmented generation pipelines with popular frameworks.

| Recipe | Description | Links |
|--------|-------------|-------|
| RAG with LangChain | RAG using Redis and LangChain | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/02_langchain.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/02_langchain.ipynb) |
| RAG with LlamaIndex | RAG using Redis and LlamaIndex | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/03_llamaindex.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/03_llamaindex.ipynb) |
| Advanced RAG | Advanced RAG techniques with RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/04_advanced_redisvl.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/04_advanced_redisvl.ipynb) |
| NVIDIA RAG | RAG using Redis and NVIDIA NIMs | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/05_nvidia_ai_rag_redis.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/05_nvidia_ai_rag_redis.ipynb) |
| RAGAS Evaluation | Evaluate RAG performance with RAGAS | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/06_ragas_evaluation.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/06_ragas_evaluation.ipynb) |
| Role-Based RAG | Implement RBAC policies with vector search | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/07_user_role_based_rag.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/RAG/07_user_role_based_rag.ipynb) |

### Agents

Build AI agents with memory, tools, and multi-agent workflows.

| Recipe | Description | Links |
|--------|-------------|-------|
| LangGraph Agents | Get started with LangGraph and agentic RAG | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/00_langgraph_redis_agentic_rag.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/00_langgraph_redis_agentic_rag.ipynb) |
| CrewAI Agents | Multi-agent systems with CrewAI and LangGraph | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/01_crewai_langgraph_redis.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/01_crewai_langgraph_redis.ipynb) |
| Full-Featured Agent | Tool-calling agent with semantic cache and router | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/02_full_featured_agent.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/02_full_featured_agent.ipynb) |
| Memory Agent | Agent with short-term and long-term memory | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/03_memory_agent.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/03_memory_agent.ipynb) |
| Autogen Agent | Blog writing agent with Autogen and Redis | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/04_autogen_agent.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/agents/04_autogen_agent.ipynb) |

### Recommendation Systems

Build personalized recommendation engines with Redis.

| Recipe | Description | Links |
|--------|-------------|-------|
| Content Filtering | Content-based filtering with RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/00_content_filtering.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/00_content_filtering.ipynb) |
| Collaborative Filtering | Collaborative filtering with RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/01_collaborative_filtering.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/01_collaborative_filtering.ipynb) |
| Two Towers | Deep learning two-tower model with RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/02_two_towers.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/recommendation-systems/02_two_towers.ipynb) |

### Specialized Applications

Explore Redis for computer vision, feature stores, and AI gateways.

| Recipe | Description | Links |
|--------|-------------|-------|
| Facial Recognition | Build a facial recognition system with Facenet and RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/computer-vision/00_facial_recognition_facenet.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/computer-vision/00_facial_recognition_facenet.ipynb) |
| Credit Scoring | Credit scoring with Feast and Redis as online store | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/feature-store/00_feast_credit_score.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/feature-store/00_feast_credit_score.ipynb) |
| Transaction Search | Real-time transaction feature search | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/feature-store/01_card_transaction_search.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/feature-store/01_card_transaction_search.ipynb) |
| LiteLLM Gateway | Getting started with LiteLLM proxy and Redis | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/gateway/00_litellm_proxy_redis.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/gateway/00_litellm_proxy_redis.ipynb) |

### Vector Search Deep Dives

Advanced vector search techniques and optimizations.

| Recipe | Description | Links |
|--------|-------------|-------|
| Vector Search with redis-py | Low-level vector search with Redis Python client | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/00_redispy.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/00_redispy.ipynb) |
| Hybrid Search | Combine BM25 and vector search | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/02_hybrid_search.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/02_hybrid_search.ipynb) |
| Data Type Support | Convert float32 index to float16 or integer | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/03_dtype_support.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/03_dtype_support.ipynb) |
| Benchmarking Basics | Search benchmarking with RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/04_redisvl_benchmarking_basics.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/04_redisvl_benchmarking_basics.ipynb) |
| Multi-Vector Search | Multi-vector queries with RedisVL | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/05_multivector_search.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/05_multivector_search.ipynb) |
| HNSW to SVS-VAMANA Migration | Migrate HNSW indices to SVS-VAMANA | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/06_hnsw_to_svs_vamana_migration.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/vector-search/06_hnsw_to_svs_vamana_migration.ipynb) |

### LLM Optimization

Reduce costs and latency with caching and routing.

| Recipe | Description | Links |
|--------|-------------|-------|
| Gemini Semantic Cache | Semantic caching with Redis and Google Gemini | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/00_semantic_caching_gemini.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/00_semantic_caching_gemini.ipynb) |
| Doc2Cache with Llama3.1 | Semantic caching with Doc2Cache framework | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/01_doc2cache_llama3_1.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/01_doc2cache_llama3_1.ipynb) |
| Cache Optimization | Optimize cache thresholds with redis-retrieval-optimizer | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/02_semantic_cache_optimization.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/02_semantic_cache_optimization.ipynb) |
| Context-Enabled Caching | Context-aware semantic caching | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/03_context_enabled_semantic_caching.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-cache/03_context_enabled_semantic_caching.ipynb) |
| Router Optimization | Optimize router thresholds | [GitHub](https://github.com/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-router/01_routing_optimization.ipynb) \| [Colab](https://colab.research.google.com/github/redis-developer/redis-ai-resources/blob/main/python-recipes/semantic-router/01_routing_optimization.ipynb) |

---

## More Resources

Looking for more examples and tutorials?

- **[Redis AI Resources](https://github.com/redis-developer/redis-ai-resources)** -- Comprehensive collection of code recipes, demos, and tutorials
- **[Java Recipes](https://github.com/redis-developer/redis-ai-resources/tree/main/java-recipes)** -- Spring AI, Redis OM Spring, and semantic routing examples
- **[Redis Developer Hub](https://redis.io/developers/)** -- Official Redis developer resources

