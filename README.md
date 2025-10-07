<div align="center">
    <img width="300" src="https://raw.githubusercontent.com/redis/redis-vl-python/main/docs/_static/Redis_Logo_Red_RGB.svg" alt="Redis">
    <h1>Redis Vector Library</h1>
    <p><strong>The AI-native Redis Python client</strong></p>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pypi](https://badge.fury.io/py/redisvl.svg)](https://pypi.org/project/redisvl/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/redisvl)
[![GitHub stars](https://img.shields.io/github/stars/redis/redis-vl-python)](https://github.com/redis/redis-vl-python/stargazers)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Language](https://img.shields.io/github/languages/top/redis/redis-vl-python)
![GitHub last commit](https://img.shields.io/github/last-commit/redis/redis-vl-python)

**[Documentation](https://docs.redisvl.com)** • **[Recipes](https://github.com/redis-developer/redis-ai-resources)** • **[GitHub](https://github.com/redis/redis-vl-python)**

</div>

---

## Introduction

Redis Vector Library (RedisVL) is the production-ready Python client for AI applications built on Redis. **Lightning-fast vector search meets enterprise-grade reliability.**

<div align="center">

| **🎯 Core Capabilities** | **🚀 AI Extensions** | **🛠️ Dev Utilities** |
|:---:|:---:|:---:|
| **[Index Management](#index-management)**<br/>*Schema design, data loading, CRUD ops* | **[Semantic Caching](#semantic-caching)**<br/>*Reduce LLM costs & boost throughput* | **[CLI](#command-line-interface)**<br/>*Index management from terminal* |
| **[Vector Search](#retrieval)**<br/>*Similarity search with metadata filters* | **[LLM Memory](#llm-memory)**<br/>*Agentic AI context management* | **Async Support**<br/>*Async indexing and search for improved performance* |
| **[Hybrid Queries](#retrieval)**<br/>*Vector + text + metadata combined* | **[Semantic Routing](#semantic-routing)**<br/>*Intelligent query classification* | **[Vectorizers](#vectorizers)**<br/>*8+ embedding provider integrations* |
| **[Multi-Query Types](#retrieval)**<br/>*Vector, Range, Filter, Count queries* | **[Embedding Caching](#embedding-caching)**<br/>*Cache embeddings for efficiency* | **[Rerankers](#rerankers)**<br/>*Improve search result relevancy* |

</div>

### **Built for Modern AI Workloads**

- **RAG Pipelines** → Real-time retrieval with hybrid search capabilities
- **AI Agents** → Short term & long term memory and semantic routing for intent-based decisions
- **Recommendation Systems** → Fast retrieval and reranking

# 💪 Getting Started

## Installation

Install `redisvl` into your Python (>=3.9) environment using `pip`:

```bash
pip install redisvl
```

> For more detailed instructions, visit the [installation guide](https://docs.redisvl.com/en/stable/overview/installation.html).

## Redis

Choose from multiple Redis deployment options:

1. [Redis Cloud](https://redis.io/try-free): Managed cloud database (free tier available)
2. [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/): Docker image for development

    ```bash
    docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
    ```

3. [Redis Enterprise](https://redis.io/enterprise/): Commercial, self-hosted database
4. [Redis Sentinel](https://redis.io/docs/management/sentinel/): High availability with automatic failover

    ```python
    # Connect via Sentinel
    redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
    ```

5. [Azure Managed Redis](https://azure.microsoft.com/en-us/products/managed-redis): Fully managed Redis Enterprise on Azure

> Enhance your experience and observability with the free [Redis Insight GUI](https://redis.io/insight/).

# Overview

## Index Management

1. [Design a schema for your use case](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html#define-an-indexschema) that models your dataset with built-in Redis  and indexable fields (*e.g. text, tags, numerics, geo, and vectors*). [Load a schema](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html#example-schema-creation) from a YAML file:

    ```yaml
    index:
      name: user-idx
      prefix: user
      storage_type: json

    fields:
      - name: user
        type: tag
      - name: credit_score
        type: tag
      - name: job_title
        type: text
        attrs:
          sortable: true
          no_index: false  # Index for search (default)
          unf: false       # Normalize case for sorting (default)
      - name: embedding
        type: vector
        attrs:
          algorithm: flat
          dims: 4
          distance_metric: cosine
          datatype: float32
    ```

    ```python
    from redisvl.schema import IndexSchema

    schema = IndexSchema.from_yaml("schemas/schema.yaml")
    ```

    Or load directly from a Python dictionary:

    ```python
    schema = IndexSchema.from_dict({
        "index": {
            "name": "user-idx",
            "prefix": "user",
            "storage_type": "json"
        },
        "fields": [
            {"name": "user", "type": "tag"},
            {"name": "credit_score", "type": "tag"},
            {
                "name": "job_title",
                "type": "text",
                "attrs": {
                    "sortable": True,
                    "no_index": False,  # Index for search
                    "unf": False        # Normalize case for sorting
                }
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "datatype": "float32",
                    "dims": 4,
                    "distance_metric": "cosine"
                }
            }
        ]
    })
    ```

2. [Create a SearchIndex](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html#create-a-searchindex) class with an input schema to perform admin and search operations on your index in Redis:

    ```python
    from redis import Redis
    from redisvl.index import SearchIndex

    # Define the index
    index = SearchIndex(schema, redis_url="redis://localhost:6379")

    # Create the index in Redis
    index.create()
    ```

    > An async-compatible index class also available: [AsyncSearchIndex](https://docs.redisvl.com/en/stable/api/searchindex.html#redisvl.index.AsyncSearchIndex).

3. [Load](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html#load-data-to-searchindex)
and [fetch](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html#fetch-an-object-from-redis) data to/from your Redis instance:

    ```python
    data = {"user": "john", "credit_score": "high", "embedding": [0.23, 0.49, -0.18, 0.95]}

    # load list of dictionaries, specify the "id" field
    index.load([data], id_field="user")

    # fetch by "id"
    john = index.fetch("john")
    ```

## Retrieval

Define queries and perform advanced searches over your indices, including the combination of vectors, metadata filters, and more.

- [VectorQuery](https://docs.redisvl.com/en/stable/api/query.html#vectorquery) - Flexible vector queries with customizable filters enabling semantic search:

    ```python
    from redisvl.query import VectorQuery

    query = VectorQuery(
      vector=[0.16, -0.34, 0.98, 0.23],
      vector_field_name="embedding",
      num_results=3
    )
    # run the vector search query against the embedding field
    results = index.query(query)
    ```

    Incorporate complex metadata filters on your queries:

    ```python
    from redisvl.query.filter import Tag

    # define a tag match filter
    tag_filter = Tag("user") == "john"

    # update query definition
    query.set_filter(tag_filter)

    # execute query
    results = index.query(query)
    ```

- [RangeQuery](https://docs.redisvl.com/en/stable/api/query.html#rangequery) - Vector search within a defined range paired with customizable filters
- [FilterQuery](https://docs.redisvl.com/en/stable/api/query.html#filterquery) - Standard search using filters and the full-text search
- [CountQuery](https://docs.redisvl.com/en/stable/api/query.html#countquery) - Count the number of indexed records given attributes
- [TextQuery](https://docs.redisvl.com/en/stable/api/query.html#textquery) - Full-text search with support for field weighting and BM25 scoring

> Read more about building [advanced Redis queries](https://docs.redisvl.com/en/stable/user_guide/02_hybrid_queries.html).

## Dev Utilities

### Vectorizers

Integrate with popular embedding providers to greatly simplify the process of vectorizing unstructured data for your index and queries:

- [AzureOpenAI](https://docs.redisvl.com/en/stable/api/vectorizer.html#azureopenaitextvectorizer)
- [Cohere](https://docs.redisvl.com/en/stable/api/vectorizer.html#coheretextvectorizer)
- [Custom](https://docs.redisvl.com/en/stable/api/vectorizer.html#customtextvectorizer)
- [GCP VertexAI](https://docs.redisvl.com/en/stable/api/vectorizer.html#vertexaitextvectorizer)
- [HuggingFace](https://docs.redisvl.com/en/stable/api/vectorizer.html#hftextvectorizer)
- [Mistral](https://docs.redisvl.com/en/stable/api/vectorizer/html#mistralaitextvectorizer)
- [OpenAI](https://docs.redisvl.com/en/stable/api/vectorizer.html#openaitextvectorizer)
- [VoyageAI](https://docs.redisvl.com/en/stable/api/vectorizer/html#voyageaitextvectorizer)

```python
from redisvl.utils.vectorize import CohereTextVectorizer

# set COHERE_API_KEY in your environment
co = CohereTextVectorizer()

embedding = co.embed(
    text="What is the capital city of France?",
    input_type="search_query"
)

embeddings = co.embed_many(
    texts=["my document chunk content", "my other document chunk content"],
    input_type="search_document"
)
```

> Learn more about using [vectorizers]((https://docs.redisvl.com/en/stable/user_guide/04_vectorizers.html)) in your embedding workflows.

### Rerankers

[Integrate with popular reranking providers](https://docs.redisvl.com/en/stable/user_guide/06_rerankers.html) to improve the relevancy of the initial search results from Redis

## Extensions

We're excited to announce the support for **RedisVL Extensions**. These modules implement interfaces exposing best practices and design patterns for working with LLM memory and agents. We've taken the best from what we've learned from our users (that's you) as well as bleeding-edge customers, and packaged it up.

*Have an idea for another extension? Open a PR or reach out to us at <applied.ai@redis.com>. We're always open to feedback.*

### Semantic Caching

Increase application throughput and reduce the cost of using LLM models in production by leveraging previously generated knowledge with the [`SemanticCache`](https://docs.redisvl.com/en/stable/api/cache.html#semanticcache).

```python
from redisvl.extensions.cache.llm import SemanticCache

# init cache with TTL and semantic distance threshold
llmcache = SemanticCache(
    name="llmcache",
    ttl=360,
    redis_url="redis://localhost:6379",
    distance_threshold=0.1
)

# store user queries and LLM responses in the semantic cache
llmcache.store(
    prompt="What is the capital city of France?",
    response="Paris"
)

# quickly check the cache with a slightly different prompt (before invoking an LLM)
response = llmcache.check(prompt="What is France's capital city?")
print(response[0]["response"])
```

```stdout
>>> Paris
```

> Learn more about [semantic caching]((https://docs.redisvl.com/en/stable/user_guide/03_llmcache.html)) for LLMs.

### Embedding Caching

Reduce computational costs and improve performance by caching embedding vectors with their associated text and metadata using the [`EmbeddingsCache`](https://docs.redisvl.com/en/stable/api/cache.html#embeddingscache).

```python
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import HFTextVectorizer

# Initialize embedding cache
embed_cache = EmbeddingsCache(
    name="embed_cache",
    redis_url="redis://localhost:6379",
    ttl=3600  # 1 hour TTL
)

# Initialize vectorizer with cache
vectorizer = HFTextVectorizer(
    model="sentence-transformers/all-MiniLM-L6-v2",
    cache=embed_cache
)

# First call computes and caches the embedding
embedding = vectorizer.embed("What is machine learning?")

# Subsequent calls retrieve from cache (much faster!)
cached_embedding = vectorizer.embed("What is machine learning?")
```

```stdout
>>> Cache hit! Retrieved from Redis in <1ms
```

> Learn more about [embedding caching](https://docs.redisvl.com/en/stable/user_guide/10_embeddings_cache.html) for improved performance.

### LLM Memory

Improve personalization and accuracy of LLM responses by providing user conversation context. Manage access to memory data using recency or relevancy, *powered by vector search* with the [`MessageHistory`](https://docs.redisvl.com/en/stable/api/message_history.html).

```python
from redisvl.extensions.message_history import SemanticMessageHistory

history = SemanticMessageHistory(
    name="my-session",
    redis_url="redis://localhost:6379",
    distance_threshold=0.7
)

# Supports roles: system, user, llm, tool
# Optional metadata field for additional context
history.add_messages([
    {"role": "user", "content": "hello, how are you?"},
    {"role": "llm", "content": "I'm doing fine, thanks."},
    {"role": "user", "content": "what is the weather going to be today?"},
    {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}}
])
```

Get recent chat history:

```python
history.get_recent(top_k=1)
```

```stdout
>>> [{"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}}]
```

Get relevant chat history (powered by vector search):

```python
history.get_relevant("weather", top_k=1)
```

```stdout
>>> [{"role": "user", "content": "what is the weather going to be today?"}]
```

Filter messages by role:

```python
# Get only user messages
history.get_recent(role="user")
# Or multiple roles
history.get_recent(role=["user", "system"])
```

> Learn more about [LLM memory]((https://docs.redisvl.com/en/stable/user_guide/07_message_history.html)).

### Semantic Routing

Build fast decision models that run directly in Redis and route user queries to the nearest "route" or "topic".

```python
from redisvl.extensions.router import Route, SemanticRouter

routes = [
    Route(
        name="greeting",
        references=["hello", "hi"],
        metadata={"type": "greeting"},
        distance_threshold=0.3,
    ),
    Route(
        name="farewell",
        references=["bye", "goodbye"],
        metadata={"type": "farewell"},
        distance_threshold=0.3,
    ),
]

# build semantic router from routes
router = SemanticRouter(
    name="topic-router",
    routes=routes,
    redis_url="redis://localhost:6379",
)


router("Hi, good morning")
```

```stdout
>>> RouteMatch(name='greeting', distance=0.273891836405)
```

> Learn more about [semantic routing](https://docs.redisvl.com/en/stable/user_guide/08_semantic_router.html).

## Command Line Interface

Create, destroy, and manage Redis index configurations from a purpose-built CLI interface: `rvl`.

```bash
$ rvl -h

usage: rvl <command> [<args>]

Commands:
        index       Index manipulation (create, delete, etc.)
        version     Obtain the version of RedisVL
        stats       Obtain statistics about an index
```

> Read more about [using the CLI](https://docs.redisvl.com/en/latest/overview/cli.html).

## 🚀 Why RedisVL?

Redis is a proven, high-performance database that excels at real-time workloads. With RedisVL, you get a production-ready Python client that makes Redis's vector search, caching, and session management capabilities easily accessible for AI applications.

Built on the [Redis Python](https://github.com/redis/redis-py/tree/master) client, RedisVL provides an intuitive interface for vector search, LLM caching, and conversational AI memory - all the core components needed for modern AI workloads.

## 😁 Helpful Links

For additional help, check out the following resources:

- [Getting Started Guide](https://docs.redisvl.com/en/stable/user_guide/01_getting_started.html)
- [API Reference](https://docs.redisvl.com/en/stable/api/index.html)
- [Redis AI Recipes](https://github.com/redis-developer/redis-ai-resources)

## 🫱🏼‍🫲🏽 Contributing

Please help us by contributing PRs, opening GitHub issues for bugs or new feature ideas, improving documentation, or increasing test coverage. [Read more about how to contribute!](CONTRIBUTING.md)

## 🚧 Maintenance

This project is supported by [Redis, Inc](https://redis.io) on a good faith effort basis. To report bugs, request features, or receive assistance, please [file an issue](https://github.com/redis/redis-vl-python/issues).
