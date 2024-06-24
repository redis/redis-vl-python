<h1 align="center">🔥 Redis Vector Library</h1>

<div align="center">
        <span style="font-size: smaller;">the AI-native Redis Python client</span>
        <br />

[![Codecov](https://img.shields.io/codecov/c/github/redis/redis-vl-python/dev?label=Codecov&logo=codecov&token=E30WxqBeJJ)](https://codecov.io/gh/redis/redis-vl-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis/redis-vl-python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub last commit](https://img.shields.io/github/last-commit/redis/redis-vl-python)
![GitHub deployments](https://img.shields.io/github/deployments/redis/redis-vl-python/github-pages?label=doc%20build)
[![pypi](https://badge.fury.io/py/redisvl.svg)](https://pypi.org/project/redisvl/)

</div>

<div align="center">
<div display="inline-block">
    <a href="https://github.com/redis/redis-vl-python"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.redisvl.com"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/redis-developer"><b>More Projects</b></a>&nbsp;&nbsp;&nbsp;
  </div>
    <br />
</div>


# Introduction

The Python Redis Vector Library (RedisVL) is a tailor-made client for AI applications leveraging [Redis](https://redis.com).

It's specifically designed for:

- Information retrieval & vector similarity search
- Real-time RAG pipelines
- Recommendation engines

Enhance your applications with Redis' **speed**, **flexibility**, and **reliability**, incorporating capabilities like vector-based semantic search, full-text search, and geo-spatial search.

# 🚀 Why RedisVL?

The emergence of the modern GenAI stack, including **vector databases** and **LLMs**, has become increasingly popular due to accelerated innovation & research in information retrieval, the ubiquity of tools & frameworks (e.g. [LangChain](https://github.com/langchain-ai/langchain), [LlamaIndex](https://www.llamaindex.ai/), [EmbedChain](https://github.com/embedchain/embedchain)), and the never-ending stream of business problems addressable by AI.

However, organizations still struggle with delivering reliable solutions **quickly** (*time to value*) at **scale** (*beyond a demo*).

[Redis](https://redis.io) has been a staple for over a decade in the NoSQL world, and boasts a number of flexible [data structures](https://redis.io/docs/data-types/) and [processing engines](https://redis.io/docs/interact/) to handle realtime application workloads like caching, session management, and search. Most notably, Redis has been used as a vector database for RAG, as an LLM cache, and chat session memory store for conversational AI applications.

The vector library **bridges the gap between** the emerging AI-native developer ecosystem and the capabilities of Redis by providing a lightweight, elegant, and intuitive interface. Built on the back of the popular Python client, [`redis-py`](https://github.com/redis/redis-py/tree/master), it abstracts the features Redis into a grammar that is more aligned to the needs of today's AI/ML Engineers or Data Scientists.

# 💪 Getting Started

## Installation

Install `redisvl` into your Python (>=3.8) environment using `pip`:

```bash
pip install redisvl
```
> For more instructions, visit the `redisvl` [installation guide](https://www.redisvl.com/overview/installation.html).

## Setting up Redis

Choose from multiple Redis deployment options:


1. [Redis Cloud](https://redis.com/try-free): Managed cloud database (free tier available)
2. [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/): Docker image for development
    ```bash
    docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
    ```
3. [Redis Enterprise](https://redis.com/redis-enterprise/): Commercial, self-hosted database
4. [Azure Cache for Redis Enterprise](https://learn.microsoft.com/azure/azure-cache-for-redis/quickstart-create-redis-enterprise): Fully managed Redis Enterprise on Azure

> Enhance your experience and observability with the free [Redis Insight GUI](https://redis.com/redis-enterprise/redis-insight/).


## What's included?


### 🗃️ Redis Index Management
1. [Design an `IndexSchema`](https://www.redisvl.com/user_guide/getting_started_01.html#define-an-indexschema) that models your dataset with built-in Redis [data structures](https://www.redisvl.com/user_guide/hash_vs_json_05.html) (*Hash or JSON*) and indexable fields (*e.g. text, tags, numerics, geo, and vectors*).

    [Load a schema](https://www.redisvl.com/user_guide/getting_started_01.html#example-schema-creation) from a [YAML file](schemas/schema.yaml):
    ```yaml
    index:
        name: user-index-v1
        prefix: user
        storage_type: json

    fields:
      - name: user
        type: tag
      - name: credit_score
        type: tag
      - name: embedding
        type: vector
        attrs:
          algorithm: flat
          dims: 3
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
            "name": "user-index-v1",
            "prefix": "user",
            "storage_type": "json"
        },
        "fields": [
            {"name": "user", "type": "tag"},
            {"name": "credit_score", "type": "tag"},
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

2. [Create a SearchIndex](https://www.redisvl.com/user_guide/getting_started_01.html#create-a-searchindex) class with an input schema and client connection in order to perform admin and search operations on your index in Redis:
    ```python
    from redis import Redis
    from redisvl.index import SearchIndex

    # Establish Redis connection and define index
    client = Redis.from_url("redis://localhost:6379")
    index = SearchIndex(schema, client)

    # Create the index in Redis
    index.create()
    ```
    > Async compliant search index class also available: `AsyncSearchIndex`

3. [Load](https://www.redisvl.com/user_guide/getting_started_01.html#load-data-to-searchindex)
and [fetch](https://www.redisvl.com/user_guide/getting_started_01.html#fetch-an-object-from-redis) data to/from your Redis instance:
    ```python
    data = {"user": "john", "credit_score": "high", "embedding": [0.23, 0.49, -0.18, 0.95]}

    # load list of dictionaries, specify the "id" field
    index.load([data], id_field="user")

    # fetch by "id"
    john = index.fetch("john")
    ```

### 🔍 Realtime Search

Define queries and perform advanced searches over your indices, including the combination of vectors, metadata filters, and more.

- [VectorQuery](https://www.redisvl.com/api/query.html#vectorquery) - Flexible vector queries with customizable filters enabling semantic search:

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

- [RangeQuery](https://www.redisvl.com/api/query.html#rangequery) - Vector search within a defined range paired with customizable filters
- [FilterQuery](https://www.redisvl.com/api/query.html#filterquery) - Standard search using filters and the full-text search
- [CountQuery](https://www.redisvl.com/api/query.html#countquery) - Count the number of indexed records given attributes

> Read more about building advanced Redis queries [here](https://www.redisvl.com/user_guide/hybrid_queries_02.html).


### 🖥️ Command Line Interface
Create, destroy, and manage Redis index configurations from a purpose-built CLI interface: `rvl`.

```bash
$ rvl -h

usage: rvl <command> [<args>]

Commands:
        index       Index manipulation (create, delete, etc.)
        version     Obtain the version of RedisVL
        stats       Obtain statistics about an index
```

> Read more about using the `redisvl` CLI [here](https://www.redisvl.com/user_guide/cli.html).

### ⚡  Community Integrations
Integrate with popular embedding models and providers to greatly simplify the process of vectorizing unstructured data for your index and queries:
- [Cohere](https://www.redisvl.com/api/vectorizer/html#coheretextvectorizer)
- [Mistral](https://www.redisvl.com/api/vectorizer/html#mistralaitextvectorizer)
- [OpenAI](https://www.redisvl.com/api/vectorizer.html#openaitextvectorizer)
- [HuggingFace](https://www.redisvl.com/api/vectorizer.html#hftextvectorizer)
- [GCP VertexAI](https://www.redisvl.com/api/vectorizer.html#vertexaitextvectorizer)

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

> Learn more about using `redisvl` Vectorizers in your workflows [here](https://www.redisvl.com/user_guide/vectorizers_04.html).

### 💫 Beyond Vector Search
In order to perform well in production, modern GenAI applications require much more than vector search for retrieval. `redisvl` provides some common extensions that
aim to improve applications working with LLMs:

- **LLM Semantic Caching** is designed to increase application throughput and reduce the cost of using LLM models in production by leveraging previously generated knowledge.

    ```python
    from redisvl.extensions.llmcache import SemanticCache

    # init cache with TTL (expiration) policy and semantic distance threshhold
    llmcache = SemanticCache(
        name="llmcache",
        ttl=360,
        redis_url="redis://localhost:6379"
    )
    llmcache.set_threshold(0.2) # can be changed on-demand

    # store user queries and LLM responses in the semantic cache
    llmcache.store(
        prompt="What is the capital city of France?",
        response="Paris",
        metadata={}
    )

    # quickly check the cache with a slightly different prompt (before invoking an LLM)
    response = llmcache.check(prompt="What is France's capital city?")
    print(response[0]["response"])
    ```
    ```stdout
    >>> "Paris"
    ```

    > Learn more about Semantic Caching [here](https://www.redisvl.com/user_guide/llmcache_03.html).

- **LLM Session Management (COMING SOON)** aims to improve personalization and accuracy of the LLM application by providing user chat session information and conversational memory.
- **LLM Contextual Access Control (COMING SOON)** aims to improve security concerns by preventing malicious, irrelevant, or problematic user input from reaching LLMs and infrastructure.


## Helpful Links

To get started, check out the following guides:
 - [Getting Started Guide](https://www.redisvl.com/user_guide/getting_started_01.html)
 - [API Reference](https://www.redisvl.com/api/index.html)
 - [Example Gallery](https://www.redisvl.com/examples/index.html)
 - [Official Redis Vector Search Docs](https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/)


## 🫱🏼‍🫲🏽 Contributing

Please help us by contributing PRs, opening GitHub issues for bugs or new feature ideas, improving documentation, or increasing test coverage. [Read more about how to contribute!](CONTRIBUTING.md)

## 🚧 Maintenance
This project is supported by [Redis, Inc](https://redis.com) on a good faith effort basis. To report bugs, request features, or receive assistance, please [file an issue](https://github.com/redis/redis-vl-python/issues).
