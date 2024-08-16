<div align="center" style="margin-bottom: 20px;">
    <div><img src="docs/_static/Redis_Logo_Red_RGB.svg" style="width: 200px; margin-bottom: 20px;"></div>
    <h1>🔥 Vector Library</h1>
</div>

<div align="center" style="margin-top: 20px;">
    <span style="display: block; margin-bottom: 10px;">the *AI-native* Redis Python client</span>
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
    <a href="https://github.com/redis-developer/redis-ai-resources"><b>Recipes</b></a>&nbsp;&nbsp;&nbsp;
  </div>
    <br />
</div>


# Introduction

Welcome to the Redis Vector Library – the ultimate Python client designed for AI applications harnessing the power of [Redis](https://redis.io).

[redisvl](https://pypi.org/project/redisvl/) is your go-to tool for:

- Lightning-fast information retrieval & vector similarity search
- Real-time RAG pipelines
- Agentic memory structures
- Smart recommendation engines


# 💪 Getting Started

## Installation

Install `redisvl` into your Python (>=3.8) environment using `pip`:

```bash
pip install redisvl
```
> For more detailed instructions, visit the [installation guide](https://www.redisvl.com/overview/installation.html).

## Setting up Redis

Choose from multiple Redis deployment options:


1. [Redis Cloud](https://redis.io/try-free): Managed cloud database (free tier available)
2. [Redis Stack](https://redis.io/docs/getting-started/install-stack/docker/): Docker image for development
    ```bash
    docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
    ```
3. [Redis Enterprise](https://redis.io/enterprise/): Commercial, self-hosted database
4. [Azure Cache for Redis Enterprise](https://learn.microsoft.com/azure/azure-cache-for-redis/quickstart-create-redis-enterprise): Fully managed Redis Enterprise on Azure

> Enhance your experience and observability with the free [Redis Insight GUI](https://redis.com/redis-enterprise/redis-insight/).


# Overview


## 🗃️ Redis Index Management
1. [Design a schema for your use case](https://www.redisvl.com/user_guide/getting_started_01.html#define-an-indexschema) that models your dataset with built-in Redis  and indexable fields (*e.g. text, tags, numerics, geo, and vectors*). [Load a schema](https://www.redisvl.com/user_guide/getting_started_01.html#example-schema-creation) from a YAML file:
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
    > Async compliant search index class also available: [AsyncSearchIndex](https://www.redisvl.com/api/searchindex.html#redisvl.index.AsyncSearchIndex).

3. [Load](https://www.redisvl.com/user_guide/getting_started_01.html#load-data-to-searchindex)
and [fetch](https://www.redisvl.com/user_guide/getting_started_01.html#fetch-an-object-from-redis) data to/from your Redis instance:
    ```python
    data = {"user": "john", "credit_score": "high", "embedding": [0.23, 0.49, -0.18, 0.95]}

    # load list of dictionaries, specify the "id" field
    index.load([data], id_field="user")

    # fetch by "id"
    john = index.fetch("john")
    ```

## 🔍 Retrieval

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

> Read more about building [advanced Redis queries](https://www.redisvl.com/user_guide/hybrid_queries_02.html).


## 🔧  Utilities

### Vectorizers
Integrate with popular embedding providers to greatly simplify the process of vectorizing unstructured data for your index and queries:
- [AzureOpenAI](https://www.redisvl.com/api/vectorizer.html#azureopenaitextvectorizer)
- [Cohere](https://www.redisvl.com/api/vectorizer.html#coheretextvectorizer)
- [Custom](https://www.redisvl.com/api/vectorizer.html#customtextvectorizer)
- [GCP VertexAI](https://www.redisvl.com/api/vectorizer.html#vertexaitextvectorizer)
- [HuggingFace](https://www.redisvl.com/api/vectorizer.html#hftextvectorizer)
- [Mistral](https://www.redisvl.com/api/vectorizer/html#mistralaitextvectorizer)
- [OpenAI](https://www.redisvl.com/api/vectorizer.html#openaitextvectorizer)

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

> Learn more about using [vectorizers]((https://www.redisvl.com/user_guide/vectorizers_04.html)) in your embedding workflows.


### Rerankers
[Integrate with popular reranking providers](https://www.redisvl.com/user_guide/rerankers_06.html) to improve the relevancy of the initial search results from Redis



## 💫 Extensions
We're excited to announce the support for **RedisVL Extensions**. These modules implement interfaces exposing best practices and design patterns for working with LLM memory and agents. We've taken the best from what we've learned from our users (that's you) as well as bleeding-edge customers, and packaged it up.

*Have an idea for another extension? Open a PR or reach out to us at applied.ai@redis.com. We're always open to feedback.*

### LLM Semantic Caching
Increase application throughput and reduce the cost of using LLM models in production by leveraging previously generated knowledge with the [`SemanticCache`](https://www.redisvl.com/api/cache.html#semanticcache).

```python
from redisvl.extensions.llmcache import SemanticCache

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

> Learn more about [semantic caching]((https://www.redisvl.com/user_guide/llmcache_03.html)) for LLMs.

### LLM Session Management

Improve personalization and accuracy of LLM responses by providing user chat history as context. Manage access to the session data using recency or relevancy, *powered by vector search* with the [`SemanticSessionManager`](https://www.redisvl.com/api/session_manager.html).

```python
from redisvl.extensions.session_manager import SemanticSessionManager

session = SemanticSessionManager(
    name="my-session",
    redis_url="redis://localhost:6379",
    distance_threshold=0.7
)

session.add_messages([
    {"role": "user", "content": "hello, how are you?"},
    {"role": "assistant", "content": "I'm doing fine, thanks."},
    {"role": "user", "content": "what is the weather going to be today?"},
    {"role": "assistant", "content": "I don't know"}
])
```
Get recent chat history:
```python
session.get_recent(top_k=1)
```
```stdout
>>> [{"role": "assistant", "content": "I don't know"}]
```
Get relevant chat history (powered by vector search):
```python
session.get_relevant("weather", top_k=1)
```
```stdout
>>> [{"role": "user", "content": "what is the weather going to be today?"}]
```
> Learn more about [LLM session management]((https://www.redisvl.com/user_guide/session_manager_07.html)).


### LLM Semantic Routing
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
> Learn more about [semantic routing](https://www.redisvl.com/user_guide/semantic_router_08.html).

## 🖥️ Command Line Interface
Create, destroy, and manage Redis index configurations from a purpose-built CLI interface: `rvl`.

```bash
$ rvl -h

usage: rvl <command> [<args>]

Commands:
        index       Index manipulation (create, delete, etc.)
        version     Obtain the version of RedisVL
        stats       Obtain statistics about an index
```

> Read more about [using the CLI](https://www.redisvl.com/user_guide/cli.html).

## 🚀 Why RedisVL?

In the age of GenAI, **vector databases** and **LLMs** are transforming information retrieval systems. With emerging and popular frameworks like [LangChain](https://github.com/langchain-ai/langchain) and [LlamaIndex](https://www.llamaindex.ai/), innovation is rapid. Yet, many organizations face the challenge of delivering AI solutions **quickly** and at **scale**.

Enter [Redis](https://redis.io) – a cornerstone of the NoSQL world, renowned for its versatile [data structures](https://redis.io/docs/data-types/) and [processing engines](https://redis.io/docs/interact/). Redis excels in real-time workloads like caching, session management, and search. It's also a powerhouse as a vector database for RAG, an LLM cache, and a chat session memory store for conversational AI.

The Redis Vector Library bridges the gap between the AI-native developer ecosystem and Redis's robust capabilities. With a lightweight, elegant, and intuitive interface, RedisVL makes it easy to leverage Redis's power. Built on the [Redis Python](https://github.com/redis/redis-py/tree/master) client, `redisvl` transforms Redis's features into a grammar perfectly aligned with the needs of today's AI/ML Engineers and Data Scientists.


## 😁 Helpful Links

For additional help, check out the following resources:
 - [Getting Started Guide](https://www.redisvl.com/user_guide/getting_started_01.html)
 - [API Reference](https://www.redisvl.com/api/index.html)
 - [Example Gallery](https://www.redisvl.com/examples/index.html)
 - [Redis AI Recipes](https://github.com/redis-developer/redis-ai-resources)
 - [Official Redis Vector API Docs](https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/)


## 🫱🏼‍🫲🏽 Contributing

Please help us by contributing PRs, opening GitHub issues for bugs or new feature ideas, improving documentation, or increasing test coverage. [Read more about how to contribute!](CONTRIBUTING.md)

## 🚧 Maintenance
This project is supported by [Redis, Inc](https://redis.com) on a good faith effort basis. To report bugs, request features, or receive assistance, please [file an issue](https://github.com/redis/redis-vl-python/issues).
