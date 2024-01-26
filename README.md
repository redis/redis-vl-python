# RedisVL: Python Client Library for Redis as a Vector Database

<div align="center">
<div display="inline-block">
    <a href="https://github.com/RedisVentures/RedisVL"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.redisvl.com"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/RedisVentures"><b>More Projects</b></a>&nbsp;&nbsp;&nbsp;
  </div>
    <br />
</div>


<div align="center">

[![Codecov](https://img.shields.io/codecov/c/github/RedisVentures/RedisVL/dev?label=Codecov&logo=codecov&token=E30WxqBeJJ)](https://codecov.io/gh/RedisVentures/RedisVL)
[![License](https://img.shields.io/badge/License-BSD-3--blue.svg)](https://opensource.org/licenses/mit/)
![Language](https://img.shields.io/github/languages/top/RedisVentures/RedisVL)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub last commit](https://img.shields.io/github/last-commit/RedisVentures/RedisVL)
![GitHub deployments](https://img.shields.io/github/deployments/RedisVentures/RedisVL/github-pages?label=doc%20build)
[![pypi](https://badge.fury.io/py/redisvl.svg)](https://pypi.org/project/redisvl/)

</div>

RedisVL provides a powerful Python client library for using Redis as a Vector Database. Leverage the speed and reliability of Redis along with vector-based semantic search capabilities to supercharge your application!

**Note**: This supported by Redis, Inc. on a good faith effort basis. To report bugs, request features, or receive assistance, please [file an issue](https://github.com/RedisVentures/redisvl/issues).


------------


## 🚀 What is RedisVL?

Vector databases have become increasingly popular in recent years due to their ability to store and retrieve vectors efficiently. However, most vector databases are complex to use and require a lot of time and effort to set up. RedisVL aims to solve this problem by providing a simple and intuitive interface for using Redis as a vector database.

RedisVL provides a client library that enables you to harness the power and flexibility of Redis as a vector database. This library simplifies the process of storing, retrieving, and performing complex semantic and hybrid searches over vectors in Redis. It also provides a robust index management system that allows you to create, update, and delete indices with ease.


### Capabilities

RedisVL has a host of powerful features designed to streamline your vector database operations.

1. **Index Management**: RedisVL allows for indices to be created, updated, and deleted with ease. A schema for each index can be defined in yaml or directly in python code and used throughout the lifetime of the index.
    - [Getting Started with SearchIndex](https://www.redisvl.com/user_guide/getting_started_01.html)
    - [``rvl`` Command Line Interface](https://www.redisvl.com/user_guide/cli.html)

2. **Embedding Creation**: RedisVLs [Vectorizers](https://www.redisvl.com/user_guide/vectorizers_04.html) integrate with common embedding model services to simplify the process of vectorizing unstructured data.
   - [OpenAI](https://www.redisvl.com/api/vectorizer.html#openaitextvectorizer)
   - [HuggingFace](https://www.redisvl.com/api/vectorizer.html#hftextvectorizer)
   - [GCP VertexAI](https://www.redisvl.com/api/vectorizer.html#vertexaitextvectorizer)

3. **Vector Search**: RedisVL provides robust search capabilities that enable you quickly define complex search queries with flexible abstractions.
   - [VectorQuery](https://www.redisvl.com/api/query.html#vectorquery) - Flexible vector queries with filters
   - [RangeQuery](https://www.redisvl.com/api/query.html#rangequery) - Vector search within a defined range
   - [CountQuery](https://www.redisvl.com/api/query.html#countquery) - Count the number of records given attributes
   - [FilterQuery](https://www.redisvl.com/api/query.html#filterquery) - Filter records given attributes

3. **[Hybrid (Filtered) queries](https://www.redisvl.com/user_guide/hybrid_queries_02.html)** that utilize tag, geographic, numeric, and other filters like full-text search are also supported.

4. **Semantic Caching**: [`LLMCache`](https://www.redisvl.com/user_guide/llmcache_03.html) is a semantic caching interface built directly into RedisVL. Semantic caching is a popular technique to increase the QPS and reduce the cost of using LLM models in production.

5. [**JSON Storage**](https://www.redisvl.com/user_guide/hash_vs_json_05.html): RedisVL supports storing JSON objects, including vectors, in Redis.

## Installation

Install `redisvl` using `pip`:

```bash
pip install redisvl
```

For more instructions, see the [installation guide](https://www.redisvl.com/overview/installation.html).

## Getting Started

To get started with RedisVL, check out the
 - [Getting Started Guide](https://www.redisvl.com/user_guide/getting_started_01.html)
 - [API Reference](https://www.redisvl.com/api/index.html)
 - [Example Gallery](https://www.redisvl.com/examples/index.html)


## Contributing

Please help us by contributing PRs or opening GitHub issues for desired behaviors or discovered bugs. [Read more about how to contribute to RedisVL!](CONTRIBUTING.md)
