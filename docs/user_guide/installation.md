---
myst:
  html_meta:
    "description lang=en": |
      Installation instructions for RedisVL
---

# Install RedisVL

There are a few ways to install RedisVL. The easiest way is to use pip.

## Install RedisVL with Pip

Install `redisvl` into your Python (>=3.9) environment using `pip`:

```bash
$ pip install -U redisvl
```

RedisVL comes with a few dependencies that are automatically installed, however, several optional
dependencies can be installed separately based on your needs:

```bash
# Vectorizer providers
$ pip install redisvl[openai]              # OpenAI embeddings
$ pip install redisvl[cohere]              # Cohere embeddings and reranking
$ pip install redisvl[mistralai]           # Mistral AI embeddings
$ pip install redisvl[voyageai]            # Voyage AI embeddings and reranking
$ pip install redisvl[sentence-transformers]  # HuggingFace local embeddings
$ pip install redisvl[vertexai]            # Google Vertex AI embeddings
$ pip install redisvl[bedrock]             # AWS Bedrock embeddings

# Other optional features
$ pip install redisvl[langcache]           # LangCache managed service integration
$ pip install redisvl[sql-redis]           # SQL query support
```

If you use ZSH, remember to escape the brackets:

```bash
$ pip install redisvl\[openai\]
```

You can install multiple optional dependencies at once:

```bash
$ pip install redisvl[openai,cohere,sentence-transformers]
```

To install **all** optional dependencies at once:

```bash
$ pip install redisvl[all]
```

## Install RedisVL from Source

To install RedisVL from source, clone the repository and install the package using `pip`:

```bash
$ git clone https://github.com/redis/redis-vl-python.git && cd redis-vl-python
$ pip install .

# or for an editable installation (for developers of RedisVL)
$ pip install -e .
```

## Development Installation

For contributors who want to develop RedisVL, we recommend using [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Clone the repository
$ git clone https://github.com/redis/redis-vl-python.git && cd redis-vl-python

# Install uv if you don't have it
$ pip install uv

# Install all dependencies (including dev and docs)
$ uv sync

# Or use make
$ make install
```

This installs the package in editable mode along with all development dependencies (testing, linting, type checking) and documentation dependencies.

### Running Tests and Linting

```bash
# Run tests (no external APIs required)
$ make test

# Run all tests (includes API-dependent tests)
$ make test-all

# Format code
$ make format

# Run type checking
$ make check-types

# Run full check (lint + test)
$ make check
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them with:

```bash
$ pre-commit install
```

Run hooks manually on all files:

```bash
$ pre-commit run --all-files
```

## Installing Redis

RedisVL requires a distribution of Redis that supports the [Search and Query](https://redis.com/modules/redis-search/) capability. There are several options:

1. [Redis Cloud](https://redis.io/cloud), a fully managed cloud offering with a free tier
2. [Redis 8+ (Docker)](https://redis.io/downloads/), for local development and testing
3. [Redis Enterprise](https://redis.com/redis-enterprise/), a commercial self-hosted option

### Redis Cloud

Redis Cloud is the easiest way to get started with RedisVL. You can sign up for a free account [here](https://redis.io/cloud). Make sure to have the `Search and Query`
capability enabled when creating your database.

### Redis 8+ (local development)

For local development and testing, we recommend running Redis 8+ in a Docker container:

```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

Redis 8 includes built-in vector search capabilities.

### Redis Enterprise (self-hosted)

Redis Enterprise is a commercial offering that can be self-hosted. You can download the latest version [here](https://redis.io/downloads/).

If you are considering a self-hosted Redis Enterprise deployment on Kubernetes, there is the [Redis Enterprise Operator](https://docs.redis.com/latest/kubernetes/) for Kubernetes. This will allow you to easily deploy and manage a Redis Enterprise cluster on Kubernetes.

### Redis Sentinel

For high availability deployments, RedisVL supports connecting to Redis through Sentinel. Use the `redis+sentinel://` URL scheme to connect:

```python
from redisvl.index import SearchIndex

# Connect via Sentinel
# Format: redis+sentinel://[username:password@]host1:port1,host2:port2/service_name[/db]
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
)

# With authentication
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://user:pass@sentinel1:26379,sentinel2:26379/mymaster/0"
)
```

The Sentinel URL format supports:

- Multiple sentinel hosts (comma-separated)
- Optional authentication (username:password)
- Service name (required - the name of the Redis master)
- Optional database number (defaults to 0)
