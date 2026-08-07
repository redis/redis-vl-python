---
myst:
  html_meta:
    "description lang=en": |
      Installation instructions for RedisVL
---

# Install RedisVL

There are a few ways to install RedisVL. The easiest way is to use pip.

## Install RedisVL with Pip

Install `redisvl` into your Python (>=3.10) environment using `pip`:

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
$ pip install redisvl[google-genai]        # Google embeddings (Vertex AI + Gemini API)
$ pip install redisvl[vertexai]            # Google Vertex AI embeddings (legacy; deprecated, multimodal only)
$ pip install redisvl[bedrock]             # AWS Bedrock embeddings

# Other optional features
$ pip install redisvl[mcp]                 # RedisVL MCP server support (Python 3.10+)
$ pip install redisvl[langcache]           # LangCache managed service integration
$ pip install redisvl[sql-redis]           # SQL query support
```

If you use ZSH, remember to escape the brackets:

```bash
$ pip install redisvl\[openai\]
```

You can install multiple optional dependencies at once:

```bash
$ pip install redisvl[mcp,openai,cohere,sentence-transformers]
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

RedisVL requires Redis with [Redis Search](https://redis.io/docs/latest/develop/ai/search-and-query/) available. There are several options:

1. [Redis Cloud](https://redis.io/cloud), a fully managed cloud offering with a free tier
2. [Redis 8+ (Docker)](https://redis.io/downloads/), for local development and testing
3. [Redis Enterprise](https://redis.io/software/), a commercial self-hosted option

### Redis Cloud

Redis Cloud is the easiest way to get started with RedisVL. You can sign up for a free account [here](https://redis.io/cloud). Make sure to have `Redis Search`
enabled when creating your database.

### Redis 8+ (local development)

For local development and testing, we recommend running Redis 8+ in a Docker container:

```bash
docker run -d --name redis -p 6379:6379 redis:8.4
```

Redis 8 includes Redis Search and built-in vector search capabilities.

### Redis Enterprise (self-hosted)

Redis Enterprise is a commercial offering that can be self-hosted. You can download the latest version [here](https://redis.io/downloads/).

If you are considering a self-hosted Redis Enterprise deployment on Kubernetes, there is the [Redis Enterprise Operator](https://docs.redis.com/latest/kubernetes/) for Kubernetes. This will allow you to easily deploy and manage a Redis Enterprise cluster on Kubernetes.

### Redis Sentinel

For high availability deployments, RedisVL supports connecting to Redis through Sentinel. Use the `redis+sentinel://` URL scheme to connect. Both sync and async connections are fully supported.

```python
from redisvl.index import SearchIndex, AsyncSearchIndex

# Sync connection via Sentinel
# Format: redis+sentinel://[username:password@]host1:port1,host2:port2/service_name[/db]
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
)

# Async connection via Sentinel
async_index = AsyncSearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
)

# With authentication and database selection
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://user:pass@sentinel1:26379,sentinel2:26379/mymaster/0"
)
```

The Sentinel URL format supports:

- Multiple sentinel hosts (comma-separated)
- Optional authentication (username:password)
- Service name (defaults to `mymaster` if not specified)
- Optional database number (defaults to 0)
- Both sync (`SearchIndex`) and async (`AsyncSearchIndex`) connections

## Redis permissions (ACLs)

RedisVL works through Redis Search commands, so a connecting credential needs the `@search` ACL category, or the individual `FT.*` commands. Reading an index additionally requires key permissions covering its prefix: the [ACL documentation](https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/#command-categories) describes this rule for creating, modifying, and reading an index, and in practice `FT.INFO`, `FT.SEARCH`, and `FT.AGGREGATE` are denied when the index prefix falls outside the allowed key patterns. `FT.CREATE` is not checked this way, so a credential can create an index it is then unable to read.

One command needs more than `@search`. Redis tags `FT._LIST` as `@admin` as well as `@search` and `@slow`, and ACL rules are applied left to right — so a rule that grants search access and then takes back administrative commands, such as `+@search -@admin` or `+@all -@admin`, denies it:

```text
User <name> has no permissions to run the 'FT._LIST' command
```

Note that `FT._LIST` does not *require* `@admin`: granting `+@search` on its own permits it. Only rules that subtract `@admin` after granting search are affected. To keep such a policy and still enumerate indexes, grant the command back explicitly with `+ft._list`.

| Operation | Redis command | Permitted by `+@all -@admin` |
|---|---|---|
| `index.create()`, `index.exists()` | `FT.CREATE`, `FT.INFO` | Yes |
| `index.query()`, `index.search()`, `index.aggregate()` | `FT.SEARCH`, `FT.AGGREGATE` | Yes |
| `index.info()`, `rvl index info`, `rvl stats` | `FT.INFO` | Yes |
| `index.load()` | `HSET` or `JSON.SET` (needs `@write` and key access) | Yes |
| `index.delete()`, `rvl index delete`, `rvl index destroy` | `FT.DROPINDEX` | Yes |
| Enumerating indexes (see below) | `FT._LIST` | No |

`SemanticCache`, `SemanticMessageHistory`, `MessageHistory`, and `SemanticRouter` each call `index.create()` while being constructed, so they are covered by the first row.

Index enumeration is the only thing an `-@admin` rule breaks. It is reached by `SearchIndex.listall()` and `AsyncSearchIndex.listall()`, by `rvl index listall`, and by the migration entry points that discover indexes for you: `rvl migrate helper`, `rvl migrate wizard` when no `-i/--index` is given, and `rvl migrate batch-plan --pattern`.

Other categories gate different operations. `FT.DROPINDEX` is tagged `@dangerous` and `@write` as well as `@search`, so a policy that subtracts `@dangerous` denies `index.delete()`, `rvl index delete`, and `rvl index destroy`.

Outside of Redis Search, RedisVL identifies itself on connect with `CLIENT SETINFO` and falls back to `ECHO`. A credential permitted to run neither currently fails when the connection is created, before any index operation.

### Redis Cloud and Redis Software

Both manage ACLs through their own control plane rather than the `ACL SETUSER` command, so permissions are attached to roles and users instead of being set on a connection.

- **Redis Cloud** provides three predefined ACL rules that cannot be edited — Full-Access, Read-Write ("read and write commands and excludes dangerous commands"), and Read-Only — which you assign to a data access role. See [Configure permissions with Redis ACLs](https://redis.io/docs/latest/operate/rc/security/access-control/data-access-control/configure-acls/). Custom rules use the same syntax as above.
- **Redis Software** ships one predefined ACL, Full Access, and you define others in the Cluster Manager UI or with a [`POST /v1/redis_acls`](https://redis.io/docs/latest/operate/rs/security/access-control/create-db-roles/) request. It [does not support every `ACL` command](https://redis.io/docs/latest/operate/rs/security/access-control/redis-acl-overview/#acl-command-support), nor nested selectors, nor `(` and `)` in key patterns.

Because the predefined rules' exact command sets are not published, confirm a credential against the database rather than inferring what its policy name implies. Redis Software's documentation uses `+@read +FT.INFO +FT.SEARCH` as an example rule, which is a good illustration: it permits querying and `index.exists()`, but not `index.create()` or index enumeration. Grant `FT.CREATE` explicitly when the application creates its own index, which `SemanticCache`, `SemanticMessageHistory`, `MessageHistory`, and `SemanticRouter` all do.
