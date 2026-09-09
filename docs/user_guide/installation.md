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

RedisVL reaches Redis through Redis Search commands, but not all of them need the `@search` category. Querying and loading work under an ordinary `+@read +@write` role; it is the commands that inspect or manage an index — `FT.INFO`, `FT.CREATE`, `FT._LIST` — that need `@search` or an explicit grant.

The command-to-category mapping below was measured against live servers rather than quoted from published documentation, which does not list ACL categories per `FT.*` command. It was identical on Redis 8.0.6, 8.2.7, 8.4.5, 8.6.4, 8.8.0 and 8.8.1. Check your own deployment with `COMMAND INFO ft.info`, `ACL CAT search`, or `ACL DRYRUN <user> FT.INFO <index>`.

### What each operation needs

| Operation | Redis command | `+@all -@admin` | `+@read +@write` |
|---|---|---|---|
| `index.query()`, `index.search()`, `index.aggregate()` | `FT.SEARCH`, `FT.AGGREGATE` | Yes | Yes |
| `index.load()` | `HSET` or `JSON.SET` (needs key access) | Yes | Yes |
| `index.clear()` | `FT.SEARCH`, then `DEL` per batch | Yes | Yes |
| `index.exists()`, `index.info()`, `SearchIndex.from_existing()`, `rvl index info`, `rvl stats` | `FT.INFO` | Yes | **No** |
| `index.create()` | `FT.CREATE` | Yes | **No** |
| `index.delete()`, `rvl index delete`, `rvl index destroy` | `FT.DROPINDEX` | Yes | Yes |
| Enumerating indexes (see below) | `FT._LIST` | **No** | **No** |

Except for `FT.CREATE`, every `Yes` above assumes key patterns that cover the index prefix — see [Key permissions](#key-permissions). `FT.CREATE` is not checked against those patterns, so a credential can create an index it cannot query. Adding `-@dangerous` to the second column additionally denies `FT.DROPINDEX`, so `index.delete()` becomes `No`. An SVS-VAMANA schema needs more than `FT.CREATE`: `index.create()` first probes capabilities with `INFO` (`@slow @dangerous`) and `MODULE LIST` (`@admin @slow @dangerous`), so both `-@admin` and `-@dangerous` policies break creation for those schemas.

Two of the rows above deserve their own explanation.

`FT._LIST` is tagged `@admin` as well as `@search` and `@slow`, and ACL rules are applied left to right — so a rule that grants search access and then takes back administrative commands, such as `+@search -@admin` or `+@all -@admin`, denies it:

```text
User <name> has no permissions to run the 'FT._LIST' command
```

`FT._LIST` does not *require* `@admin`: granting `+@search` on its own permits it. Only rules that subtract `@admin` after granting search are affected. To keep such a policy and still enumerate indexes, grant the command back explicitly with `+ft._list`. Enumeration is reached by `SearchIndex.listall()` and `AsyncSearchIndex.listall()`, by `rvl index listall`, and by the migration entry points that discover indexes for you: `rvl migrate helper`, `rvl migrate wizard` when no `-i/--index` is given, and `rvl migrate batch-plan --pattern`.

`FT.DROPINDEX` is tagged `@dangerous` and `@write` as well as `@search`, so a policy that subtracts `@dangerous` denies `index.delete()`, `rvl index delete`, and `rvl index destroy`.

### Roles built from `@read` and `@write`

`FT.INFO` is in neither `@read` nor `@write` — its only category is `@search` — and `FT.CREATE` is the same. So an application role assembled from `+@read +@write -@dangerous` — a natural least-privilege shape for a runtime that must query and load but must not manage indexes — can query and load, but cannot ask whether an index exists and cannot create or drop one. Subtracting `@dangerous` is not what denies `FT.INFO` or `FT.CREATE`: `+@all -@dangerous` permits both. Those commands are simply never granted by `@read` or `@write`.

Every extension constructor checks whether its index exists, so under such a credential all of them fail while being constructed:

```text
RedisSearchError: Error while fetching llmcache index info:
User <name> has no permissions to run the 'FT.INFO' command
```

### "no permissions to run the 'FT.INFO' command"

RedisVL does not guess its way around this. A credential that cannot ask whether the index exists also cannot create one, so there is nothing useful to infer — instead, tell RedisVL that the index is already there:

```python
cache = SemanticCache(
    name="llmcache",
    redis_url="redis://localhost:6379",
    create_index=False,
)
```

`create_index=False` is available on `SemanticCache`, `MessageHistory`, `SemanticMessageHistory` and `SemanticRouter`. It skips the existence check, the comparison of your schema against the live index, and index creation — the constructor issues no index command at all. Pass it when the index is managed externally, or when the credential cannot run `FT.INFO`. It cannot be combined with `overwrite=True`, which asks for the opposite.

A `SearchIndex` used directly needs nothing special: build it with `from_dict()` or `from_yaml()`, then load and query. The methods that stay unavailable are the ones that read index metadata — `exists()`, `info()`, and `from_existing()`, which reconstructs a schema out of Redis. `clear()` is not among them: it enumerates with `FT.SEARCH` and deletes in batches, so it needs no more than querying does.

The flag also skips the SVS-VAMANA capability probe described above, since that runs inside `create()`.

### Provisioning a router

`SemanticRouter` with `create_index=False` writes nothing at all: not the reference vectors for its routes, and not the stored route config that `SemanticRouter.from_existing()` reads. Preparing a router for this mode therefore means constructing it once with a privileged credential — a hand-written `FT.CREATE` is not enough, because the reference vectors have to be embedded and written too. Without them the router matches nothing, which looks like a distance-threshold problem rather than an empty index.

Afterwards, `SemanticRouter.from_existing(name, create_index=False)` is the way to attach to it: it recovers the routes and thresholds with `JSON.GET` and needs no `FT.INFO`. The stored config must contain the full route set. Each route's distance threshold is applied from that recovered list, so an incomplete stored config silently narrows matching — and `add_route()` and `remove_route()` rewrite the stored config from the same list, so mutating an incomplete config permanently drops the omitted routes from the config every other client reads.

### When the schema diverges

With `create_index=False` nothing verifies that the live index matches the schema you described. Some mismatches are loud on first use, and several are silent:

| Mismatch | What happens |
|---|---|
| The index does not exist | `RedisSearchError` on the first query, naming the missing index |
| Vector dimensions disagree | `Error parsing vector similarity query: query vector blob size (32) does not match index's expected size (16)` — but only once the index holds a document. On an empty index the same query returns nothing, so a freshly provisioned index hides this until the first write lands |
| The prefix does not cover your keys | **Silent.** Documents are written but never indexed, so queries return nothing, forever |
| The index is `ON JSON` and you write hashes (or the reverse) | **Silent**, the same way |
| The datatype or distance metric differs | **Silent.** Neither is restated by a query, so nothing compares them — results come back ranked by the index's metric, not yours |

For the silent cases the tell is `FT.INFO`'s `key_type`, `prefixes` and `attributes` — not `hash_indexing_failures`, which stays `0` because those keys were never indexing candidates. Diagnosing it therefore needs a credential that can run `FT.INFO`.

### What an attach-only instance may still do

Removing *entries* is available on every path, and is how a caller invalidates an externally managed cache without holding the provisioning credential: `clear()` (plus `SemanticCache.aclear()`), and targeted removal of a specific cache entry, message or route. None of it removes the index, and all of it runs under `+@read +@write`.

What `create_index=False` refuses is `delete()` (and `SemanticCache.adelete()`), because that drops the index. Refusing it protects an externally managed index — including one reached through an alias — from being destroyed through an attach-only instance. Drop the index through the privileged provisioning path that owns it.

The two kinds of `clear()` decide *which keys go* differently, and neither choice is verified against the live index under this flag:

| Method | Deletes | Chooses keys by |
|---|---|---|
| `SemanticCache.clear()`, `aclear()` | every key under `{name}:` | `SCAN`/`DEL` on the prefix this instance declares — no index command at all |
| `MessageHistory.clear()`, `SemanticMessageHistory.clear()`, `SemanticRouter.clear()` | every document the live index covers | `FT.SEARCH` paging via `SearchIndex.clear()` |

`FT.SEARCH` is in `@read` as well as `@search`, so a `+@read +@write` credential is granted it — unlike `FT.INFO`, which is in neither and is what made these three unavailable before. Note that `FT.SEARCH` additionally requires the credential's key patterns to be a superset of the index prefixes, the same rule described under [Key permissions](#key-permissions).

Because the two enumerate differently, they fail differently, and the section above is what decides which failure you get. Both are silent:

- **Prefix-based clearing deletes too much, or nothing.** `SCAN`/`DEL` is blind to the index and to the key type, so it removes every key under `{name}:` — another writer's entries, and unrelated application data sharing that namespace root. And if the live index covers a *different* prefix, or is an alias onto one, `clear()` deletes only what this instance itself wrote and leaves every served entry in place: it reports success and the cache still returns the stale hits you called it to invalidate.
- **Index-based clearing deletes documents you never wrote.** `SearchIndex.clear()` deletes what the live index covers, so against an index on a different prefix — or a multi-`PREFIX` index, or an alias — it removes another application's documents while leaving this instance's own unindexed entries behind.

Diagnosing either needs `FT.INFO`, which is the command an attach-only credential does not have. If the index is provisioned for you, get its `prefixes` and `key_type` from whoever provisions it and make your extension's name match, rather than inferring it from a successful query.

### Key permissions

Command categories are only half of it. Redis also scopes the search commands by key pattern: [the ACL documentation](https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/#command-categories) states that only users with access to a *superset* of the prefixes defined at index creation can create, modify, or read an index.

Measured on 8.4.5 against an index prefixed `doc:`, with the command categories held constant at `+@all`:

| Key patterns | `FT.SEARCH`, `FT.INFO`, `FT.AGGREGATE` |
|---|---|
| `~doc:*` (superset) | Permitted |
| `%R~doc:*` (read permission only) | Permitted |
| `~doc:1` (partial overlap) | `NOPERM User does not have the required permissions to query the index` |
| `~other:*` (no overlap) | The same denial |

Partial overlap is worth emphasising: it fails exactly like no overlap at all, rather than returning the subset you can read. `FT.CREATE` is not checked this way, so a credential can create an index it is then unable to query.

`create_index=False` does not help here — the very commands it lets you avoid are joined by the ones it cannot, so widen the key patterns instead.

Outside of Redis Search, RedisVL identifies itself on connect with `CLIENT SETINFO`. That command is tagged `@connection` and `@slow`, and belongs to neither `@read` nor `@write`, so a rule built up from those categories never grants it. A credential that cannot run it still connects: identification only populates the `lib-name` field that `CLIENT LIST` and `CLIENT INFO` display, so a refusal is ignored (and logged, if you have configured logging at debug level). Grant `+client|setinfo` if you want RedisVL to appear as the connecting library there — note that this labels the connection RedisVL opens, while redis-py labels the rest of the pool as plain `redis-py`.

Cluster deployments need one more grant. `RedisCluster` discovers the topology with `CLUSTER SLOTS`, which is tagged `@slow` only, so a credential assembled from `+@read +@write` cannot open a clustered connection at all — redis-py reports this as `Redis Cluster cannot be connected`, with the underlying permission error chained beneath it. Grant `+cluster|slots` alongside the rules above.

### Redis Cloud and Redis Software

Both manage ACLs through their own control plane rather than the `ACL SETUSER` command, so permissions are attached to roles and users instead of being set on a connection.

- **Redis Cloud** provides three predefined ACL rules that cannot be edited — Full-Access, Read-Write ("read and write commands and excludes dangerous commands"), and Read-Only — which you assign to a data access role. See [Configure permissions with Redis ACLs](https://redis.io/docs/latest/operate/rc/security/access-control/data-access-control/configure-acls/). Custom rules use the same syntax as above.
- **Redis Software** ships one predefined ACL, Full Access, and you define others in the Cluster Manager UI or with a [`POST /v1/redis_acls`](https://redis.io/docs/latest/operate/rs/security/access-control/create-db-roles/) request. It [does not support every `ACL` command](https://redis.io/docs/latest/operate/rs/security/access-control/redis-acl-overview/#acl-command-support), nor nested selectors, nor `(` and `)` in key patterns.

Because the predefined rules' exact command sets are not published, confirm a credential against the database rather than inferring what its policy name implies — `ACL DRYRUN <user> FT.INFO <index>` answers it directly. Read the descriptions carefully before assuming you are unaffected: Read-Only allows read commands, and Read-Write "allows read and write commands and excludes dangerous commands", so both read as `@read`/`@write`-shaped — the shape that denies `FT.INFO` and `FT.CREATE` and wants `create_index=False`. Only Full-Access is clearly unaffected.

Redis Software's documentation uses `+@read +FT.INFO +FT.SEARCH` as an example rule, which is a good illustration: it permits querying and `index.exists()`, but not `index.create()` or index enumeration. Grant `FT.CREATE` explicitly when the application creates its own index. When the index is provisioned for the application instead, leave it out and construct with `create_index=False`.
