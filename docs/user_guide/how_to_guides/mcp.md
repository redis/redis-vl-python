---
myst:
  html_meta:
    "description lang=en": |
      How to run the RedisVL MCP server, configure it, and use its search and upsert tools.
---

# Run RedisVL MCP

This guide shows how to run the RedisVL MCP server against an existing Redis index, configure its behavior, and use the MCP tools it exposes.

For the higher-level design, see {doc}`/concepts/mcp`.

## Before You Start

RedisVL MCP assumes all of the following are already true:

- you have Python 3.10 or newer
- you have Redis with Search capabilities available
- the Redis index already exists
- you know which text field and vector field the server should use
- you have installed the vectorizer provider dependencies your config needs

Install the MCP extra:

```bash
pip install redisvl[mcp]
```

If your vectorizer needs a provider extra, install that too:

```bash
pip install redisvl[mcp,openai]
```

## Start the Server

Run the server over stdio:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml
```

Run it in read-only mode to expose search without upsert:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml --read-only
```

You can also control boot settings through environment variables:

| Variable | Purpose |
|----------|---------|
| `REDISVL_MCP_CONFIG` | Path to the MCP YAML config |
| `REDISVL_MCP_READ_ONLY` | Disable `upsert-records` when set to `true` |
| `REDISVL_MCP_TOOL_SEARCH_DESCRIPTION` | Override the search tool description |
| `REDISVL_MCP_TOOL_UPSERT_DESCRIPTION` | Override the upsert tool description |

## Example Config

This example binds one logical MCP server to one existing Redis index called `knowledge`.

The config uses `${REDIS_URL}` and `${OPENAI_API_KEY}` as environment-variable placeholders. These values are resolved when the server starts. You can also use `${VAR:-default}` to provide a fallback value.

```yaml
server:
  redis_url: ${REDIS_URL}

indexes:
  knowledge:
    redis_name: knowledge

    vectorizer:
      class: OpenAITextVectorizer
      model: text-embedding-3-small
      api_config:
        api_key: ${OPENAI_API_KEY}

    schema_overrides:
      fields:
        - name: embedding
          type: vector
          attrs:
            dims: 1536
            datatype: float32

    search:
      type: hybrid
      params:
        text_scorer: BM25STD
        stopwords: english
        vector_search_method: KNN
        combination_method: LINEAR
        linear_text_weight: 0.3

    runtime:
      text_field_name: content
      vector_field_name: embedding
      default_embed_text_field: content
      default_limit: 10
      max_limit: 25
      max_result_window: 1000
      max_upsert_records: 64
      skip_embedding_if_present: true
      startup_timeout_seconds: 30
      request_timeout_seconds: 60
      max_concurrency: 16
```

### What This Config Means

- `redis_name` must point to an index that already exists in Redis
- `search.type` fixes retrieval behavior for every MCP caller
- `runtime.text_field_name` tells full-text and hybrid search which field to search
- `runtime.vector_field_name` tells the server which vector field to use
- `runtime.default_embed_text_field` tells upsert which text field to embed when a record needs embedding
- `runtime.max_result_window` caps deep paging by limiting the maximum `offset + limit`
- `schema_overrides` is only for patching incomplete field attrs discovered from Redis

## Tool Contracts

RedisVL MCP exposes a small, implementation-owned contract.

### `search-records`

Arguments:

- `query`
- `limit`
- `offset`
- `filter`
- `return_fields`

Example request payload:

```json
{
  "query": "incident response runbook",
  "limit": 2,
  "offset": 0,
  "filter": {
    "and": [
      { "field": "category", "op": "eq", "value": "operations" },
      { "field": "rating", "op": "gte", "value": 4 }
    ]
  },
  "return_fields": ["title", "content", "category", "rating"]
}
```

Example response payload:

```json
{
  "search_type": "hybrid",
  "offset": 0,
  "limit": 2,
  "results": [
    {
      "id": "knowledge:runbook:eu-failover",
      "score": 0.82,
      "score_type": "hybrid_score",
      "record": {
        "title": "EU failover runbook",
        "content": "Restore traffic after a regional failover.",
        "category": "operations",
        "rating": 5
      }
    }
  ]
}
```

Notes:

- `search_type` is response metadata, not a request argument
- when `return_fields` is omitted, RedisVL MCP returns all non-vector fields
- returning the configured vector field is rejected
- `filter` accepts either a raw string or a JSON DSL object
- `offset + limit` must stay within `runtime.max_result_window`

### `upsert-records`

Arguments:

- `records`
- `id_field`
- `skip_embedding_if_present`

Example request payload:

```json
{
  "records": [
    {
      "doc_id": "doc-42",
      "content": "Updated operational guidance for failover handling.",
      "category": "operations",
      "rating": 5
    }
  ],
  "id_field": "doc_id"
}
```

Example response payload:

```json
{
  "status": "success",
  "keys_upserted": 1,
  "keys": ["knowledge:doc-42"]
}
```

Notes:

- this tool is not registered in read-only mode
- records that need embedding must contain `runtime.default_embed_text_field`
- when `skip_embedding_if_present` is `true`, records that already contain the vector field can skip re-embedding

## Search Examples

### Read-Only Vector Search

Use read-only mode when assistants should only retrieve data:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml --read-only
```

With a `search.type` of `vector`, callers send only the query, filters, pagination, and field projection:

```json
{
  "query": "cache invalidation incident",
  "limit": 3,
  "return_fields": ["title", "content", "category"]
}
```

### Raw String Filter

Pass a raw Redis filter string through unchanged:

```json
{
  "query": "science",
  "filter": "@category:{science}",
  "return_fields": ["content", "category"]
}
```

### JSON DSL Filter

The DSL supports logical operators and type-checked field operators:

```json
{
  "query": "science",
  "filter": {
    "and": [
      { "field": "category", "op": "eq", "value": "science" },
      { "field": "rating", "op": "gte", "value": 4 }
    ]
  },
  "return_fields": ["content", "category", "rating"]
}
```

### Pagination and Field Projection

```json
{
  "query": "science",
  "limit": 1,
  "offset": 1,
  "return_fields": ["content", "category"]
}
```

### Hybrid Search With `schema_overrides`

Use `schema_overrides` when Redis inspection cannot recover complete vector attrs, then keep hybrid behavior in config:

```yaml
schema_overrides:
  fields:
    - name: embedding
      type: vector
      attrs:
        algorithm: flat
        dims: 1536
        datatype: float32
        distance_metric: cosine

search:
  type: hybrid
  params:
    text_scorer: BM25STD
    stopwords: english
    vector_search_method: KNN
    combination_method: LINEAR
    linear_text_weight: 0.3
```

The MCP caller still sends the same request shape:

```json
{
  "query": "legacy cache invalidation flow",
  "filter": { "field": "category", "op": "eq", "value": "release-notes" },
  "return_fields": ["title", "content", "release_version"]
}
```

## Upsert Examples

### Auto-Embed New Records

If a record does not include the configured vector field, RedisVL MCP embeds `runtime.default_embed_text_field` and writes the result:

```json
{
  "records": [
    {
      "content": "First upserted document",
      "category": "science",
      "rating": 5
    },
    {
      "content": "Second upserted document",
      "category": "health",
      "rating": 4
    }
  ]
}
```

### Update Existing Records With `id_field`

```json
{
  "records": [
    {
      "doc_id": "doc-1",
      "content": "Updated content",
      "category": "engineering",
      "rating": 5
    }
  ],
  "id_field": "doc_id"
}
```

### Control Re-Embedding With `skip_embedding_if_present`

```json
{
  "records": [
    {
      "doc_id": "doc-2",
      "content": "Existing content",
      "category": "science",
      "rating": 4
    }
  ],
  "id_field": "doc_id",
  "skip_embedding_if_present": false
}
```

Set `skip_embedding_if_present` to `false` when you want the server to regenerate embeddings during upsert. In most cases, the caller should omit the vector field and let the server manage embeddings from `runtime.default_embed_text_field`.

## Troubleshooting

### Missing MCP Dependencies

If `rvl mcp` reports missing optional dependencies, install the MCP extra:

```bash
pip install redisvl[mcp]
```

If the configured vectorizer needs a provider SDK, install that provider extra too.

### Unsupported Python Runtime

RedisVL MCP requires Python 3.10 or newer even though the core package supports Python 3.9. Use a newer interpreter for the MCP server process.

### Configured Redis Index Does Not Exist

The server only binds to an existing index. Create the index first, then point `indexes.<id>.redis_name` at that index name.

### Missing Required Environment Variables

YAML values support `${VAR}` and `${VAR:-default}` substitution. Missing required variables fail startup before the server registers tools.

### Vectorizer Dimension Mismatch

If the vectorizer dims do not match the configured vector field dims, startup fails. Make sure the embedding model and the effective vector field dimensions are aligned.

### Hybrid Config Requires Native Runtime Support

Some hybrid params depend on native hybrid support in Redis and redis-py. If your environment does not support that path, remove native-only params such as `knn_ef_runtime` or upgrade Redis and redis-py.
