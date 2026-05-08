---
description: Run the RedisVL MCP server, configure it, and use its search and upsert tools.
---

# Run RedisVL MCP

This guide shows how to run the RedisVL MCP server against an existing Redis index, configure its behavior, and use the MCP tools it exposes.

For the higher-level design, see [MCP](../../concepts/mcp.md).

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

Run the server over stdio (default):

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml
```

Run it over Streamable HTTP for remote MCP clients:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml --transport streamable-http --host 0.0.0.0 --port 8000
```

Run it over SSE:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml --transport sse --host 0.0.0.0 --port 9000
```

!!! warning

    Streamable HTTP and SSE endpoints are **unauthenticated by default**. Only bind to public interfaces (`--host 0.0.0.0`) on trusted networks or behind an authenticating reverse proxy. When not using `--read-only`, the `upsert-records` tool is also exposed to any client that can reach the server.

Run it in read-only mode to expose search without upsert:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp.yaml --read-only
```

### CLI Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--config` | — | Path to the MCP YAML config (required) |
| `--transport` | `stdio` | Transport protocol: `stdio`, `sse`, or `streamable-http` |
| `--host` | `127.0.0.1` | Bind address (only used with `sse` and `streamable-http`) |
| `--port` | `8000` | Bind port (only used with `sse` and `streamable-http`) |
| `--read-only` | off | Disable the `upsert-records` tool |

### Environment Variables

You can also control boot settings through environment variables:

| Variable | Purpose |
|----------|---------|
| `REDISVL_MCP_CONFIG` | Path to the MCP YAML config |
| `REDISVL_MCP_READ_ONLY` | Disable `upsert-records` when set to `true` |
| `REDISVL_MCP_TOOL_SEARCH_DESCRIPTION` | Set the base search tool description text; RedisVL still appends schema-derived typed filter, `exists`, and `return_fields` hints |
| `REDISVL_MCP_TOOL_UPSERT_DESCRIPTION` | Override the upsert tool description |

## Connect a Remote MCP Client

When using Streamable HTTP or SSE transport, point your MCP client at the server URL:

- **Streamable HTTP**: `http://<host>:<port>/mcp`
- **SSE**: `http://<host>:<port>/sse`

> **Note:** `<host>` here is the bind address the server was started with. The default `127.0.0.1` only accepts connections from the same machine. To allow connections from other machines, start the server with `--host 0.0.0.0` and use the machine's actual IP or hostname in the client URL.

For example, to configure a remote MCP client to connect to a Streamable HTTP server running on `192.168.1.10:8000`:

```json
{
  "mcpServers": {
    "redisvl": {
      "url": "http://192.168.1.10:8000/mcp",
      "transport": "streamable-http"
    }
  }
}
```

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
- `runtime.text_field_name` is required for `fulltext` and `hybrid` search
- `runtime.vector_field_name` is required for `vector` and `hybrid` search, and optional for plain full-text deployments
- `runtime.default_embed_text_field` is only required when the server should generate embeddings during upsert
- `vectorizer` is required for query embedding and server-side embedding, but optional for fulltext-only configs
- `runtime.max_result_window` caps deep paging by limiting the maximum `offset + limit`
- `schema_overrides` is only for patching incomplete field attrs discovered from Redis
