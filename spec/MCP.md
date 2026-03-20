---
name: redisvl-mcp-server-spec
description: Implementation specification for a RedisVL MCP server with deterministic, agent-friendly contracts for development and testing.
metadata:
  status: draft
  audience: RedisVL maintainers and coding agents
  objective: Define a deterministic, testable MCP server contract so agents can implement safely without relying on implicit behavior.
---

# RedisVL MCP Server Specification

## Overview

This specification defines a Model Context Protocol (MCP) server for RedisVL that allows MCP clients to search and upsert data in an existing Redis index.

The MCP design targets indexes hosted on open-source Redis Stack, Redis Cloud, or Redis Enterprise, provided the required Search capabilities are available for the configured tool behavior.

The server is designed for stdio transport first and must be runnable via:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp_config.yaml
```

For a production-oriented usage narrative and end-to-end example, see [MCP-production-example.md](./MCP-production-example.md).

### Goals

1. Expose RedisVL search capabilities (`vector`, `fulltext`, `hybrid`) through stable MCP tools.
2. Support controlled write access via an upsert tool.
3. Automatically reconstruct the index schema from an existing Redis index instead of requiring a full manual schema definition.
4. Keep the vectorizer configuration explicit and user-defined.
5. Provide deterministic contracts for tool inputs, outputs, and errors.
6. Align implementation with existing RedisVL architecture and CLI patterns.

### Non-Goals (v1)

1. Multi-index routing in a single server process.
2. Remote transports (SSE/HTTP).
3. Index creation or schema provisioning from MCP config.
4. Delete/count/info tools (future scope).
5. Automatic vectorizer selection from Redis metadata.

---

## Compatibility Matrix

These are hard compatibility expectations for v1.

| Component | Requirement | Notes |
|----------|-------------|-------|
| Core RedisVL package | Python `>=3.9.2,<3.15` | Match current project constraints |
| MCP feature | Python `>=3.10,<3.15` | `redisvl[mcp]` may have a stricter floor than the core package |
| RedisVL | current repo version | Server lives inside this package |
| redis-py | `>=5.0,<7.2` | Already required by project |
| FastMCP server SDK | `fastmcp>=2.0.0` | Standalone FastMCP package used for server implementation |
| Redis server | Redis Stack / Redis with Search module | Required for all search modes |
| Hybrid search | Prefer native implementation on Redis `>=8.4.0` with redis-py `>=7.1.0`; otherwise fall back to `AggregateHybridQuery` | Hybrid search remains available across both paths |

Notes:
- This spec standardizes on the standalone `fastmcp` package for server implementation. It does not assume the official `mcp` package is on a 2.x line.
- Client SDK examples may still use whichever client-side MCP package their ecosystem requires.
- Native hybrid support is preferred when available because it aligns with current Redis runtime capabilities, but lack of native support is not a blocker for `search_type=\"hybrid\"`.

---

## Architecture

### Module Structure

```text
redisvl/
├── mcp/
│   ├── __init__.py
│   ├── server.py             # RedisVLMCPServer
│   ├── settings.py           # MCPSettings
│   ├── config.py             # Config models + loader + validation
│   ├── errors.py             # MCP error mapping helpers
│   ├── filters.py            # Filter parser (DSL + raw string handling)
│   └── tools/
│       ├── __init__.py
│       ├── search.py         # search-records
│       └── upsert.py         # upsert-records
└── cli/
    ├── main.py               # Add `mcp` command dispatch
    └── mcp.py                # MCP command handler class
```

### Dependency Groups

Add optional extras for explicit install intent.

```toml
[project.optional-dependencies]
mcp = [
  "fastmcp>=2.0.0",
  "pydantic-settings>=2.0",
]
```

Notes:
- `fulltext` and both hybrid implementations (`HybridQuery` and `AggregateHybridQuery`) rely on the same query-time stopword handling. If `nltk` is not installed and stopwords are enabled, server must return a structured dependency error.
- Provider vectorizer dependencies remain provider-specific (`openai`, `cohere`, `vertexai`, etc.).

---

## Configuration

Configuration is composed from environment + YAML:

1. `MCPSettings` from env/CLI.
2. YAML file referenced by `config` setting.
3. Env substitution inside YAML with strict validation.

The normal v1 path is inspection-first: the YAML identifies a single configured index binding, the server discovers that Redis index's schema at startup, and optional overrides patch only discovery gaps. The YAML shape is future-friendly for multi-index support even though v1 allows exactly one configured index.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDISVL_MCP_CONFIG` | str | required | Path to MCP YAML config |
| `REDISVL_MCP_READ_ONLY` | bool | `false` | If true, do not register upsert tool |
| `REDISVL_MCP_TOOL_SEARCH_DESCRIPTION` | str | default text | MCP tool description override |
| `REDISVL_MCP_TOOL_UPSERT_DESCRIPTION` | str | default text | MCP tool description override |

### YAML Schema (Normative)

```yaml
server:
  redis_url: redis://localhost:6379

indexes:
  knowledge:
    redis_name: knowledge

    vectorizer:
      class: OpenAITextVectorizer
      model: text-embedding-3-small
      # kwargs passed to vectorizer constructor
      # for providers using api_config, pass as nested object:
      # api_config:
      #   api_key: ${OPENAI_API_KEY}

    schema_overrides:
      fields:
        - name: embedding
          type: vector
          attrs:
            dims: 1536
            datatype: float32

    runtime:
      # required explicit field mapping for tool behavior
      text_field_name: content
      vector_field_name: embedding
      default_embed_text_field: content

      # request constraints
      default_limit: 10
      max_limit: 100
      max_upsert_records: 64

      # default overwrite behavior for existing vectors
      skip_embedding_if_present: true

      # timeouts
      startup_timeout_seconds: 30
      request_timeout_seconds: 60

      # server-side concurrency guard
      max_concurrency: 16
```

### Schema Discovery and Override Rules

1. `server.redis_url` is required.
2. `indexes` is required and must contain exactly one configured binding in v1.
3. The `indexes` mapping key is the logical binding id. It is stable for future routing and does not need to equal the Redis index name.
4. `indexes.<id>.redis_name` is required and must refer to an existing Redis index.
5. The server must reconstruct the base schema from Redis metadata, preferably via existing RedisVL inspection primitives built on `FT.INFO`.
6. `indexes.<id>.vectorizer` remains fully manual and is never inferred from Redis index metadata in v1.
7. `indexes.<id>.schema_overrides` is optional and exists only to supplement incomplete inspection data.
8. Discovered index identity is authoritative:
   - `indexes.<id>.redis_name`
   - storage type
   - field identity (`name`, `type`, and `path` when applicable)
9. Overrides may:
   - add missing attrs for a discovered field
   - replace discovered attrs for a discovered field when needed for compatibility
10. Overrides must not:
   - redefine index identity
   - add entirely new fields that do not exist in the inspected index
   - change a discovered field's `name`, `type`, or `path`
11. Override conflicts must fail startup with a config error.
12. Tool requests implicitly target the sole configured index binding in v1. No `index` request parameter is exposed yet.

### Env Substitution Rules

Supported patterns in YAML values:
- `${VAR}`: required variable. Fail startup if unset.
- `${VAR:-default}`: optional variable with fallback.

Unresolved required vars must fail startup with config error.

### Config Validation Rules

Server startup must fail fast if:
1. Config file missing/unreadable.
2. YAML invalid.
3. `server.redis_url` missing or blank.
4. `indexes` missing, empty, or containing more than one entry.
5. The configured binding id is blank.
6. `indexes.<id>.redis_name` missing or blank.
7. The referenced Redis index does not exist.
8. Schema inspection fails and no valid `indexes.<id>.schema_overrides` resolve the issue.
9. `indexes.<id>.runtime.text_field_name` not in the effective schema.
10. `indexes.<id>.runtime.vector_field_name` not in the effective schema or not vector type.
11. `indexes.<id>.runtime.default_embed_text_field` not in the effective schema.
12. `default_limit <= 0` or `max_limit < default_limit`.
13. `max_upsert_records <= 0`.

---

## Lifecycle and Resource Management

### Startup Sequence (Normative)

On server startup:

1. Load settings and config.
2. Resolve the sole configured index binding from `indexes`.
3. Create or obtain an async Redis client using `server.redis_url`.
4. Validate Redis connectivity by performing a lightweight call (`info` or equivalent search operation).
5. Inspect the existing index named by `indexes.<id>.redis_name`, preferably via `AsyncSearchIndex.from_existing(...)` or an equivalent `FT.INFO`-backed flow.
6. Convert the inspected index metadata into an `IndexSchema`.
7. Apply any validated `indexes.<id>.schema_overrides` to produce the effective schema.
8. Instantiate `AsyncSearchIndex` from the effective schema.
9. Instantiate the configured `indexes.<id>.vectorizer`.
10. Validate vectorizer dimensions against the effective vector field dims when available.
11. Register tools (omit upsert in read-only mode).

If vector field attributes cannot be reconstructed from Redis metadata on the target Redis version, startup must fail with an actionable error unless `indexes.<id>.schema_overrides` provides the missing attrs.

### Shutdown Sequence

On shutdown, disconnect Redis client owned by `AsyncSearchIndex` and release vectorizer resources if applicable.

### Concurrency Guard

Tool executions are bounded by an async semaphore (`runtime.max_concurrency`). Requests exceeding capacity wait, then may timeout according to `request_timeout_seconds`.

---

## Filter Contract (Normative)

`search-records.filter` follows RedisVL convention and accepts either:
- `string`: raw RedisVL/Redis Search filter string (passed through to query filter).
- `object`: JSON DSL described below.

### Operators

- Logical: `and`, `or`, `not`
- Comparison: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `like`
- Utility: `exists`

### Atomic Expression Shape

```json
{ "field": "category", "op": "eq", "value": "science" }
```

### Composite Shape

```json
{
  "and": [
    { "field": "category", "op": "eq", "value": "science" },
    {
      "or": [
        { "field": "rating", "op": "gte", "value": 4.5 },
        { "field": "is_pinned", "op": "eq", "value": true }
      ]
    }
  ]
}
```

### Parsing Rules

1. Unknown `op` fails with `invalid_filter`.
2. Unknown `field` fails with `invalid_filter`.
3. Type mismatches fail with `invalid_filter`.
4. Empty logical arrays fail with `invalid_filter`.
5. Object DSL parser translates to `redisvl.query.filter.FilterExpression`.
6. String filter is treated as raw filter expression and passed through.

---

## Tools

## Tool: `search-records`

Search records using vector, full-text, or hybrid query.

### Request Contract

| Parameter | Type | Required | Default | Constraints |
|----------|------|----------|---------|-------------|
| `query` | str | yes | - | non-empty |
| `search_type` | enum | no | `vector` | `vector` \| `fulltext` \| `hybrid` |
| `limit` | int | no | `runtime.default_limit` | `1..runtime.max_limit` |
| `offset` | int | no | `0` | `>=0` |
| `filter` | string \\| object | no | `null` | Raw RedisVL filter string or DSL object |
| `return_fields` | list[str] | no | all non-vector fields | Unknown fields rejected |

### Response Contract

```json
{
  "search_type": "vector",
  "offset": 0,
  "limit": 10,
  "results": [
    {
      "id": "doc:123",
      "score": 0.93,
      "score_type": "vector_distance_normalized",
      "record": {
        "content": "The document text...",
        "category": "science"
      }
    }
  ]
}
```

### Search Semantics

- `vector`: embeds `query` with configured vectorizer, builds `VectorQuery`.
- `fulltext`: builds `TextQuery`.
- `hybrid`: embeds `query` and selects the query implementation by runtime capability:
  - use native `HybridQuery` when Redis `>=8.4.0` and redis-py `>=7.1.0` are available
  - otherwise fall back to `AggregateHybridQuery`
- The MCP request/response contract for `hybrid` is identical across both implementation paths.
- In v1, `filter` is applied uniformly to the hybrid query rather than allowing separate text-side and vector-side filters. This is intentional to keep the API simple; future versions may expose finer-grained hybrid filtering controls.

### Errors

| Code | Meaning | Retryable |
|------|---------|-----------|
| `invalid_request` | bad query params | no |
| `invalid_filter` | filter parse/type failure | no |
| `dependency_missing` | missing optional lib/provider SDK | no |
| `backend_unavailable` | Redis unavailable/timeout | yes |
| `internal_error` | unexpected failure | maybe |

---

## Tool: `upsert-records`

Upsert records with automatic embedding.

Not registered when read-only mode is enabled.

### Request Contract

| Parameter | Type | Required | Default | Constraints |
|----------|------|----------|---------|-------------|
| `records` | list[object] | yes | - | non-empty and `len(records) <= runtime.max_upsert_records` |
| `id_field` | str | no | `null` | if set, must exist in every record |
| `embed_text_field` | str | no | `runtime.default_embed_text_field` | must exist in every record |
| `skip_embedding_if_present` | bool | no | `runtime.skip_embedding_if_present` | if false, always re-embed |

### Response Contract

```json
{
  "status": "success",
  "keys_upserted": 3,
  "keys": ["doc:abc123", "doc:def456", "doc:ghi789"]
}
```

### Upsert Semantics

1. Validate input records before writing.
2. Resolve `embed_text_field`.
3. Respect `skip_embedding_if_present` (default true): only generate embeddings for records missing configured vector field.
4. Populate configured vector field.
5. Call `AsyncSearchIndex.load`.

### Error Semantics

- Validation failures return `invalid_request`.
- Provider errors return `dependency_missing` or `internal_error` with actionable message.
- Redis write failures return `backend_unavailable`.
- On write failure, response must include `partial_write_possible: true` (conservative signal).

---

## Server Implementation

### Core Class Contract

```python
class RedisVLMCPServer(FastMCP):
    settings: MCPSettings
    config: MCPConfig

    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...

    async def get_index(self) -> AsyncSearchIndex: ...
    async def get_vectorizer(self): ...
```

Tool implementations must always call `await server.get_index()` and `await server.get_vectorizer()`; never read uninitialized attributes directly.

### Field Mapping Requirements

For the sole configured binding in v1, the server owns these validated values:
- `text_field_name`
- `vector_field_name`
- `default_embed_text_field`

Schema discovery is automatic in v1. Field mapping is not. Runtime field mappings remain explicit so the server does not guess among multiple valid text or vector fields.

---

## CLI Integration

Current RedisVL CLI is command-dispatch based (not argparse subparsers), so MCP integration must follow existing pattern.

### User Commands

```bash
rvl mcp --config path/to/mcp_config.yaml
rvl mcp --config path/to/mcp_config.yaml --read-only
```

### Required CLI Changes

1. Add `mcp` command to usage/help in `redisvl/cli/main.py`.
2. Add `RedisVlCLI.mcp()` method that dispatches to new `MCP` handler class.
3. Implement `redisvl/cli/mcp.py` similar to existing command modules.
4. Gracefully report missing optional deps (`pip install redisvl[mcp]`).
5. Clearly report when the current Python runtime is unsupported for the MCP extra.

---

## Client Configuration Examples

### Claude Desktop

```json
{
  "mcpServers": {
    "redisvl": {
      "command": "uvx",
      "args": ["--from", "redisvl[mcp]", "rvl", "mcp", "--config", "/path/to/mcp_config.yaml"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Claude Agents SDK (Python)

```python
from agents import Agent
from agents.mcp import MCPServerStdio

async def main():
    async with MCPServerStdio(
        command="uvx",
        args=["--from", "redisvl[mcp]", "rvl", "mcp", "--config", "mcp_config.yaml"],
    ) as server:
        agent = Agent(
            name="search-agent",
            instructions="Search and maintain Redis-backed knowledge.",
            mcp_servers=[server],
        )
```

### Google ADK (Python)

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="redis_search_agent",
    instruction="Search and maintain Redis-backed knowledge using vector search.",
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uvx",
                    args=["--from", "redisvl[mcp]", "rvl", "mcp", "--config", "/path/to/mcp_config.yaml"],
                    env={
                        "OPENAI_API_KEY": "sk-..."  # Or other vectorizer API key
                    }
                ),
            ),
            # Optional: filter to specific tools
            # tool_filter=["search-records"]
        )
    ],
)
```

### n8n

n8n supports MCP servers via the MCP Server Trigger node. Configure the RedisVL MCP server as an external MCP tool source:

1. **Using SSE transport** (if supported in future versions):
   ```json
   {
     "mcpServers": {
       "redisvl": {
         "url": "http://localhost:9000/sse"
       }
     }
   }
   ```

2. **Using stdio transport** (via n8n's Execute Command node as a workaround):
   Configure a workflow that spawns the MCP server process:
   ```bash
   uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp_config.yaml
   ```

Note: Full n8n MCP client support depends on n8n's MCP implementation. Refer to [n8n MCP documentation](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-langchain.mcptrigger/) for current capabilities.

---

## Observability and Security

### Logging

- Use structured logs with operation name, latency, and error code.
- Never log secrets (API keys, auth headers, full DSNs with credentials).
- Log config path but not raw config values for sensitive keys.

### Timeouts

- Startup timeout: `runtime.startup_timeout_seconds`
- Tool request timeout: `runtime.request_timeout_seconds`

### Secret Handling

- Support env-injected secrets via `${VAR}` substitution.
- Fail fast for required missing vars.

---

## Testing Strategy

### Unit Tests (`tests/unit/test_mcp/`)

- `test_settings.py`
  - env parsing and overrides
  - read-only behavior
- `test_config.py`
  - YAML validation
  - env substitution success/failure
  - schema inspection merge and override validation
  - field mapping validation
- `test_filters.py`
  - DSL parsing, invalid operators, type mismatches
- `test_errors.py`
  - internal exception -> MCP error code mapping

### Integration Tests (`tests/integration/test_mcp/`)

- `test_server_startup.py`
  - startup success path against the sole configured index binding
  - missing index failure
  - vector field inspection gap resolved by `indexes.<id>.schema_overrides`
  - conflicting override failure
- `test_search_tool.py`
  - vector/fulltext/hybrid success paths
  - native hybrid path on Redis `>=8.4.0`
  - aggregate hybrid fallback path on older supported runtimes
  - pagination and field projection
  - filter behavior
- `test_upsert_tool.py`
  - insert/update success
  - id_field/embed_text_field validation failures
  - read-only mode excludes tool

### Deterministic Verification Commands

```bash
uv run python -m pytest tests/unit/test_mcp -q
uv run python -m pytest tests/integration/test_mcp -q
```

---

## Implementation Plan and DoD

### Phase 1: Framework

Deliverables:
1. `redisvl/mcp/` scaffolding.
2. Config/settings models with strict validation.
3. Inspection-first startup/shutdown lifecycle.
4. Error mapping helpers.

DoD:
1. Server boots successfully with valid config against the sole configured index binding.
2. Server fails fast with actionable config errors.
3. Unit tests for config/settings pass.

### Phase 2: Search Tool

Deliverables:
1. `search-records` request/response contract.
2. Filter parser (JSON DSL + raw string pass-through).
3. Hybrid query selection between native and aggregate implementations.

DoD:
1. All search modes tested.
2. Invalid filter returns `invalid_filter`.
3. `hybrid` uses native execution when available and `AggregateHybridQuery` otherwise, without changing the MCP contract.

### Phase 3: Upsert Tool

Deliverables:
1. `upsert-records` implementation.
2. Record pre-validation.
3. Read-only exclusion.

DoD:
1. Upsert works end-to-end.
2. Invalid records fail before writes.
3. Read-only mode verified.

### Phase 4: CLI and Packaging

Deliverables:
1. `rvl mcp` command via current CLI pattern.
2. Optional dependency group updates.
3. User-facing error messages for missing extras and unsupported Python runtime.

DoD:
1. `uvx --from redisvl[mcp] rvl mcp --config ...` runs successfully.
2. CLI help includes `mcp` command.

### Phase 5: Documentation

Deliverables:
1. Config reference and examples.
2. Client setup examples.
3. Companion production example document.
4. Troubleshooting guide with common errors and fixes.

DoD:
1. Docs reflect normative contracts in this spec.
2. Companion example is aligned with the config and lifecycle contract.

---

## Risks and Mitigations

1. Runtime mismatch for hybrid search.
   - Native hybrid requires newer Redis and redis-py capabilities, while older supported environments may still need the aggregate fallback path.
   - Mitigation: explicitly detect runtime capability and select native `HybridQuery` or `AggregateHybridQuery` deterministically.
2. Dependency drift across provider vectorizers.
   - Mitigation: dependency matrix and startup validation.
3. Ambiguous filter behavior causing agent retries.
   - Mitigation: explicit raw-string pass-through semantics and deterministic DSL parser errors.
4. Hidden partial writes during failures.
   - Mitigation: conservative `partial_write_possible` signaling.
5. Incomplete schema reconstruction on older Redis versions.
   - `FT.INFO` may not return enough vector metadata on some older Redis versions to fully reconstruct vector field attrs.
   - Mitigation: fail fast with an actionable error and support targeted `indexes.<id>.schema_overrides` for missing attrs.
6. Security and deployment limitations (v1 scope).
   - This implementation is stdio-first and not production-hardened by itself. It does not include:
     - Authentication/authorization mechanisms.
     - Remote transports (SSE/HTTP) that would enable multi-tenant or networked deployments.
     - Rate limiting or request validation beyond basic input constraints.
   - Mitigation: Document clearly that v1 can be used against Redis Enterprise, Redis Cloud, or OSS Redis deployments, but production use requires the operator to supply the surrounding controls for auth, process isolation, and network boundaries. Users wanting built-in remote transport and auth should wait for future RedisVL MCP versions.
   - For production deployments requiring authentication, users can:
     - Deploy behind an authenticating proxy.
     - Use environment-based secrets for Redis and vectorizer credentials.
     - Restrict network access to the MCP server process.
