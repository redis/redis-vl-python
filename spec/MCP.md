# RedisVL MCP Server Specification

## Document Status

- Status: Draft for implementation
- Audience: RedisVL maintainers and coding agents implementing MCP support
- Primary objective: Define a deterministic, testable MCP server contract so agents can implement safely without relying on implicit behavior

---

## Overview

This specification defines a Model Context Protocol (MCP) server for RedisVL that allows MCP clients to search and upsert data in a Redis index.

The server is designed for stdio transport first and must be runnable via:

```bash
uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp_config.yaml
```

### Goals

1. Expose RedisVL search capabilities (`vector`, `fulltext`, `hybrid`) through stable MCP tools.
2. Support controlled write access via an upsert tool.
3. Provide deterministic contracts for tool inputs, outputs, and errors.
4. Align implementation with existing RedisVL architecture and CLI patterns.

### Non-Goals (v1)

1. Multi-index routing in a single server process.
2. Remote transports (SSE/HTTP).
3. Delete/count/info tools (future scope).

---

## Compatibility Matrix

These are hard compatibility expectations for v1.

| Component | Requirement | Notes |
|----------|-------------|-------|
| Python | `>=3.9.2,<3.15` | Match project constraints |
| RedisVL | current repo version | Server lives inside this package |
| redis-py | `>=5.0,<7.2` | Already required by project |
| MCP SDK | `mcp>=1.9.0` | Provides FastMCP |
| Redis server | Redis Stack / Redis with Search module | Required for all search modes |
| Hybrid search | Redis `>=8.4.0` and redis-py `>=7.1.0` runtime capability | If unavailable, `hybrid` returns structured error |

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
│   ├── filters.py            # Filter DSL -> FilterExpression parser
│   └── tools/
│       ├── __init__.py
│       ├── search.py         # redisvl-search
│       └── upsert.py         # redisvl-upsert
└── cli/
    ├── main.py               # Add `mcp` command dispatch
    └── mcp.py                # MCP command handler class
```

### Dependency Groups

Add optional extras for explicit install intent.

```toml
[project.optional-dependencies]
mcp = [
  "mcp>=1.9.0",
  "pydantic-settings>=2.0",
]
```

Notes:
- `fulltext`/`hybrid` use `TextQuery`/`HybridQuery`, which rely on NLTK stopwords when defaults are used. If `nltk` is not installed and stopwords are enabled, server must return a structured dependency error.
- Provider vectorizer dependencies remain provider-specific (`openai`, `cohere`, `vertexai`, etc.).

---

## Configuration

Configuration is composed from environment + YAML:

1. `MCPSettings` from env/CLI.
2. YAML file referenced by `config` setting.
3. Env substitution inside YAML with strict validation.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDISVL_MCP_CONFIG` | str | required | Path to MCP YAML config |
| `REDISVL_MCP_READ_ONLY` | bool | `false` | If true, do not register upsert tool |
| `REDISVL_MCP_TOOL_SEARCH_DESCRIPTION` | str | default text | MCP tool description override |
| `REDISVL_MCP_TOOL_UPSERT_DESCRIPTION` | str | default text | MCP tool description override |

### YAML Schema (Normative)

```yaml
redis_url: redis://localhost:6379

index:
  name: my_index
  prefix: doc
  storage_type: hash

fields:
  - name: content
    type: text
  - name: category
    type: tag
  - name: embedding
    type: vector
    attrs:
      algorithm: hnsw
      dims: 1536
      distance_metric: cosine
      datatype: float32

vectorizer:
  class: OpenAITextVectorizer
  model: text-embedding-3-small
  # kwargs passed to vectorizer constructor
  # for providers using api_config, pass as nested object:
  # api_config:
  #   api_key: ${OPENAI_API_KEY}

runtime:
  # index lifecycle mode:
  # validate_only (default) | create_if_missing
  index_mode: validate_only

  # required explicit field mapping for tool behavior
  text_field_name: content
  vector_field_name: embedding
  default_embed_field: content

  # request constraints
  default_limit: 10
  max_limit: 100

  # timeouts
  startup_timeout_seconds: 30
  request_timeout_seconds: 60

  # server-side concurrency guard
  max_concurrency: 16
```

### Env Substitution Rules

Supported patterns in YAML values:
- `${VAR}`: required variable. Fail startup if unset.
- `${VAR:-default}`: optional variable with fallback.

Unresolved required vars must fail startup with config error.

### Config Validation Rules

Server startup must fail fast if:
1. Config file missing/unreadable.
2. YAML invalid.
3. `runtime.text_field_name` not in schema.
4. `runtime.vector_field_name` not in schema or not vector type.
5. `runtime.default_embed_field` not in schema.
6. `default_limit <= 0` or `max_limit < default_limit`.

---

## Lifecycle and Resource Management

### Startup Sequence (Normative)

On server startup:

1. Load settings and config.
2. Build `IndexSchema`.
3. Create `AsyncSearchIndex` with `redis_url`.
4. Validate Redis connectivity by performing a lightweight call (`info` or equivalent search operation).
5. Handle index lifecycle:
   - `validate_only`: verify index exists; fail if missing.
   - `create_if_missing`: create index when absent; do not overwrite existing index.
6. Instantiate vectorizer.
7. Validate vectorizer dimensions match configured vector field dims when available.
8. Register tools (omit upsert in read-only mode).

### Shutdown Sequence

On shutdown, disconnect Redis client owned by `AsyncSearchIndex` and release vectorizer resources if applicable.

### Concurrency Guard

Tool executions are bounded by an async semaphore (`runtime.max_concurrency`). Requests exceeding capacity wait, then may timeout according to `request_timeout_seconds`.

---

## Filter DSL (Normative)

`redisvl-search.filter` accepts JSON in this DSL.

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
5. Parser translates DSL to `redisvl.query.filter.FilterExpression`.

---

## Tools

## Tool: `redisvl-search`

Search records using vector, full-text, or hybrid query.

### Request Contract

| Parameter | Type | Required | Default | Constraints |
|----------|------|----------|---------|-------------|
| `query` | str | yes | - | non-empty |
| `search_type` | enum | no | `vector` | `vector` \| `fulltext` \| `hybrid` |
| `limit` | int | no | `runtime.default_limit` | `1..runtime.max_limit` |
| `offset` | int | no | `0` | `>=0` |
| `filter` | object | no | `null` | Must satisfy filter DSL |
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
- `hybrid`: embeds `query`, builds `HybridQuery`.
- `hybrid` must fail with structured capability error if runtime support is unavailable.

### Errors

| Code | Meaning | Retryable |
|------|---------|-----------|
| `invalid_request` | bad query params | no |
| `invalid_filter` | filter parse/type failure | no |
| `dependency_missing` | missing optional lib/provider SDK | no |
| `capability_unavailable` | hybrid unsupported in runtime | no |
| `backend_unavailable` | Redis unavailable/timeout | yes |
| `internal_error` | unexpected failure | maybe |

---

## Tool: `redisvl-upsert`

Upsert records with automatic embedding.

Not registered when read-only mode is enabled.

### Request Contract

| Parameter | Type | Required | Default | Constraints |
|----------|------|----------|---------|-------------|
| `records` | list[object] | yes | - | non-empty |
| `id_field` | str | no | `null` | if set, must exist in every record |
| `embed_field` | str | no | `runtime.default_embed_field` | must exist in every record |
| `skip_embedding_if_present` | bool | no | `true` | if false, always re-embed |

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
2. Resolve `embed_field`.
3. Generate embeddings for required records (`aembed_many`).
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

Server owns these validated values:
- `text_field_name`
- `vector_field_name`
- `default_embed_field`

No implicit auto-detection is allowed in v1.

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

## Unit Tests (`tests/unit/test_mcp/`)

- `test_settings.py`
  - env parsing and overrides
  - read-only behavior
- `test_config.py`
  - YAML validation
  - env substitution success/failure
  - field mapping validation
- `test_filters.py`
  - DSL parsing, invalid operators, type mismatches
- `test_errors.py`
  - internal exception -> MCP error code mapping

## Integration Tests (`tests/integration/test_mcp/`)

- `test_server_startup.py`
  - startup success path
  - missing index in `validate_only`
  - create in `create_if_missing`
- `test_search_tool.py`
  - vector/fulltext/hybrid success paths
  - hybrid capability failure path
  - pagination and field projection
  - filter behavior
- `test_upsert_tool.py`
  - insert/update success
  - id_field/embed_field validation failures
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
3. Startup/shutdown lifecycle.
4. Error mapping helpers.

DoD:
1. Server boots successfully with valid config.
2. Server fails fast with actionable config errors.
3. Unit tests for config/settings pass.

### Phase 2: Search Tool

Deliverables:
1. `redisvl-search` request/response contract.
2. Filter DSL parser.
3. Capability checks for hybrid support.

DoD:
1. All search modes tested.
2. Invalid filter returns `invalid_filter`.
3. Capability failures are deterministic and non-ambiguous.

### Phase 3: Upsert Tool

Deliverables:
1. `redisvl-upsert` implementation.
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
3. User-facing error messages for missing extras.

DoD:
1. `uvx --from redisvl[mcp] rvl mcp --config ...` runs successfully.
2. CLI help includes `mcp` command.

### Phase 5: Documentation

Deliverables:
1. Config reference and examples.
2. Client setup examples.
3. Troubleshooting guide with common errors and fixes.

DoD:
1. Docs reflect normative contracts in this spec.
2. Examples are executable and tested.

---

## Risks and Mitigations

1. Runtime mismatch for hybrid search.
   - Mitigation: explicit capability check + clear error code.
2. Dependency drift across provider vectorizers.
   - Mitigation: dependency matrix and startup validation.
3. Ambiguous filter behavior causing agent retries.
   - Mitigation: strict DSL and deterministic parser errors.
4. Hidden partial writes during failures.
   - Mitigation: conservative `partial_write_possible` signaling.

---

## Open Design Questions

1. Should `upsert` preserve user-provided vectors by default when the vector field already exists (`skip_embedding_if_present=true`), or always re-embed?
2. Do we want `index_mode=create_if_missing` as the default instead of `validate_only`?
3. Should v1 support string-based raw Redis filter expressions in addition to the JSON filter DSL, or keep JSON-only?
4. Is there a hard maximum payload size for `records` in one upsert request (count/bytes) for guardrails?

