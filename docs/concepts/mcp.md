---
myst:
  html_meta:
    "description lang=en": |
      RedisVL MCP concepts: how the RedisVL MCP server exposes an existing Redis index to MCP clients.
---

# RedisVL MCP

RedisVL includes an MCP server that exposes a Redis-backed retrieval surface through a small, deterministic tool contract. It is designed for AI applications that want to search or maintain data in an existing Redis index without each client reimplementing Redis query logic.

## What RedisVL MCP Does

The RedisVL MCP server sits between an MCP client and Redis:

1. It connects to an existing Redis Search index.
2. It inspects that index at startup and reconstructs its schema.
3. It instantiates the configured vectorizer for query embedding and optional upsert embedding.
4. It exposes stable MCP tools for search, and optionally upsert.

This keeps the Redis index as the source of truth for search behavior while giving MCP clients a predictable interface.

## How RedisVL MCP Runs

RedisVL MCP works with a focused model:

- One server process binds to exactly one existing Redis index.
- The server supports stdio (default), Streamable HTTP, and SSE transports.
- Search behavior is owned by configuration, not by MCP callers.
- The vectorizer is configured explicitly.
- Upsert is optional and can be disabled with read-only mode.

## Config-Owned Search Behavior

MCP callers can control:

- `query`
- `limit`
- `offset`
- `filter`
- `return_fields`

These request-time controls are still bounded by runtime config. In particular,
deep paging is limited by a configured maximum result window, enforced as
`offset + limit`.

MCP callers do not choose:

- which index to target
- whether retrieval is `vector`, `fulltext`, or `hybrid`
- query tuning parameters such as hybrid fusion or vector runtime settings

That behavior lives in the server config under `indexes.<id>.search`. The response includes `search_type` as informational metadata, but it is not a request parameter.

## Single Index Binding

The YAML config uses an `indexes` mapping with one configured entry. That binding points to an existing Redis index through `redis_name`, and every tool call targets that configured index.

## Schema Inspection and Overrides

RedisVL MCP is inspection-first:

- the Redis index must already exist
- the server reconstructs the schema from Redis metadata at startup
- runtime field mappings remain explicit in config

In some environments, Redis metadata can be incomplete for vector field attributes. When that happens, `schema_overrides` can patch missing attrs for fields that were already discovered. It does not create new fields or change discovered field identity.

Startup also validates that the inspected schema does not collide with
MCP-reserved score metadata field names for the configured search mode.

## Read-Only and Read-Write Modes

RedisVL MCP always registers `search-records`.

`upsert-records` is only registered when the server is not in read-only mode. Read-only mode is controlled by:

- the CLI flag `--read-only`
- or the environment variable `REDISVL_MCP_READ_ONLY=true`

Use read-only mode when Redis is serving approved content to assistants and another system owns ingestion.

## Tool Surface

RedisVL MCP exposes two tools:

- `search-records` searches the configured index using the server-owned search mode
- `upsert-records` validates and upserts records, embedding them when needed

These tools follow a stable contract:

- request validation happens before query or write execution
- filters support either raw strings or a RedisVL-backed JSON DSL
- `search-records` describes the inspected schema by advertising JSON DSL filter fields and valid `return_fields`
- error codes are mapped into a stable set of MCP-facing categories

## Why Use MCP Instead of Direct RedisVL Calls

Use RedisVL MCP when you want a standard tool boundary for agent frameworks or assistants that already speak MCP.

Use direct RedisVL client code when your application should own index lifecycle, search construction, data loading, or richer RedisVL features directly in Python.

RedisVL MCP is a good fit when:

- multiple assistants should share one approved retrieval surface
- you want search behavior fixed by deployment config
- you need a read-only or tightly controlled write boundary
- you want to reuse an existing Redis index without rebuilding retrieval logic in every client

For setup steps, config, commands, and examples, see {doc}`/user_guide/how_to_guides/mcp`.
