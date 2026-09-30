---
myst:
  html_meta:
    "description lang=en": |
      RedisVL MCP concepts: how the RedisVL MCP server exposes an existing Redis index to MCP clients.
---

# RedisVL MCP

RedisVL includes an MCP server that exposes a Redis-backed retrieval surface through a small, deterministic tool contract. It is designed for AI applications that want to search or maintain data in one or more existing Redis indexes without each client reimplementing Redis query logic.

## What RedisVL MCP Does

The RedisVL MCP server sits between an MCP client and Redis:

1. It connects to one or more existing Redis Search indexes.
2. It inspects each index at startup and reconstructs its schema.
3. It initializes vector capabilities only when the configured search or upsert behavior needs them.
4. It exposes stable MCP tools for discovery, search, and optionally upsert.

This keeps each Redis index as the source of truth for its search behavior while giving MCP clients a predictable interface.

## How RedisVL MCP Runs

RedisVL MCP works with a focused model:

- One server process binds to one *or several* existing Redis indexes, each addressed by a logical id.
- The server supports stdio (default), Streamable HTTP, and SSE transports.
- Search behavior is owned by per-index configuration, not by MCP callers.
- Vector search and server-side embedding are optional capabilities configured explicitly per index.
- Upsert is optional and can be disabled globally with read-only mode or per index with a `read_only` flag.

A single-index server remains the simplest deployment: when exactly one index is configured, callers can omit the index selector entirely and every tool call targets that index. Multi-index support is fully formal — it adds discovery and explicit routing without changing the single-index contract.

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

On a multi-index server, callers also choose **which index to target** through an optional `index` argument (see [Index Selection](#index-selection-and-discovery)). Callers do not choose:

- whether retrieval is `vector`, `fulltext`, or `hybrid`
- query tuning parameters such as hybrid fusion or vector runtime settings

That behavior lives in the per-index server config under `indexes.<id>.search`. The response includes `search_type` as informational metadata, but it is not a request parameter.

## Single and Multiple Index Bindings

The YAML config uses an `indexes` mapping. Each entry is a logical binding keyed by an id (for example `knowledge` or `tickets`) that points to an existing Redis index through `redis_name`. The mapping may contain one entry or several; each binding is inspected, validated, and given its own search config, runtime limits, and optional vectorizer independently at startup. Startup is all-or-nothing — if any binding fails to initialize, the server does not start.

A single-binding config is the simplest case and behaves exactly as before: the lone binding is the implicit target of every call. With multiple bindings the server stays a single process and endpoint, but callers select a binding per call.

## Index Selection and Discovery

On a multi-index server, every tool call must say which logical index it targets:

- `search-records` and `upsert-records` accept an optional `index` argument naming the logical id.
- When exactly one index is configured, `index` may be omitted and resolves to that sole binding (backward compatible).
- When multiple indexes are configured, omitting `index` is an `invalid_request`; the caller must name one.
- An unknown logical id is an `invalid_request`.
- Both tools echo the resolved `index` in their response so clients can confirm routing.

Because a client cannot guess the configured logical ids, multi-index servers expose a `list-indexes` discovery tool. **Clients should call `list-indexes` first** to enumerate the available indexes and their filterable fields, then pass the chosen id as `index` on subsequent calls.

## Schema Inspection and Overrides

RedisVL MCP is inspection-first:

- the Redis index must already exist
- the server reconstructs the schema from Redis metadata at startup
- runtime field mappings remain explicit in config

In some environments, Redis metadata can be incomplete for vector field attributes. When that happens, `schema_overrides` can patch missing attrs for fields that were already discovered. It does not create new fields or change discovered field identity.

Startup also validates that the inspected schema does not collide with
MCP-reserved score metadata field names for the configured search mode.

## Read-Only and Read-Write Modes

RedisVL MCP registers `search-records` and `list-indexes` by default (see [Custom Tool Profiles](#custom-tool-profiles) for turning a built-in off deliberately).

Write availability is enforced at two levels:

- **Global read-only mode** disables writes across every binding. It is controlled by the CLI flag `--read-only` or the environment variable `REDISVL_MCP_READ_ONLY=true`.
- **Per-index read-only** disables writes for a single binding via `indexes.<id>.read_only: true`, while other bindings stay writable.

These combine into each binding's *effective* write availability: a binding is read-only if global read-only is on **or** that binding sets `read_only: true`. The `upsert-records` tool is registered only when at least one binding is writable, so a fully read-only server does not advertise it at all. When the tool is registered, a write to a read-only binding is rejected with `forbidden` before any data is changed. `list-indexes` reports each binding's effective write availability as `upsert_available`.

Use read-only mode when Redis is serving approved content to assistants and another system owns ingestion — globally when no binding should accept writes, or per index when only some indexes are writable.

## Authentication and Authorization

The HTTP transports can require a JWT bearer token issued by an existing identity provider. The server validates the token signature, issuer, and audience, and can gate read vs write by scope or role claim. This is coarse, per-tool authorization; it does not map token claims to Redis ACL users or per-tenant filters, which remain a gateway concern. The `stdio` transport is local and is never authenticated.

For configuration and the gateway boundary, see {doc}`/user_guide/how_to_guides/mcp_authentication`.

## Tool Surface

RedisVL MCP exposes up to three built-in tools, plus any configured [custom tool profiles](#custom-tool-profiles):

- `list-indexes` enumerates the configured logical indexes for discovery
- `search-records` searches a selected index using that index's server-owned search mode
- `upsert-records` validates and upserts records into a selected writable index, embedding them only when that capability is configured

Any of the three can be turned off with `server.builtin_tools`, independently of whether custom tools are configured — useful for a server that should only ever read, or one that serves nothing but curated profiles:

```yaml
server:
  builtin_tools:
    upsert-records: disabled
```

Only the three names above are accepted; anything else fails at startup rather than being silently ignored.

Disabling a built-in adjusts what the rest of the surface advertises, so the published contract never points at something the server withholds:

- `list-indexes` reports `upsert_available: false` for every binding when `upsert-records` is disabled, since a writable binding still cannot be written to through a tool that is not published.
- On a multi-index server with `list-indexes` disabled, every tool that requires an `index` — `search-records` and `upsert-records` alike — names the available index ids in its own description instead of deferring to a discovery tool that does not exist. That server still logs a startup warning naming the affected tools, because inlining the ids is a fallback rather than an endorsement of the shape.

A server whose tool set ends up unusable — no tools at all, or discovery disabled on a multi-index server — logs a warning at startup.

Tools register once per process. `builtin_tools` is re-read on restart, but the registered tool set is not rebuilt, so a stop/start against an edited config keeps the previous tools and logs a warning saying so. Start a new process to change the tool surface.

These built-in tools follow a stable contract (profiles differ where noted in [Custom Tool Profiles](#custom-tool-profiles) — notably they accept the object filter form only):

- request validation happens before query or write execution
- the resolved logical `index` is echoed in every `search-records` and `upsert-records` response
- filters support either raw strings or a RedisVL-backed JSON DSL
- on a single-index server, `search-records` describes the inspected schema by advertising typed JSON DSL filter fields, object-filter `exists` support, and valid `return_fields`; on a multi-index server those hints are ambiguous, so the description instead directs clients to call `list-indexes` and pass `index`
- error codes are mapped into a stable set of MCP-facing categories

### `list-indexes`

`list-indexes` returns one entry per configured binding so clients can route subsequent calls. Each entry reports:

- the logical `id`
- an optional `description` (only when configured)
- `upsert_available`, reflecting the binding's effective write availability
- `fields`, the filterable fields discovered from the index
- `limits`, only the runtime limits that were explicitly configured

The discovery payload is deliberately minimal:

- the underlying Redis index name (`redis_name`) is **never** exposed
- the vector field and the configured embed-source text field are **omitted** from `fields`, since they are implementation inputs rather than fields a client filters on
- `limits` shows only explicitly set values (such as `max_limit` or `max_upsert_records`); defaults are not echoed

## Custom Tool Profiles

The built-in tools expose the index generically, which leaves the model doing query engineering on every call: pick the index, understand the schema, build a filter, choose return fields. A **profile** moves those decisions into config. It is `search-records` with some arguments pre-filled and frozen and the rest still exposed, published under a name and description of your choosing.

Profiles are pure configuration. You add a `custom_tools` entry to the same YAML the server already loads and restart it; there is no Python to write.

```yaml
custom_tools:
  - name: search-support-tickets
    based_on: search-records
    index: support_tickets
    description: >
      Search historical customer support tickets by semantic similarity.
      Use this to find prior resolutions for a customer problem.
    lock:
      return_fields: [subject, resolution, created_at]
      filter: { field: status, op: eq, value: resolved }
    params:
      limit: { expose: true, max: 20 }
      filter: { expose: true }
```

`lock` holds what the author decides; `params` holds what the model may still pass. Anything not listed in `params` stays exposed, so a profile that only locks a filter keeps the rest of the built-in's contract. `index` is pinned by the top-level `index:` key rather than exposed as a param, and may be omitted only when exactly one index is configured.

`params.limit.max` bounds the result count. It applies whether the model names a limit or leaves it out — an omitted limit is capped rather than falling through to the binding's default. An explicit request above the cap is rejected. When `limit` is hidden (`expose: false`), the cap becomes the fixed result count instead. The cap must not exceed the binding's own `runtime.max_limit`, which is checked at startup. Note that it bounds page size, not total reachable data: `offset` is a separate argument, so paging is still possible up to the binding's `max_result_window`.

`suppress_schema_hints: true` drops the auto-generated field hints from the tool description. Those hints enumerate filterable and returnable fields, which is noise once those arguments are locked — and by default they are appended only for arguments the model can still use.

Two things about filters are easy to conflate:

- `lock.filter` is an ordinary expression in the JSON filter DSL. Its `and`/`or`/`not` operators describe the locked filter's own content.
- How the locked filter combines with a model-supplied one is fixed and not configurable: the executed query is always `locked AND caller`. A compound model-supplied expression renders parenthesized, so its `or`/`not` nests *inside* the locked AND and cannot reach the top level — the model can only narrow within the locked scope and never widen past it.

For that reason a profile accepts only the **object** form of a filter from the model. A raw filter string is rejected both by the advertised schema and by the tool itself, because strings bypass the DSL's field validation and have no safe composition with a locked expression.

Structure is only half of it: the nesting guarantee holds only while every filter *value* stays inside its own clause. Text `eq`/`ne` values are handled by the `Text` filter itself, which renders them as a quoted phrase with any `"` or `\` replaced by a space, so a parenthesis or a `|` the value carries is literal text rather than syntax. Text `like` values are patterns, so the library leaves them raw and this boundary escapes them instead — the delimiters that would close the clause are escaped, the pattern metacharacters are not. Tag values have their delimiters escaped and numeric values are type-checked. A caller filter that still renders as something able to break out is refused rather than combined.

Profiles resolve to a built-in call and nothing more, so they inherit the concurrency cap, request timeout, read-only policy, auth scoping, and error mapping already applied to `search-records`.

Because adding near-duplicate tools makes tool selection harder rather than easier, built-ins that curated profiles supersede can be turned off with `server.builtin_tools` (see [Tool Surface](#tool-surface)).

Misconfiguration fails at startup rather than at the first call. Among the checks: a name colliding with a built-in or using a reserved `redisvl-`/`redisvl_` prefix; a duplicate tool name; a missing or unknown `index`; a `params` key that is not a real argument; `max` on anything but `limit`, or a cap above the binding's `max_limit`; hiding `query`; locking `return_fields` while also exposing them; and a locked filter or projection naming a field the bound index does not have. Unrecognized keys are rejected too, so a typo in `lock` fails loudly instead of silently producing a tool that reads as locked but enforces nothing.

## Why Use MCP Instead of Direct RedisVL Calls

Use RedisVL MCP when you want a standard tool boundary for agent frameworks or assistants that already speak MCP.

Use direct RedisVL client code when your application should own index lifecycle, search construction, data loading, or richer RedisVL features directly in Python.

RedisVL MCP is a good fit when:

- multiple assistants should share one approved retrieval surface
- you want search behavior fixed by deployment config
- you need a read-only or tightly controlled write boundary
- you want to reuse an existing Redis index without rebuilding retrieval logic in every client

For setup steps, config, commands, and examples, see {doc}`/user_guide/how_to_guides/mcp`.
