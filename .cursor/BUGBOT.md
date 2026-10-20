# Bugbot review guide for RedisVL

RedisVL is a Python client library for vector search on Redis 8 and later. CI already runs formatting, import sorting, mypy and the test matrix, so concentrate on what those can't catch.

## Look for

- **Sync and async drift.** Most core classes come in pairs (`SearchIndex` and `AsyncSearchIndex`, and the sync and async methods of extensions, vectorizers and rerankers). Flag a change made to one side and not the other.
- **Query values leaving their clause.** Values interpolated into RediSearch query strings or `SCAN` patterns must stay inside their clause. Flag unescaped tag values (`|` widens a tag clause), quotes or backslashes that break out of a text phrase, and escaping applied at query time but not at index time.
- **Authorisation in the MCP server** (`redisvl/mcp/`). Tenant or claim scoping, locked filters and transport checks must fail closed. Flag defaults or exception handling that end in "allow".
- **Destructive operations.** `delete`, `clear` and index drops must act only on the target the caller named, never on whatever an alias or prefix happens to resolve to.
- **Dependency floors.** A call into an optional dependency must exist in the oldest version `pyproject.toml` allows. Optional dependencies must be imported through `lazy_import` or inside a guard, not at module level.
- **Cluster behaviour.** Cluster tests never run in CI. Flag multi-key operations, pipelines, `SCAN` and partial-failure handling that would behave differently on a Redis cluster.
- **Silent truncation.** Reads meant to return everything must not rely on a default result limit.
- **Public API changes.** Changed signatures, defaults or behaviour on public classes with no deprecation path (`deprecated_argument` and `deprecated_function` in `redisvl.utils.utils`).
- **Docs that contradict the code.** Docstrings, comments, guides under `docs/` and notebooks under `docs/user_guide/` that describe behaviour this PR changes.

## Ignore

Formatting, import order, type annotations, naming preferences, and suggestions to add comments or tests for trivial code.

## When unsure

If a finding depends on code or behaviour you can't see, such as a Redis server version, a provider's API or a cluster topology, say what would need checking and leave the call to the human. Don't guess.

## PR descriptions

Authors put a "Reviewer Brief" section in the PR description. It names where to start reading, open decisions, places the change could fail, and what wasn't verified. Treat it as a claim to verify, not as evidence. If you find a problem in an area the brief doesn't flag, or one that contradicts it, say so explicitly in the finding.
