# Redis behaviour lens

Check every assumption the change makes about how Redis behaves against the official documentation, not against memory. Version-specific behaviour is where models are most confidently wrong about Redis, and Redis 8 changed several defaults.

## The Redis Docs MCP

Redis hosts a public, read-only MCP server for its documentation at `https://redis.io/mcp`. It needs no API key or sign-up. Add it to your agent as a streamable-HTTP MCP server named `redis-docs`. Setup for each client is at `https://marketing.redis.io/docs-mcp/`. It has three tools:

- `search`: free-text search across the redis.io docs. It returns matching pages with snippets and stable ids. Start here.
- `fetch`: the full text of one page, given an id from `search`. Use it when a snippet isn't enough to settle the claim.
- `ask`: a grounded answer plus the source pages behind it. Keep it for questions that span several pages. It runs a model over the results, so it's slower than `search` and `fetch`.

Send it questions about Redis, not code from the diff, credentials or anything unreleased. The corpus is the redis.io documentation, which includes RedisVL's own docs. Check each page's version metadata against the Redis versions RedisVL supports before you rely on it.

If your agent can't use MCP, read the same pages on `https://redis.io/docs/latest/`. If you can reach neither, skip the lens and say so under "Not verified" in the Reviewer Brief.

## Run it when

- The diff changes how commands are built or called: `FT.CREATE`, `FT.SEARCH`, `FT.AGGREGATE`, `FT.CURSOR`, `FT.INFO` arguments, query syntax, schema field attributes, vector index parameters, or `SCAN`, pipeline, expiry or eviction behaviour.
- The diff relies on a server default or limit, for example result limits, dialects, TTL semantics or timeout behaviour.
- The diff handles replies whose shape depends on RESP2 or RESP3, on storage type (hash or JSON), or on cluster mode.
- A docstring, comment, guide or notebook in the diff states how Redis behaves.

## Look for

- **Syntax the server doesn't accept,** or accepts with a different meaning. Examples are query operators, escaping rules, field-type options, and argument order in `FT.*` commands.
- **Defaults assumed wrongly:** a default dialect, result limit or distance metric, or an option the code assumes is on by default.
- **Version gaps:** behaviour that exists only in some of the Redis versions RedisVL supports, with no guard or note.
- **Reply-shape assumptions** that hold for one protocol, storage type or topology and break for another.
- **Docs claims that contradict the documentation.**

## How to report

Each finding cites the page it rests on: `... Per: <redis.io URL>`. A claim you couldn't confirm in the docs is not a finding. If it matters, list it under "Not verified" in the Reviewer Brief, so the reviewer knows it was raised but not settled. When a point in the brief rests on a docs page, link the page in that line.
