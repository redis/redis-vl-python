# Redis behaviour lens

Check every assumption the change makes about how Redis behaves against the official documentation, not against memory. Version-specific behaviour is where models are most confidently wrong about Redis, and Redis 8 changed several defaults.

## The Redis Docs MCP

Redis hosts a public, read-only MCP server for its documentation at `https://redis.io/mcp`. It needs no API key or sign-up. Add it to your agent as a streamable-HTTP MCP server named `redis-docs`. Setup for each client is at `https://marketing.redis.io/docs-mcp/`. It has three tools:

- `search`: free-text search across the redis.io docs. It returns matching pages with snippets and stable ids. Start here.
- `fetch`: the full text of one page, given an id from `search`. Use it when a snippet isn't enough to settle the claim.
- `ask`: a grounded answer plus the source pages behind it. Keep it for questions that span several pages. It runs a model over the results, so it's slower than `search` and `fetch`.

Send it questions about Redis, not code from the diff, credentials or anything unreleased. The corpus is the redis.io documentation, which includes RedisVL's own docs. Check each page's version metadata against the Redis versions RedisVL supports before you rely on it.

If your agent can't use MCP, read the same pages on `https://redis.io/docs/latest/`. If you can reach neither, skip the lens and say so under "Not verified" in the Reviewer Brief.

## Measure what the change depends on

The documentation says what Redis accepts. It rarely says how a value is tokenised, trimmed or ordered in a reply, and those are the claims reviewers overturn most often by running the server. If the change, its tests, its comments or its Release Notes rest on one of the following, run it against a real server before you rely on it (`make redis-start`, then `redis-cli` or a ten-line script), and keep the command and the reply:

- **What gets indexed.** How a stored value is split, trimmed or normalised: separators in text fields (`don't` is indexed as `don` and `t`), leading and trailing whitespace in tag values, coordinate limits on GEO fields. An escaped query term only matches if the indexed value kept the same characters.
- **Hash and JSON separately.** They differ in exactly these details; for example, hash indexing trims whitespace from tag values and JSON indexing doesn't. A result measured on one storage type says nothing about the other.
- **Reply shapes.** The same command on RESP2 and RESP3, and with `decode_responses` on and off: key types, value types (`'1'` against `1.0`), and what the leading element of a reply is.
- **Error order and wording.** Which check the server runs first when two would fail, and whether an error arrives as a typed exception or a bare `ResponseError`. Don't build a test on an order the documentation doesn't promise.
- **Whether a query changes results.** A clause that parses is not a clause that matches. If the change adds a weight, an escape or an operator, show a document it now matches, or a score it changes.

State the Redis version next to each measurement. Use a Redis 8 release; RedisVL doesn't target Redis Stack. A claim you neither measured nor found in the documentation goes under "Not verified", and comes out of comments, docstrings and Release Notes.

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

Each finding cites what it rests on: the page (`... Per: <redis.io URL>`), or the measurement (`... Measured on Redis 8.4, JSON: <command> -> <reply>`). A claim you could confirm neither way is not a finding. If it matters, list it under "Not verified" in the Reviewer Brief, so the reviewer knows it was raised but not settled. When a point in the brief rests on a docs page, link the page in that line.
