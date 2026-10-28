# Correctness lens

Find changes that make the code behave wrongly for some input, state or environment the project supports. This lens runs on every PR.

## Look for

- **Line-level mistakes:** wrong or inverted conditions, off-by-one errors, a missing `await`, swallowed or over-broad exceptions, mutable default arguments, late-binding closures in loops.
- **Removed behaviour:** for every deleted or rewritten line, name the invariant it enforced, such as a validation, an escape, a guard or a cleanup. Then find where the new code re-establishes it. If you can't find that place, report it.
- **Broken contracts:** callers that rely on a changed return shape, a new exception, a new precondition or a changed default. Search for every caller, including tests, docs and notebooks.
- **The sync or async twin:** a fix to `SearchIndex` that `AsyncSearchIndex` lacks, or the other way round. The same applies to every sync/async pair in `redisvl/extensions/` and to the async vectorizer and reranker methods.
- **Wrappers and proxies:** a wrapper that drops arguments, ignores a return value, or forgets to close or release what it opened.
- **Setup and teardown asymmetry:** something opened, locked or registered on one path and not released on another, especially on error paths.

## RedisVL specifics

- **Dependency floors.** Check every call into an optional dependency against the oldest version `pyproject.toml` allows. A new attribute or keyword that only exists in a later release crashes users on the floor version. This is the most common high-severity finding in this repository.
- **Redis replies.** The code has to handle RESP2 and RESP3 reply shapes, and both hash and JSON storage.
- **Result limits.** Default result counts (`num_results`, `LIMIT`, `MAXSEARCHRESULTS`) silently truncate. Check any read that is meant to return everything.
- **Cluster mode.** Multi-key operations, `SCAN` and pipelines behave differently on a cluster, and partial failures are possible. CI never runs cluster tests, so reason about these cases from the code. If you can't settle one that way, put it under "Not verified" in the Reviewer Brief.
- **Client ownership.** Who creates, owns and closes a Redis client, and what happens on reconnect, especially for async clients reused across event loops.

## Depth

For diffs under 100 lines, and for chore, CI or refactor changes, do a single pass over the diff and its direct callers. For anything larger, also trace each changed public function through its callers and its twin.
