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

- **Dependency floors and ceilings.** Check every call into a dependency against the oldest version `pyproject.toml` allows and the newest. A new attribute or keyword that only exists in a later release crashes users on the floor version. This is the most common high-severity finding in this repository. The newest allowed release fails differently: redis-py changes defaults between majors, for example which protocol a client negotiates and which keywords a connection pool fills in by itself. For a change that reads client or pool state, or handles reply shapes, run the affected tests on both ends (`uv run --with "redis==<version>" pytest <file>`) and say which versions you ran.
- **Sibling sites.** The same defect in the other places that use the same expression: every provider's constructor, both executors and the planner, each escaper constant. `SKILL.md` step 1 lists them; check that the diff covers each one, or that Notes says why not.
- **Inputs the fix wasn't aimed at.** For each changed condition, try `None`, an empty string, `0`, `False` and an empty collection, and compare the result with `main`. A key that is present with the value `None` is not a missing key, and a truthiness test treats a genuine zero as unset. Check also which error wins when two checks fail: moving a new check ahead of an old one changes the exception callers see.
- **Redis replies.** The code has to handle RESP2 and RESP3 reply shapes, and both hash and JSON storage.
- **Result limits.** Default result counts (`num_results`, `LIMIT`, `MAXSEARCHRESULTS`) silently truncate. Check any read that is meant to return everything.
- **Cluster mode.** Multi-key operations, `SCAN` and pipelines behave differently on a cluster, and partial failures are possible. CI never runs cluster tests, so reason about these cases from the code. If you can't settle one that way, put it under "Not verified" in the Reviewer Brief.
- **Client ownership.** Who creates, owns and closes a Redis client, and what happens on reconnect, especially for async clients reused across event loops.

## Depth

For diffs under 100 lines, and for chore, CI or refactor changes, do a single pass over the diff and its direct callers. For anything larger, also trace each changed public function through its callers and its twin.
