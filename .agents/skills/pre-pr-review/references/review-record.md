# Review record

The record is the content under the `## Review record` heading of the PR description, which the PR template provides. It replaces whatever is under that heading, and the heading itself stays as it is. It's written for a human approver who has a minute to decide whether to trust it, so keep it short and concrete.

## Template

```markdown
Reviewed at `<short sha>` by <agent> (<model>) with `pre-pr-review`.
Lenses: <lens> (<reason>); <lens> (<reason>). Skipped: <lens> (<reason>).

- <n> findings raised, <k> kept after challenge.
- Fixed: <claim> (`path.py:LINE`); <claim> (`path.py:LINE`).
- Rejected: <claim> (<reason>).
- Open for the reviewer: <item needing a decision>, or "none".

Review comments:
- <source>: <point> — <disposition>.

Verified: <command> (<result>); <command> (<result>).
Docs consulted: <redis.io URL>; <redis.io URL>.
Not checked: <what wasn't exercised, and why>.
```

Leave out the "Review comments" list until there are comments to respond to, and the "Docs consulted" line when the Redis behaviour lens didn't run.

## Rules

- **Translate, never paste.** One line per finding, in plain words, with a file and line. Leave raw model output out of the record.
- **Rejections stay in.** The rejected findings and their reasons show that the review discriminated. Don't drop them to make the record look tidier.
- **"Verified" lists only commands you ran and whose output you saw,** with the result, for example "31 passed" or "no issues".
- **"Not checked" is mandatory.** If the change touches cluster behaviour, provider APIs, Redis versions below the CI matrix, or anything you reasoned about without running, say so. "Nothing" is a valid answer only for a change that CI fully exercises.
- **The sha names what was reviewed.** Re-run the review after any substantive push. For a push that changes nothing a reviewer would care about, such as a rebase or a typo fix, keep the reviewed sha and add a line: "Later commits: `<sha>` (<what changed>)." Never move the sha to code the review didn't see.

## Example

An illustrative record for a small fix. The paths and counts are invented.

```markdown
Reviewed at `7188c45` by Codex CLI (`<model>`) with `pre-pr-review`.
Lenses: correctness (always); tests (adds integration tests); docs (changes the `messages` docstrings). Skipped: security (no query-building, MCP or destructive paths); design (40-line fix, no API change).

- 9 findings raised, 4 kept after challenge.
- Fixed: semantic history used `FilterQuery`'s default of 10 results (`semantic.py:212`); the new test wrote fewer than 10 messages, so it passed without the fix (`test_message_history.py:88`).
- Rejected: "unbounded read may be slow" (the property is documented as unbounded; `get_recent` is the bounded path).
- Open for the reviewer: none.

Verified: `make check-types` (no issues); `pytest tests/integration/test_message_history.py` (31 passed, Redis 8.4).
Not checked: cluster mode.
```
