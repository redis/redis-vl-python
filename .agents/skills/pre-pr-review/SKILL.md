---
name: pre-pr-review
description: Review a RedisVL branch before its PR is marked ready, fix what the review finds, and write a Review record into the PR description for the human approver. Use before opening or marking a PR ready, after a substantive push, or when responding to Bugbot or reviewer comments on a PR.
compatibility: Works with any coding agent. Needs git, make and pytest; gh is optional and only used to read and update the PR.
---

# Pre-PR review

This review runs before a human sees the PR. Its output is a short Review record in the PR description, telling the approver what was reviewed and what wasn't, so they can check the evidence instead of re-reviewing the change.

The review is only useful if the approver can trust it. Report only what you can tie to the code, fix only what survives challenge, and never claim a check you didn't run.

## 1. Scope the change

- Diff the branch against its merge base with `main`: `git diff $(git merge-base origin/main HEAD)...HEAD`.
- Read each changed file in full, not just the hunks.
- For every changed public function, method or class, read its callers and its sync or async twin (`SearchIndex` and `AsyncSearchIndex`, and so on).
- Note the size (lines added outside `tests/` and `docs/`), the type (fix, feat, refactor, docs, test, chore or CI) and the paths touched. Step 2 uses all three.

## 2. Choose lenses

Pick the lenses this diff needs, and write down the reason for each one you run and each one you skip. Most PRs need two or three.

| Lens | File | Run it when |
|---|---|---|
| Correctness | `references/correctness.md` | Always. A single quick pass for diffs under 100 lines and for chore, CI or refactor changes |
| Security | `references/security.md` | The diff touches `redisvl/mcp/`, `redisvl/query/filter.py`, `redisvl/utils/token_escaper.py`, code that builds query strings or `SCAN` patterns, destructive methods (`delete`, `clear`, drop), `redisvl/redis/connection.py`, `.github/`, or dependencies |
| Design | `references/design.md` | Feature PRs over about 300 lines; new or changed public API or MCP surface; work on deprecated APIs; new vectorizer, reranker or provider modules |
| Tests | `references/tests.md` | Tests are added or changed, or the change sits in an area CI can't exercise (cluster, live provider APIs) |
| Docs | `references/docs.md` | `docs/`, notebooks, docstrings or comments change, or public behaviour changes |

If you're unsure whether a lens applies, run it. A lens you skip is a gap the approver should know about, so the record lists it.

## 3. Review, one lens at a time

Run one pass per chosen lens. Start each pass from the diff and that lens's file, not from an earlier pass's conclusions.

- If your agent can start isolated sub-tasks (subagents, sub-sessions, parallel workers), give each lens its own, and run them in parallel.
- If it can't, run the lenses one after another, and write each pass's findings to a scratch file before starting the next.

Every finding takes the same shape:

```
[lens] path/to/file.py:LINE: claim, in plain words. Failure: the input or state that goes wrong, and what happens. Fix: the smallest change that resolves it.
```

Drop formatting, import order and typing nits. CI and `make lint` already handle them.

## 4. Challenge the findings

Run a separate challenger pass, as described in `references/challenger.md`. It starts from the code and the list of findings, not from the reasoning that produced them, and scores each finding from 1 to 10. Keep findings that score 7 or more and have a file, a line and a concrete failure or cost. Record the rest as rejected, each with its one-line reason.

Run a second challenge round only if the first round's fixes changed the code materially.

## 5. Fix and re-check

- Fix only the kept findings, and only within the PR's scope. A real problem outside scope goes into the record as an open item, not into the diff.
- Fix the root cause. Add the narrowest regression test that fails without the fix, or say why none applies.
- Re-run whatever the fixes touched: `make format`, `make check-types`, and the affected tests (`make test ARGS="tests/unit/test_x.py"` or `pytest` on the specific file).

## 6. Write the Review record

Fill in the template in `references/review-record.md`. It replaces everything under the `## Review record` heading of the PR description, including the template's placeholder comment or any earlier record, and leaves the rest of the description untouched.

With `gh`: fetch the current description (`gh pr view <number> --json body -q .body`), replace only that section, write the whole description to a file and run `gh pr edit <number> --body-file <file>`. `--body-file` replaces the entire description, so never pass it the record alone. Without `gh`, print the record for the author to paste.

## Responding to review

When Bugbot or a person comments on the PR, handle each point like this:

1. **Gather everything open.** That means review threads that are neither resolved nor outdated, review summaries and top-level PR comments. With `gh`, `gh pr view <number> --comments` shows reviews and top-level comments, and `gh api --paginate repos/{owner}/{repo}/pulls/<number>/comments` lists inline comments. Resolution state is only exposed through the GraphQL `reviewThreads` field, so check it there or on GitHub.
2. **Verify each point against the current code.** The reviewer's wording is a claim to check, not a verdict.
3. **Give it exactly one disposition:**
   - `fixed in <sha>`
   - `already fixed`
   - `invalid`, with the evidence
   - `duplicate of <item>`
   - `needs a decision`, for the author or a maintainer to settle
4. **Before you reply "fixed", check the fix.** Re-read the exact file and line in the fixed tree, and search for other instances of the same pattern.
5. **Reply and record it.** Answer each thread with its disposition, then add one line per point to the Review record's "Review comments" list.

Commit fixes locally; the author pushes. Ask the author before you post replies on the PR, unless they've already told you to.
